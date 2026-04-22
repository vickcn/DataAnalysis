const $ = (sel) => document.querySelector(sel);

const state = {
  apiBase: localStorage.getItem('API_BASE') || 'http://127.0.0.1:8000',
  files: [],
  directories: [],
  browseEntries: [],
  sourceDir: '',
  currentDir: '',
  parentDir: null,
  selectedFile: null,
  columns: [],
  preview: null,
  picks: { x: null, y: null },
  lastResponse: null,
  plotlySpec: null,
  isRunning: false,
  busyTimer: null,
  busyStartedAt: 0,
  jsonExpanded: false,
  resolvedXCols: [],
  resolvedYCol: null,
};

function escapeHtml(s){
  return String(s ?? '')
    .replaceAll('&','&amp;')
    .replaceAll('<','&lt;')
    .replaceAll('>','&gt;')
    .replaceAll('"','&quot;')
    .replaceAll("'","&#039;");
}

function log(msg){
  const box = $('#log');
  const t = new Date().toLocaleTimeString();
  const div = document.createElement('div');
  div.className = 'line';
  div.textContent = `[${t}] ${msg}`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function setApiStatus(ok){
  const dot = $('#apiStatus');
  dot.classList.remove('ok','bad');
  dot.classList.add(ok ? 'ok' : 'bad');
}

function setActionButtonsDisabled(disabled){
  const ids = ['btnCompute', 'btnSave', 'btnPlot'];
  for(const id of ids){
    const el = document.getElementById(id);
    if(el) el.disabled = !!disabled;
  }
}

function setResultBusy(isBusy, text='計算中...'){
  const mask = $('#resultBusyMask');
  const label = $('#resultBusyText');
  const elapsed = $('#resultBusyElapsed');
  if(!mask) return;
  if(isBusy){
    if(label) label.textContent = text;
    state.busyStartedAt = Date.now();
    if(elapsed) elapsed.textContent = '0.0s';
    if(state.busyTimer){
      clearInterval(state.busyTimer);
      state.busyTimer = null;
    }
    state.busyTimer = setInterval(()=>{
      if(!elapsed) return;
      const sec = Math.max(0, (Date.now() - state.busyStartedAt) / 1000);
      elapsed.textContent = `${sec.toFixed(1)}s`;
    }, 100);
    mask.classList.add('show');
    mask.setAttribute('aria-hidden', 'false');
    return;
  }

  if(state.busyTimer){
    clearInterval(state.busyTimer);
    state.busyTimer = null;
  }
  if(elapsed) elapsed.textContent = '0.0s';
  mask.classList.remove('show');
  mask.setAttribute('aria-hidden', 'true');
}

function setJsonExpanded(expanded){
  state.jsonExpanded = !!expanded;
  const section = $('#jsonSection');
  const btn = $('#btnToggleJson');
  if(section) section.classList.toggle('collapsed', !state.jsonExpanded);
  if(btn) btn.textContent = state.jsonExpanded ? '收合 JSON' : '顯示 JSON';
}

async function api(path, opts={}){
  const url = state.apiBase.replace(/\/+$/,'') + path;
  const r = await fetch(url, {
    headers: { 'Content-Type':'application/json' },
    ...opts
  });
  if(!r.ok){
    const text = await r.text().catch(()=> '');
    throw new Error(`HTTP ${r.status} ${r.statusText} :: ${text}`);
  }
  return r.json();
}

function parseNullableNumber(value){
  const raw = String(value ?? '').trim();
  if(!raw) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

function parseMaybeBool(value){
  if(value === 'auto') return null;
  return value === 'true';
}

function getSelectedLayers(){
  const vals = [...document.querySelectorAll('.layerChk:checked')].map(el => Number(el.value)).filter(Boolean);
  return vals.length ? vals : [1,2,3];
}

function buildExtraPayload(mode){
  const extra = {
    is_discrete: parseMaybeBool($('#isDiscrete').value),
    discrete_threshold: Number($('#discreteThreshold').value || 10),
    shrink_k: Number($('#shrinkK').value || 20),
    model: $('#model').value,
    norm_method: $('#normMethod').value,
    q_low: parseNullableNumber($('#qLow').value),
    q_high: parseNullableNumber($('#qHigh').value),
    c: parseNullableNumber($('#cValue').value),
    tol: parseNullableNumber($('#tol').value),
    invert: $('#invert').value === 'true',
    cv_folds: Number($('#cvFolds').value || 5),
    preview_n: Number($('#previewN').value || 8),
    density_shrink: $('#densityShrink').value === 'true',
    density_bandwidth: parseNullableNumber($('#densityBandwidth').value),
    density_lambda: parseNullableNumber($('#densityLambda').value),
    raw_layers_fn: $('#rawLayersFn').value.trim() || 'raw_with_instability_layers',
  };
  if(mode === 'plot'){
    const figTitle = $('#figTitle').value.trim();
    if(figTitle) extra.fig_title = figTitle;
  }
  return extra;
}

function buildRequestBody(mode){
  if(!state.selectedFile) throw new Error('請先選擇檔案');
  if(!state.picks.x || !state.picks.y) throw new Error('請先選擇 x / y 欄位');

  const body = {
    filePath: state.selectedFile,
    sheet: Number($('#sheetInput').value || 0),
    x_col: state.picks.x,
    y_col: state.picks.y,
    layers: getSelectedLayers(),
    ret: buildExtraPayload(mode),
  };

  const outputDir = $('#outputDir').value.trim();
  if(mode !== 'compute' && outputDir){
    body.output_dir = outputDir;
  }
  return body;
}

function scoreColumn(col, intent){
  const dtype = String(col.dtype || '').toLowerCase();
  const n = col.non_null_count ?? 0;
  const z = col.null_count ?? 0;
  const u = col.unique_count ?? 0;
  const uniqueRate = n > 0 ? u / n : 0;
  let s = Math.min(1, n / Math.max(1, n + z)) * 40;
  if(uniqueRate < 0.002) s -= 40;
  else if(uniqueRate < 0.02) s += 4;
  else if(uniqueRate < 0.2) s += 12;
  else if(uniqueRate < 0.8) s += 18;
  else s += 8;
  if(intent === 'x') s += dtype.includes('object') ? 8 : 18;
  if(intent === 'y') s += dtype.includes('int') || dtype.includes('float') || dtype.includes('double') ? 28 : -6;
  return s;
}

function fuzzyRank(cols, q){
  const query = String(q || '').trim().toLowerCase();
  return cols
    .map(col => {
      const name = String(col.name).toLowerCase();
      let s = 0;
      if(!query) return { col, score: 0 };
      if(name === query) s += 1000;
      if(name.startsWith(query)) s += 180;
      if(name.includes(query)) s += 90;
      for(const token of query.split(/\s+/).filter(Boolean)){
        if(name.includes(token)) s += 18;
      }
      return { col, score: s };
    })
    .sort((a,b)=>b.score-a.score);
}

function colMetaText(col){
  return `${col.dtype} · uniq=${col.unique_count} · nn=${col.non_null_count}`;
}

function colOptionText(col){
  return `${col.name} ｜ ${colMetaText(col)}`;
}

function renderPickMeta(key){
  const metaMap = { x: $('#xMeta'), y: $('#yMeta') };
  const picked = state.columns.find(c => c.name === state.picks[key]);
  if(!metaMap[key]) return;
  if(!picked){
    metaMap[key].textContent = '尚未選擇';
    return;
  }
  metaMap[key].textContent = colMetaText(picked);
}

function applyPick(key, colName){
  state.picks[key] = colName || null;
  renderPickMeta(key);
}

function parseResolvedXCols(value){
  if(Array.isArray(value)){
    return [...new Set(value
      .flatMap(v => String(v ?? '').split(','))
      .map(v => v.trim())
      .filter(Boolean))];
  }
  if(typeof value === 'string'){
    return [...new Set(value.split(',').map(v => v.trim()).filter(Boolean))];
  }
  return [];
}

function parseResolvedYCol(value){
  if(Array.isArray(value)){
    const first = value.map(v => String(v ?? '').trim()).find(Boolean);
    return first || null;
  }
  if(typeof value === 'string'){
    const first = value.split(',').map(v => v.trim()).find(Boolean);
    return first || null;
  }
  return null;
}

function applyResolvedPicksIfPossible(){
  if(!state.columns.length) return false;
  if(!state.resolvedYCol && state.resolvedXCols.length === 0) return false;

  const colSet = new Set(state.columns.map(c => c.name));
  const yCandidate = state.resolvedYCol && colSet.has(state.resolvedYCol)
    ? state.resolvedYCol
    : null;
  const xCandidate = state.resolvedXCols.find(x => colSet.has(x) && x !== yCandidate) || null;

  let changed = false;
  if(yCandidate && state.picks.y !== yCandidate){
    state.picks.y = yCandidate;
    changed = true;
  }
  if(xCandidate && state.picks.x !== xCandidate){
    state.picks.x = xCandidate;
    changed = true;
  }
  if(changed){
    state.resolvedXCols = [];
    state.resolvedYCol = null;
  }
  return changed;
}

function rankColumnsForPicker(intent, query=''){
  const normalizedQuery = String(query || '').trim();
  const byQuery = fuzzyRank(state.columns, normalizedQuery);
  let ranked = [];

  if(normalizedQuery){
    ranked = byQuery
      .filter(item => item.score > 0)
      .sort((a,b)=>{
        if(a.score !== b.score) return b.score - a.score;
        return scoreColumn(b.col, intent) - scoreColumn(a.col, intent);
      })
      .map(item => item.col);
  }else{
    ranked = [...state.columns].sort((a,b)=>scoreColumn(b, intent)-scoreColumn(a, intent));
  }

  const selectedName = state.picks[intent];
  if(selectedName){
    const selectedCol = state.columns.find(c => c.name === selectedName);
    if(selectedCol){
      ranked = [selectedCol, ...ranked.filter(c => c.name !== selectedName)];
    }
  }
  return ranked;
}

function renderPicker(targetEl, intent, cols){
  targetEl.innerHTML = '';
  if(!cols.length){
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '（無可用欄位）';
    targetEl.appendChild(opt);
    targetEl.disabled = true;
    if(state.picks[intent]){
      state.picks[intent] = null;
      renderPickMeta(intent);
    }
    return;
  }

  targetEl.disabled = false;
  for(const col of cols){
    const opt = document.createElement('option');
    opt.value = col.name;
    opt.textContent = colOptionText(col);
    targetEl.appendChild(opt);
  }

  const selected = state.picks[intent];
  const hasSelected = cols.some(c => c.name === selected);
  if(hasSelected){
    targetEl.value = selected;
  }else{
    targetEl.value = cols[0].name;
    state.picks[intent] = cols[0].name;
  }
  renderPickMeta(intent);
}

function renderPickers(query=''){
  const maxItems = 400;
  const rankedX = rankColumnsForPicker('x', query).slice(0, maxItems);
  const rankedY = rankColumnsForPicker('y', query).slice(0, maxItems);
  renderPicker($('#xPicker'), 'x', rankedX);
  renderPicker($('#yPicker'), 'y', rankedY);
}

function ensureValidPicks(){
  const rankedX = [...state.columns].sort((a,b)=>scoreColumn(b,'x')-scoreColumn(a,'x'));
  const rankedY = [...state.columns].sort((a,b)=>scoreColumn(b,'y')-scoreColumn(a,'y'));
  const hasX = rankedX.some(c => c.name === state.picks.x);
  const hasY = rankedY.some(c => c.name === state.picks.y);

  if(!hasX){
    state.picks.x = rankedX[0]?.name || null;
  }
  if(!hasY){
    state.picks.y = rankedY[0]?.name || null;
  }
  if(state.picks.x && state.picks.y && state.picks.x === state.picks.y && rankedY.length > 1){
    const altY = rankedY.find(c => c.name !== state.picks.x);
    if(altY) state.picks.y = altY.name;
  }
}

function renderPreviewTable(preview){
  const wrap = $('#previewTableWrap');
  const data = preview?.preview?.data || {};
  const columns = Object.keys(data);
  const rowCount = preview?.preview?.rows_shown || 0;
  if(!columns.length){
    wrap.className = 'table-wrap empty-hint';
    wrap.textContent = '預覽資料為空';
    return;
  }
  let html = '<table><thead><tr><th>#</th>';
  html += columns.map(c=>`<th>${escapeHtml(c)}</th>`).join('');
  html += '</tr></thead><tbody>';
  for(let i=0;i<rowCount;i++){
    html += `<tr><td class="mono">${i}</td>`;
    html += columns.map(c=>`<td>${escapeHtml(data[c]?.[i] ?? '')}</td>`).join('');
    html += '</tr>';
  }
  html += '</tbody></table>';
  wrap.className = 'table-wrap';
  wrap.innerHTML = html;
}

function renderObjectTable(targetSel, rows){
  const wrap = $(targetSel);
  if(!Array.isArray(rows) || !rows.length){
    wrap.className = 'table-wrap empty-hint';
    wrap.textContent = '尚無資料';
    return;
  }
  const headers = [...new Set(rows.flatMap(r => Object.keys(r || {})))];
  let html = '<table><thead><tr>' + headers.map(h=>`<th>${escapeHtml(h)}</th>`).join('') + '</tr></thead><tbody>';
  for(const row of rows){
    html += '<tr>' + headers.map(h=>`<td>${escapeHtml(row?.[h] ?? '')}</td>`).join('') + '</tr>';
  }
  html += '</tbody></table>';
  wrap.className = 'table-wrap';
  wrap.innerHTML = html;
}

function renderPaths(data){
  const wrap = $('#pathsWrap');
  const items = [];
  const res = data?.res || {};
  if(data?.output_dir) items.push({ tag:'output_dir', value:data.output_dir });
  if(Array.isArray(res.saved_paths)){
    for(const p of res.saved_paths) items.push({ tag:'saved', value:p });
  }
  if(Array.isArray(res.figure_paths)){
    for(const p of res.figure_paths) items.push({ tag:'figure', value:p });
  }
  if(!items.length){
    wrap.className = 'paths-wrap empty-hint';
    wrap.textContent = '尚無輸出路徑';
    return;
  }
  wrap.className = 'paths-wrap';
  wrap.innerHTML = items.map(it => `
    <div class="path-item">
      <div class="path-tag mono">${escapeHtml(it.tag)}</div>
      <div class="mono">${escapeHtml(it.value)}</div>
    </div>
  `).join('');
}

function normalizePlotlySpec(runData){
  const plotly = runData?.res?.plotly;
  if(!plotly || typeof plotly !== 'object') return null;
  if(!Array.isArray(plotly.data) || plotly.data.length === 0) return null;
  return {
    data: plotly.data,
    layout: (plotly.layout && typeof plotly.layout === 'object') ? plotly.layout : {},
    config: (plotly.config && typeof plotly.config === 'object') ? plotly.config : {},
    meta: (plotly.meta && typeof plotly.meta === 'object') ? plotly.meta : {},
  };
}

function clearInteractivePlot(message='尚無互動圖資料'){
  const wrap = $('#plotlyWrap');
  if(!wrap) return;
  if(typeof Plotly !== 'undefined'){
    try{ Plotly.purge(wrap); }catch(e){}
  }
  wrap.className = 'plotly-wrap empty-hint';
  wrap.textContent = message;
  const meta = $('#plotlyMeta');
  if(meta) meta.textContent = message;
  state.plotlySpec = null;
}

// async function renderInteractivePlot(runData){
//   const wrap = $('#plotlyWrap');
//   const meta = $('#plotlyMeta');
//   if(!wrap || !meta) return;
//   if(typeof Plotly === 'undefined'){
//     clearInteractivePlot('Plotly 未載入');
//     return;
//   }

//   const spec = normalizePlotlySpec(runData);
//   if(!spec){
//     clearInteractivePlot('後端未提供 plotly JSON');
//     return;
//   }

//   state.plotlySpec = spec;
//   wrap.className = 'plotly-wrap';
//   wrap.innerHTML = '';
//   await Plotly.newPlot(wrap, state.plotlySpec.data, state.plotlySpec.layout, state.plotlySpec.config);
//   const points = Number(state.plotlySpec.meta?.point_count || 0);
//   const sampled = Number(state.plotlySpec.meta?.sample_n || 0);
//   meta.textContent = points > 0 ? `互動圖資料點: ${points}（sample_n=${sampled || '-'})` : '互動圖已載入';
// }

let heteroPlotResizeTimer = null;
function scheduleHeteroPlotlyResize(){
  const wrap = document.getElementById('plotlyWrap');
  if(!wrap || wrap.classList.contains('empty-hint')) return;
  if(typeof Plotly === 'undefined') return;
  if(heteroPlotResizeTimer) clearTimeout(heteroPlotResizeTimer);
  heteroPlotResizeTimer = setTimeout(()=>{
    heteroPlotResizeTimer = null;
    try{
      Plotly.Plots.resize(wrap);
    }catch(_){}
  }, 120);
}

async function renderInteractivePlot(runData){
  const wrap = $('#plotlyWrap');
  const meta = $('#plotlyMeta');
  if(!wrap || !meta) return;
  if(typeof Plotly === 'undefined'){
    clearInteractivePlot('Plotly 未載入');
    return;
  }

  const spec = normalizePlotlySpec(runData);
  if(!spec){
    clearInteractivePlot('後端未提供 plotly JSON');
    return;
  }

  state.plotlySpec = spec;
  wrap.className = 'plotly-wrap';
  wrap.innerHTML = '';

  const oldLayout = { ...(state.plotlySpec.layout || {}) };
  delete oldLayout.height;
  delete oldLayout.width;
  const oldMargin = oldLayout.margin || {};

  state.plotlySpec.layout = {
    ...oldLayout,
    autosize: true,
    margin: {
      l: Math.max(70, oldMargin.l || 0),
      r: Math.max(56, oldMargin.r || 0),
      t: Math.max(72, oldMargin.t || 0),
      b: Math.max(70, oldMargin.b || 0),
    },
    legend: {
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.02,
      xanchor: 'left',
      x: 0
    }
  };

  state.plotlySpec.config = {
    responsive: true,
    displayModeBar: true,
    scrollZoom: true,
    ...state.plotlySpec.config
  };

  await Plotly.newPlot(
    wrap,
    state.plotlySpec.data,
    state.plotlySpec.layout,
    state.plotlySpec.config
  );

  requestAnimationFrame(() => {
    try{
      Plotly.Plots.resize(wrap);
    }catch(_){}
    scheduleHeteroPlotlyResize();
  });

  const points = Number(state.plotlySpec.meta?.point_count || 0);
  const sampled = Number(state.plotlySpec.meta?.sample_n || 0);
  meta.textContent = points > 0 ? `互動圖資料點: ${points}（sample_n=${sampled || '-'})` : '互動圖已載入';
}

function renderSummary(data, mode){
  const res = data?.res || {};
  const note = Array.isArray(res.notes) && res.notes.length ? res.notes.join(' | ') : (data?.message || '-');
  $('#resultSummary').innerHTML = `
    <div class="metric-box"><div class="metric-label">mode</div><div class="metric-value mono">${escapeHtml(mode)}</div></div>
    <div class="metric-box"><div class="metric-label">x / y</div><div class="metric-value mono">${escapeHtml((data?.x_col || state.picks.x || '-') + ' / ' + (data?.y_col || state.picks.y || '-'))}</div></div>
    <div class="metric-box"><div class="metric-label">layers</div><div class="metric-value mono">${escapeHtml(JSON.stringify(res.layers_resolved || data?.layers || []))}</div></div>
    <div class="metric-box"><div class="metric-label">note</div><div class="metric-value mono">${escapeHtml(note)}</div></div>
  `;
  $('#resultMeta').textContent = `${data?.success ? 'success' : 'failed'} · ${data?.message || ''}`;
}

function renderResponse(data, mode){
  state.lastResponse = data;
  $('#jsonOut').textContent = JSON.stringify(data, null, 2);
  renderSummary(data, mode);
  renderObjectTable('#layersPreviewWrap', data?.res?.layers_table_preview || []);
  renderPaths(data);
}

async function ping(){
  try{
    const j = await api('/config');
    setApiStatus(true);
    log(`API ok: ${j.host_ip}:${j.host_port}`);
  }catch(e){
    setApiStatus(false);
    log(`API ping failed: ${e.message}`);
  }
}

function selectedBrowseEntry(){
  const sel = $('#fileSelect');
  const idx = Number(sel?.value);
  if(!Number.isFinite(idx)) return null;
  return state.browseEntries[idx] || null;
}

function applyBrowseSelection(){
  const entry = selectedBrowseEntry();
  if(!entry){
    state.selectedFile = null;
    return;
  }
  state.selectedFile = entry.type === 'file' ? entry.path : null;
}

function renderBrowsePath(){
  const source = String(state.sourceDir || '');
  const rel = String(state.currentDir || '');
  if(!rel){
    $('#browsePath').value = source;
    return;
  }
  if(!source){
    $('#browsePath').value = rel;
    return;
  }
  const relClean = rel.replace(/^[\\/]+/, '');
  $('#browsePath').value = (source.endsWith('\\') || source.endsWith('/'))
    ? (source + relClean)
    : (source + '\\' + relClean);
}

async function refreshFiles(directory){
  try{
    let query = '';
    if(directory !== undefined){
      const dirParam = String(directory ?? '').trim();
      // directory="" (explicit) means browse root; omit param means default (source_dir)
      query = `?directory=${encodeURIComponent(dirParam)}`;
    }
    const j = await api(`/files${query}`);
    state.files = j.files || [];
    state.directories = j.directories || [];
    state.sourceDir = j.source_dir || state.sourceDir;
    state.currentDir = j.relative_directory || '';
    state.parentDir = (j.parent_relative_directory ?? null);

    const entries = [];
    for(const d of state.directories){
      entries.push({
        type: 'dir',
        name: d.name,
        path: d.path,
        relative_path: d.relative_path
      });
    }
    for(const f of state.files){
      entries.push({
        type: 'file',
        name: f.name,
        path: f.path,
        relative_path: f.relative_path
      });
    }
    state.browseEntries = entries;

    const sel = $('#fileSelect');
    sel.innerHTML = '';
    for(let i = 0; i < entries.length; i++){
      const entry = entries[i];
      const opt = document.createElement('option');
      opt.value = String(i);
      opt.textContent = entry.type === 'dir' ? `[DIR] ${entry.name}` : entry.name;
      sel.appendChild(opt);
    }
    if(entries.length === 0){
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '(empty)';
      sel.appendChild(opt);
      state.selectedFile = null;
    }else{
      const firstFileIndex = entries.findIndex(e => e.type === 'file');
      sel.value = String(firstFileIndex >= 0 ? firstFileIndex : 0);
      applyBrowseSelection();
    }
    renderBrowsePath();
    log(`Browse loaded: ${state.currentDir || '.'} (dirs=${state.directories.length}, files=${state.files.length})`);
  }catch(e){
    log(`瀏覽目錄失敗: ${e.message}`);
  }
}

async function browseSelected(){
  const entry = selectedBrowseEntry();
  if(!entry){
    log('請先選擇資料夾或檔案');
    return;
  }
  if(entry.type === 'dir'){
    await refreshFiles(entry.relative_path);
    return;
  }
  state.selectedFile = entry.path;
  log(`已選擇檔案: ${entry.name}，開始預覽`);
  await previewFile();
}

async function browseUp(){
  try{
    if(state.parentDir === null){
      // At root: keep browsing root
      await refreshFiles('');
      return;
    }
    await refreshFiles(state.parentDir);
  }catch(e){
    log(`回上層失敗: ${e.message}`);
  }
}

async function previewFile(){
  if(!state.selectedFile){ log('請先選擇檔案'); return; }
  try{
    const j = await api('/file/preview', {
      method:'POST',
      body: JSON.stringify({
        filePath: state.selectedFile,
        sheet: Number($('#sheetInput').value || 0),
        preview_rows: Number($('#previewRows').value || 5),
      })
    });
    state.preview = j;
    state.columns = j.columns || [];
    $('#previewMeta').textContent = `${j.file_type} · ${j.shape?.rows || 0} x ${j.shape?.columns || 0}`;
    renderPreviewTable(j);
    ensureValidPicks();
    const autoApplied = applyResolvedPicksIfPossible();
    renderPickers($('#colSearch').value);
    if(autoApplied){
      log(`已依設定帶入 x / y：${state.picks.x} / ${state.picks.y}`);
    }
    clearInteractivePlot('請先執行 Compute / Plot');
    log(`Preview ok: columns=${state.columns.length}`);
  }catch(e){
    log(`預覽失敗: ${e.message}`);
  }
}

async function runMode(mode){
  if(state.isRunning){
    log('已有計算執行中，請稍候');
    return;
  }
  state.isRunning = true;
  setActionButtonsDisabled(true);
  setResultBusy(true, `${mode.toUpperCase()} 計算中...`);
  try{
    const body = buildRequestBody(mode);
    const endpoint = mode === 'compute' ? '/instability/compute' : mode === 'save' ? '/instability/save' : '/instability/plot';
    log(`${mode.toUpperCase()} 開始：${body.x_col} vs ${body.y_col}`);
    const j = await api(endpoint, {
      method:'POST',
      body: JSON.stringify(body)
    });
    renderResponse(j, mode);
    try{
      await renderInteractivePlot(j);
    }catch(plotErr){
      clearInteractivePlot(`互動圖載入失敗: ${plotErr.message}`);
      log(`Plotly render failed: ${plotErr.message}`);
    }
    log(`${mode.toUpperCase()} 完成`);
  }catch(e){
    const msg = `${mode.toUpperCase()} 失敗: ${e.message}`;
    $('#jsonOut').textContent = JSON.stringify({ success:false, message: msg }, null, 2);
    clearInteractivePlot(msg);
    log(msg);
  }finally{
    state.isRunning = false;
    setActionButtonsDisabled(false);
    setResultBusy(false);
  }
}

async function tryResolveXyFromConfig(){
  try{
    const base = (state.apiBase || '').replace(/\/+$/, '');
    const params = new URLSearchParams(window.location.search || '');
    const query = new URLSearchParams();
    for(const key of ['config_path', 'default_data_path', 'model_config_path']){
      const value = String(params.get(key) || '').trim();
      if(value) query.set(key, value);
    }
    const url = query.size
      ? `${base}/config/resolve-xy?${query.toString()}`
      : `${base}/config/resolve-xy`;
    const r = await fetch(url);
    if(!r.ok) return;
    const j = await r.json();
    if(!j || j.success === false || j.implemented === false) return;

    state.resolvedXCols = parseResolvedXCols(j.x_cols);
    state.resolvedYCol = parseResolvedYCol(j.y_col);

    if(!state.selectedFile && typeof j.input_file === 'string' && j.input_file.trim()){
      state.selectedFile = j.input_file.trim();
    }

    if(applyResolvedPicksIfPossible()){
      renderPickers($('#colSearch').value);
      log(`已依設定帶入 x / y：${state.picks.x} / ${state.picks.y}`);
    }
  }catch(_){
    /* 保持相容：無 API 或錯誤時略過 */
  }
}

function init(){
  $('#apiBase').value = state.apiBase;
  $('#btnSaveApi').addEventListener('click', async ()=>{
    state.apiBase = $('#apiBase').value.trim() || state.apiBase;
    localStorage.setItem('API_BASE', state.apiBase);
    log(`API_BASE saved: ${state.apiBase}`);
    await ping();
    await refreshFiles();
  });
  $('#btnPing').addEventListener('click', ping);
  $('#btnRefreshFiles').addEventListener('click', ()=>refreshFiles(state.currentDir));
  $('#btnBrowse').addEventListener('click', browseSelected);
  $('#btnUpDir').addEventListener('click', browseUp);
  $('#browsePath').addEventListener('keydown', (e)=>{
    if(e.key !== 'Enter') return;
    const raw = String($('#browsePath').value || '').trim();
    refreshFiles(raw);
  });
  $('#fileSelect').addEventListener('change', applyBrowseSelection);
  $('#btnPreview').addEventListener('click', previewFile);
  $('#xPicker').addEventListener('change', (e)=>applyPick('x', e.target.value));
  $('#yPicker').addEventListener('change', (e)=>applyPick('y', e.target.value));
  $('#btnCompute').addEventListener('click', ()=>runMode('compute'));
  $('#btnSave').addEventListener('click', ()=>runMode('save'));
  $('#btnPlot').addEventListener('click', ()=>runMode('plot'));
  $('#btnToggleJson').addEventListener('click', ()=>setJsonExpanded(!state.jsonExpanded));
  $('#btnCopyJson').addEventListener('click', async ()=>{
    try{
      await navigator.clipboard.writeText($('#jsonOut').textContent);
      log('JSON copied');
    }catch(e){
      log(`Copy failed: ${e.message}`);
    }
  });
  $('#colSearch').addEventListener('input', ()=>{
    renderPickers($('#colSearch').value);
  });

  setJsonExpanded(false);
  clearInteractivePlot('請先讀取欄位並執行 Compute / Plot');
  if(!window.__heteroPlotResizeBound){
    window.__heteroPlotResizeBound = true;
    window.addEventListener('resize', scheduleHeteroPlotlyResize);
  }
  ping();
  refreshFiles();
  tryResolveXyFromConfig();
}

init();
