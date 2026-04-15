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

function chipHTML(col, extra=''){
  const meta = `${col.dtype} · uniq=${col.unique_count} · nn=${col.non_null_count}`;
  return `
    <div class="chip ${extra}" data-col="${escapeHtml(col.name)}">
      <span class="mono">${escapeHtml(col.name)}</span>
      <span class="meta mono">${escapeHtml(meta)}</span>
    </div>
  `;
}

function setPicker(el, colName){
  const col = state.columns.find(c => c.name === colName);
  if(!col) return;
  el.innerHTML = chipHTML(col, 'active');
  el.dataset.value = colName;
}

function applyPick(key, colName){
  state.picks[key] = colName;
  const map = { x: $('#xPicker'), y: $('#yPicker') };
  setPicker(map[key], colName);
}

function renderPickers(query=''){
  const ranked = fuzzyRank(state.columns, query).map(x => x.col);
  const topX = [...ranked].sort((a,b)=>scoreColumn(b,'x')-scoreColumn(a,'x')).slice(0,8);
  const topY = [...ranked].sort((a,b)=>scoreColumn(b,'y')-scoreColumn(a,'y')).slice(0,8);

  if(!$('#xPicker').dataset.value){
    $('#xPicker').innerHTML = topX.map(c=>chipHTML(c,'suggest')).join('');
    $('#xPicker').querySelectorAll('.chip').forEach(el=>el.addEventListener('click', ()=>applyPick('x', el.dataset.col)));
  }
  if(!$('#yPicker').dataset.value){
    $('#yPicker').innerHTML = topY.map(c=>chipHTML(c,'suggest')).join('');
    $('#yPicker').querySelectorAll('.chip').forEach(el=>el.addEventListener('click', ()=>applyPick('y', el.dataset.col)));
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

    const rankedX = [...state.columns].sort((a,b)=>scoreColumn(b,'x')-scoreColumn(a,'x'));
    const rankedY = [...state.columns].sort((a,b)=>scoreColumn(b,'y')-scoreColumn(a,'y'));
    if(!state.picks.x && rankedX[0]) applyPick('x', rankedX[0].name);
    if(!state.picks.y && rankedY[0]) applyPick('y', rankedY[0].name);
    renderPickers($('#colSearch').value);
    log(`Preview ok: columns=${state.columns.length}`);
  }catch(e){
    log(`預覽失敗: ${e.message}`);
  }
}

async function runMode(mode){
  try{
    const body = buildRequestBody(mode);
    const endpoint = mode === 'compute' ? '/instability/compute' : mode === 'save' ? '/instability/save' : '/instability/plot';
    log(`${mode.toUpperCase()} 開始：${body.x_col} vs ${body.y_col}`);
    const j = await api(endpoint, {
      method:'POST',
      body: JSON.stringify(body)
    });
    renderResponse(j, mode);
    log(`${mode.toUpperCase()} 完成`);
  }catch(e){
    const msg = `${mode.toUpperCase()} 失敗: ${e.message}`;
    $('#jsonOut').textContent = JSON.stringify({ success:false, message: msg }, null, 2);
    log(msg);
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
  $('#btnCompute').addEventListener('click', ()=>runMode('compute'));
  $('#btnSave').addEventListener('click', ()=>runMode('save'));
  $('#btnPlot').addEventListener('click', ()=>runMode('plot'));
  $('#btnCopyJson').addEventListener('click', async ()=>{
    try{
      await navigator.clipboard.writeText($('#jsonOut').textContent);
      log('JSON copied');
    }catch(e){
      log(`Copy failed: ${e.message}`);
    }
  });
  $('#colSearch').addEventListener('input', ()=>{
    $('#xPicker').dataset.value = state.picks.x || '';
    $('#yPicker').dataset.value = state.picks.y || '';
    if(!state.picks.x) $('#xPicker').innerHTML = '';
    if(!state.picks.y) $('#yPicker').innerHTML = '';
    renderPickers($('#colSearch').value);
    if(state.picks.x) setPicker($('#xPicker'), state.picks.x);
    if(state.picks.y) setPicker($('#yPicker'), state.picks.y);
  });

  ping();
  refreshFiles();
}

init();
