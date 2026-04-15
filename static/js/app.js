/* Scientific Data Lab - app.js
 * - Uses backend FastAPI: /files, /file/preview, /calculate-correlation, /dppc
 * - For interactive scatter points, recommended to add backend endpoint: /data/points (patch below)
 */

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
  sheet: 0,
  preview: null,         // from /file/preview
  columns: [],           // normalized columns info
  focusPicker: null,     // which picker is focused for quick search apply
  picks: { x: null, y: null, x1: null, x2: null },
  micManual: new Set(),
  lastPoints: null,      // points data for plot
};

function log(msg){
  const box = $('#log');
  const t = new Date();
  const stamp = t.toLocaleTimeString();
  const div = document.createElement('div');
  div.className = 'line';
  div.textContent = `[${stamp}] ${msg}`;
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

/* ---------- Column intelligence ---------- */

function isNumericDtype(dtype){
  const d = String(dtype || '').toLowerCase();
  return d.includes('int') || d.includes('float') || d.includes('double') || d.includes('number');
}
function isCategoricalLike(col){
  // heuristic: object/category/bool or low unique rate
  const d = String(col.dtype||'').toLowerCase();
  if(d.includes('bool') || d.includes('object') || d.includes('category')) return true;
  const u = col.unique_count ?? 0;
  const n = col.non_null_count ?? 0;
  if(n > 0 && u > 0 && (u / n) < 0.08) return true;
  return false;
}

function scoreColumn(col, intent){
  // intent: 'x','y','mic'
  // Use non-null, unique, dtype heuristics
  const n = col.non_null_count ?? 0;
  const z = col.null_count ?? 0;
  const u = col.unique_count ?? 0;

  // base: prefer many non-null, penalize too many nulls
  let s = 0;
  s += Math.min(1, n / Math.max(1, n + z)) * 40;

  // unique rate: prefer mid for MIC, high for numeric analysis, avoid near-constant
  const uniqueRate = (n > 0) ? (u / n) : 0;
  if(uniqueRate < 0.002) s -= 40; // almost constant
  else if(uniqueRate < 0.02) s += 4;
  else if(uniqueRate < 0.20) s += 12;
  else if(uniqueRate < 0.80) s += 18;
  else s += 10;

  if(intent === 'y'){
    // Y usually numeric
    s += isNumericDtype(col.dtype) ? 28 : -8;
  }else if(intent === 'x'){
    // X can be numeric or categorical
    s += isNumericDtype(col.dtype) ? 18 : 8;
    if(isCategoricalLike(col)) s += 6;
  }else if(intent === 'mic'){
    // MIC: prefer informative columns; avoid ultra-high-cardinality id-like columns
    s += 16;
    if(uniqueRate > 0.95) s -= 10;
  }

  return s;
}

function normalizeColumns(preview){
  const cols = (preview?.columns || []).map(c => ({
    name: c.name,
    dtype: c.dtype,
    non_null_count: c.non_null_count,
    null_count: c.null_count,
    unique_count: c.unique_count,
    sample_values: c.sample_values || [],
  }));
  return cols;
}

function fuzzyRank(cols, q){
  const query = String(q||'').trim().toLowerCase();
  if(!query) return cols.map(c => ({ col:c, score:0 }));
  return cols.map(c => {
    const nm = String(c.name).toLowerCase();
    let s = 0;
    if(nm === query) s += 1000;
    if(nm.startsWith(query)) s += 180;
    if(nm.includes(query)) s += 90;
    // token bonus
    const parts = query.split(/\s+/).filter(Boolean);
    for(const p of parts){
      if(nm.includes(p)) s += 18;
    }
    return { col:c, score:s };
  }).sort((a,b)=>b.score-a.score);
}

/* ---------- UI render helpers ---------- */

function chipHTML(col, extra=''){
  const sv = (col.sample_values||[]).slice(0,2).join(', ');
  const meta = `${col.dtype} · uniq=${col.unique_count} · nn=${col.non_null_count}`;
  return `
    <div class="chip ${extra}" data-col="${escapeHtml(col.name)}">
      <span class="mono">${escapeHtml(col.name)}</span>
      <span class="meta mono">${escapeHtml(meta)}</span>
    </div>
  `;
}

function listItemHTML(col){
  const sv = (col.sample_values||[]).slice(0,3).join(', ');
  const meta = `${col.dtype} · uniq=${col.unique_count} · null=${col.null_count}`;
  return `
    <div class="list-item" data-col="${escapeHtml(col.name)}">
      <div class="left">
        <div class="name">${escapeHtml(col.name)}</div>
        <div class="sub mono">${escapeHtml(meta)}</div>
      </div>
      <div class="sub mono">${escapeHtml(sv)}</div>
    </div>
  `;
}

function escapeHtml(s){
  return String(s)
    .replaceAll('&','&amp;')
    .replaceAll('<','&lt;')
    .replaceAll('>','&gt;')
    .replaceAll('"','&quot;')
    .replaceAll("'","&#039;");
}

function setPicker(pickerEl, colName){
  const col = state.columns.find(c => c.name === colName);
  if(!col) return;
  pickerEl.innerHTML = chipHTML(col, 'active');
  pickerEl.dataset.value = colName;
}

function bindPickerFocus(pickerEl, key){
  pickerEl.addEventListener('click', () => {
    state.focusPicker = { el: pickerEl, key };
    // subtle focus style
    for(const p of document.querySelectorAll('.picker')) p.style.outline = 'none';
    pickerEl.style.outline = '2px solid rgba(102,227,255,.35)';
    log(`Picker focused: ${key}`);
  });
}

function applyPick(key, colName){
  state.picks[key] = colName;
  const map = { x:$('#xSingle'), y:$('#ySingle'), x1:$('#x1Dual'), x2:$('#x2Dual') };
  setPicker(map[key], colName);
}

/* ---------- App init ---------- */

async function init(){
  $('#apiBase').value = state.apiBase;

  bindPickerFocus($('#xSingle'), 'x');
  bindPickerFocus($('#ySingle'), 'y');
  bindPickerFocus($('#x1Dual'), 'x1');
  bindPickerFocus($('#x2Dual'), 'x2');

  $('#btnSaveApi').addEventListener('click', async ()=>{
    state.apiBase = $('#apiBase').value.trim() || state.apiBase;
    localStorage.setItem('API_BASE', state.apiBase);
    log(`API_BASE saved: ${state.apiBase}`);
    await ping();
    await refreshFiles();
  });

  $('#btnPing').addEventListener('click', ping);
  $('#btnRefreshFiles').addEventListener('click', ()=> refreshFiles(state.currentDir));
  $('#btnBrowse').addEventListener('click', browseSelected);
  $('#btnUpDir').addEventListener('click', browseUp);
  $('#browsePath').addEventListener('keydown', (e)=>{
    if(e.key !== 'Enter') return;
    const raw = String($('#browsePath').value || '').trim();
    refreshFiles(raw);
  });
  $('#fileSelect').addEventListener('change', applyBrowseSelection);
  $('#sheetInput').addEventListener('change', ()=>{
    state.sheet = parseInt($('#sheetInput').value||'0',10) || 0;
  });

  $('#btnPreview').addEventListener('click', previewFile);

  // Tabs
  document.querySelectorAll('.tab').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
      document.querySelectorAll('.tabpane').forEach(p=>p.classList.remove('active'));
      btn.classList.add('active');
      $('#'+btn.dataset.tab).classList.add('active');
    });
  });

  // MIC
  $('#micMode').addEventListener('change', renderMicManual);
  $('#btnMic').addEventListener('click', runMic);
  $('#btnMicPlot').addEventListener('click', runMicPlot);

  // XY
  $('#btnFetchPoints').addEventListener('click', fetchPointsAndPlot);

  // Plot controls
  $('#btnResetView').addEventListener('click', () => {
    Plotly.relayout('plot', { 'scene.camera': null });
    log('Plot view reset');
  });
  $('#btnExportPng').addEventListener('click', async () => {
    try{
      const img = await Plotly.toImage('plot', {format:'png', height: 900, width: 1400});
      const a = document.createElement('a');
      a.href = img;
      a.download = 'scatter.png';
      a.click();
      log('Exported PNG');
    }catch(e){
      log('Export PNG failed: ' + e.message);
    }
  });

  // Detail
  $('#btnCloseDetail').addEventListener('click', ()=> $('#detailCard').classList.remove('show'));

  // Column search (sidebar)
  $('#colSearch').addEventListener('input', ()=>{
    renderSuggestedPickers($('#colSearch').value);
  });

  // Quick search overlay
  setupQuickSearch();

  await ping();
  await refreshFiles();

  // initial plot placeholder
  renderEmptyPlot();
}

async function ping(){
  try{
    const j = await api('/');
    setApiStatus(true);
    log(`API ok: ${j?.message || 'root'}`);
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
      // directory="" means browse root; omit param means backend default dir
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
      const firstFileIndex = entries.findIndex((entry) => entry.type === 'file');
      sel.value = String(firstFileIndex >= 0 ? firstFileIndex : 0);
      applyBrowseSelection();
    }

    renderBrowsePath();
    log(`Browse loaded: ${state.currentDir || '.'} (dirs=${state.directories.length}, files=${state.files.length})`);
  }catch(e){
    log('List files failed: ' + e.message);
  }
}

async function browseSelected(){
  const entry = selectedBrowseEntry();
  if(!entry){
    log('請先選擇項目');
    return;
  }
  if(entry.type === 'dir'){
    await refreshFiles(entry.relative_path);
    return;
  }
  state.selectedFile = entry.path;
  log(`Selected file: ${entry.name}`);
  await previewFile();
}

async function browseUp(){
  try{
    if(state.parentDir === null){
      await refreshFiles('');
      return;
    }
    await refreshFiles(state.parentDir);
  }catch(e){
    log('Browse up failed: ' + e.message);
  }
}

async function previewFile(){
  if(!state.selectedFile){
    log('請先選擇檔案');
    return;
  }
  const sheet = parseInt($('#sheetInput').value||'0',10) || 0;
  try{
    log('Previewing file columns...');
    const j = await api('/file/preview', {
      method:'POST',
      body: JSON.stringify({ filePath: state.selectedFile, sheet, preview_rows: 5 })
    });
    state.preview = j;
    state.columns = normalizeColumns(j);
    log(`Preview ok: ${j.shape?.rows}x${j.shape?.columns}, columns=${state.columns.length}`);

    // Smart defaults: pick best X/Y if empty
    const rankedX = [...state.columns].sort((a,b)=>scoreColumn(b,'x')-scoreColumn(a,'x'));
    const rankedY = [...state.columns].sort((a,b)=>scoreColumn(b,'y')-scoreColumn(a,'y'));

    if(!state.picks.x && rankedX[0]) applyPick('x', rankedX[0].name);
    if(!state.picks.y && rankedY[0]) applyPick('y', rankedY[0].name);

    // dual defaults
    if(!state.picks.x1 && rankedX[0]) applyPick('x1', rankedX[0].name);
    if(!state.picks.x2 && rankedX[1]) applyPick('x2', rankedX[1].name);

    renderSuggestedPickers($('#colSearch').value);
    renderMicManual();
  }catch(e){
    log('Preview failed: ' + e.message);
  }
}

function renderSuggestedPickers(query){
  // For each picker, show top suggestions based on scoring + fuzzy match
  const ranked = fuzzyRank(state.columns, query);
  const cols = ranked.map(x=>x.col);

  const topX = [...cols].sort((a,b)=>scoreColumn(b,'x')-scoreColumn(a,'x')).slice(0,8);
  const topY = [...cols].sort((a,b)=>scoreColumn(b,'y')-scoreColumn(a,'y')).slice(0,8);

  // If picker already has selection, keep it; otherwise show suggestions
  const px = $('#xSingle');
  if(!px.dataset.value){
    px.innerHTML = topX.map(c=>chipHTML(c,'suggest')).join('');
    px.querySelectorAll('.chip').forEach(el=>{
      el.addEventListener('click', ()=> applyPick('x', el.dataset.col));
    });
  }
  const py = $('#ySingle');
  if(!py.dataset.value){
    py.innerHTML = topY.map(c=>chipHTML(c,'suggest')).join('');
    py.querySelectorAll('.chip').forEach(el=>{
      el.addEventListener('click', ()=> applyPick('y', el.dataset.col));
    });
  }

  const p1 = $('#x1Dual');
  if(!p1.dataset.value){
    p1.innerHTML = topX.map(c=>chipHTML(c,'suggest')).join('');
    p1.querySelectorAll('.chip').forEach(el=>{
      el.addEventListener('click', ()=> applyPick('x1', el.dataset.col));
    });
  }
  const p2 = $('#x2Dual');
  if(!p2.dataset.value){
    p2.innerHTML = topX.slice(1).map(c=>chipHTML(c,'suggest')).join('');
    p2.querySelectorAll('.chip').forEach(el=>{
      el.addEventListener('click', ()=> applyPick('x2', el.dataset.col));
    });
  }
}

/* ---------- MIC ---------- */

function micAutoHeaders(limit=30){
  // Auto choose a reasonable subset for MIC when columns are huge:
  // - avoid near-constant, avoid extremely ID-like high-cardinality
  const ranked = [...state.columns].sort((a,b)=>scoreColumn(b,'mic')-scoreColumn(a,'mic'));
  return ranked.slice(0, Math.max(8, Math.min(limit, ranked.length))).map(c=>c.name);
}

function renderMicManual(){
  const box = $('#micManualList');
  box.innerHTML = '';
  if(!state.columns.length){
    box.innerHTML = `<div class="mono small" style="padding:8px;color:var(--faint);">請先按「讀取欄位/預覽」</div>`;
    return;
  }
  const mode = $('#micMode').value;
  if(mode !== 'manual'){
    box.innerHTML = `<div class="mono small" style="padding:8px;color:var(--faint);">Auto 模式不需要手動勾選</div>`;
    return;
  }

  const q = $('#colSearch').value || '';
  const ranked = fuzzyRank(state.columns, q).slice(0, 200).map(x=>x.col); // 防爆量
  for(const c of ranked){
    const item = document.createElement('div');
    item.className = 'list-item';
    item.dataset.col = c.name;
    const checked = state.micManual.has(c.name);
    item.innerHTML = `
      <div class="left">
        <div class="name">${escapeHtml(c.name)}</div>
        <div class="sub mono">${escapeHtml(`${c.dtype} · uniq=${c.unique_count} · nn=${c.non_null_count}`)}</div>
      </div>
      <div class="sub mono">${checked ? '✓ selected' : ''}</div>
    `;
    item.addEventListener('click', ()=>{
      if(state.micManual.has(c.name)) state.micManual.delete(c.name);
      else state.micManual.add(c.name);
      renderMicManual();
    });
    box.appendChild(item);
  }
}

function parseFixedValues(){
  const raw = ($('#fixedJson').value || '').trim();
  if(!raw) return null;
  try{ return JSON.parse(raw); }catch{ return null; }
}

async function runMic(){
  if(!state.selectedFile){ log('請先選檔案'); return; }
  if(!state.columns.length){ log('請先讀取欄位/預覽'); return; }

  const mode = $('#micMode').value;
  const headers = (mode === 'manual')
    ? Array.from(state.micManual)
    : micAutoHeaders(40);

  if(headers.length < 2){
    log('MIC 欄位太少，至少選 2 個');
    return;
  }

  // 你後端 /calculate-correlation 目前接受 CorrelationRequest：
  // {filePath, sheet, method, exp_fd, stamps, ret}
  // 但無 selectHeader，因此建議直接用 /dppc（它有 selectHeader + isAggregate）
  // 這裡先走 /dppc（更符合你需求：欄位很多時可聚合離散以加速）
  const sheet = parseInt($('#sheetInput').value||'0',10) || 0;

  try{
    log(`MIC correlation (dppc) headers=${headers.length} ...`);
    const j = await api('/dppc', {
      method:'POST',
      body: JSON.stringify({
        filePath: state.selectedFile,
        sheet,
        method: 'mic',
        exp_fd: 'tmp/micCorr',
        stamps: ['ui'],
        selectHeader: headers,
        isAggregate: true,
        ret: {}
      })
    });
    log(`MIC done: success=${j.success}`);
    // 這裡 j.res 會是 safe_serialize 過的內容（多為 metadata）
  }catch(e){
    log('MIC failed: ' + e.message);
  }
}

async function runMicPlot(){
  if(!state.selectedFile){ log('請先選檔案'); return; }
  const sheet = parseInt($('#sheetInput').value||'0',10) || 0;
  try{
    log('Plot correlation via backend /plot-correlation ...');
    const j = await api('/plot-correlation', {
      method:'POST',
      body: JSON.stringify({
        filePath: state.selectedFile,
        sheet,
        method: 'mic',
        exp_fd: 'tmp/micCorr',
        stamps: ['ui'],
        ret: {}
      })
    });
    log(`Plot generated: success=${j.success}, output=${j.output_directory}`);
  }catch(e){
    log('Plot failed: ' + e.message);
  }
}

/* ---------- Plot ---------- */

function renderEmptyPlot(){
  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin:{l:40,r:20,t:20,b:40},
    xaxis:{ title:'', gridcolor:'rgba(29,42,85,.35)', zerolinecolor:'rgba(29,42,85,.4)', color:'#9FB2E6' },
    yaxis:{ title:'', gridcolor:'rgba(29,42,85,.35)', zerolinecolor:'rgba(29,42,85,.4)', color:'#9FB2E6' },
    font:{ color:'#EAF0FF' },
    annotations:[{
      text:'載入資料點後才會顯示散點圖（X–Y / 3D）',
      x:0.5, y:0.5, xref:'paper', yref:'paper',
      showarrow:false, font:{size:14, color:'#9FB2E6'}
    }]
  };
  Plotly.newPlot('plot', [], layout, {displayModeBar:false, responsive:true});
}

function showDetail(obj){
  const card = $('#detailCard');
  const body = $('#detailBody');
  body.textContent = JSON.stringify(obj, null, 2);
  card.classList.add('show');
}

function bindPlotEvents(){
  const plot = document.getElementById('plot');
  plot.on('plotly_click', (ev)=>{
    if(!ev?.points?.length) return;
    const p = ev.points[0];
    const custom = p.customdata || {};
    showDetail(custom);
  });
}

/* ---------- Fetch points + plot (requires backend patch) ---------- */

async function fetchPointsAndPlot(){
  if(!state.selectedFile){ log('請先選檔案'); return; }
  if(!state.columns.length){ log('請先讀取欄位/預覽'); return; }

  const sheet = parseInt($('#sheetInput').value||'0',10) || 0;
  const mode = $('#xyMode').value;
  const sampleN = parseInt($('#sampleN').value||'2000',10) || 2000;
  const fixed_values = parseFixedValues();

  try{
    if(mode === '2d'){
      const x = state.picks.x;
      const y = state.picks.y;
      if(!x || !y){ log('請先選 X 與 Y'); return; }

      log(`Fetching 2D points: x=${x}, y=${y}, sample=${sampleN}`);
      const j = await api('/data/points', {   // 需要後端補強（見下方 patch）
        method:'POST',
        body: JSON.stringify({
          filePath: state.selectedFile,
          sheet,
          columns: [x, y],
          fixed_values,
          sample_n: sampleN,
          seed: 7
        })
      });
      state.lastPoints = j;
      plot2d(j, x, y);

    }else{
      const x1 = state.picks.x1;
      const x2 = state.picks.x2;
      const y = state.picks.y;
      if(!x1 || !x2 || !y){ log('請先選 X1 / X2 / Y'); return; }

      log(`Fetching 3D points: x1=${x1}, x2=${x2}, y=${y}, sample=${sampleN}`);
      const j = await api('/data/points', {
        method:'POST',
        body: JSON.stringify({
          filePath: state.selectedFile,
          sheet,
          columns: [x1, x2, y],
          fixed_values,
          sample_n: sampleN,
          seed: 7
        })
      });
      state.lastPoints = j;
      plot3d(j, x1, x2, y);
    }
  }catch(e){
    log('Fetch points failed: ' + e.message);
  }
}

function plot2d(payload, xName, yName){
  const rows = payload.rows || [];
  const xs = rows.map(r => r[xName]);
  const ys = rows.map(r => r[yName]);

  const trace = {
    type:'scattergl',
    mode:'markers',
    x: xs,
    y: ys,
    marker:{ size:7, opacity:0.85 },
    customdata: rows,
    hovertemplate:
      `<b>${escapeHtml(xName)}</b>=%{x}<br>`+
      `<b>${escapeHtml(yName)}</b>=%{y}<br>`+
      `<span class="mono">click for details</span><extra></extra>`
  };

  const layout = {
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',
    margin:{l:55,r:20,t:20,b:55},
    xaxis:{ title:xName, gridcolor:'rgba(29,42,85,.35)', zerolinecolor:'rgba(29,42,85,.4)', color:'#9FB2E6' },
    yaxis:{ title:yName, gridcolor:'rgba(29,42,85,.35)', zerolinecolor:'rgba(29,42,85,.4)', color:'#9FB2E6' },
    font:{ color:'#EAF0FF' }
  };

  Plotly.newPlot('plot', [trace], layout, {displayModeBar:false, responsive:true});
  $('#plotSubtitle').textContent = `2D: ${xName} vs ${yName} · n=${rows.length}`;
  bindPlotEvents();
  log('2D plot rendered');
}

function plot3d(payload, x1Name, x2Name, yName){
  const rows = payload.rows || [];
  const x1 = rows.map(r => r[x1Name]);
  const x2 = rows.map(r => r[x2Name]);
  const y = rows.map(r => r[yName]);

  const trace = {
    type:'scatter3d',
    mode:'markers',
    x: x1,
    y: x2,
    z: y,
    marker:{ size:3, opacity:0.85 },
    customdata: rows,
    hovertemplate:
      `<b>${escapeHtml(x1Name)}</b>=%{x}<br>`+
      `<b>${escapeHtml(x2Name)}</b>=%{y}<br>`+
      `<b>${escapeHtml(yName)}</b>=%{z}<br>`+
      `<span class="mono">click for details</span><extra></extra>`
  };

  const layout = {
    paper_bgcolor:'rgba(0,0,0,0)',
    margin:{l:0,r:0,t:0,b:0},
    scene:{
      xaxis:{ title:x1Name, gridcolor:'rgba(29,42,85,.35)', color:'#9FB2E6' },
      yaxis:{ title:x2Name, gridcolor:'rgba(29,42,85,.35)', color:'#9FB2E6' },
      zaxis:{ title:yName, gridcolor:'rgba(29,42,85,.35)', color:'#9FB2E6' },
      bgcolor:'rgba(0,0,0,0)'
    },
    font:{ color:'#EAF0FF' }
  };

  Plotly.newPlot('plot', [trace], layout, {displayModeBar:false, responsive:true});
  $('#plotSubtitle').textContent = `3D: ${x1Name} vs ${x2Name} vs ${yName} · n=${rows.length}`;
  bindPlotEvents();
  log('3D plot rendered');
}

/* ---------- Quick Search Overlay ---------- */

function setupQuickSearch(){
  const overlay = $('#overlay');
  const input = $('#quickSearch');
  const list = $('#quickList');

  function open(){
    if(!state.columns.length){ log('請先讀取欄位/預覽'); return; }
    overlay.classList.add('show');
    overlay.setAttribute('aria-hidden','false');
    input.value = '';
    renderQuickList('');
    input.focus();
  }
  function close(){
    overlay.classList.remove('show');
    overlay.setAttribute('aria-hidden','true');
  }

  function renderQuickList(q){
    const ranked = fuzzyRank(state.columns, q).slice(0, 80);
    list.innerHTML = ranked.map(({col})=>{
      const meta = `${col.dtype} · uniq=${col.unique_count} · nn=${col.non_null_count}`;
      return `
        <div class="quick-item" data-col="${escapeHtml(col.name)}">
          <div class="nm">${escapeHtml(col.name)}</div>
          <div class="mt mono">${escapeHtml(meta)}</div>
        </div>
      `;
    }).join('');

    list.querySelectorAll('.quick-item').forEach(el=>{
      el.addEventListener('click', ()=>{
        applyFromQuick(el.dataset.col);
        close();
      });
    });
  }

  function applyFromQuick(colName){
    if(!state.focusPicker){
      log('請先點一下 X/Y/X1/X2 的 picker 讓它成為焦點');
      return;
    }
    applyPick(state.focusPicker.key, colName);
    log(`Applied: ${state.focusPicker.key} = ${colName}`);
  }

  document.addEventListener('keydown', (e)=>{
    const isMac = navigator.platform.toLowerCase().includes('mac');
    const mod = isMac ? e.metaKey : e.ctrlKey;
    if(mod && e.key.toLowerCase() === 'k'){
      e.preventDefault();
      open();
    }
    if(e.key === 'Escape' && overlay.classList.contains('show')){
      close();
    }
  });

  overlay.addEventListener('click', (e)=>{
    if(e.target === overlay) close();
  });

  input.addEventListener('input', ()=> renderQuickList(input.value));
  input.addEventListener('keydown', (e)=>{
    if(e.key === 'Enter'){
      // apply first item
      const first = list.querySelector('.quick-item');
      if(first){
        applyFromQuick(first.dataset.col);
        close();
      }
    }
  });
}

init();
