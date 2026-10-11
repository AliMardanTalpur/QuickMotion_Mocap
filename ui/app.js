const app = document.getElementById('app');
const fileInput = document.getElementById('video-file');
const toastNode = document.getElementById('toast');
const profileButton = document.getElementById('profile-button');
const profilePopover = document.getElementById('profile-popover');

const state = {
  jobs: [], system: null, uploadPercent: null, uploadError: '',
  qualityMode: true, viewer: null, refreshKey: '',
};

const edges = [
  [0, 3], [3, 6], [6, 9], [9, 12], [12, 15],
  [0, 1], [1, 4], [4, 7], [7, 10],
  [0, 2], [2, 5], [5, 8], [8, 11],
  [9, 13], [13, 16], [16, 18], [18, 20],
  [9, 14], [14, 17], [17, 19], [19, 21],
];

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString();
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

function showToast(message) {
  toastNode.textContent = message;
  toastNode.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toastNode.hidden = true; }, 4000);
}

async function api(path, options = {}) {
  const response = await fetch(path, { cache: 'no-store', ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

async function loadData() {
  const [jobs, system] = await Promise.all([
    api('/api/jobs'), api('/api/system'),
  ]);
  state.jobs = jobs.jobs;
  state.system = system;
}

function currentPath() { return decodeURI(location.pathname); }

function navigate(path) {
  if (currentPath() !== path) history.pushState({}, '', path);
  state.viewer = null;
  render();
  window.scrollTo(0, 0);
}

function setActiveNav(path) {
  let key = 'home';
  if (path.startsWith('/assets') || path.startsWith('/asset/')) key = 'assets';
  else if (path.startsWith('/capture')) key = 'capture';
  else if (path.startsWith('/share')) key = 'share';
  else if (path.startsWith('/docs')) key = 'docs';
  document.querySelectorAll('[data-nav]').forEach(link => {
    link.classList.toggle('active', link.dataset.nav === key);
    if (link.dataset.nav === key) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
}

function fileUrl(job, kind) {
  return `/api/jobs/${encodeURIComponent(job.id)}/files/${kind}`;
}

function assetCard(job) {
  return `<a class="asset-card" href="/asset/${encodeURIComponent(job.id)}" data-route="/asset/${encodeURIComponent(job.id)}" aria-label="Open ${esc(job.name)}">
    <span class="asset-card-image"><img src="${fileUrl(job, 'thumbnail')}" alt="" loading="lazy"><span class="status-pill ${esc(job.status)}">${esc(job.status)}</span></span>
    <span class="asset-card-footer"><strong title="${esc(job.name)}">${esc(job.name)}</strong><small>${esc(formatDate(job.created))}</small></span>
  </a>`;
}

function emptyCard(index) {
  if (index === 0) {
    return `<a class="asset-card asset-placeholder" href="/capture" data-route="/capture" aria-label="Create a new capture">
      <span class="asset-card-image" style="display:grid;place-items:center"><span class="plus">+</span></span>
      <span class="asset-card-footer"><strong>New capture</strong><small>Upload a video</small></span>
    </a>`;
  }
  return `<div class="asset-card asset-placeholder" aria-hidden="true"><span class="asset-card-image"></span><span class="asset-card-footer"></span></div>`;
}

function renderHome() {
  const recent = state.jobs.slice(0, 8);
  const cards = recent.map(assetCard);
  while (cards.length < 8) cards.push(emptyCard(cards.length === 0 ? 0 : cards.length));
  app.innerHTML = `<section class="home-page"><h1 class="page-title">Recent assets</h1><div class="asset-grid">${cards.join('')}</div></section>`;
}

function renderAssets() {
  const cutoff = Date.now() - 30 * 24 * 3600 * 1000;
  const recent = state.jobs.filter(job => new Date(job.created).getTime() >= cutoff);
  const old = state.jobs.filter(job => new Date(job.created).getTime() < cutoff);
  const grid = (jobs, empty) => jobs.length
    ? `<div class="asset-grid">${jobs.map(assetCard).join('')}</div>`
    : `<div class="empty-section">${empty}</div>`;
  app.innerHTML = `<section class="assets-page">
    <h1 class="page-title">Your Assets</h1>
    <p class="page-subtitle">View and manage your motion capture assets</p>
    <section class="assets-section"><h2 class="section-heading">New</h2>${grid(recent, 'No recent assets yet. Upload a video to create one.')}</section>
    <section class="assets-section"><h2 class="section-heading">Old</h2>${grid(old, 'Older assets will appear here after 30 days.')}</section>
  </section>`;
}

function renderCapture() {
  const system = state.system;
  const note = system?.ready
    ? 'GVHMR is ready. Processing runs locally with one model at a time on CPU.'
    : `Setup needed: ${esc(system?.missing?.join('; ') || 'Checking models…')}`;
  app.innerHTML = `<section class="capture-page"><div class="capture-card">
    <h1>Single Camera: motion capture from video</h1><div class="capture-rule"></div>
    <div id="drop-zone" class="drop-zone" role="button" tabindex="0" aria-label="Upload or drop a video">
      <span class="upload-cloud" aria-hidden="true">☁</span>
      <strong>Click to upload or drop a video</strong>
      <small>MP4, AVI or MOV up to 1080p 60fps</small>
    </div>
    <div class="capture-options"><label><input id="quality-mode" type="checkbox" ${state.qualityMode ? 'checked' : ''}> Higher quality mirrored pass</label>
      <span class="system-note ${system?.ready ? '' : 'error'}">${note}</span></div>
    ${state.uploadPercent !== null ? `<div class="upload-progress"><p>Uploading video · ${state.uploadPercent}%</p><div class="progress-track"><span class="progress-fill" style="width:${state.uploadPercent}%"></span></div></div>` : ''}
    ${state.uploadError ? `<p class="system-note error" role="alert">${esc(state.uploadError)}</p>` : ''}
  </div></section>`;
  const zone = document.getElementById('drop-zone');
  zone.addEventListener('click', () => fileInput.click());
  zone.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); }
  });
  for (const type of ['dragenter', 'dragover']) zone.addEventListener(type, event => {
    event.preventDefault(); zone.classList.add('dragging');
  });
  for (const type of ['dragleave', 'drop']) zone.addEventListener(type, event => {
    event.preventDefault(); zone.classList.remove('dragging');
  });
  zone.addEventListener('drop', event => {
    const file = event.dataTransfer?.files?.[0];
    if (file) startUpload(file);
  });
  document.getElementById('quality-mode').addEventListener('change', event => {
    state.qualityMode = event.target.checked;
  });
}

function detailActions(job) {
  const isComplete = job.status === 'complete';
  return `<div class="detail-tools">
    ${isComplete ? `
      <a class="action-button primary" href="${fileUrl(job, 'fbx')}">Download FBX</a>
      <a class="action-button" href="${fileUrl(job, 'bvh')}">Download BVH</a>
      <a class="action-button" href="${fileUrl(job, 'bundle')}">Download share package</a>
      <a class="action-button" href="${fileUrl(job, 'report')}">Quality report</a>
      <button class="action-button" data-action="copy-link" data-id="${job.id}">Copy local link</button>` : ''}
    ${['failed', 'interrupted'].includes(job.status) ? `<button class="action-button primary" data-action="retry" data-id="${job.id}">Retry processing</button>` : ''}
    ${!job.demo && job.status !== 'running' ? `<button class="action-button danger" data-action="delete" data-id="${job.id}">Delete asset</button>` : ''}
  </div>`;
}

function renderDetail(job) {
  const complete = job.status === 'complete';
  const time = formatDate(job.created);
  const frameCount = Math.round((Number(job.fps) || 0) * (Number(job.duration) || 0));
  const stageNumber = {
    'Recovering 3D motion': 1, 'Refining limbs': 2,
    'Preparing previews': 3, 'Checking video alignment': 4,
    'Exporting FBX and BVH': 5,
  }[job.stage];
  const recoveryNote = job.stage === 'Recovering 3D motion'
    ? `Recovering ${frameCount || 'all'} frames at ${job.width || '?'}×${job.height || '?'} on CPU. YOLO tracking, ViTPose, HMR2, and GVHMR run in sequence. Full HD clips can take tens of minutes; this stage's bar shows activity, not a frame percentage. Keep the PowerShell window open.`
    : `${stageNumber ? `Stage ${stageNumber} of 5. ` : ''}Processing is sequential to fit this computer’s GPU memory.`;
  app.innerHTML = `<section class="detail-page">
    <div class="detail-heading"><button class="back-button" data-action="back" aria-label="Back to assets">‹</button><h1>Motion Capture Asset</h1></div>
    <div class="detail-meta">
      <div class="meta-card"><small>Model</small><span class="model-badge">${esc(job.model || 'GVHMR · single camera')}</span></div>
      <div class="meta-card"><small>${complete ? 'Created' : 'Uploaded'}</small><strong>${esc(time)}</strong></div>
      <div class="meta-card"><small>Asset ID</small><strong>${esc(job.id)} <button class="copy-id" data-action="copy-id" data-id="${job.id}" aria-label="Copy asset ID">⧉</button></strong></div>
    </div>
    <div class="viewer-panel">
      <div class="video-panel"><video controls preload="metadata" playsinline poster="${fileUrl(job, 'thumbnail')}" src="${fileUrl(job, 'playback')}"></video>
        <div class="video-fallback" hidden><img src="${fileUrl(job, 'thumbnail')}" alt="Video thumbnail"><div>This browser cannot play this video's codec.<br><a href="${fileUrl(job, 'video')}" download>Download source video</a></div></div>
        <div class="panel-label"><strong>Source video</strong><small>${esc(job.name)} · ${esc(formatTime(job.duration))}</small></div></div>
      <div class="skeleton-panel">${complete
        ? `<canvas id="skeleton-canvas" aria-label="Rotatable 3D skeleton preview"></canvas><div class="skeleton-controls"><button id="skeleton-play" title="Play or pause" aria-label="Play or pause">▶</button><input id="skeleton-frame" type="range" min="0" max="0" value="0" aria-label="Animation frame"><small id="skeleton-time">0:00</small></div>`
        : `<div class="panel-unavailable">${esc(job.stage || 'Waiting to process')}<br>${stageNumber ? `Stage ${stageNumber} of 5` : esc(job.status)}</div>`}
        <div class="panel-label"><strong>3D skeleton</strong><small>${complete ? 'Drag to rotate' : esc(job.status)}</small></div>
      </div>
    </div>
    ${detailActions(job)}
    ${complete ? `<section class="status-panel"><h2>Review the motion</h2><p>Compare the recovered 3D joints with the source video. The preview is a diagnostic contact sheet; the FBX and BVH contain the animation.</p><img class="preview-image" src="${fileUrl(job, 'overlay')}" alt="Video frames with 3D skeleton overlay" loading="lazy"></section>` : ''}
    ${!complete ? `<section class="status-panel"><h2>${esc(job.stage)}</h2><p>${job.status === 'failed' ? esc(job.error || 'Processing failed.') : esc(recoveryNote)}</p><div class="progress-track"><span class="progress-fill ${job.stage === 'Recovering 3D motion' ? 'indeterminate' : ''}" style="width:${Number(job.progress) || 0}%"></span></div><h3>Job log</h3><pre class="log-window">${esc((job.logs || []).slice(-35).join('\n'))}</pre></section>` : ''}
  </section>`;
  const video = app.querySelector('.video-panel video');
  video.addEventListener('error', () => {
    video.hidden = true;
    app.querySelector('.video-fallback').hidden = false;
  });
  if (complete) setupViewer(job);
}

function renderShare() {
  const complete = state.jobs.filter(job => job.status === 'complete');
  app.innerHTML = `<section class="utility-page"><h1 class="page-title">Share</h1><p class="page-subtitle">Package and share your completed captures</p>
    <div class="utility-panel"><h2>Completed assets</h2><p>Download a package containing the source video, FBX, BVH, preview, and quality report. Local links work on this computer while QuickMotion is running.</p>
      ${complete.length ? complete.map(job => `<div class="share-row"><div><strong>${esc(job.name)}</strong><small>${esc(formatDate(job.created))}</small></div><div class="share-row-actions"><a class="action-button" href="${fileUrl(job, 'bundle')}">Download package</a><button class="action-button" data-action="copy-link" data-id="${job.id}">Copy local link</button></div></div>`).join('') : '<p>No completed assets yet.</p>'}
    </div></section>`;
}

function renderDocs() {
  app.innerHTML = `<section class="utility-page"><h1 class="page-title">Docs</h1><p class="page-subtitle">QuickMotion single-camera capture</p>
    <div class="utility-panel"><h2>How to capture motion</h2><ol class="steps"><li>Open <a href="/capture" data-route="/capture">Single Cam</a> or select Upload.</li><li>Choose an MP4, AVI, or MOV video up to 1080p and 60 FPS.</li><li>Wait for 3D recovery, limb refinement, previews, and export to finish.</li><li>Open the asset, compare the source and 3D preview, then download FBX or BVH.</li></ol><p>Processing runs locally. Only one GVHMR job executes at a time; its models run sequentially on CPU to fit the 4 GB RTX 2050.</p></div>
    <div class="utility-panel"><h2>Formats and limitations</h2><p>FBX and BVH contain an animated skeleton. Import FBX in Blender through File → Import → FBX. A single camera cannot fully establish depth or hidden limbs, so inspect the animation from several angles.</p><p>New jobs are stored in <code>runs/quickmotion/</code>. The existing dance output appears as a sample asset when present.</p></div>
    <div class="utility-panel"><h2>Current setup</h2><p class="system-note ${state.system?.ready ? '' : 'error'}">${state.system?.ready ? 'Ready to process videos.' : `Missing: ${esc(state.system?.missing?.join('; ') || 'Checking…')}`}</p></div>
  </section>`;
}

function render() {
  const path = currentPath();
  setActiveNav(path);
  if (path === '/' || path === '/home') renderHome();
  else if (path === '/assets') renderAssets();
  else if (path === '/capture') renderCapture();
  else if (path === '/share') renderShare();
  else if (path === '/docs') renderDocs();
  else if (path.startsWith('/asset/')) {
    const id = path.split('/')[2];
    const job = state.jobs.find(item => item.id === id);
    if (job) renderDetail(job);
    else app.innerHTML = '<section class="utility-page"><h1>Asset not found</h1><p><a href="/assets" data-route="/assets">Return to assets</a></p></section>';
  } else navigate('/');
}

function startUpload(file) {
  if (!file) return;
  if (!/\.(mp4|avi|mov)$/i.test(file.name)) { state.uploadError = 'Choose an MP4, AVI, or MOV video.'; renderCapture(); return; }
  if (file.size > 2 * 1024 ** 3) { state.uploadError = 'The maximum video size is 2 GB.'; renderCapture(); return; }
  if (currentPath() !== '/capture') navigate('/capture');
  state.uploadError = '';
  state.uploadPercent = 0;
  renderCapture();
  const request = new XMLHttpRequest();
  request.open('POST', `/api/jobs?quality=${state.qualityMode ? 1 : 0}`);
  request.setRequestHeader('X-Filename', encodeURIComponent(file.name));
  request.setRequestHeader('Content-Type', 'application/octet-stream');
  request.upload.onprogress = event => {
    if (event.lengthComputable) {
      state.uploadPercent = Math.round(event.loaded / event.total * 100);
      const text = document.querySelector('.upload-progress p');
      const fill = document.querySelector('.upload-progress .progress-fill');
      if (text) text.textContent = `Uploading video · ${state.uploadPercent}%`;
      if (fill) fill.style.width = `${state.uploadPercent}%`;
    }
  };
  request.onload = async () => {
    state.uploadPercent = null;
    let response = {};
    try { response = JSON.parse(request.responseText); } catch (_) { /* server reply */ }
    if (request.status >= 200 && request.status < 300) {
      showToast('Video uploaded. Motion recovery is queued.');
      await refreshData();
      navigate(`/asset/${response.job.id}`);
    } else {
      state.uploadError = response.error || `Upload failed (${request.status})`;
      renderCapture();
    }
  };
  request.onerror = () => { state.uploadPercent = null; state.uploadError = 'Upload connection failed.'; renderCapture(); };
  request.send(file);
}

async function refreshData() {
  const response = await api('/api/jobs');
  state.jobs = response.jobs;
}

async function act(action, id) {
  try {
    if (action === 'back') { navigate('/assets'); return; }
    if (action === 'copy-id') { await navigator.clipboard.writeText(id); showToast('Asset ID copied.'); return; }
    if (action === 'copy-link') { await navigator.clipboard.writeText(`${location.origin}/asset/${id}`); showToast('Local asset link copied.'); return; }
    if (action === 'retry') {
      await api(`/api/jobs/${id}/retry`, { method: 'POST' });
      await refreshData(); render(); showToast('Processing queued again.'); return;
    }
    if (action === 'delete') {
      if (!confirm('Delete this asset and its files?')) return;
      await api(`/api/jobs/${id}`, { method: 'DELETE' });
      await refreshData(); navigate('/assets'); showToast('Asset deleted.');
    }
  } catch (error) { showToast(error.message); }
}

function setupViewer(job) {
  const canvas = document.getElementById('skeleton-canvas');
  if (!canvas) return;
  const viewer = { canvas, frames: [], fps: 30, frame: 0, yaw: .45, pitch: .2, zoom: 1,
    playing: true, drag: false, lastX: 0, lastY: 0, lastTick: 0 };
  state.viewer = viewer;
  fetch(`/api/jobs/${encodeURIComponent(job.id)}/points`).then(response => response.json()).then(data => {
    if (state.viewer !== viewer) return;
    viewer.frames = data.frames || [];
    viewer.fps = data.fps || 30;
    const slider = document.getElementById('skeleton-frame');
    if (slider) slider.max = Math.max(0, viewer.frames.length - 1);
  }).catch(() => showToast('3D preview is unavailable.'));
  const button = document.getElementById('skeleton-play');
  const slider = document.getElementById('skeleton-frame');
  button.addEventListener('click', () => {
    viewer.playing = !viewer.playing;
    button.textContent = viewer.playing ? '❚❚' : '▶';
  });
  slider.addEventListener('input', event => { viewer.frame = Number(event.target.value); viewer.playing = false; button.textContent = '▶'; });
  canvas.addEventListener('pointerdown', event => { viewer.drag = true; viewer.lastX = event.clientX; viewer.lastY = event.clientY; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener('pointerup', () => { viewer.drag = false; });
  canvas.addEventListener('pointermove', event => {
    if (!viewer.drag) return;
    viewer.yaw += (event.clientX - viewer.lastX) * .01;
    viewer.pitch = Math.max(-.8, Math.min(.8, viewer.pitch + (event.clientY - viewer.lastY) * .007));
    viewer.lastX = event.clientX; viewer.lastY = event.clientY;
  });
  canvas.addEventListener('wheel', event => {
    event.preventDefault(); viewer.zoom = Math.max(.5, Math.min(2.8, viewer.zoom * (event.deltaY > 0 ? .9 : 1.1)));
  }, { passive: false });
  requestAnimationFrame(now => drawViewer(viewer, now));
}

function drawViewer(viewer, now) {
  if (state.viewer !== viewer || !viewer.canvas.isConnected) return;
  const canvas = viewer.canvas;
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = '#1e1e1e'; ctx.fillRect(0, 0, rect.width, rect.height);
  if (!viewer.frames.length) {
    ctx.fillStyle = '#a7a7a7'; ctx.font = '15px Arial'; ctx.textAlign = 'center';
    ctx.fillText('Loading 3D skeleton…', rect.width / 2, rect.height / 2);
    requestAnimationFrame(next => drawViewer(viewer, next));
    return;
  }
  if (viewer.playing && viewer.lastTick) viewer.frame = (viewer.frame + (now - viewer.lastTick) * viewer.fps / 1000) % viewer.frames.length;
  viewer.lastTick = now;
  const index = Math.floor(viewer.frame);
  const frame = viewer.frames[index];
  const pelvis = frame[0];
  const scale = Math.min(rect.width, rect.height) * .39 * viewer.zoom;
  const project = point => {
    const x = point[0] - pelvis[0], y = point[1] - pelvis[1], z = point[2] - .85;
    const rx = x * Math.cos(viewer.yaw) - y * Math.sin(viewer.yaw);
    const ry = x * Math.sin(viewer.yaw) + y * Math.cos(viewer.yaw);
    const rz = z * Math.cos(viewer.pitch) - ry * Math.sin(viewer.pitch);
    return [rect.width / 2 + rx * scale, rect.height / 2 - rz * scale];
  };
  // Subtle floor and axes make the third dimension readable while orbiting.
  ctx.strokeStyle = '#353535'; ctx.lineWidth = 1;
  for (let line = -3; line <= 3; line++) {
    const a = project([pelvis[0] + line * .4, pelvis[1] - 1.2, 0]);
    const b = project([pelvis[0] + line * .4, pelvis[1] + 1.2, 0]);
    const c = project([pelvis[0] - 1.2, pelvis[1] + line * .4, 0]);
    const d = project([pelvis[0] + 1.2, pelvis[1] + line * .4, 0]);
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.moveTo(...c); ctx.lineTo(...d); ctx.stroke();
  }
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  for (const [a, b] of edges) {
    const start = project(frame[a]), end = project(frame[b]);
    ctx.strokeStyle = '#68caff'; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(...start); ctx.lineTo(...end); ctx.stroke();
  }
  for (let j = 0; j < 22; j++) {
    const [x, y] = project(frame[j]);
    ctx.fillStyle = [10, 11, 20, 21].includes(j) ? '#ffe15d' : '#f3f8fa';
    ctx.beginPath(); ctx.arc(x, y, 4.2, 0, Math.PI * 2); ctx.fill();
  }
  const slider = document.getElementById('skeleton-frame');
  const time = document.getElementById('skeleton-time');
  if (slider) slider.value = index;
  if (time) time.textContent = formatTime(index / viewer.fps);
  requestAnimationFrame(next => drawViewer(viewer, next));
}

document.addEventListener('click', event => {
  const link = event.target.closest('[data-route]');
  if (link) { event.preventDefault(); navigate(link.dataset.route); return; }
  const action = event.target.closest('[data-action]');
  if (action) act(action.dataset.action, action.dataset.id);
});

document.getElementById('header-upload').addEventListener('click', () => {
  navigate('/capture'); fileInput.click();
});

profileButton.addEventListener('click', () => {
  const open = profilePopover.hidden;
  profilePopover.hidden = !open;
  profileButton.setAttribute('aria-expanded', String(open));
  profilePopover.innerHTML = `<strong>QuickMotion · local</strong><p>${state.system?.ready ? 'Ready to process videos' : 'Setup needed'}</p><p>${esc(state.system?.device || '')}</p>${state.system?.missing?.length ? `<p>${esc(state.system.missing.join('; '))}</p>` : ''}`;
});

fileInput.addEventListener('change', event => {
  const file = event.target.files?.[0];
  if (file) startUpload(file);
  fileInput.value = '';
});

window.addEventListener('popstate', () => { state.viewer = null; render(); });

setInterval(async () => {
  try {
    const before = state.jobs.map(job => `${job.id}:${job.status}:${job.stage}:${job.progress}`).join('|');
    await refreshData();
    const after = state.jobs.map(job => `${job.id}:${job.status}:${job.stage}:${job.progress}`).join('|');
    if (before !== after) render();
    else if (currentPath().startsWith('/asset/')) {
      const id = currentPath().split('/')[2];
      const job = state.jobs.find(item => item.id === id);
      const logs = document.querySelector('.log-window');
      if (logs && job) logs.textContent = (job.logs || []).slice(-35).join('\n');
    }
  } catch (_) { /* Show stale data until the server returns. */ }
}, 2500);

loadData().then(render).catch(error => {
  app.innerHTML = `<section class="utility-page"><h1>QuickMotion is unavailable</h1><p>${esc(error.message)}</p><p>Restart the Python UI server and refresh this page.</p></section>`;
});
