const user = requireLogin('contractor');
renderNav('contractor');

const slot = document.getElementById('jobs-slot');
let capturedPosition = {}; // jobId -> {lat, lon}
let photoFreshness = {}; // jobId -> {ok, message}

// Same camera-only freshness heuristic used on the citizen report form — see
// report.js for the full explanation. Enforced again server-side via EXIF.
const PHOTO_FRESHNESS_MAX_AGE_MS = 20 * 60 * 1000;
function checkFileFreshness(file) {
  if (!file.lastModified) return { ok: true };
  const age = Date.now() - file.lastModified;
  if (age > PHOTO_FRESHNESS_MAX_AGE_MS) {
    return {
      ok: false,
      message: 'This file looks older than a few minutes — please take a fresh photo with your camera right now at the repair site. Photos from your gallery will be flagged and are likely to be rejected.',
    };
  }
  return { ok: true };
}

async function loadJobs() {
  try {
    const data = await API.get('/api/contractor/jobs');
    renderJobs(data.jobs || []);
  } catch (e) {
    slot.innerHTML = `<div class="banner error">${escapeHTML(e.message)}</div>`;
  }
}

function renderJobs(jobs) {
  if (jobs.length === 0) {
    slot.innerHTML = '<div class="empty">No jobs assigned to you right now.</div>';
    return;
  }
  slot.innerHTML = jobs.map(job => `
    <div class="panel" data-job="${job.id}">
      <div class="case-header" style="margin-bottom:1rem">
        <div>
          <span class="id mono" style="color:var(--ink-soft)">#${String(job.id).padStart(4, '0')}</span>
          <h3 style="margin-top:0.2rem">${escapeHTML(job.description)}</h3>
          <p class="hint" style="margin:0">${escapeHTML(job.category)}${job.address_text ? ' · ' + escapeHTML(job.address_text) : ''}
             · reported at ${fmtNum(job.lat)}, ${fmtNum(job.lon)}</p>
        </div>
        ${stampHTML(job.status)}
      </div>
      <div class="case-photos" style="margin-bottom:1rem">
        <figure>
          <img src="${job.before_photo_url}" alt="Original complaint photo">
          <figcaption>Original complaint photo — match this angle & background as closely as you can</figcaption>
        </figure>
        <figure>
          <img class="preview-${job.id}" style="display:none;width:100%;aspect-ratio:4/3;object-fit:cover;border:1px solid var(--hairline);border-radius:var(--radius)">
        </figure>
      </div>
      <div class="field">
        <label>After-repair photo</label>
        <input type="file" accept="image/*" capture="environment" id="file-${job.id}">
        <p class="hint" style="margin-top:0.4rem">Take this with your camera at the repair site right now — gallery photos are flagged and likely to be rejected.</p>
      </div>
      <div class="geo-status" id="geo-${job.id}">Requesting device location…</div>
      <div id="banner-${job.id}" style="margin-top:0.8rem"></div>
      <button id="submit-${job.id}" style="margin-top:0.5rem">Submit repair photo</button>
      <div id="result-${job.id}" style="margin-top:1rem"></div>
    </div>
  `).join('');

  jobs.forEach(job => wireJob(job.id));
}

function fmtNum(n) { return n === null || n === undefined ? '—' : Number(n).toFixed(5); }

function wireJob(jobId) {
  requestLocation(jobId);

  const fileInput = document.getElementById(`file-${jobId}`);
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    const preview = document.querySelector(`.preview-${jobId}`);
    const banner = document.getElementById(`banner-${jobId}`);
    if (!file) { preview.style.display = 'none'; return; }
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';

    const fresh = checkFileFreshness(file);
    photoFreshness[jobId] = fresh;
    banner.innerHTML = fresh.ok ? '' : `<div class="banner error">${escapeHTML(fresh.message)}</div>`;
  });

  document.getElementById(`submit-${jobId}`).addEventListener('click', () => submitRepair(jobId));
}

function requestLocation(jobId) {
  const el = document.getElementById(`geo-${jobId}`);
  if (!('geolocation' in navigator)) {
    el.textContent = 'Location unavailable in this browser.';
    el.className = 'geo-status bad';
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      capturedPosition[jobId] = { lat: pos.coords.latitude, lon: pos.coords.longitude };
      el.textContent = `Location captured (±${Math.round(pos.coords.accuracy)}m): ${pos.coords.latitude.toFixed(6)}, ${pos.coords.longitude.toFixed(6)}`;
      el.className = 'geo-status ok';
    },
    (err) => {
      el.textContent = `Could not get location (${err.message}). GPS is required to submit — please enable location access and retry.`;
      el.className = 'geo-status bad';
    },
    { enableHighAccuracy: true, timeout: 10000 }
  );
}

async function submitRepair(jobId) {
  const banner = document.getElementById(`banner-${jobId}`);
  const resultEl = document.getElementById(`result-${jobId}`);
  const btn = document.getElementById(`submit-${jobId}`);
  banner.innerHTML = '';
  const file = document.getElementById(`file-${jobId}`).files[0];
  if (!file) { banner.innerHTML = '<div class="banner error">Please attach the after-repair photo.</div>'; return; }
  const fresh = photoFreshness[jobId] || checkFileFreshness(file);
  if (!fresh.ok) { banner.innerHTML = `<div class="banner error">${escapeHTML(fresh.message)}</div>`; return; }
  const pos = capturedPosition[jobId];
  if (!pos) { banner.innerHTML = '<div class="banner error">Waiting on device location — please allow location access.</div>'; return; }

  btn.disabled = true;
  btn.textContent = 'Submitting & running verification…';

  const fd = new FormData();
  fd.append('after_photo', file);
  fd.append('lat', pos.lat);
  fd.append('lon', pos.lon);

  try {
    const res = await API.postForm(`/api/complaints/${jobId}/repair`, fd);
    resultEl.innerHTML = renderVerificationSummary(res.verification, jobId);
    btn.style.display = 'none';
  } catch (e) {
    banner.innerHTML = `<div class="banner error">${escapeHTML(e.message)}</div>`;
    btn.disabled = false;
    btn.textContent = 'Submit repair photo';
  }
}

function renderVerificationSummary(v, jobId) {
  const cls = v.overall_status === 'verified' ? 'ok' : v.overall_status === 'rejected' ? 'error' : 'info';
  return `
    <div class="banner ${cls}">
      <b>${v.overall_status.toUpperCase()}</b> — ${escapeHTML(v.headline)}
      <div style="margin-top:0.5rem"><a href="/case.html?id=${jobId}">View full evidence on the public ledger</a></div>
    </div>`;
}

loadJobs();
