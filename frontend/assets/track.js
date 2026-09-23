renderNav('ledger');

let currentStatus = '';

async function loadStats() {
  try {
    const s = await API.get('/api/stats');
    const by = s.by_status || {};
    const awaiting = (by.assigned || 0) + (by.disputed || 0);
    const verified = by.verified || 0;
    const rejected = by.rejected || 0;
    const nums = document.querySelectorAll('#stats-strip .n');
    nums[0].textContent = s.total ?? 0;
    nums[1].textContent = awaiting;
    nums[2].textContent = verified;
    nums[3].textContent = rejected;
  } catch (e) { /* stats are non-critical */ }
}

async function loadCases() {
  const grid = document.getElementById('case-grid');
  grid.innerHTML = '<p class="hint">Loading complaints…</p>';
  try {
    const qs = currentStatus ? `?status=${encodeURIComponent(currentStatus)}` : '';
    const data = await API.get('/api/complaints' + qs);
    const complaints = data.complaints || [];
    if (complaints.length === 0) {
      grid.innerHTML = '<div class="empty">No complaints in this category yet.</div>';
      return;
    }
    grid.innerHTML = complaints.map(c => `
      <a class="case-card" href="/case.html?id=${c.id}" style="text-decoration:none;color:inherit">
        <div class="thumb" style="background-image:url('${c.before_photo_url}')"></div>
        <div class="body">
          <div class="id-row">
            <span class="id">#${String(c.id).padStart(4, '0')}</span>
            ${stampHTML(c.status)}
          </div>
          <p class="desc">${escapeHTML(c.description)}</p>
          <span class="meta">${escapeHTML(c.category)} · filed ${fmtTime(c.created_at)}</span>
        </div>
      </a>
    `).join('');
  } catch (e) {
    grid.innerHTML = `<div class="empty">Could not load complaints: ${escapeHTML(e.message)}</div>`;
  }
}

async function loadActivity() {
  const el = document.getElementById('activity-feed');
  try {
    const data = await API.get('/api/activity?limit=25');
    const events = data.events || [];
    if (events.length === 0) {
      el.innerHTML = '<li class="d hint">No activity yet.</li>';
      return;
    }
    el.innerHTML = events.map((e, i) => `
      <li>
        <span class="t">${fmtTime(e.created_at)}</span>
        <span class="d">
          <a href="/case.html?id=${e.complaint_id}">#${String(e.complaint_id).padStart(4, '0')}</a>
          — ${escapeHTML(eventLabel(e))}
        </span>
      </li>
    `).join('');
  } catch (e) {
    el.innerHTML = `<li class="d hint">Could not load activity feed.</li>`;
  }
}

function eventLabel(e) {
  const map = {
    reported: 'complaint filed',
    assigned: e.detail || 'assigned to a contractor',
    repair_submitted: 'repair photo submitted',
    verification_verified: 'auto-verification: VERIFIED',
    verification_rejected: 'auto-verification: REJECTED',
    verification_flagged: 'flagged for manual review',
    disputed: 'disputed by citizen',
    admin_review: e.detail || 'reviewed by admin',
  };
  return map[e.event_type] || e.event_type;
}

document.getElementById('filters').addEventListener('click', (ev) => {
  const btn = ev.target.closest('button');
  if (!btn) return;
  document.querySelectorAll('#filters button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentStatus = btn.dataset.status;
  loadCases();
});

loadStats();
loadCases();
loadActivity();
setInterval(() => { loadStats(); loadActivity(); }, 15000);
