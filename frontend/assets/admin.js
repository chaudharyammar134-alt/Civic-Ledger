const user = requireLogin('admin');
renderNav('admin');

let currentStatus = '';
let allComplaints = [];

async function load() {
  const tbody = document.getElementById('admin-tbody');
  try {
    const data = await API.get('/api/admin/complaints');
    allComplaints = data.complaints || [];
    renderTable();
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="6" class="hint">Could not load: ${escapeHTML(e.message)}</td></tr>`;
  }
}

function renderTable() {
  const tbody = document.getElementById('admin-tbody');
  const rows = currentStatus ? allComplaints.filter(c => c.status === currentStatus) : allComplaints;
  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="hint">Nothing here.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(c => `
    <tr>
      <td class="mono">#${String(c.id).padStart(4, '0')}</td>
      <td>${stampHTML(c.status)}</td>
      <td>${escapeHTML(c.category)}</td>
      <td>${escapeHTML(truncate(c.description, 70))}</td>
      <td class="mono">${fmtTime(c.created_at)}</td>
      <td><a href="/case.html?id=${c.id}">${c.status === 'rejected' ? 'Open case & reassign' : 'Open case'}</a></td>
    </tr>
  `).join('');
}

async function loadContractors() {
  const tbody = document.getElementById('contractors-tbody');
  try {
    const data = await API.get('/api/admin/contractors');
    const contractors = data.contractors || [];
    if (contractors.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="hint">No contractors yet.</td></tr>';
      return;
    }
    tbody.innerHTML = contractors.map(ct => `
      <tr>
        <td>${escapeHTML(ct.name)}<div class="hint" style="font-size:0.78rem">${escapeHTML(ct.email)}</div></td>
        <td class="mono">${ct.total_submissions}</td>
        <td class="mono">${ct.rejected_submissions} (${Math.round(ct.rejection_rate * 100)}%)</td>
        <td class="mono">${ct.gaming_attempts}</td>
        <td>${ct.risk_level === 'low' ? '<span class="hint">Low</span>' : `<span class="risk-badge ${ct.risk_level}">${ct.risk_level === 'high' ? 'High risk' : 'Elevated'}</span>`}</td>
      </tr>
    `).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="5" class="hint">Could not load: ${escapeHTML(e.message)}</td></tr>`;
  }
}

function truncate(s, n) { return s && s.length > n ? s.slice(0, n - 1) + '…' : s; }

document.getElementById('filters').addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (!btn) return;
  document.querySelectorAll('#filters button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentStatus = btn.dataset.status;
  renderTable();
});

load();
loadContractors();
