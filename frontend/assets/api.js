// api.js — thin fetch wrapper + session storage. No build step, no framework.
const API = {
  base: '',

  token() { return localStorage.getItem('pt_token'); },
  user() {
    try { return JSON.parse(localStorage.getItem('pt_user') || 'null'); }
    catch { return null; }
  },
  setSession(token, user) {
    localStorage.setItem('pt_token', token);
    localStorage.setItem('pt_user', JSON.stringify(user));
  },
  clearSession() {
    localStorage.removeItem('pt_token');
    localStorage.removeItem('pt_user');
  },
  isLoggedIn() { return !!this.token(); },

  async _handle(res) {
    let body = null;
    try { body = await res.json(); } catch { /* no body */ }
    if (!res.ok) {
      const msg = (body && body.error) ? body.error : `Request failed (${res.status})`;
      throw new Error(msg);
    }
    return body;
  },

  async get(path) {
    const res = await fetch(this.base + path, { headers: this._authHeaders() });
    return this._handle(res);
  },

  async postJSON(path, data) {
    const res = await fetch(this.base + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...this._authHeaders() },
      body: JSON.stringify(data),
    });
    return this._handle(res);
  },

  async postForm(path, formData) {
    const res = await fetch(this.base + path, {
      method: 'POST',
      headers: this._authHeaders(), // NOTE: no Content-Type — browser sets multipart boundary
      body: formData,
    });
    return this._handle(res);
  },

  _authHeaders() {
    const t = this.token();
    return t ? { Authorization: `Bearer ${t}` } : {};
  },
};

// --- small shared UI helpers -------------------------------------------------
function fmtTime(unixSeconds) {
  if (!unixSeconds) return '—';
  const d = new Date(unixSeconds * 1000);
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

function statusLabel(status) {
  const map = {
    reported: 'Reported', assigned: 'Assigned', repair_submitted: 'Repair submitted',
    verified: 'Verified', rejected: 'Rejected', disputed: 'Disputed', resolved: 'Resolved',
  };
  return map[status] || status;
}

function stampHTML(status) {
  return `<span class="stamp ${status}">${statusLabel(status)}</span>`;
}

function escapeHTML(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function requireLogin(role) {
  const user = API.user();
  if (!API.isLoggedIn() || !user) {
    window.location.href = '/login.html';
    return null;
  }
  if (role && user.role !== role) {
    window.location.href = '/index.html';
    return null;
  }
  return user;
}

function renderNav(activePage) {
  const user = API.user();
  const el = document.getElementById('nav-slot');
  if (!el) return;
  const links = [
    { href: '/index.html', label: 'Ledger', key: 'ledger' },
    { href: '/report.html', label: 'Report a pothole', key: 'report' },
  ];
  if (user?.role === 'contractor') links.push({ href: '/contractor.html', label: 'My jobs', key: 'contractor' });
  if (user?.role === 'admin') links.push({ href: '/admin.html', label: 'Admin', key: 'admin' });

  let html = links.map(l =>
    `<a href="${l.href}" class="${l.key === activePage ? 'active' : ''}">${l.label}</a>`
  ).join('');

  if (user) {
    html += `<span class="who">${escapeHTML(user.name)} · ${user.role}</span>`;
    html += `<a href="#" id="logout-link">Log out</a>`;
  } else {
    html += `<a href="/login.html" class="${activePage === 'login' ? 'active' : ''}">Log in</a>`;
  }
  el.innerHTML = html;

  const logout = document.getElementById('logout-link');
  if (logout) logout.addEventListener('click', (e) => {
    e.preventDefault();
    API.clearSession();
    window.location.href = '/index.html';
  });
}
