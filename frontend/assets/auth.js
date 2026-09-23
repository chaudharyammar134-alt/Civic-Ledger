renderNav('login');

let mode = 'login';
let role = 'citizen';

const banner = document.getElementById('banner-slot');
const form = document.getElementById('auth-form');
const submitBtn = form.querySelector('button[type=submit]');
const nameField = document.getElementById('name-field');
const roleSlot = document.getElementById('register-role-slot');

function setMode(newMode) {
  mode = newMode;
  document.querySelectorAll('#mode-tabs button').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  nameField.style.display = mode === 'register' ? 'block' : 'none';
  roleSlot.style.display = mode === 'register' ? 'flex' : 'none';
  submitBtn.textContent = mode === 'register' ? 'Create account' : 'Log in';
  banner.innerHTML = '';
}

document.getElementById('mode-tabs').addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (btn) setMode(btn.dataset.mode);
});

roleSlot.addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (!btn) return;
  role = btn.dataset.role;
  document.querySelectorAll('#register-role-slot button').forEach(b => b.classList.toggle('active', b === btn));
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  banner.innerHTML = '';
  submitBtn.disabled = true;
  const email = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value;
  try {
    let result;
    if (mode === 'register') {
      const name = document.getElementById('name').value.trim();
      if (!name) throw new Error('Please enter your name.');
      result = await API.postJSON('/api/auth/register', { name, email, password, role });
    } else {
      result = await API.postJSON('/api/auth/login', { email, password });
    }
    API.setSession(result.token, result.user);
    const dest = result.user.role === 'admin' ? '/admin.html'
      : result.user.role === 'contractor' ? '/contractor.html'
      : '/index.html';
    window.location.href = dest;
  } catch (err) {
    banner.innerHTML = `<div class="banner error">${escapeHTML(err.message)}</div>`;
    submitBtn.disabled = false;
  }
});
