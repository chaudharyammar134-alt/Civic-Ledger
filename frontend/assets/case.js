renderNav('');

const params = new URLSearchParams(window.location.search);
const caseId = params.get('id');
const justFiled = params.get('justFiled') === '1';
const justMerged = params.get('merged') === '1';
const mergedDistance = params.get('mergedDistance');
const main = document.getElementById('case-main');

if (!caseId) {
  main.innerHTML = '<div class="banner error">No case id given.</div>';
} else {
  loadCase();
}

async function loadCase() {
  try {
    const c = await API.get(`/api/complaints/${caseId}`);
    render(c);
  } catch (e) {
    main.innerHTML = `<div class="banner error">Could not load this case: ${escapeHTML(e.message)}</div>`;
  }
}

function checkRow(pass, title, explanation) {
  return `
    <div class="check-row">
      <div class="check-glyph ${pass ? 'pass' : 'fail'}">${pass ? '✓' : '✕'}</div>
      <div class="check-body">
        <b>${escapeHTML(title)}</b>
        <span>${escapeHTML(explanation || '')}</span>
      </div>
    </div>`;
}

function verificationPanel(sub) {
  const v = sub.verification;
  if (!v) return '';
  const conf = Math.max(0, Math.min(100, v.overall_confidence ?? 0));
  const barColor = v.overall_status === 'verified' ? 'var(--green)' : v.overall_status === 'rejected' ? 'var(--red)' : 'var(--yellow)';
  const ai = v.ai_authenticity || {};
  const aiPass = !ai.flagged;

  return `
  <div class="panel">
    <h3>Automated verification report</h3>
    <p style="margin-top:-0.4rem">${escapeHTML(v.headline || '')}</p>
    <div class="hint" style="display:flex;justify-content:space-between">
      <span>Confidence score</span><span class="mono">${conf}/100</span>
    </div>
    <div class="confidence-bar"><div style="width:${conf}%;background:${barColor}"></div></div>

    ${checkRow(v.gps?.pass, 'Location match (GPS)', v.gps?.explanation)}
    ${checkRow(v.landmark_match?.pass, 'Background landmarks & camera angle match', v.landmark_match?.explanation)}
    ${checkRow(v.region_change?.pass, 'Repair actually visible in the pothole area', v.region_change?.explanation)}
    ${checkRow(aiPass, 'Photo authenticity (not AI-generated / synthetic)', ai.explanation)}

    <details style="margin-top:0.9rem">
      <summary class="hint" style="cursor:pointer">Raw technical measurements</summary>
      <table class="data-table" style="margin-top:0.6rem">
        <tbody>
          <tr><td>GPS distance</td><td class="mono">${fmtNum(v.gps?.distance_m)} m (tolerance ${fmtNum(v.gps?.tolerance_m)} m)</td></tr>
          <tr><td>Matched background keypoints</td><td class="mono">${v.landmark_match?.good_matches ?? '—'}</td></tr>
          <tr><td>Geometrically consistent (inliers)</td><td class="mono">${v.landmark_match?.inliers ?? '—'} (${fmtPct(v.landmark_match?.inlier_ratio)})</td></tr>
          <tr><td>Repair-area similarity (SSIM, before vs after)</td><td class="mono">${fmtNum(v.region_change?.ssim_inside, 3)}</td></tr>
          <tr><td>Background stability (SSIM outside repair area)</td><td class="mono">${fmtNum(v.region_change?.ssim_outside, 3)}</td></tr>
          <tr><td>AI-authenticity suspicion score</td><td class="mono">${fmtNum(ai.score, 0)}/100 (lower = more camera-authentic signals)</td></tr>
        </tbody>
      </table>
    </details>
  </div>`;
}

function fmtNum(n, digits = 1) {
  if (n === null || n === undefined) return '—';
  return Number(n).toFixed(digits);
}
function fmtPct(n) {
  if (n === null || n === undefined) return '—';
  return Math.round(n * 100) + '%';
}

function render(c) {
  const user = API.user();
  const latestSub = (c.repair_submissions || [])[0];

  let html = '';

  if (justFiled) {
    html += `<div class="banner ok">Complaint #${String(c.id).padStart(4, '0')} filed. You can track every step of its repair right here.</div>`;
  }
  if (justMerged) {
    html += `<div class="banner info">This looks like the same pothole someone already reported${mergedDistance ? ` (${escapeHTML(mergedDistance)}m away)` : ''} — your photo has been added as extra evidence on this existing case instead of opening a duplicate.</div>`;
  }

  html += `
    <div class="case-header">
      <div>
        <span class="id mono" style="color:var(--ink-soft)">#${String(c.id).padStart(4, '0')}</span>
        <h1 style="margin-top:0.2rem">${escapeHTML(c.description)}</h1>
        <p class="hint">
          ${escapeHTML(c.category)} · filed by ${escapeHTML(c.citizen_name)} on ${fmtTime(c.created_at)}
          ${c.address_text ? ' · ' + escapeHTML(c.address_text) : ''}
          ${c.report_count > 1 ? ` · reported by ${c.report_count} citizens` : ''}
        </p>
      </div>
      ${stampHTML(c.status)}
    </div>
  `;

  html += `<div class="case-photos">
    <figure>
      <img src="${c.before_photo_url}" alt="Before photo">
      <figcaption>BEFORE — submitted by citizen · lat ${fmtNum(c.lat, 5)}, lon ${fmtNum(c.lon, 5)}</figcaption>
    </figure>
    ${latestSub ? `
    <figure>
      <img src="${latestSub.after_photo_url}" alt="After photo">
      <figcaption>AFTER — submitted by contractor · lat ${fmtNum(latestSub.submitted_lat, 5)}, lon ${fmtNum(latestSub.submitted_lon, 5)}</figcaption>
    </figure>` : `
    <figure>
      <div style="aspect-ratio:4/3;border:1.5px dashed var(--hairline);border-radius:var(--radius);display:flex;align-items:center;justify-content:center;color:var(--ink-soft);font-size:0.85rem;text-align:center;padding:1rem">
        No repair photo submitted yet
      </div>
    </figure>`}
  </div>`;

  if (latestSub) html += verificationPanel(latestSub);

  if ((c.additional_reports || []).length > 0) {
    html += `<div class="panel">
      <h3>Also reported by other citizens</h3>
      <p class="hint" style="margin-top:-0.4rem">Same pothole, matched by location${c.additional_reports.some(r => r.match_method === 'gps_and_visual') ? ' and background matching' : ''} — merged here instead of opening duplicate cases.</p>
      <div class="mini-photo-grid">
        ${c.additional_reports.map(r => `
          <figure>
            <img src="${r.photo_url}" alt="Additional photo of the same pothole">
            <figcaption>${escapeHTML(r.citizen_name)} · ${fmtNum(r.distance_m, 1)}m away · ${fmtTime(r.created_at)}</figcaption>
          </figure>
        `).join('')}
      </div>
    </div>`;
  }

  if (c.assigned_contractor_name) {
    html += `<div class="banner info">Assigned to contractor: <b>${escapeHTML(c.assigned_contractor_name)}</b></div>`;
  }

  // --- role-gated actions -----------------------------------------------
  html += '<div id="actions-slot"></div>';

  if ((c.admin_reviews || []).length > 0) {
    html += `<div class="panel"><h3>Admin review history</h3>${c.admin_reviews.map(r => `
      <div class="check-row"><div class="check-glyph pass">•</div><div class="check-body">
        <b>${escapeHTML(r.decision)}</b>
        <span>${fmtTime(r.created_at)}${r.note ? ' — ' + escapeHTML(r.note) : ''}</span>
      </div></div>`).join('')}</div>`;
  }

  html += `<div class="panel">
    <h3>Timeline</h3>
    <ul class="timeline">
      ${(c.timeline || []).map((e, i) => `
        <li><span class="n">${String(i + 1).padStart(2, '0')}</span>
        <span class="t">${fmtTime(e.created_at)}</span>
        <span class="d">${escapeHTML(eventLabel(e))}</span></li>
      `).join('')}
    </ul>
  </div>`;

  main.innerHTML = html;
  renderActions(c, user);
}

function riskBadge(level) {
  if (!level || level === 'low') return '';
  const label = level === 'high' ? 'High-risk contractor' : 'Elevated risk';
  return ` <span class="risk-badge ${level}">${label}</span>`;
}

function contractorOptionLabel(ct) {
  const stats = ct.total_submissions
    ? ` — ${ct.rejected_submissions}/${ct.total_submissions} rejected${ct.gaming_attempts ? `, ${ct.gaming_attempts} gaming flag(s)` : ''}`
    : '';
  return `${ct.name} (${ct.email})${stats}`;
}

async function loadContractorsInto(selectEl, currentContractorId) {
  selectEl.innerHTML = '<option>Loading contractors…</option>';
  const data = await API.get('/api/admin/contractors');
  const contractors = data.contractors || [];
  selectEl.innerHTML = contractors.map(ct => `
    <option value="${ct.id}" ${ct.id === currentContractorId ? 'selected' : ''}>
      ${escapeHTML(contractorOptionLabel(ct))}${ct.risk_level !== 'low' ? `  [${ct.risk_level.toUpperCase()} RISK]` : ''}
    </option>`).join('');
  return contractors;
}

function renderActions(c, user) {
  const slot = document.getElementById('actions-slot');
  if (!user) return;

  // Admin: assign a contractor
  if (user.role === 'admin' && c.status === 'reported') {
    slot.innerHTML = `<div class="panel">
      <h3>Assign a contractor</h3>
      <div id="assign-banner"></div>
      <div class="field"><select id="contractor-select"><option>Loading contractors…</option></select></div>
      <button id="assign-btn">Assign</button>
    </div>`;
    loadContractorsInto(document.getElementById('contractor-select'), null);
    document.getElementById('assign-btn').addEventListener('click', async () => {
      const banner = document.getElementById('assign-banner');
      const contractorId = document.getElementById('contractor-select').value;
      try {
        await API.postJSON(`/api/admin/complaints/${c.id}/assign`, { contractor_id: parseInt(contractorId, 10) });
        window.location.reload();
      } catch (e) {
        banner.innerHTML = `<div class="banner error">${escapeHTML(e.message)}</div>`;
      }
    });
    return;
  }

  // Admin: reassign a rejected repair — no dispute required. This is the
  // direct fix for "a rejected job just sits there": the admin can send it
  // straight back out, optionally to a different contractor if the one who
  // did it looks unreliable (risk badge shown per contractor in the list).
  if (user.role === 'admin' && c.status === 'rejected') {
    slot.innerHTML = `<div class="panel">
      <h3>This repair was rejected — reassign the job</h3>
      <p class="hint" style="margin-top:-0.4rem">The automated check rejected the repair photo (see the report above). You can send this job back out — to the same contractor for another attempt, or to a different one.</p>
      <div id="reassign-banner"></div>
      <div class="field"><label>Contractor</label><select id="reassign-contractor-select"><option>Loading contractors…</option></select></div>
      <div class="field"><textarea id="reassign-note" rows="2" placeholder="Note (optional but recommended for the public record)"></textarea></div>
      <button id="reassign-btn" class="btn-outline">Reassign job</button>
    </div>`;
    loadContractorsInto(document.getElementById('reassign-contractor-select'), c.assigned_contractor_id);
    document.getElementById('reassign-btn').addEventListener('click', async () => {
      const banner = document.getElementById('reassign-banner');
      const contractorId = document.getElementById('reassign-contractor-select').value;
      try {
        await API.postJSON(`/api/admin/complaints/${c.id}/review`, {
          decision: 'reassign',
          contractor_id: parseInt(contractorId, 10),
          note: document.getElementById('reassign-note').value.trim(),
        });
        window.location.reload();
      } catch (e) {
        banner.innerHTML = `<div class="banner error">${escapeHTML(e.message)}</div>`;
      }
    });
    return;
  }

  // Admin: review a disputed / flagged case
  if (user.role === 'admin' && c.status === 'disputed') {
    slot.innerHTML = `<div class="panel">
      <h3>Review this case</h3>
      <p class="hint" style="margin-top:-0.4rem">This case needs a human decision — either the automated check flagged it, or the citizen disputed the result.</p>
      <div id="review-banner"></div>
      <div class="field"><textarea id="review-note" rows="2" placeholder="Note (optional but recommended for the public record)"></textarea></div>
      <div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-bottom:0.8rem">
        <button id="rv-verify">Confirm verified</button>
        <button id="rv-reject" class="btn-danger">Confirm rejected</button>
      </div>
      <div class="field"><label>Or send back to a contractor instead</label><select id="reassign-contractor-select"><option>Loading contractors…</option></select></div>
      <button id="rv-reassign" class="btn-outline">Send back to contractor</button>
    </div>`;
    loadContractorsInto(document.getElementById('reassign-contractor-select'), c.assigned_contractor_id);
    const doReview = async (decision, extra = {}) => {
      const banner = document.getElementById('review-banner');
      try {
        await API.postJSON(`/api/admin/complaints/${c.id}/review`, {
          decision, note: document.getElementById('review-note').value.trim(), ...extra,
        });
        window.location.reload();
      } catch (e) {
        banner.innerHTML = `<div class="banner error">${escapeHTML(e.message)}</div>`;
      }
    };
    document.getElementById('rv-verify').addEventListener('click', () => doReview('confirm_verified'));
    document.getElementById('rv-reject').addEventListener('click', () => doReview('confirm_rejected'));
    document.getElementById('rv-reassign').addEventListener('click', () => {
      const contractorId = document.getElementById('reassign-contractor-select').value;
      doReview('reassign', { contractor_id: parseInt(contractorId, 10) });
    });
    return;
  }

  // Citizen: dispute a verified/rejected result
  if (user.role === 'citizen' && user.id === c.citizen_id && (c.status === 'verified' || c.status === 'rejected')) {
    slot.innerHTML = `<div class="panel">
      <h3>Disagree with this outcome?</h3>
      <p class="hint" style="margin-top:-0.4rem">If you don't think this result is accurate, explain why and an admin will take a manual look.</p>
      <div id="dispute-banner"></div>
      <div class="field"><textarea id="dispute-reason" rows="3" placeholder="What looks wrong about this result?"></textarea></div>
      <button id="dispute-btn" class="btn-outline">Dispute this result</button>
    </div>`;
    document.getElementById('dispute-btn').addEventListener('click', async () => {
      const banner = document.getElementById('dispute-banner');
      const reason = document.getElementById('dispute-reason').value.trim();
      if (!reason) { banner.innerHTML = '<div class="banner error">Please explain the issue.</div>'; return; }
      try {
        await API.postJSON(`/api/complaints/${c.id}/dispute`, { reason });
        window.location.reload();
      } catch (e) {
        banner.innerHTML = `<div class="banner error">${escapeHTML(e.message)}</div>`;
      }
    });
  }
}

function eventLabel(e) {
  const map = {
    reported: 'Complaint filed',
    assigned: e.detail || 'Assigned to a contractor',
    repair_submitted: 'Repair photo submitted by contractor',
    verification_verified: 'Automated verification: VERIFIED',
    verification_rejected: 'Automated verification: REJECTED',
    verification_flagged: 'Automated verification: flagged for manual review',
    disputed: `Disputed by citizen — "${e.detail || ''}"`,
    admin_review: e.detail || 'Reviewed by admin',
    duplicate_merged: e.detail || 'Additional citizen report merged into this case',
  };
  return map[e.event_type] || e.event_type;
}
