const user = requireLogin('citizen');
renderNav('report');

let currentPosition = null; // { lat, lon, accuracy }

function setGeoStatus(text, cls) {
  const el = document.getElementById('geo-status');
  el.textContent = text;
  el.className = 'geo-status' + (cls ? ' ' + cls : '');
}

function requestLocation() {
  if (!('geolocation' in navigator)) {
    setGeoStatus('This browser does not support location — you can still submit, but the location match check will be skipped.', 'bad');
    return;
  }
  setGeoStatus('Requesting device location…');
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      currentPosition = {
        lat: pos.coords.latitude,
        lon: pos.coords.longitude,
        accuracy: pos.coords.accuracy,
      };
      setGeoStatus(
        `Location captured (±${Math.round(pos.coords.accuracy)}m accuracy): ${pos.coords.latitude.toFixed(6)}, ${pos.coords.longitude.toFixed(6)}`,
        'ok'
      );
    },
    (err) => {
      setGeoStatus(`Could not get device location (${err.message}). You can still submit if your photo carries GPS EXIF data, or the complaint will need manual location entry by an admin.`, 'bad');
    },
    { enableHighAccuracy: true, timeout: 10000 }
  );
}

// Camera-only enforcement: `capture="environment"` on the file input is a
// best-effort hint that many mobile browsers honor by opening the camera
// directly, but it isn't guaranteed everywhere (iOS Safari, for one, still
// offers "Photo Library" in the picker sheet). As a second line of defense,
// check the selected file's own lastModified timestamp: a photo the camera
// JUST took will have a timestamp from the last few seconds/minutes, while
// one pulled from the gallery will carry its original, much older date. This
// is backed up server-side by an EXIF-timestamp check that can't be spoofed
// by simply skipping this client-side one.
const PHOTO_FRESHNESS_MAX_AGE_MS = 20 * 60 * 1000; // 20 minutes, matches the backend

function checkFileFreshness(file) {
  if (!file.lastModified) return { ok: true }; // some browsers don't expose it — don't block on absence
  const age = Date.now() - file.lastModified;
  if (age > PHOTO_FRESHNESS_MAX_AGE_MS) {
    return {
      ok: false,
      message: 'This photo looks like it was taken a while ago, not just now. Please take a fresh photo with your camera — photos chosen from your gallery are not accepted.',
    };
  }
  return { ok: true };
}

let photoFreshnessOk = true;

document.getElementById('photo').addEventListener('change', (e) => {
  const file = e.target.files[0];
  const preview = document.getElementById('preview');
  const banner = document.getElementById('banner-slot');
  if (!file) { preview.style.display = 'none'; return; }
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.style.display = 'block';

  const fresh = checkFileFreshness(file);
  photoFreshnessOk = fresh.ok;
  banner.innerHTML = fresh.ok ? '' : `<div class="banner error">${escapeHTML(fresh.message)}</div>`;
});

document.getElementById('report-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const banner = document.getElementById('banner-slot');
  const btn = document.getElementById('submit-btn');
  banner.innerHTML = '';

  const photoFile = document.getElementById('photo').files[0];
  if (!photoFile) {
    banner.innerHTML = '<div class="banner error">Please attach a photo of the pothole.</div>';
    return;
  }
  if (!photoFreshnessOk) {
    banner.innerHTML = `<div class="banner error">${escapeHTML(checkFileFreshness(photoFile).message)}</div>`;
    return;
  }
  const description = document.getElementById('description').value.trim();
  if (!description) {
    banner.innerHTML = '<div class="banner error">Please describe the issue.</div>';
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Submitting…';

  const fd = new FormData();
  fd.append('photo', photoFile);
  fd.append('description', description);
  fd.append('category', document.getElementById('category').value);
  fd.append('address_text', document.getElementById('address').value.trim());
  if (currentPosition) {
    fd.append('lat', currentPosition.lat);
    fd.append('lon', currentPosition.lon);
  }

  try {
    const result = await API.postForm('/api/complaints', fd);
    if (result.merged) {
      window.location.href = `/case.html?id=${result.id}&merged=1&mergedDistance=${encodeURIComponent(result.merged_distance_m ?? '')}`;
    } else {
      window.location.href = `/case.html?id=${result.id}&justFiled=1`;
    }
  } catch (err) {
    banner.innerHTML = `<div class="banner error">${escapeHTML(err.message)}</div>`;
    btn.disabled = false;
    btn.textContent = 'Submit complaint';
  }
});

requestLocation();
