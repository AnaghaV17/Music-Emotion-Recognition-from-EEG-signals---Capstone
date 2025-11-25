document.getElementById('upload-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const data = new FormData(form);
  const btn = form.querySelector('button');
  btn.disabled = true;
  btn.textContent = 'Predicting...';

  try {
    const res = await fetch('/predict', { method: 'POST', body: data, headers: { 'Accept': 'application/json' } });
    const json = await res.json();
    const resultsDiv = document.getElementById('results');
    if (json.error) {
      resultsDiv.innerHTML = '<div class="card"><pre>' + JSON.stringify(json, null, 2) + '</pre></div>';
    } else {
      let html = '<h2>Results</h2><div class="cards">';
      for (const k of Object.keys(json)) {
        html += `<div class="card"><h3>${k}</h3><pre>${JSON.stringify(json[k], null, 2)}</pre></div>`;
      }
      html += '</div>';
      resultsDiv.innerHTML = html;
    }
  } catch (err) {
    document.getElementById('results').innerHTML = '<pre>' + err.toString() + '</pre>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Upload & Predict';
  }
});
