const byId = id => document.getElementById(id);
async function health() {
  try {
    const response = await fetch('/health');
    const status = await response.json();
    byId('health').textContent = status.status === 'ok'
      ? `Pipeline ready · ${status.ai_provider}`
      : `Setup needed · ${Object.values(status.errors).join(' ')}`;
  } catch { byId('health').textContent = 'Cannot reach the local API.'; }
}
health();
byId('analysis-form').addEventListener('submit', async event => {
  event.preventDefault();
  byId('analyze').disabled = true;
  byId('analyze').textContent = 'Analyzing…';
  byId('error').hidden = true;
  byId('results').hidden = true;
  try {
    const response = await fetch('/analyze?debug=true', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({feedback:byId('feedback').value})});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail));
    if (!['positive', 'negative'].includes(result.sentiment)) throw new Error('The server returned an incompatible sentiment schema. Restart the binary pipeline.');
    for (const key of ['sentiment','language','provider','reason']) byId(key).textContent = result[key];
    byId('sentiment').textContent = result.sentiment.toUpperCase();
    byId('sentiment').dataset.sentiment = result.sentiment;
    byId('confidence').textContent = `${(result.confidence * 100).toFixed(1)}%`;
    byId('positive-evidence').textContent = `${(result.scores.positive * 100).toFixed(1)}%`;
    byId('negative-evidence').textContent = `${(result.scores.negative * 100).toFixed(1)}%`;
    byId('topic').textContent = `${result.topic.name} · ${(result.topic.probability * 100).toFixed(1)}%`;
    byId('keywords').textContent = result.topic.keywords.join(', ') || 'No known topic terms in this feedback.';
    byId('provider-note').textContent = result.provider === 'mock' ? 'Simulated result for interface testing.'
      : result.provider === 'rag_weighted_fallback' ? 'Provisional neighbor vote. Vote share is not calibrated confidence.'
      : 'AI provider result. Confidence is a model estimate.';
    byId('references').replaceChildren();
    for (const reference of result.rag.retrieved_examples) {
      const card = document.createElement('article'); card.className = 'reference';
      const sentence = document.createElement('p'); sentence.textContent = reference.text;
      const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = reference.sentiment;
      const meta = document.createElement('small');
      meta.textContent = `Cosine ${reference.similarity.toFixed(3)} · Domain ${reference.domain_relevance_score.toFixed(3)} · ${reference.source_name} · ${reference.aspect} · ${reference.language} · ${reference.split}`;
      card.append(sentence,badge,meta); byId('references').append(card);
    }
    byId('debug-panel').hidden = !result.debug;
    const evidenceBody = byId('evidence-table').querySelector('tbody');
    evidenceBody.replaceChildren();
    if (result.debug) {
      byId('debug-formula').textContent = `${result.debug.formula}. Positive total: ${result.debug.positive_score.toFixed(4)}; negative total: ${result.debug.negative_score.toFixed(4)}. ${result.debug.tie_break}.`;
      for (const [index, evidence] of result.debug.evidence.entries()) {
        const row = document.createElement('tr');
        for (const value of [index + 1, evidence.sentiment, evidence.similarity.toFixed(3), evidence.domain_relevance_score.toFixed(3), evidence.source_weight.toFixed(2), evidence.weighted_contribution.toFixed(4)]) {
          const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
        }
        evidenceBody.append(row);
      }
    }
    byId('json').textContent = JSON.stringify(result,null,2);
    byId('results').hidden = false;
  } catch (error) { byId('error').textContent = error.message; byId('error').hidden = false; }
  finally { byId('analyze').disabled = false; byId('analyze').textContent = 'Analyze feedback →'; }
});
