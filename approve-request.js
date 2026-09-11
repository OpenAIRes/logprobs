/* Show the request that is about to be sent, and wait for a verdict.

   The point is that what is on screen IS what goes out: the caller builds the
   request body, hands that same object here, and posts the very object that was
   approved. Nothing is rebuilt in between, so the dialog cannot drift from the
   call.

   Returns 'yes' (send this one), 'all' (send this and the rest without asking
   again), 'skip' (leave this one alone) or 'stop' (stop here). Escape and the
   backdrop mean 'stop', because the safe reading of a dismissed dialog is
   "don't spend anything". */
(function (root) {

  const CSS = `
  .arq-back { position: fixed; inset: 0; background: rgba(0,0,0,.45); z-index: 4000;
              display: flex; align-items: center; justify-content: center; padding: 1rem; }
  .arq-box { background: var(--panel, #fff); color: var(--ink, #111);
             border: 1px solid var(--line, #ddd); border-radius: 10px;
             box-shadow: 0 10px 40px rgba(0,0,0,.3);
             width: min(46rem, 100%); max-height: 90vh; overflow: auto; padding: 1rem 1.1rem; }
  .arq-box h2 { font-size: 1rem; margin: 0 0 .2rem; }
  .arq-why { font-size: .8125rem; color: var(--ink-dim, #666); margin: 0 0 .7rem; }
  .arq-grid { display: grid; grid-template-columns: max-content 1fr; gap: .15rem .7rem;
              font-size: .8125rem; margin-bottom: .6rem; }
  .arq-grid dt { color: var(--ink-dim, #666); }
  .arq-grid dd { margin: 0; font-variant-numeric: tabular-nums; }
  .arq-label { font-size: .75rem; color: var(--ink-dim, #666); margin: .5rem 0 .2rem; }
  .arq-pre { background: var(--chip, #f3f3f3); border: 1px solid var(--line, #ddd);
             border-radius: 6px; padding: .5rem .6rem; margin: 0;
             font-size: .78125rem; white-space: pre-wrap; overflow-wrap: anywhere;
             max-height: 15rem; overflow: auto; }
  .arq-row { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .9rem; }
  .arq-row .arq-spacer { flex: 1; }
  `;

  let styled = false;
  function ensureStyle() {
    if (styled) return;
    styled = true;
    const el = document.createElement('style');
    el.textContent = CSS;
    document.head.appendChild(el);
  }

  const esc = t => String(t)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  /* Whitespace has to be visible: a prompt whose last token is a newline or a
     space is a different prompt, and "…the " reads identically to "…the". */
  const visible = t => String(t)
    .replace(/\n/g, '⏎\n').replace(/\t/g, '⇥').replace(/ /g, '·');

  function approve(request, info) {
    ensureStyle();
    const i = (info && info.index) || 1;
    const total = (info && info.total) || 1;
    const remaining = Math.max(0, total - i + 1);

    return new Promise(resolve => {
      const back = document.createElement('div');
      back.className = 'arq-back';
      back.setAttribute('role', 'dialog');
      back.setAttribute('aria-modal', 'true');

      const params = Object.keys(request)
        .filter(k => k !== 'prompt')
        .map(k => `<dt>${esc(k)}</dt><dd>${esc(JSON.stringify(request[k]))}</dd>`)
        .join('');

      back.innerHTML = `
        <div class="arq-box">
          <h2>Poslat tento požadavek na API?</h2>
          <p class="arq-why">${i} z ${total} · tohle je přesné tělo požadavku,
            které se odešle na <code>/v1/completions</code>. Nic se neposílá,
            dokud to neodsouhlasíš.</p>
          <dl class="arq-grid">${params}
            <dt>prompt</dt><dd>${request.prompt.length} znaků</dd>
          </dl>
          <p class="arq-label">prompt (· mezera, ⏎ nový řádek)</p>
          <pre class="arq-pre" id="arqPrompt"></pre>
          <p class="arq-label">JSON, jak půjde na drát</p>
          <pre class="arq-pre" id="arqJson"></pre>
          <div class="arq-row">
            <button type="button" class="primary" id="arqYes">Poslat</button>
            <button type="button" id="arqSkip">Přeskočit</button>
            <span class="arq-spacer"></span>
            <button type="button" id="arqAll">Poslat i zbývajících ${remaining} bez dotazu</button>
            <button type="button" class="danger" id="arqStop">Zastavit</button>
          </div>
        </div>`;

      // textContent, not innerHTML: a prompt is arbitrary text and must never be
      // parsed as markup on its way to being approved.
      back.querySelector('#arqPrompt').textContent = visible(request.prompt);
      back.querySelector('#arqJson').textContent = JSON.stringify(request, null, 2);

      const done = verdict => {
        document.removeEventListener('keydown', onKey, true);
        back.remove();
        resolve(verdict);
      };
      const onKey = ev => {
        if (ev.key === 'Escape') { ev.preventDefault(); done('stop'); }
        if (ev.key === 'Enter') { ev.preventDefault(); done('yes'); }
      };
      back.querySelector('#arqYes').addEventListener('click', () => done('yes'));
      back.querySelector('#arqSkip').addEventListener('click', () => done('skip'));
      back.querySelector('#arqAll').addEventListener('click', () => done('all'));
      back.querySelector('#arqStop').addEventListener('click', () => done('stop'));
      back.addEventListener('click', ev => { if (ev.target === back) done('stop'); });

      document.addEventListener('keydown', onKey, true);
      document.body.appendChild(back);
      back.querySelector('#arqYes').focus();
    });
  }

  root.ApproveRequest = {approve};
})(typeof module === 'object' ? module.exports : window);
