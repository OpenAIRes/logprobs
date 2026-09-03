/* The settings bar: one implementation, on every page that shows results.

   It replaces the old start page as the place changes get made. The landing
   experience is now the result itself -- greedy at temperature 0, the full 4096
   tokens we have on record, opened token by token -- and the bar sits above it.

   Two rules keep it honest:

     * State lives in the URL, so what you see is what a link reproduces. Values
       absent from the URL fall back to what was last used, and the default with
       nothing remembered is exactly the landing case (greedy, one result).
     * Only the four controls that change what you are looking at stay on the
       strip. Everything else -- prompt, model, the by-hand endings, extend,
       walk's base and steps -- lives behind `more`, because a strip of fourteen
       controls is a page you have to read rather than a bar you can use.

   Changing a control navigates, because `results` moves between two different
   pages: one result is a single string in the token browser, more than one is a
   ranked list. Deciding that here rather than in each page is the point.
*/
(() => {
  const STORE_KEY = 'bar_state';

  /* Every setting the bar owns, with the value that means "not set". A param is
     only written to a link when it differs, so a shared URL says what it means
     and nothing more. */
  const FIELDS = {
    view: 'greedy',
    top: '1',
    sort: '',
    prompt: '',
    model: 'gpt-3.5-turbo-instruct',
    prefix: '',
    ends: '',
    nodes: '',
    chosen_only: '',
    extend: '',
    base_id: '',
    steps: '',
    forward_only: '',
  };

  const VIEWS = [
    ['greedy', 'greedy path', 'What temperature 0 produces: the argmax at every position. One result is that string; more than one adds its next-best siblings.'],
    ['completions', 'completions', 'One entry per recorded call — the whole string it produced, prompt included.'],
    ['prefixes', 'prefixes', 'Every prefix in the token trie, in exact n-best order.'],
    ['sweep', 'alternatives grid', 'Per-position top-k for one base completion.'],
    ['walk', 'cheapest-deviation walk', 'Repeatedly take the cheapest single-token edit, then regenerate.'],
  ];

  const SORTS = [
    ['cost', 'deviation cost', v => v === 'greedy'],
    ['sum', 'Σ logprob', v => true],
    ['ppl', 'perplexity', v => true],
    ['length', 'length', v => true],
  ];

  /* The prefix-to-complete axis, nested: each stop is a subset of the one to its
     left, so stepping right only narrows. Kept in step with store.py's ENDS and
     NODE_KINDS. */
  const STOPS = [
    { name: 'prefix', ends: '', nodes: '',
      why: 'Everything, prefixes included. There is rarely a reason to want the prefixes and not the rest, so this end of the axis is no filter.' },
    { name: 'no continuation on record', ends: '', nodes: 'leaf',
      why: 'Nothing further on record: the strings the model finished, plus the prefixes whose continuation we do not have.' },
    { name: 'complete', ends: 'stop', nodes: '',
      why: 'The model emitted EOS — the only strings that are not prefixes of anything.' },
  ];

  const RANKED = v => v === 'completions' || v === 'prefixes';
  const LISTY = v => RANKED(v) || v === 'greedy';
  const SCOPED = v => LISTY(v);

  // ---------------------------------------------------------------- state

  const params = new URLSearchParams(location.search);
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(STORE_KEY) || '{}'); } catch {}

  const state = {};
  for (const [k, dflt] of Object.entries(FIELDS)) {
    state[k] = params.has(k) ? params.get(k)
      : (saved[k] !== undefined ? saved[k] : dflt);
  }
  // `id` is a destination, not a setting: it says which record the token browser
  // is showing, and it must never be carried over to a different query.
  const landedOnId = params.has('id');

  function persist() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch {}
  }

  function query() {
    const p = new URLSearchParams();
    for (const [k, dflt] of Object.entries(FIELDS)) {
      const v = String(state[k] ?? '');
      if (v && v !== String(dflt)) p.set(k, v);
    }
    return p;
  }

  // ---------------------------------------------------------------- destination

  /* One result is a single string, which belongs in the token browser; more than
     one is a ranked list. The bar owns this decision so neither page has to. */
  async function destination() {
    const single = LISTY(state.view) && String(state.top) === '1';
    if (!single) {
      const p = query();
      p.set('view', state.view);        // always explicit in a list link
      return '/dijkstra.html?' + p.toString();
    }
    let id = null;
    try {
      if (state.view === 'greedy') {
        // Resolved by walking argmax records rather than by ranking anything, so
        // it has its own endpoint; its first hop is where the browser lands.
        const q = new URLSearchParams({ prompt: state.prompt, model: state.model });
        const g = await (await fetch('/api/greedy?' + q, { cache: 'no-store' })).json();
        id = g.first_id || null;
      } else {
        const q = new URLSearchParams({ view: state.view, top: '1' });
        if (state.prefix) q.set('prefix', state.prefix);
        if (state.sort) q.set('sort', state.sort);
        if (state.ends) q.set('ends', state.ends);
        if (state.nodes) q.set('nodes', state.nodes);
        if (state.chosen_only) q.set('chosen_only', '1');
        const db = await (await fetch('/api/query?' + q, { cache: 'no-store' })).json();
        id = ((db.entries || [])[0] || {}).id || null;
      }
    } catch { /* store not answering: open on whatever the browser holds locally */ }
    const out = query();
    out.delete('id');
    if (id) out.set('id', id);
    else if (state.prompt) out.set('prompt', state.prompt);
    return '/logprobs.html?' + out.toString();
  }

  let navigating = false;
  async function go() {
    if (navigating) return;
    navigating = true;
    persist();
    const url = await destination();
    // Re-rendering in place would mean two code paths for every control; a local
    // navigation is a few milliseconds and keeps the URL and the view in step.
    location.href = url;
  }

  // ---------------------------------------------------------------- markup

  const host = document.getElementById('settingsbar');
  if (!host) return;

  const opt = (v, label, sel) =>
    `<option value="${v}"${sel === v ? ' selected' : ''}>${label}</option>`;

  host.innerHTML = `
    <div class="sbar">
      <label class="sfield"><span>view</span>
        <select id="sView">${VIEWS.map(([v, l]) => opt(v, l, state.view)).join('')}</select>
      </label>
      <label class="sfield" id="sResultsWrap"><span>results</span>
        <select id="sResults">
          ${['1', '20', '50', '200', '500', '1000', '2000']
            .map(n => opt(n, n === '1' ? '1 — token by token' : n, String(state.top))).join('')}
        </select>
      </label>
      <label class="sfield" id="sSortWrap"><span>ranked by</span>
        <select id="sSort"></select>
      </label>
      <span class="sfield sscope" id="sScopeWrap"><span>how complete</span>
        <input type="range" id="sScope" min="0" max="2" step="1" list="sTicks">
        <datalist id="sTicks"><option value="0"></option><option value="1"></option><option value="2"></option></datalist>
        <b id="sScopeName"></b>
      </span>
      <button type="button" class="smore" id="sMore" aria-expanded="false">more ▾</button>
      <span class="sinfo" id="sInfo"></span>
    </div>
    <div class="smorepanel" id="sPanel" hidden>
      <label class="sfield"><span>starting prompt</span>
        <input type="text" id="sPrompt" placeholder="(empty = unconditional)" spellcheck="false" value="${state.prompt.replace(/"/g, '&quot;')}">
      </label>
      <label class="sfield"><span>starts with</span>
        <input type="text" id="sPrefix" placeholder="e.g. I have" spellcheck="false" value="${state.prefix.replace(/"/g, '&quot;')}">
      </label>
      <label class="sfield"><span>model</span>
        <select id="sModel">
          ${['gpt-3.5-turbo-instruct', 'davinci-002', 'babbage-002']
            .map(m => opt(m, m, state.model)).join('')}
        </select>
      </label>
      <label class="sfield" id="sBaseWrap"><span>base completion</span>
        <select id="sBase"></select>
      </label>
      <label class="sfield" id="sStepsWrap"><span>steps</span>
        <input type="number" id="sSteps" min="1" max="200" value="${state.steps || 20}">
      </label>
      <label class="sfield" id="sFwdWrap"><span>positions</span>
        <span><input type="checkbox" id="sFwd"${state.forward_only ? ' checked' : ''}> forward only</span>
      </label>
      <label class="sfield" id="sExtendWrap"><span>rows</span>
        <span><input type="checkbox" id="sExtend"${state.extend ? ' checked' : ''}> extend each to the end on record</span>
      </label>
      <label class="sfield" id="sRecWrap"><span>tokens</span>
        <span><input type="checkbox" id="sRec"${state.chosen_only ? '' : ' checked'}> include tokens known only from top_logprobs</span>
      </label>
      <div class="sfield"><span>endings</span>
        <span class="sends" id="sEnds"></span>
      </div>
      <div class="sfoot">
        <a href="/help.html#scope">what do these mean?</a>
        <span class="sep">·</span>
        <a href="/index.html">the full settings page, with the counts</a>
      </div>
    </div>`;

  const el = id => document.getElementById(id);

  // ---------------------------------------------------------------- scope stepper

  const ENDS_ALL = ['stop', 'length', 'open'];
  const NODES_ALL = ['continues', 'leaf'];
  el('sEnds').innerHTML = ENDS_ALL.map(v =>
    `<label><input type="checkbox" class="sEndBox" value="${v}"> ${v}</label>`).join('')
    + NODES_ALL.map(v =>
    `<label><input type="checkbox" class="sNodeBox" value="${v}"> ${v === 'leaf' ? 'dead end' : v}</label>`).join('');

  const endBoxes = () => [...document.querySelectorAll('.sEndBox')];
  const nodeBoxes = () => [...document.querySelectorAll('.sNodeBox')];
  const listOf = s => (s ? s.split(',') : []);

  function paintBoxes() {
    const e = listOf(state.ends), n = listOf(state.nodes);
    for (const b of endBoxes()) b.checked = !e.length || e.includes(b.value);
    for (const b of nodeBoxes()) b.checked = !n.length || n.includes(b.value);
  }
  function boxesToState() {
    const e = endBoxes().filter(b => b.checked).map(b => b.value);
    const n = nodeBoxes().filter(b => b.checked).map(b => b.value);
    state.ends = e.length === ENDS_ALL.length ? '' : e.join(',');
    state.nodes = n.length === NODES_ALL.length ? '' : n.join(',');
  }
  /* Which stop the current ends/nodes amount to, or null when they say something
     the axis has no stop for -- `continues` alone, say. */
  function stopOf() {
    for (let i = 0; i < STOPS.length; i++) {
      if (STOPS[i].ends === state.ends && STOPS[i].nodes === state.nodes) return i;
    }
    return null;
  }
  function paintScope() {
    const at = stopOf();
    el('sScope').value = String(at === null ? 0 : at);
    el('sScope').classList.toggle('off', at === null);
    el('sScopeName').textContent = at === null ? 'combination' : (STOPS[at].name === 'prefix'
      ? 'prefix' : STOPS[at].name === 'complete' ? 'complete' : '·');
    el('sScopeWrap').title = at === null
      ? 'The endings below say something this axis has no stop for.'
      : STOPS[at].why;
  }

  // ---------------------------------------------------------------- per-view

  function paintSorts() {
    const sel = el('sSort');
    const allowed = SORTS.filter(([, , ok]) => ok(state.view));
    const want = state.sort || (state.view === 'greedy' ? 'cost' : 'sum');
    sel.innerHTML = allowed.map(([v, l]) => opt(v, l, want)).join('');
  }

  function paintVisibility() {
    const v = state.view;
    el('sResultsWrap').hidden = !LISTY(v);
    el('sSortWrap').hidden = !(v === 'completions' || v === 'greedy');
    el('sScopeWrap').hidden = !SCOPED(v);
    el('sBaseWrap').hidden = LISTY(v);
    el('sStepsWrap').hidden = v !== 'walk';
    el('sFwdWrap').hidden = v !== 'walk';
    el('sExtendWrap').hidden = !LISTY(v) || String(state.top) === '1';
    el('sRecWrap').hidden = v !== 'prefixes';
    el('sEnds').parentElement.hidden = !SCOPED(v);
    el('sPrompt').parentElement.hidden = v !== 'greedy';
    el('sPrefix').parentElement.hidden = !RANKED(v);
    const view = VIEWS.find(([x]) => x === v);
    el('sView').title = view ? view[2] : '';
  }

  /* How many strings the settings on screen match. The list can never show them
     all -- the prefixes view matches 176,649 -- so the count is the only honest
     way to say what a screenful is a screenful of. */
  let infoSeq = 0;
  async function paintInfo() {
    const mine = ++infoSeq;
    const v = state.view;
    if (!LISTY(v)) { el('sInfo').textContent = ''; return; }
    const p = new URLSearchParams({ top: '1' });
    if (v !== 'greedy') p.set('view', v);
    if (state.ends) p.set('ends', state.ends);
    if (state.nodes) p.set('nodes', state.nodes);
    if (state.prefix && RANKED(v)) p.set('prefix', state.prefix);
    if (state.chosen_only && v === 'prefixes') p.set('chosen_only', '1');
    if (state.prompt && v === 'greedy') p.set('prompt', state.prompt);
    try {
      const route = v === 'greedy' ? '/api/greedy_alternatives?' : '/api/query?';
      const db = await (await fetch(route + p, { cache: 'no-store' })).json();
      if (mine !== infoSeq) return;
      const n = db.available;
      if (n == null) { el('sInfo').textContent = ''; return; }
      const want = Number(state.top) || 1;
      el('sInfo').textContent = want >= n
        ? `${n.toLocaleString('en-US')} match`
        : `${want.toLocaleString('en-US')} of ${n.toLocaleString('en-US')} match`;
    } catch {
      if (mine === infoSeq) el('sInfo').textContent = 'store not answering';
    }
  }

  async function ensureBases() {
    const sel = el('sBase');
    if (sel.options.length) return;
    try {
      const { bases } = await (await fetch('/api/bases', { cache: 'no-store' })).json();
      sel.innerHTML = bases.map(b =>
        opt(b.id, `${b.text.slice(0, 40)}${b.text.length > 40 ? '…' : ''}`, state.base_id)).join('');
      if (!state.base_id && bases[0]) state.base_id = bases[0].id;
    } catch { sel.innerHTML = '<option value="">store not answering</option>'; }
  }

  function repaint() {
    paintSorts();
    paintVisibility();
    paintBoxes();
    paintScope();
    paintInfo();
    if (!LISTY(state.view)) ensureBases();
  }

  // ---------------------------------------------------------------- wiring

  el('sView').addEventListener('change', () => {
    state.view = el('sView').value;
    // The sort belongs to the view: deviation cost IS the greedy criterion, and
    // a ranking has no notion of it, so it is never carried across.
    state.sort = '';
    repaint();
    go();
  });
  el('sResults').addEventListener('change', () => { state.top = el('sResults').value; repaint(); go(); });
  el('sSort').addEventListener('change', () => { state.sort = el('sSort').value; go(); });
  el('sScope').addEventListener('input', () => {
    const s = STOPS[Number(el('sScope').value) || 0];
    state.ends = s.ends; state.nodes = s.nodes;
    repaint();
    go();
  });
  for (const b of [...endBoxes(), ...nodeBoxes()]) {
    b.addEventListener('change', () => {
      // Every box of a group off asks for nothing; put the one just cleared back.
      if (!endBoxes().some(x => x.checked) || !nodeBoxes().some(x => x.checked)) {
        b.checked = true;
        return;
      }
      boxesToState();
      repaint();
      go();
    });
  }
  el('sModel').addEventListener('change', () => { state.model = el('sModel').value; go(); });
  el('sBase').addEventListener('change', () => { state.base_id = el('sBase').value; go(); });
  el('sSteps').addEventListener('change', () => { state.steps = el('sSteps').value; go(); });
  el('sFwd').addEventListener('change', () => { state.forward_only = el('sFwd').checked ? '1' : ''; go(); });
  el('sExtend').addEventListener('change', () => { state.extend = el('sExtend').checked ? '1' : ''; go(); });
  el('sRec').addEventListener('change', () => { state.chosen_only = el('sRec').checked ? '' : '1'; repaint(); go(); });

  let typing = null;
  for (const [id, key] of [['sPrompt', 'prompt'], ['sPrefix', 'prefix']]) {
    el(id).addEventListener('input', () => {
      state[key] = el(id).value;
      clearTimeout(typing);
      typing = setTimeout(() => { paintInfo(); }, 350);   // one query per pause
    });
    el(id).addEventListener('change', () => { state[key] = el(id).value; go(); });
  }

  const panel = el('sPanel'), more = el('sMore');
  const MORE_KEY = 'bar_more';
  let open = false;
  try { open = localStorage.getItem(MORE_KEY) === '1'; } catch {}
  function paintMore() {
    panel.hidden = !open;
    more.setAttribute('aria-expanded', String(open));
    more.textContent = open ? 'less ▴' : 'more ▾';
  }
  more.addEventListener('click', () => {
    open = !open;
    try { localStorage.setItem(MORE_KEY, open ? '1' : '0'); } catch {}
    paintMore();
  });
  paintMore();

  repaint();

  // What the page can ask the bar, rather than reading its internals.
  window.settingsBar = { state, query, destination, repaint, landedOnId };
})();
