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
/* ---------------------------------------------------------------- shared bits

   Token colouring by probability lived only in the lists view, so the same token
   was plain in the browser and coloured in the table. It belongs to the data, not
   to one page, so it sits here -- the one script both pages load -- and both use
   it. Green above 0.6, amber above 0.15, red below; alpha tracks the probability
   so a near-certain token reads as solid and an unlikely one as a faint wash. */
window.tokColor = function tokColor(logprob) {
  if (logprob === null || logprob === undefined || !isFinite(logprob)) return '';
  const p = Math.exp(logprob);
  const a = 0.10 + 0.55 * Math.min(1, p);
  if (p > 0.6)  return `rgba(45,160,110,${a})`;
  if (p > 0.15) return `rgba(200,150,50,${a})`;
  return `rgba(190,70,75,${Math.max(0.12, a)})`;
};

/* Inline SVG at currentColor, 15px: no text on the strip, so each button has to
   carry its meaning in the glyph alone. A palette for colour, sun and moon for
   the palette itself, a gear for everything else. */
const ICON = {
  colour: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3a9 9 0 1 0 0 18c1.1 0 2-.9 2-2 0-1.2-1-1.7-1-2.7 0-.8.7-1.3 1.6-1.3H17a4 4 0 0 0 4-4c0-4.4-4-8-9-8Z"/><circle cx="8" cy="10" r="1.1" fill="currentColor" stroke="none"/><circle cx="12" cy="7.5" r="1.1" fill="currentColor" stroke="none"/><circle cx="16" cy="10" r="1.1" fill="currentColor" stroke="none"/></svg>',
  sun: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M18.7 5.3l-1.4 1.4M6.7 17.3l-1.4 1.4"/></svg>',
  moon: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z"/></svg>',
  gear: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3.1"/><path d="M12 2.6v2.2M12 19.2v2.2M4.4 4.4l1.6 1.6M18 18l1.6 1.6M2.6 12h2.2M19.2 12h2.2M4.4 19.6 6 18M18 6l1.6-1.6"/></svg>',
};

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
    // Off by default: the colours help when you are reading probabilities and get
    // in the way when you are reading the text.
    colour: '',
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

  /* `defaults=1` means "this URL is the whole story": start from the defaults and
     let only the params present override them. The landing sends it, because
     naming just view and top let a remembered `ends=stop` leak in -- the bar then
     reported "1 of 4 match" over a string that was not filtered at all. */
  const fromDefaults = params.get('defaults') === '1';
  const state = {};
  for (const [k, dflt] of Object.entries(FIELDS)) {
    state[k] = params.has(k) ? params.get(k)
      : (!fromDefaults && saved[k] !== undefined ? saved[k] : dflt);
  }
  // `id` is a destination, not a setting: it says which record the token browser
  // is showing, and it must never be carried over to a different query.
  const landedOnId = params.has('id');

  function persist() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch {}
  }

  /* What each view actually reads. A link that carries settings the view ignores
     -- a walk with `ends=stop` on it -- says something about the screen that is
     not true, and the next person to open it has to work out which half counts. */
  const USES = {
    greedy: ['top', 'sort', 'prompt', 'model', 'ends', 'nodes', 'extend', 'colour'],
    completions: ['top', 'sort', 'prefix', 'model', 'ends', 'nodes', 'extend', 'colour'],
    prefixes: ['top', 'prefix', 'model', 'ends', 'nodes', 'extend', 'chosen_only', 'colour'],
    sweep: ['base_id', 'model', 'colour'],
    walk: ['base_id', 'steps', 'forward_only', 'model', 'colour'],
  };

  function query() {
    const p = new URLSearchParams();
    const uses = USES[state.view] || Object.keys(FIELDS);
    for (const [k, dflt] of Object.entries(FIELDS)) {
      if (k !== 'view' && !uses.includes(k)) continue;
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

  /* One visible strip, and it holds only what is about *reading* what is on
     screen: the colour scale and the palette. Everything that changes WHICH
     strings are on screen is a click away, because those are decisions and these
     are comfort. The count stays out because it is not a setting -- it says what
     you are looking at. */
  host.innerHTML = `
    <div class="sbar">
      <button type="button" class="sicon" id="sColour" aria-pressed="false"
              title="Colour — shade each token by the probability the model gave it at that position">${ICON.colour}</button>
      <button type="button" class="sicon" id="sTheme"></button>
      <button type="button" class="sicon" id="sMore" aria-expanded="false" aria-controls="sPanel"
              title="Settings">${ICON.gear}</button>
      <span class="sinfo" id="sInfo"></span>
      <!-- Where a page hangs its own status line, so it needs no strip of its
           own. The token browser's was the third row on screen. -->
      <span class="spagestatus" id="sPageStatus"></span>
    </div>
    <div class="smorepanel" id="sPanel" hidden>
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
      <div class="sfield" id="sScopeWrap"><span>how complete</span>
        <span class="sscope">
          <input type="range" id="sScope" min="0" max="2" step="1" list="sTicks">
          <datalist id="sTicks"><option value="0"></option><option value="1"></option><option value="2"></option></datalist>
          <b id="sScopeName"></b>
        </span>
      </div>
      <label class="sfield"><span>starting prompt</span>
        <input type="text" id="sPrompt" placeholder="(empty = unconditional)" spellcheck="false" value="${state.prompt.replace(/"/g, '&quot;')}">
      </label>
      <label class="sfield"><span>starts with</span>
        <input type="text" id="sPrefix" placeholder="e.g. I have" spellcheck="false" value="${state.prefix.replace(/"/g, '&quot;')}">
      </label>
      <label class="sfield" id="sModelWrap"><span>model</span>
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
      <div class="sfield" id="sEndsWrap"><span>endings</span>
        <span class="sends" id="sEnds"></span>
      </div>
      <!-- Where a page hangs its own controls. The token browser's toolbar --
           new tree, the caches, its model select -- was a third row on screen. -->
      <div id="sPageExtra"></div>
      <!-- Help is in the nav above; a second link to it here was the same link
           twice on one screen. -->
      <div class="sfoot">
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
    el('sEndsWrap').hidden = !SCOPED(v);
    el('sPrompt').parentElement.hidden = v !== 'greedy';
    el('sPrefix').parentElement.hidden = !RANKED(v);
    const view = VIEWS.find(([x]) => x === v);
    el('sView').title = view ? view[2] : '';
  }

  /* How many strings the settings on screen match. The list can never show them
     all -- the prefixes view matches 176,649 -- so the count is the only honest
     way to say what a screenful is a screenful of. */
  let infoSeq = 0;
  /* A page with a richer counter of its own claims the slot, and the bar stops
     writing it. Two writers on one element is a race, and the lists page's text
     ("200 of 363 results · 451 unreachable · 8492 records in store") says
     strictly more than the bar's could. */
  let infoSuppressed = false;
  function suppressInfo() { infoSuppressed = true; ++infoSeq; el('sInfo').textContent = ''; }
  async function paintInfo() {
    if (infoSuppressed) return;
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

  /* The field is `base_id`, not `id` -- reading the wrong one left base_id empty,
     /api/sweep answered 400 because it is required, and the grid came up blank. */
  /* Guarded by the in-flight promise, not by the select being filled: repaint()
     and the ready promise both call this and both start before the first one has
     answered, so the option check let two identical fetches through. */
  let basesPromise = null;
  function ensureBases() {
    return (basesPromise = basesPromise || loadBases());
  }
  async function loadBases() {
    const sel = el('sBase');
    if (sel.options.length) return;
    try {
      const { bases } = await (await fetch('/api/bases', { cache: 'no-store' })).json();
      sel.innerHTML = bases.map(b => {
        const t = (b.text || b.base_id).slice(0, 40);
        return opt(b.base_id, `${b.calls}× — ${t.replace(/&/g, '&amp;').replace(/</g, '&lt;')}`,
                   state.base_id);
      }).join('');
      // The grid and the walk cannot query without one, so pick the first.
      if (!state.base_id && bases[0]) state.base_id = bases[0].base_id;
      sel.value = state.base_id;
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

  /* Applied here and on every load, not by navigating: nothing about the query
     changes, so re-fetching and re-rendering to repaint backgrounds would be work
     for its own sake. */
  function paintColour() {
    const on = !!state.colour;
    el('sColour').setAttribute('aria-pressed', String(on));
    el('sColour').classList.toggle('on', on);
    document.body.classList.toggle('tok-colour', on);
  }
  el('sColour').addEventListener('click', () => {
    state.colour = state.colour ? '' : '1';
    persist();
    paintColour();
    // Keep the address in step so a link still reproduces the screen.
    const p = query();
    const id = params.get('id');
    if (id) p.set('id', id);
    /* Keep `defaults=1` if this URL had it. Dropping it turned a
       self-contained address into one that falls back to remembered state, so a
       reload -- or a recipient with different settings -- would not see this
       screen. */
    if (fromDefaults) p.set('defaults', '1');
    history.replaceState(null, '', location.pathname + '?' + p.toString());
  });
  paintColour();

  /* The glyph shows the palette you are IN, not the one a click would give you.
     A button that pictures its own effect reads as a state indicator half the
     time and as an action the other half, with no way to tell which at a glance. */
  function paintTheme() {
    const t = window.appTheme;
    if (!t) { el('sTheme').hidden = true; return; }
    const dark = t.get() === 'dark';
    el('sTheme').innerHTML = dark ? ICON.moon : ICON.sun;
    el('sTheme').title = `Palette: ${dark ? 'dark' : 'light'} — click for ${dark ? 'light' : 'dark'}`;
  }
  el('sTheme').addEventListener('click', () => { window.appTheme.toggle(); paintTheme(); });
  window.addEventListener('themechange', paintTheme);
  paintTheme();

  const panel = el('sPanel'), more = el('sMore');
  const MORE_KEY = 'bar_more';
  let open = false;
  try { open = localStorage.getItem(MORE_KEY) === '1'; } catch {}
  function paintMore() {
    panel.hidden = !open;
    more.setAttribute('aria-expanded', String(open));
    // Icon only, so the pressed look is what says whether it is open.
    more.classList.toggle('on', open);
  }
  more.addEventListener('click', () => {
    open = !open;
    try { localStorage.setItem(MORE_KEY, open ? '1' : '0'); } catch {}
    paintMore();
  });
  paintMore();

  /* The grid and the walk cannot be queried without a base_id, and resolving it
     means a fetch. The page awaits this before its first load, or it fires the
     query with no base and gets a 400. */
  const ready = (async () => { if (!LISTY(state.view)) await ensureBases(); })();

  repaint();

  // What the page can ask the bar, rather than reading its internals.
    /* A page moves its own controls into the panel instead of keeping a strip of
     its own, and its status line onto the bar. Both were a whole extra row. */
  function adopt(node, where) {
    const slot = el(where === 'status' ? 'sPageStatus' : 'sPageExtra');
    if (slot && node) slot.appendChild(node);
  }
  function hideField(id) { const e = el(id); if (e) e.hidden = true; }

  window.settingsBar = { state, query, destination, repaint, ready, suppressInfo,
                         adopt, hideField, landedOnId };
})();
