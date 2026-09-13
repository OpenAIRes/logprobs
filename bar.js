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
  /* A branch: one line carrying straight on, one leaving it at a node -- the
     shape version control draws a branch with, because it says the same thing
     about a string. */
  branch: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="7" cy="5" r="2.1"/><circle cx="7" cy="19" r="2.1"/><circle cx="17.5" cy="12" r="2.1"/><path d="M7 7.1v9.8"/><path d="M9.1 5h2.4a4 4 0 0 1 4 4v.9"/></svg>',
  gear: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 2h4l.5 2.2 1.4.6 1.9-1.2 2.6 2.6-1.2 1.9.6 1.4 2.2.5v4l-2.2.5-.6 1.4 1.2 1.9-2.6 2.6-1.9-1.2-1.4.6L14 22h-4l-.5-2.2-1.4-.6-1.9 1.2-2.6-2.6 1.2-1.9-.6-1.4L2 14v-4l2.2-.5.6-1.4-1.2-1.9 2.6-2.6 1.9 1.2 1.4-.6L10 2Z"/><circle cx="12" cy="12" r="3"/></svg>',
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
    /* Empty means every model. It has to be expressible: a ranking over records
        can span models, and making one of them the default would quietly drop the
        davinci and babbage records from the default list. A decode path is a
        different matter -- see GREEDY_MODEL. */
    model: '',
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
    // Which databases get ranked. Empty means all of them.
    sources: '',
  };

  const VIEWS = [
    ['greedy1', 'greedy I', 'Departures from all discovered strings, chosen by prefix probability.'],
    ['greedy', 'greedy II', 'One string per top-20 first token after the starting prompt, followed by its stored continuation.'],
    ['completions', 'completions', 'One entry per recorded call — the whole string it produced, prompt included.'],
    ['prefixes', 'prefixes', 'Every prefix in the token trie, in exact n-best order.'],
    ['sweep', 'alternatives grid', 'Per-position top-k for one base completion.'],
    ['walk', 'cheapest-deviation walk', 'Repeatedly take the cheapest single-token edit, then regenerate.'],
    ['deviations', 'one-token deviations', 'Every position of one string \u00d7 every alternative recorded there, each one a new prompt the model regenerates from.'],
  ];

  /* Every criterion is offered everywhere, because picking one is how you choose
     what you are looking at -- see viewForSort. `deviation cost` used to be
     offered only on the greedy view, which made it look like an accessory of that
     view rather than the way into it. */
  const SORTS = [
    ['cost', 'deviation cost'],
    ['sum', 'Σ logprob'],
    ['ppl', 'perplexity'],
    ['length', 'length'],
  ];

  /* The criterion decides the view, not the other way round.

     `deviation cost` is measured against the greedy path -- there is no cost
     without a path to depart from -- so choosing it IS choosing greedy, and with
     nothing else said that path starts from the empty prompt, i.e. what
     temperature 0 produces. The other three rank a pool of finished strings, so
     choosing one of them leaves greedy.

     The exception, and it is the reason the greedy view has a `results` control
     at all: greedy asking for more than one result is the path plus its one-token
     siblings, and that IS a pool worth ranking by sum or perplexity. Only the
     single-string case has nothing to rank. */
  function viewForSort(next) {
    /* Both of these are keyed to one string, so there is no ranking of the whole
       store to fall back to -- changing the criterion reorders what is on screen
       rather than choosing something else to look at. */
    if (state.view === 'greedy1') return 'greedy1';
    if (state.view === 'deviations') return 'deviations';
    if (next === 'cost') return state.view === 'greedy1' ? 'greedy1' : 'greedy';
    if ((state.view === 'greedy' || state.view === 'greedy1') && String(state.top) !== '1') return state.view === 'greedy1' ? 'greedy1' : 'greedy';
    return 'completions';
  }

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

  /* What may be moved onto the visible strip, in the order it appears there. A
     fixed order matters: laying the strip out in the order the boxes happened to
     be ticked would make the same choices look different on two machines.

     The strip and the panel share one node per control -- it is moved, not
     copied. A second copy is how the last round of desync bugs started: two
     controls for one setting and no way to tell which one won. */
  const PINNABLE = [
    ['sViewWrap', 'view'],
    ['sResultsWrap', 'results'],
    ['sSortWrap', 'ranked by'],
    ['sScopeWrap', 'how complete'],
    ['sPromptWrap', 'starting prompt'],
    ['sPrefixWrap', 'starts with'],
    ['sModelWrap', 'model'],
    ['sBaseWrap', 'base completion'],
    ['sStepsWrap', 'steps'],
    ['sFwdWrap', 'positions'],
    ['sExtendWrap', 'rows — extend'],
    ['sRecWrap', 'tokens — recovered'],
    ['sTokenLimitWrap', 'tokens per string'],
    ['sEndsWrap', 'endings'],
    ['sJsonWrap', 'json — the raw record'],
    ['sSourcesWrap', 'databases'],
  ];
  const PINNED_KEY = 'bar_pinned';
  const HIDE_WHEN_DEAD = new Set(['sJsonWrap']);

  /* Page-level display, as opposed to controls. Each one is a class on <body> and
     the pages' CSS keys off it, so turning one on and off costs no request and no
     re-render -- the same arrangement as the colour box. */
  const BLOCKS = [
    ['show-source-status', 'source status — from history / cache / API', false],
    ['show-meta', 'details block — provenance, counts, filters in force', false],
    ['wide-rows', 'full text per row — no wrapping, the table scrolls sideways', true],
  ];
  const BLOCKS_KEY = 'bar_blocks';
  const TOKEN_LIMIT_KEY = 'bar_token_limit';

  /* A greedy path is per model -- argmax at every step means argmax of ONE
     model's distribution -- so unlike a ranking it cannot be asked for "any".
     Chaining records from several models would not produce a mixed path, it
     would produce a wrong one. */
  const GREEDY_MODEL = 'gpt-3.5-turbo-instruct';
  const greedyModel = () => state.model || GREEDY_MODEL;

  const RANKED = v => v === 'completions' || v === 'prefixes';
  // The one case where no ranking is involved at all.
  const singleGreedy = () => state.view === 'greedy' && String(state.top) === '1';
  const LISTY = v => RANKED(v) || (v === 'greedy' || v === 'greedy1') || v === 'deviations';
  const SCOPED = v => LISTY(v);
  /* Which views work from one named record instead of from the whole store. The
     grid and the walk already did; deviations is the third, and `base_id` being
     tied to "not a list" was what made it look as though a per-record view could
     not also be a ranked list. */
  const BASED = v => v === 'sweep' || v === 'walk' || v === 'deviations';
  /* Which views are ordered by what a one-token departure cost. There is no cost
     without something to depart from, which is why the other views cannot be. */
  const DEPARTURES = v => v === 'greedy' || v === 'greedy1' || v === 'deviations';

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
    greedy1: ['top', 'sort', 'prompt', 'model', 'ends', 'nodes', 'extend', 'colour', 'sources'],
    greedy: ['top', 'sort', 'prompt', 'model', 'ends', 'nodes', 'extend', 'colour', 'sources'],
    completions: ['top', 'sort', 'prefix', 'model', 'ends', 'nodes', 'extend', 'colour', 'sources'],
    prefixes: ['top', 'prefix', 'model', 'ends', 'nodes', 'extend', 'chosen_only', 'colour', 'sources'],
    sweep: ['base_id', 'model', 'colour'],
    walk: ['base_id', 'steps', 'forward_only', 'model', 'colour'],
    deviations: ['base_id', 'model', 'sort', 'ends', 'nodes', 'sources', 'colour'],
  };

  function query() {
    const p = new URLSearchParams();
    const uses = USES[state.view] || Object.keys(FIELDS);
    for (const [k, dflt] of Object.entries(FIELDS)) {
      if (k !== 'view' && !uses.includes(k)) continue;
      /* `sort` stays out of a single-string greedy link: `cost` is that view's
         own default, so writing it would add a parameter that says nothing, and
         no other value can survive there -- picking one moves the view. */
      if (k === 'sort' && singleGreedy() && effectiveSort() === 'cost') continue;
      const v = String(state[k] ?? '');
      if (v && v !== String(dflt)) p.set(k, v);
    }
    return p;
  }

  // ---------------------------------------------------------------- destination

  /* One result is a single string, which belongs in the token browser; more than
     one is a ranked list. The bar owns this decision so neither page has to. */
  async function destination() {
    // A per-record view is a list however few rows it has: `results` does not
    // apply to it, so a 1 left over from another view must not turn it into a
    // link to a single string.
    const single = LISTY(state.view) && !BASED(state.view) && String(state.top) === '1';
    if (!single || state.view === 'greedy1') {
      const p = query();
      p.set('view', state.view);        // always explicit in a list link
      return '/strings.html?' + p.toString();
    }
    let id = null;
    try {
      if ((state.view === 'greedy' || state.view === 'greedy1')) {
        // Resolved by walking argmax records rather than by ranking anything, so
        // it has its own endpoint; its first hop is where the browser lands.
        const q = new URLSearchParams({ prompt: state.prompt, model: greedyModel() });
        const g = await (await fetch('/api/greedy?' + q, { cache: 'no-store' })).json();
        id = g.first_id || null;
      } else {
        const q = new URLSearchParams({ view: state.view, top: '1' });
        if (state.prefix) q.set('prefix', state.prefix);
        if (state.sort) q.set('sort', state.sort);
        if (state.ends) q.set('ends', state.ends);
        if (state.nodes) q.set('nodes', state.nodes);
        if (state.chosen_only) q.set('chosen_only', '1');
        if (state.sources) q.set('sources', state.sources);
        if (state.model) q.set('model', state.model);
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
      <!-- Only on screen when there is one record to work from: with a list up
           there is no single string to perturb, and a greyed-out glyph with no
           label is worse than an absence (the same reason the json button
           disappears). Backticks are banned in here: this is inside a template
           literal, and one of them ends it. -->
      <button type="button" class="sicon" id="sVariants" hidden
              title="Jednotokenové odchylky — každá pozice tohoto řetězce × každá alternativa, kterou tam model zaznamenal, jako nový prompt; pokračování se bere z databáze, a co v ní není, se dotáhne jen na vyžádání (to je to placené). 16 tokenů ≈ 300 řetězců.">${ICON.branch}</button>
      <button type="button" class="sicon" id="sMore" aria-expanded="false" aria-controls="sPanel"
              title="Settings">${ICON.gear}</button>
      <span class="sinfo" id="sInfo"></span>
      <!-- Where a page hangs its own status line, so it needs no strip of its
           own. The token browser's was the third row on screen. -->
      <span class="spagestatus" id="sPageStatus"></span>
    </div>
    <div class="smorepanel" id="sPanel" hidden>
      <label class="sfield" id="sViewWrap"><span>view</span>
        <select id="sView">${VIEWS.map(([v, l]) => opt(v, l, state.view)).join('')}</select>
      </label>
      <label class="sfield" id="sResultsWrap"><span>results</span>
        <input type="number" id="sResults" min="1" max="5000" step="1" list="sResultPresets" value="${Number(state.top) || 20}">
        <datalist id="sResultPresets"><option value="1"></option><option value="20"></option><option value="50"></option><option value="200"></option><option value="500"></option><option value="1000"></option><option value="2000"></option></datalist>
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
      <label class="sfield" id="sPromptWrap"><span>starting prompt</span>
        <input type="text" id="sPrompt" placeholder="(empty = unconditional)" spellcheck="false" value="${state.prompt.replace(/"/g, '&quot;')}">
      </label>
      <label class="sfield" id="sPrefixWrap"><span>starts with</span>
        <input type="text" id="sPrefix" placeholder="e.g. I have" spellcheck="false" value="${state.prefix.replace(/"/g, '&quot;')}">
      </label>
      <label class="sfield" id="sModelWrap"><span>model</span>
        <select id="sModel">
          ${[['', 'any'], ['gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-instruct'],
             ['davinci-002', 'davinci-002'], ['babbage-002', 'babbage-002']]
            .map(([v, l]) => opt(v, l, state.model)).join('')}
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
      <div class="sfield" id="sSourcesWrap"><span>databases</span>
        <span class="sends" id="sSources"></span>
      </div>
      <!-- No label: the word on the button is the label. -->
      <div class="sfield sbare" id="sJsonWrap">
        <button type="button" id="sJson">json</button>
      </div>
      <div class="sfield" id="sTokenLimitWrap"><label for="sTokenLimit">tokens per string</label>
        <select id="sTokenLimit">
          <option value="0">all available</option>
          <option value="16">16</option><option value="20">20</option>
          <option value="50">50</option><option value="100">100</option>
          <option value="250">250</option><option value="500">500</option>
          <option value="1000">1,000</option><option value="custom">vlastní…</option>
        </select>
        <input type="number" id="sTokenLimitCustom" min="1" step="1" hidden aria-label="Vlastní počet tokenů na řetězec" placeholder="počet">
      </div>
      <div class="sfield" id="sAskWrap"><span title="Kdy se program zeptá, než pošle volání na API. Volání, na které je odpověď v databázi, se neděje, takže se na ně ani neptá. Nastavení platí pro celý program a je uložené na serveru, ne v prohlížeči.">ptát se před API</span>
        <span class="sends" id="sAsk"></span>
      </div>
      <label class="sfield" id="sCallMaxWrap" title="max_tokens každého volání, které tenhle prohlížeč pošle: klik na token, odchylky, větve. Prázdné = podle modelu (20 pro gpt-3.5-turbo-instruct, 5 pro base modely). Pozor: záznam vyrobený na 20 umí odpovědět na dotaz na 5 tím, že se zkrátí, ale ne naopak — víc, než na kolik běžely sweepy, tedy mění trefy v databázi na placená volání."><span>délka volání</span>
        <input type="number" id="sCallMax" min="1" max="4096" step="1" placeholder="podle modelu">
      </label>
      <details class="sadv" id="sAdv"><summary>advanced — what shows</summary>
        <p class="sadvhint">On the strip: ticked controls sit on the visible strip
          instead of in here. Nothing is duplicated — the control moves. Untick
          everything for the three icons and nothing else.</p>
        <div class="sadvlist" id="sAdvList"></div>
        <p class="sadvhint">On the page:</p>
        <div class="sadvlist" id="sBlockList"></div>
        <button type="button" id="sPinReset">back to minimal</button>
      </details>
      <!-- Where a page hangs its own controls. The token browser's toolbar --
           new tree, the caches, its model select -- was a third row on screen. -->
      <div id="sPageExtra"></div>
      <!-- Help is in the nav above; a second link to it here was the same link
           twice on one screen. -->
      <!-- The nav strip is gone -- it held one link most of the time and a record
           count that the meta block already reported -- so its links live here. -->
      <div class="sfoot">
        <a href="/">start over</a>
        <span class="sep">·</span>
        <a href="/help.html">help</a>
        <span class="sep">·</span>
        <a href="/index.html">the full settings page, with the counts</a>
      </div>
    </div>
    <div class="sjson" id="sJsonOut" hidden></div>`;

  const el = id => document.getElementById(id);

  /* Declared here, next to el(), because layout() reads it and `const` is not
     hoisted -- leaving it further down was a ReferenceError waiting for whichever
     listener fired first. */
  const panel = el('sPanel'), more = el('sMore');

  /* Which record `json` should show. Asked for on demand rather than pushed at
     us: the token browser assigns its current id in eight different places, and
     hooking all eight is eight chances to miss one. */
  let recordSourceFn = null;
  function recordId() {
    try { return (recordSourceFn && recordSourceFn()) || null; } catch { return null; }
  }
  function provideRecord(fn) { recordSourceFn = fn; repaint(); }

  /* Pretty-printing the whole thing is not always sane: the 4096-token record is
     2.9 MB formatted, and pouring that into a <pre> stalls the tab. So the block
     shows the head and says exactly how much it left out, with the untruncated
     route one click away -- the browser renders JSON itself. */
  const JSON_CAP = 200000;
  let jsonOpen = false;
  async function showJson() {
    const box = el('sJsonOut');
    const id = recordId();
    if (!id) { box.hidden = true; return; }
    jsonOpen = !jsonOpen;
    el('sJson').classList.toggle('on', jsonOpen);
    if (!jsonOpen) { box.hidden = true; return; }
    box.hidden = false;
    box.textContent = 'loading…';
    try {
      const res = await fetch('/api/record?id=' + encodeURIComponent(id), { cache: 'no-store' });
      const rec = await res.json();
      const text = JSON.stringify(rec, null, 2);
      const choice = (rec.choices || [{}])[0] || {};
      const n = (((choice.logprobs || {}).tokens) || []).length;
      const head = `${id} · ${n} generated tokens · ${(text.length / 1024).toFixed(0)} kB formatted`;
      const cut = text.length > JSON_CAP;
      box.innerHTML = `<div class="sjsonhead">${head}`
        + ` · <a href="/api/record?id=${encodeURIComponent(id)}" target="_blank" rel="noopener">open the whole thing</a></div>`
        + `<pre></pre>`;
      box.querySelector('pre').textContent = cut
        ? text.slice(0, JSON_CAP)
          + `\n\n… ${((text.length - JSON_CAP) / 1024).toFixed(0)} kB not shown; `
          + 'use the link above for the rest.'
        : text;
    } catch (err) {
      box.textContent = 'could not read the record: ' + err.message;
    }
  }
  el('sJson').addEventListener('click', showJson);

  /* Its own page rather than a block on this one: the set runs to hundreds of
     full-length strings, which wants a table and a download, not a strip. A new
     tab, so the string being read is still there to come back to.

     Deviations, not /single-token-variants.html: the two answer different
     questions and only the first one is what this icon means. Variants keeps the
     original tail and never calls a model; deviations regenerates the tail, so
     its strings are ones the model really produces. The variants page is still
     there at its own URL.

     It used to be a page of its own. It is a view of the list now, which is how
     it gets the colour key, the scope stepper and the rest of this bar -- the
     things a table of strings wants and a one-off page had none of. */
  el('sVariants').addEventListener('click', () => {
    const id = recordId();
    if (!id) return;
    const p = new URLSearchParams({ view: 'deviations', base_id: id });
    if (state.model) p.set('model', state.model);
    window.open('/strings.html?' + p, '_blank', 'noopener');
  });

  // ---------------------------------------------------------------- scope stepper

  const ENDS_ALL = ['stop', 'length', 'open'];
  const NODES_ALL = ['continues', 'leaf'];
  el('sEnds').innerHTML = ENDS_ALL.map(v =>
    `<label><input type="checkbox" class="sEndBox" value="${v}"> ${v}</label>`).join('')
    + NODES_ALL.map(v =>
    `<label><input type="checkbox" class="sNodeBox" value="${v}"> ${v === 'leaf' ? 'dead end' : v}</label>`).join('');

  /* Which record files get ranked. Two groups rather than six checkboxes,
     because the six are two kinds of thing: calls made by hand, and the sweeps --
     8081 of the 8492 records, so they dominate any ranking they are in. */
  const SOURCE_GROUPS = [
    ['history', 'history', 'calls made by hand: completion_history, builder_history, meta_resample_root — 411 records'],
    ['sweep', 'sweeps', 'exhaustive one-token perturbations of a few bases: sweep_history, sweep2, sweep3 — 8081 records'],
  ];
  el('sSources').innerHTML = SOURCE_GROUPS.map(([v, label, why]) =>
    `<label title="${why}"><input type="checkbox" class="sSrcBox" value="${v}"> ${label}</label>`).join('');
  const srcBoxes = () => [...document.querySelectorAll('.sSrcBox')];
  function paintSources() {
    const on = listOf(state.sources);
    for (const b of srcBoxes()) b.checked = !on.length || on.includes(b.value);
  }
  for (const b of srcBoxes()) {
    b.addEventListener('change', () => {
      // Every group off asks for nothing; put the one just cleared back.
      if (!srcBoxes().some(x => x.checked)) { b.checked = true; return; }
      const on = srcBoxes().filter(x => x.checked).map(x => x.value);
      state.sources = on.length === SOURCE_GROUPS.length ? '' : on.join(',');
      repaint();
      go();
    });
  }

  /* When the program asks before paying. Not a PINNABLE control: it is a policy
     for the whole program rather than something about what is on screen, and it
     is the same setting on every page and in every folder -- which is why it is
     stored on the server and not in this browser. */
  const ASK = window.AskPolicy;
  if (ASK) {
    el('sAsk').innerHTML = ASK.CONDITIONS.map(([k, label, why]) =>
      `<label title="${why.replace(/"/g, '&quot;')}"><input type="checkbox" class="sAskBox" value="${k}"> ${label}</label>`).join('')
      + '<label title="Neptat se nikdy. Volání pak odcházejí bez potvrzení."><input type="checkbox" class="sAskBox" value="never"> nikdy</label>';
  }
  const askBoxes = () => [...document.querySelectorAll('.sAskBox')];
  /* Empty means "per model" -- the box shows what that comes to for the model in
     force, as a placeholder, so the number is never a mystery. */
  function paintCallMax() {
    if (!ASK) return;
    const box = el('sCallMax');
    const set = ASK.get().max_tokens;
    if (document.activeElement !== box) box.value = set ? String(set) : '';
    box.placeholder = `podle modelu (${ASK.perModelMax(greedyModel())})`;
  }
  el('sCallMax').addEventListener('change', async () => {
    const box = el('sCallMax');
    if (box.value && !box.reportValidity()) return;
    await ASK.set({ ...ASK.get(), max_tokens: box.value ? Number(box.value) : 0 });
    paintCallMax();
  });

  function paintAsk() {
    if (!ASK) return;
    const p = ASK.get();
    const none = !p.always && !p.big && !p.batch;
    for (const b of askBoxes()) {
      b.checked = b.value === 'never' ? none : !!p[b.value];
      // `always` already covers the two conditions, so leaving them clickable
      // would offer a choice that changes nothing.
      b.disabled = p.always && (b.value === 'big' || b.value === 'batch');
    }
  }
  for (const b of askBoxes()) {
    b.addEventListener('change', async () => {
      const v = b.value;
      let next;
      if (v === 'never') next = {always: false, big: false, batch: false};
      else if (v === 'always') next = b.checked ? {always: true, big: false, batch: false}
                                                : {always: false, big: false, batch: false};
      else {
        const p = ASK.get();
        next = {always: false, big: p.big, batch: p.batch};
        next[v] = b.checked;
      }
      // Carry the length through: it lives in the same policy object.
      await ASK.set({ ...ASK.get(), ...next });
      paintAsk();
    });
  }

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

  /* Which criterion is actually in force: the chosen one, or the view's default
     when nothing was chosen. Two places needed this answer and computing it twice
     is how they would come to disagree. */
  function effectiveSort() {
    // Prefixes come out of a sum-ordered heap: the ranking IS the search order,
    // so nothing else can be in force there whatever was last chosen.
    if (state.view === 'prefixes') return 'sum';
    const want = state.sort || (DEPARTURES(state.view) ? 'cost' : 'sum');
    return (want === 'cost' && !DEPARTURES(state.view)) ? 'sum' : want;
  }

  function paintSorts() {
    const sel = el('sSort');
    sel.innerHTML = SORTS.map(([v, l]) => opt(v, v === 'cost' && state.view === 'greedy1' ? 'departure prefix cost' : l, effectiveSort())).join('');
    sel.title = state.view === 'greedy1' ? 'Discovery order maximizes the departure prefix logprob; other criteria sort the discovered set.' : 'The criterion decides what you are looking at: deviation cost is '
      + 'measured against the greedy path, so it selects that path; the others rank '
      + 'recorded strings.';
  }

  /* Why a control does not apply to the current view, or null when it does. A
     reason rather than a boolean: a pinned control is shown disabled and has to
     be able to say what is wrong. Vanishing from the strip after you deliberately
     put it there is not an answer. */
  function inapplicable() {
    const v = state.view, one = String(state.top) === '1';
    const notList = LISTY(v) ? null : `${v} is not a ranked list of strings`;
    return {
      sResultsWrap: v === 'deviations'
        ? 'every deviation of the string is a row — there is nothing to cap'
        : notList,
      // Never dead: every criterion is reachable, and picking one moves the view
      // to where it means something. That is the whole point of viewForSort.
      sSortWrap: null,
      sScopeWrap: SCOPED(v) ? null : notList,
      sPromptWrap: (v === 'greedy' || v === 'greedy1') ? null : 'only the greedy path starts from a prompt you give',
      sPrefixWrap: RANKED(v) ? null : `${v} has no ranking to filter`,
      sModelWrap: null,
      sBaseWrap: BASED(v) ? null
        : 'a base completion is what the grid, the walk and the deviations work from',
      sStepsWrap: v === 'walk' ? null : 'steps belong to the walk',
      sFwdWrap: v === 'walk' ? null : 'this belongs to the walk',
      sExtendWrap: v === 'deviations'
        ? 'a deviation is already the continuation the model regenerated'
        : !LISTY(v) ? notList
        : one ? 'one string is already shown in full' : null,
      sRecWrap: v === 'prefixes' ? null
        : 'only the trie search walks token by token, so only it can be restricted',
      sEndsWrap: SCOPED(v) ? null : notList,
      /* The raw record is a property of ONE call. A list has many, so the button
         works there only once a row is opened -- the page says which one that is
         through provideRecord(). */
      sJsonWrap: recordId() ? null
        : 'no single record on screen — open one result, or a row of a list',
      sSourcesWrap: LISTY(v) ? null : notList,
    };
  }

  function paintVisibility() {
    const view = VIEWS.find(([x]) => x === state.view);
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
    /* A per-record view already knows its own total -- the plan says how many
       deviations the string has -- and /api/query cannot answer for it anyway,
       so probing there was two 400s on every load. */
    if (!LISTY(v) || BASED(v)) { el('sInfo').textContent = ''; return; }
    const p = new URLSearchParams({ top: '1' });
    if (v !== 'greedy') p.set('view', v);
    if (state.ends) p.set('ends', state.ends);
    if (state.nodes) p.set('nodes', state.nodes);
    if (state.prefix && RANKED(v)) p.set('prefix', state.prefix);
    if (state.chosen_only && v === 'prefixes') p.set('chosen_only', '1');
    if (state.prompt && (v === 'greedy' || v === 'greedy1')) p.set('prompt', state.prompt);
    if (state.sources) p.set('sources', state.sources);
    if ((v === 'greedy' || v === 'greedy1')) p.set('model', greedyModel());
    else if (state.model) p.set('model', state.model);
    try {
      const route = (v === 'greedy' || v === 'greedy1') ? (v === 'greedy1' ? '/api/greedy_i?' : '/api/greedy_alternatives?') : '/api/query?';
      const db = await (await fetch(route + p, { cache: 'no-store' })).json();
      if (mine !== infoSeq) return;
      const n = db.available;
      if (n == null) { el('sInfo').textContent = ''; return; }
      const want = Number(state.top) || 1;
      /* With one result out of a ranking, the criterion is what picked it, and
         nothing else on screen said so. That mattered: `perplexity` and `length`
         name the same record here -- the 4096-token one is both the longest and
         the best per token -- so switching between them looks like a dead
         control unless the line says which one is in force. */
      const by = (want === 1 && !singleGreedy() && LISTY(v))
        ? ` by ${(SORTS.find(([k]) => k === effectiveSort()) || [, ''])[1]}` : '';
      el('sInfo').textContent = want >= n
        ? `${n.toLocaleString('en-US')} match`
        : `${want.toLocaleString('en-US')} of ${n.toLocaleString('en-US')} match${by}`;
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

  /* Moves each control to the side of the fence it belongs on. Called after
     paintVisibility, because a pinned control the current view has no use for
     must still be hidden -- pinning says where it goes, not that it applies. */
  function layout() {
    const on = new Set(pinned);
    const bar = host.querySelector('.sbar');
    const info = el('sInfo');
    const why = inapplicable();
    for (const [id] of PINNABLE) {
      const node = el(id);
      if (!node) continue;
      const wantBar = on.has(id);
      const inBar = node.parentElement === bar;
      if (wantBar && !inBar) bar.insertBefore(node, info);
      else if (!wantBar && inBar) panel.insertBefore(node, el('sAdv'));

      /* A control the current view cannot use: hidden in the panel, where it
         would be clutter, but on the strip it stays put and goes grey. You asked
         for it to be there, so it is there -- and it says why it is unavailable
         instead of disappearing. */
      /* Greying a pinned control keeps it where you put it and lets it explain
         itself. That only works for a control with a label to explain: `json` is
         a bare button, and a greyed-out mystery blob is worse than an absence,
         so this one goes away instead. */
      const dead = why[id];
      const hideDead = HIDE_WHEN_DEAD.has(id);
      node.hidden = !!dead && (!wantBar || hideDead);
      node.classList.toggle('dead', !!dead && wantBar && !hideDead);
      for (const f of node.querySelectorAll('select, input, button')) f.disabled = !!dead;
      node.title = dead ? `Not available: ${dead}.` : '';
    }
    // Re-assert the order every time, so unpinning and pinning again does not
    // shuffle the strip.
    for (const [id] of PINNABLE) {
      const node = el(id);
      if (node && node.parentElement === bar) bar.insertBefore(node, info);
    }
    for (const b of document.querySelectorAll('.sPinBox')) b.checked = on.has(b.value);
  }

  function repaint() {
    paintAsk();
    paintCallMax();
    // repaint() is where every other "can this control do anything right now"
    // decision is made, so putting it here is what keeps the button in step with
    // the record on screen.
    el('sVariants').hidden = !recordId();
    paintSorts();
    paintVisibility();
    paintSources();
    layout();
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
  el('sResults').addEventListener('change', () => { if (!el('sResults').reportValidity() || !el('sResults').value) return; state.top = el('sResults').value; repaint(); go(); });
  el('sSort').addEventListener('change', () => {
    const next = el('sSort').value;
    state.sort = next;
    state.view = viewForSort(next);
    repaint();
    go();
  });
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
  el('sSteps').addEventListener('change', () => { if (!el('sSteps').reportValidity() || !el('sSteps').value) return; state.steps = el('sSteps').value; go(); });
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

  /* Which controls the reader wants on the strip. Kept in localStorage, not the
     URL: it is a layout preference belonging to the person, like the palette, and
     a shared link should reproduce the RESULT, not force someone else's chrome. */
  let pinned = [];
  try {
    const raw = localStorage.getItem(PINNED_KEY);
    if (raw) pinned = raw.split(',').filter(id => PINNABLE.some(([x]) => x === id));
  } catch {}

  el('sAdvList').innerHTML = PINNABLE.map(([id, label]) =>
    `<label><input type="checkbox" class="sPinBox" value="${id}"> ${label}</label>`).join('');
  for (const b of document.querySelectorAll('.sPinBox')) {
    b.addEventListener('change', () => {
      pinned = PINNABLE.map(([x]) => x)
        .filter(x => [...document.querySelectorAll('.sPinBox')]
          .some(y => y.value === x && y.checked));
      try { localStorage.setItem(PINNED_KEY, pinned.join(',')); } catch {}
      layout();
    });
  }
  el('sPinReset').addEventListener('click', () => {
    pinned = [];
    try { localStorage.removeItem(PINNED_KEY); } catch {}
    layout();
  });

  /* Which page blocks are on. Stored, not in the URL: like the palette this is
     about the person reading rather than about which strings are on screen. */
  let blocks = null;
  try {
    const raw = localStorage.getItem(BLOCKS_KEY);
    if (raw) blocks = new Set(raw.split(',').filter(Boolean));
  } catch {}
  if (!blocks) blocks = new Set(BLOCKS.filter(([, , on]) => on).map(([k]) => k));

  function paintBlocks() {
    for (const [key] of BLOCKS) document.body.classList.toggle(key, blocks.has(key));
    for (const b of document.querySelectorAll('.sBlockBox')) b.checked = blocks.has(b.value);
    // A page may render differently, not just show or hide: the lists page caps
    // its rows when the full text is on.
    window.dispatchEvent(new CustomEvent('blockschange'));
  }
  el('sBlockList').innerHTML = BLOCKS.map(([key, label]) =>
    `<label><input type="checkbox" class="sBlockBox" value="${key}"> ${label}</label>`).join('');
  for (const b of document.querySelectorAll('.sBlockBox')) {
    b.addEventListener('change', () => {
      if (b.checked) blocks.add(b.value); else blocks.delete(b.value);
      try { localStorage.setItem(BLOCKS_KEY, [...blocks].join(',')); } catch {}
      paintBlocks();
    });
  }
  paintBlocks();
  // The server is the source of truth for the ask policy, so refresh once and
  // repaint when it answers; until then the cached value applies.
  if (ASK) ASK.load().then(() => { paintAsk(); paintCallMax(); });

  let tokenLimit = 0;
  try { tokenLimit = Math.max(0, Number(localStorage.getItem(TOKEN_LIMIT_KEY)) || 0); } catch {}
  const tokenPresets = [0, 16, 20, 50, 100, 250, 500, 1000];
  tokenLimit = Number.isSafeInteger(tokenLimit) ? tokenLimit : 0;
  const customTokenLimit = el('sTokenLimitCustom');
  el('sTokenLimit').value = tokenPresets.includes(tokenLimit) ? String(tokenLimit) : 'custom';
  customTokenLimit.hidden = el('sTokenLimit').value !== 'custom';
  customTokenLimit.value = tokenLimit > 0 ? String(tokenLimit) : '';
  function saveTokenLimit(value) {
    tokenLimit = value;
    try { localStorage.setItem(TOKEN_LIMIT_KEY, String(tokenLimit)); } catch {}
    window.dispatchEvent(new CustomEvent('tokenlimitchange'));
  }
  el('sTokenLimit').addEventListener('change', () => {
    const custom = el('sTokenLimit').value === 'custom';
    customTokenLimit.hidden = !custom;
    if (custom) {
      customTokenLimit.value = tokenLimit > 0 ? String(tokenLimit) : '';
      customTokenLimit.focus(); customTokenLimit.select();
    } else saveTokenLimit(Number(el('sTokenLimit').value));
  });
  function applyCustomTokenLimit() {
    if (!customTokenLimit.value || !customTokenLimit.reportValidity()) return;
    const value = Number(customTokenLimit.value);
    if (Number.isSafeInteger(value) && value > 0 && value !== tokenLimit) saveTokenLimit(value);
  }
  customTokenLimit.addEventListener('change', applyCustomTokenLimit);
  customTokenLimit.addEventListener('keydown', event => {
    if (event.key === 'Enter') { event.preventDefault(); applyCustomTokenLimit(); }
  });

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

  /* Every filter currently narrowing the result, in words. An empty list has to
     be able to say why it is empty, and since the details block is off by default
     there is otherwise nothing on screen that names the filters at all -- which is
     how "davinci and more than one record shows nothing" became a mystery rather
     than a message. Two remembered settings do that: `complete`, because all 7
     davinci records end on length and none on EOS, and `sweeps only`, because all
     7 are in the history group. */
  function activeFilters() {
    const out = [];
    if (state.model) out.push({ key: 'model', label: 'model', value: state.model });
    const at = stopOf();
    if (state.ends || state.nodes) {
      out.push({ key: 'scope', label: 'how complete',
                 value: at !== null ? STOPS[at].name
                        : [state.ends, state.nodes].filter(Boolean).join(' + ') });
    }
    if (state.sources) out.push({ key: 'sources', label: 'databases', value: state.sources });
    if (state.chosen_only) out.push({ key: 'chosen_only', label: 'tokens',
                                      value: 'generated only, no recovered' });
    if (state.prefix) out.push({ key: 'prefix', label: 'starts with', value: state.prefix });
    return out;
  }

  /* Puts every narrowing setting back to its default and reloads. Named
     deliberately: it does not touch the view, the count or the sort, because
     those are not why a list came back empty. */
  function clearFilters() {
    for (const k of ['model', 'ends', 'nodes', 'sources', 'chosen_only', 'prefix']) {
      state[k] = '';
    }
    persist();
    go();
  }

  /* A page moves its own controls into the panel instead of keeping a strip of
     its own, and its status line onto the bar. Both were a whole extra row. */
  function adopt(node, where) {
    const slot = el(where === 'status' ? 'sPageStatus' : 'sPageExtra');
    if (slot && node) slot.appendChild(node);
  }
  /* Exported because strings.html had its own copy of the rule, and a second
     copy of "which criterion is in force" is a second thing to remember when a
     view is added. */
  window.settingsBar = { state, query, destination, repaint, ready, suppressInfo,
                         effectiveSort,
                         adopt, provideRecord, landedOnId,
                         activeFilters, clearFilters,
                         block: key => blocks.has(key),
                         tokenLimit: () => tokenLimit };
})();
