/* What is happening around a call to the model -- said once, for everything.

   Five programs had their own answer to this: the token browser's source chip,
   the deviations page's status line, the list view's panel text, the builder's
   log, the resampling program's status pill. They said different things in
   different words, and three of them said nothing at all while a call was in
   flight -- which is how a failed deviation came out as silence and a DNS
   outage came out as a JSON parse error.

   So: one strip, one vocabulary, one place to change it. A page gets it by
   loading this script; there is no markup to add.

   `report` is enough on its own. `run` is the whole shape of a paid call --
   ask, announce, send, announce the outcome -- and exists so that a call site
   is three lines rather than a copy of this reasoning. */
(function (root) {

  /* Stage -> how it reads, and whether it is still going. The wording is the
     only place the user meets these, so it is spelled out here rather than
     assembled from fragments at each call site. */
  const STAGES = {
    looking:   ['Hledám v databázi…',            'busy'],
    asking:    ['Čekám na potvrzení…',           'busy'],
    calling:   ['Volám API…',                    'busy'],
    saving:    ['Ukládám do databáze…',          'busy'],
    cached:    ['Z databáze — nic se nevolalo',  'ok'],
    saved:     ['Hotovo, uloženo',               'ok'],
    declined:  ['Zrušeno — nic se neposlalo',    'ok'],
    stopped:   ['Zastaveno',                     'ok'],
    failed:    ['Nepovedlo se',                  'bad'],
    unreachable: ['API nedostupné — nic se neúčtovalo', 'bad'],
  };

  const CSS = `
  .crp { position: fixed; left: 50%; transform: translateX(-50%); bottom: 1rem;
         z-index: 3500; display: flex; align-items: baseline; gap: .5rem;
         max-width: min(60rem, 94vw); padding: .45rem .8rem;
         background: var(--panel, #fff); color: var(--ink, #111);
         border: 1px solid var(--line, #ddd); border-radius: 999px;
         box-shadow: 0 2px 14px rgba(0,0,0,.18); font-size: .8125rem; }
  .crp[hidden] { display: none !important; }
  .crp-dot { width: .5rem; height: .5rem; border-radius: 50%; flex: none;
             background: var(--ink-faint, #999); }
  .crp.busy .crp-dot { background: var(--mid, #b07d2b); animation: crp-pulse 1s infinite; }
  .crp.ok  .crp-dot { background: var(--good, #2d6a4f); }
  .crp.bad .crp-dot { background: var(--danger, #a63d40); }
  .crp-what { font-weight: 600; white-space: nowrap; }
  .crp-detail { color: var(--ink-dim, #666); overflow: hidden;
                text-overflow: ellipsis; white-space: nowrap; }
  @keyframes crp-pulse { 50% { opacity: .35; } }
  `;

  let strip = null, hideTimer = null;

  function mount() {
    if (strip || typeof document === 'undefined') return strip;
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    strip = document.createElement('div');
    strip.className = 'crp';
    strip.setAttribute('role', 'status');
    strip.hidden = true;
    strip.innerHTML = '<span class="crp-dot"></span><span class="crp-what"></span>'
                    + '<span class="crp-detail"></span>';
    document.body.appendChild(strip);
    return strip;
  }

  /* A finished stage clears itself; one still in flight does not, because a
     strip that vanishes mid-call is how you end up not knowing whether
     anything is happening. */
  function report(stage, detail) {
    const known = STAGES[stage] || [String(stage), 'busy'];
    if (typeof document === 'undefined') return known[0];
    mount();
    clearTimeout(hideTimer);
    strip.className = 'crp ' + known[1];
    strip.hidden = false;
    strip.querySelector('.crp-what').textContent = known[0];
    strip.querySelector('.crp-detail').textContent = detail ? '· ' + detail : '';
    if (known[1] !== 'busy') {
      hideTimer = setTimeout(() => { if (strip) strip.hidden = true; }, 6000);
    }
    return known[0];
  }

  const clear = () => { if (strip) { clearTimeout(hideTimer); strip.hidden = true; } };

  /* The whole shape of one paid call.

     `send` does the actual request and nothing else; everything around it --
     asking, announcing, telling a refusal from a failure -- happens here, so
     the five call sites stop each having their own version of it.

     Returns {ok, verdict, answer, error}. It does not throw: a call site that
     is in a loop has to be able to carry on, and one that is not can read `ok`. */
  async function run(options) {
    const o = options || {};
    const request = o.request || {};
    const label = o.label || '';
    const policy = root.AskPolicy || (typeof window !== 'undefined' && window.AskPolicy);

    if (!policy) {
      report('failed', 'ask-policy.js se nenačetlo — placené volání se neprovede');
      return {ok: false, verdict: 'stop', error: new Error('AskPolicy missing')};
    }

    report('asking', label);
    let verdict = await policy.guard(request, o.info || {total: 1});
    if (verdict === true) verdict = 'yes';
    if (verdict !== 'yes' && verdict !== 'all') {
      report(verdict === 'skip' ? 'stopped' : 'declined', label);
      return {ok: false, verdict, error: null};
    }

    report('calling', label);
    try {
      const answer = await o.send(request);
      // A server that answered from its own history did not spend anything, and
      // saying so is the difference between trusting the total and not.
      report(answer && answer.from_cache ? 'cached' : 'saved', label);
      return {ok: true, verdict, answer, error: null};
    } catch (error) {
      const unreachable = !!(error && (error.unreachable
        || /nedosa|unreachable|getaddrinfo|ENOTFOUND|Failed to fetch/i.test(error.message || '')));
      report(unreachable ? 'unreachable' : 'failed',
             [label, error && error.message].filter(Boolean).join(' · '));
      return {ok: false, verdict, error};
    }
  }

  root.CallReport = {report, clear, run, mount, STAGES};
})(typeof module === 'object' ? module.exports : window);
