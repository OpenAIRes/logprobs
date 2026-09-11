/* When does the program ask before it calls the API -- one rule for every page.

   Four ideas were doing this job separately before: this module's dialog on the
   deviation path, a one-line window.confirm in logprobs' New tree, another one in
   prompt-studio, and nothing at all on the token click (the most used paid path
   in the program). They disagreed about what they showed, and two of them showed
   nothing worth reading. Now there is one policy, one dialog, and one place that
   decides.

   The policy lives on the SERVER, not in localStorage, for a plain reason: the
   viewers are served from 127.0.0.1:8899 and prompt-studio from :8787, which are
   different origins and therefore different localStorage. A setting that claims
   to cover the whole program has to live where the whole program can see it.
   localStorage is kept as a cache, so a page still behaves sensibly with
   server.py down -- and when it cannot tell, it asks rather than assumes.

   This module also owns the per-model call limits, which had three copies
   (greedy-branches.js, logprobs.html, completion.py) of the same two numbers. */
(function (root) {

  const KEY = 'api_ask_policy';

  /* prompt-studio runs on its own origin (its dev server) while the policy lives
     on server.py, so the path has to be addressable and the fetch replaceable:
     reaching 127.0.0.1 from that page needs mode:'cors' and Private Network
     Access, which the studio already does in its own localFetch. Both hooks are
     set by the page before this script loads; the viewers, served by server.py
     itself, need neither. */
  const api = (path, init) => {
    const send = root.ASK_POLICY_FETCH || ((p, i) => fetch(p, i));
    return send(String(root.ASK_POLICY_BASE || '') + path, init);
  };

  /* Conditions, not levels: `big` and `batch` are independent and can combine,
     while `always` is the union of everything and no condition ticked means
     never. That is exactly the shape of the four options as asked for, without
     pretending that "only long calls" and "only batches" are a scale. */
  const CONDITIONS = [
    ['always', 'každé volání',
     'Ptát se před každým voláním, které opravdu odejde. Volání, na které odpověď '
     + 'už je v databázi, se neděje, takže se na ně ani neptá.'],
    ['big', 'od 4096 tokenů',
     'Jen když má volání max_tokens 4096 nebo víc -- tedy ta nejdražší, která '
     + 'generují do stropu modelu.'],
    ['batch', 'víc než jedno volání',
     'Jen když operace není jediné volání: jednotokenové odchylky řetězce jsou '
     + 'stovky volání, klik na token je jedno.'],
  ];

  const DEFAULT = {always: true, big: false, batch: false};

  const clean = p => ({
    always: !!(p && p.always),
    big: !!(p && p.big),
    batch: !!(p && p.batch),
  });

  let cached = null;

  function fromStorage() {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) return clean(JSON.parse(raw));
    } catch { /* private window, or no storage at all */ }
    return null;
  }

  function toStorage(policy) {
    try { localStorage.setItem(KEY, JSON.stringify(policy)); } catch {}
  }

  /* The policy as last known. Synchronous on purpose: a call site must not have
     to await anything to find out whether it may spend money. load() refreshes
     it from the server; until it has, the cache or the default applies. */
  function get() {
    if (cached) return cached;
    cached = fromStorage() || {...DEFAULT};
    return cached;
  }

  async function load() {
    try {
      const res = await api('/api/ask_policy', {cache: 'no-store'});
      if (res.ok) {
        const data = await res.json();
        if (data && data.policy) {
          cached = clean(data.policy);
          toStorage(cached);
          return cached;
        }
      }
    } catch { /* server.py not running: the cache stands */ }
    return get();
  }

  async function set(policy) {
    cached = clean(policy);
    toStorage(cached);
    try {
      await api('/api/ask_policy', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(cached),
      });
    } catch { /* saved locally at least */ }
    return cached;
  }

  /* Whether this particular call has to be shown first.

     `info.total` is how many calls the whole operation will make; a single call
     passes 1 or nothing. */
  function shouldAsk(request, info) {
    const p = get();
    if (p.always) return true;
    if (p.big && Number((request || {}).max_tokens) >= 4096) return true;
    if (p.batch && Number((info || {}).total) > 1) return true;
    return false;
  }

  function describe() {
    const p = get();
    if (p.always) return 'před každým voláním';
    const on = CONDITIONS.filter(([k]) => k !== 'always' && p[k]).map(([, label]) => label);
    return on.length ? 'jen ' + on.join(' nebo ') : 'nikdy';
  }

  /* The one function every paid call site goes through. Returns 'yes', 'all',
     'skip' or 'stop'; a caller that only makes one call treats anything but
     'yes'/'all' as "do not send".

     Fails closed twice over: if the policy says ask and no dialog is available,
     the answer is 'stop' rather than a silent send. */
  async function guard(request, info) {
    if (!shouldAsk(request, info)) return 'yes';
    const dialog = root.ApproveRequest || (typeof window !== 'undefined' && window.ApproveRequest);
    if (!dialog || typeof dialog.approve !== 'function') {
      if (typeof console !== 'undefined') {
        console.error('[ask-policy] the policy says ask, but approve-request.js is not loaded — refusing to call');
      }
      return 'stop';
    }
    return dialog.approve(request, info);
  }

  /* The per-model call size, in one place.

     20 for gpt-3.5-turbo-instruct because every sweep was run at 20, so asking
     for exactly that turns a cache near-miss into a hit. 5 for the base models,
     which have no sweep data at all and also return only 5 alternatives per
     position however many are asked for (measured: ntop=5 on all 17 base-model
     records, against 20 for gpt). */
  const BASE_MODEL = /^(ada|babbage|curie|davinci)/;
  const maxTokensFor = model => (BASE_MODEL.test(String(model || '')) ? 5 : 20);
  const logprobsFor = model => (BASE_MODEL.test(String(model || '')) ? 5 : 20);

  root.AskPolicy = {
    CONDITIONS, DEFAULT, KEY,
    get, load, set, shouldAsk, guard, describe,
    maxTokensFor, logprobsFor,
    // Testing seam: drops the memoised value so the next get() re-reads.
    forget: () => { cached = null; },
  };
})(typeof module === 'object' ? module.exports : window);
