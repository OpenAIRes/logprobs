/* One-token deviations of a given string.

   For a string of n tokens, every position i and every alternative the model
   recorded at i gives a prompt -- tokens[:i] + [alternative] -- and the answer
   is what the model actually generates from that prompt. A 16-token string
   therefore yields about 16 x 19 strings, not 20.

   That is what `run` does now. It used to take the whole string as the prompt
   and branch only at the position AFTER it, which is a different question --
   "what comes next instead" rather than "what if this token had been different"
   -- and it produced 20 strings from any string of any length. That behaviour is
   kept as `runNext`, because it is the right question when starting from a
   prompt (or from nothing) rather than from a string you already have.

   Network requests are made only by run()/runNext(), from an explicit button,
   and only for the deviations the store cannot answer. */
(function (root) {

  const poster = async (url, body) => {
    const response = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(data.error || `HTTP ${response.status}`);
      // A paid call that could not be saved must not vanish: the caller offers
      // it as a download instead of buying the same answer twice.
      error.record = data.api_completed ? data.record : null;
      throw error;
    }
    return data;
  };

  /* The fourth argument used to be the poster itself. Both shapes are accepted
     so the callers and the test that pass a bare function keep working. */
  const options = o => (typeof o === 'function' ? {post: o} : (o || {}));

  /* The per-model call size lives in ask-policy.js -- it had three copies of
     the same two numbers, which is two too many. Resolved at call time so this
     file stays loadable both in the browser (script tag before this one) and in
     node (the test requires it). No fallback on purpose: a wrong max_tokens is a
     wrong record, so failing loudly beats guessing. */
  const limits = () => {
    if (typeof globalThis !== 'undefined' && globalThis.AskPolicy) return globalThis.AskPolicy;
    if (typeof require === 'function') {
      try { return require('./ask-policy.js').AskPolicy; } catch { /* fall through */ }
    }
    throw new Error('ask-policy.js must be loaded before greedy-branches.js');
  };
  const maxTokensFor = model => limits().maxTokensFor(model);
  const logprobsFor = model => limits().logprobsFor(model);

  /* Which of a row's tokens are the given prompt and which were generated.

     Only the generated ones can be deviated: the API returns logprobs for what
     it generated, so a prompt token has no alternatives and no question to ask.
     `opts.fixed` is how many leading tokens are the prompt -- the views that put
     a prompt in the row carry it as `prompt_len`. */
  function split(parent, o) {
    const all = (parent.tokens || []).map(t => t.token);
    const fixed = Math.max(0, Math.min(Number(o.fixed) || 0, all.length));
    return {prompt: all.slice(0, fixed), tokens: all.slice(fixed)};
  }

  /* What deviating this string would involve, before anything is bought: the
     rows the store can already answer, and the prompts it cannot. */
  async function plan(parent, model, opts) {
    const o = options(opts);
    const post = o.post || poster;
    const parts = split(parent, o);
    return post('/api/deviations', {
      tokens: parts.tokens, prompt_tokens: parts.prompt,
      model, sort: o.sort || 'cost',
      ends: o.ends || null, nodes: o.nodes || null, sources: o.sources || null,
    });
  }

  /* Buy the missing continuations, then ask for the plan again.

     Re-planning rather than assembling the new rows here is deliberate: every
     row then comes out of the same scoring code, so a bought row and a stored
     row cannot disagree about Σ, cost or where the string ends. */
  async function fill(parent, model, first, opts) {
    const o = options(opts);
    const post = o.post || poster;
    const progress = o.progress || (() => {});
    const missing = first.missing || [];
    /* The prompt is rebuilt here rather than sent: see the note in
       store.deviations. Same rule on both sides -- the fixed prompt, the
       generated tokens before the position, then the alternative. The fixed part
       is taken from the plan's own answer when it is there, so the two cannot
       disagree about where the prompt ends. */
    const parts = split(parent, o);
    const fixedText = first.prompt !== undefined ? first.prompt : parts.prompt.join('');
    const promptFor = cell => (cell.prompt !== undefined ? cell.prompt
      : fixedText + parts.tokens.slice(0, cell.position).join('') + cell.alternative);
    let calls = 0, skipped = 0, stopped = false, approvedAll = false;
    const failures = [];
    for (let i = 0; i < missing.length; i++) {
      const cell = missing[i];
      if (o.stopped && o.stopped()) { stopped = true; break; }

      /* Built first, shown, then posted -- the same object. Rebuilding it after
         the approval would mean the dialog and the call could differ, which
         would make the approval worthless. */
      const request = {
        prompt: promptFor(cell), model, temperature: 0,
        max_tokens: maxTokensFor(model), logprobs: logprobsFor(model),
        confirmed: true,
      };

      if (o.approve && !approvedAll) {
        let verdict = await o.approve(request, {index: i + 1, total: missing.length, cell});
        if (verdict === true) verdict = 'yes';
        // Anything that is not an explicit go means do not spend: a dialog that
        // was dismissed, closed or returned nothing stops the run.
        if (verdict === 'all') approvedAll = true;
        else if (verdict === 'skip') { skipped++; continue; }
        else if (verdict !== 'yes') { stopped = true; break; }
      }

      progress(`Volám API: ${++calls}/${missing.length} · pozice ${cell.position + 1}, ${JSON.stringify(cell.alternative)}`);
      try {
        await post('/api/complete', request);
      } catch (error) {
        // One refused or unsaveable call must not throw away the other 300.
        failures.push({cell, error});
        if (error.record) throw error;
      }
    }
    progress('Přepočítávám…');
    const again = await plan(parent, model, opts);
    return {...again, calls, failures, skipped, stopped};
  }

  /* plan -> (optionally) buy -> plan again.

     Two ways to be asked, and neither is the default: `opts.approve(request,
     info)` is asked before EVERY call, with the exact request body, and returns
     'yes' / 'all' / 'skip' / 'stop'; `opts.confirm(plan)` is the older one-off
     summary, used only when there is no approve. With neither, nothing is
     bought: no confirm and no approve means run() returns what the store had.
     Nothing here spends money on its own. */
  async function run(parent, model, progress, opts) {
    const o = {...options(opts)};
    o.progress = progress || o.progress;
    (o.progress || (() => {}))('Hledám odchylky v databázi…');
    const first = await plan(parent, model, o);
    if (first.error) throw new Error(first.error);
    if (!first.missing_count) return {...first, calls: 0, failures: []};
    /* Fail closed. A caller that passes neither approve nor confirm gets the
       plan and nothing else -- spending 1655 calls because an option was
       forgotten is exactly the accident this whole path is built to avoid. */
    if (!o.approve && !o.confirm) {
      return {...first, calls: 0, failures: [], declined: true, needs_approval: true};
    }
    // Two dialogs for one decision is one too many: when every call is approved
    // individually, the up-front summary is skipped.
    if (o.confirm && !o.approve && !(await o.confirm(first))) {
      return {...first, calls: 0, failures: [], declined: true};
    }
    return fill(parent, model, first, o);
  }

  /* The old behaviour: the whole string as the prompt, branching at the position
     after it. Answers "what if the NEXT token were different", which is the
     question when there is no string yet -- only a starting prompt, or nothing. */
  async function runNext(parent, model, progress, opts) {
    const o = options(opts);
    const post = o.post || poster;
    progress = progress || (() => {});
    let calls = 0, hits = 0, approvedAll = false;
    const get = async prompt => {
      const lookup = {prompt, model, temperature: 0};
      const cached = await post('/api/lookup', lookup);
      if (cached.hit) { hits++; return cached.record; }

      /* Same rule as fill(): the body is built, shown, and then posted -- and
         without a gate nothing is bought at all. This path is not on any button
         today, but an exported function that can spend money unasked is a trap
         waiting for whoever wires it up next. */
      const request = {...lookup, max_tokens: maxTokensFor(model),
                       logprobs: logprobsFor(model), confirmed: true};
      if (!o.approve && !o.confirm) {
        throw new Error('runNext needs opts.approve (or opts.confirm) before it can call the API');
      }
      if (o.approve && !approvedAll) {
        let verdict = await o.approve(request, {index: calls + 1, total: null});
        if (verdict === true) verdict = 'yes';
        if (verdict === 'all') approvedAll = true;
        // No 'skip' here: a branch without its continuation has nothing to show,
        // so the only answers are go on or stop.
        else if (verdict !== 'yes') throw new Error('Zastaveno před voláním API.');
      }
      progress(`Načítám pokračování přes API (${++calls})…`);
      return post('/api/complete', request);
    };
    const base = await get(parent.text);
    const choice = base.choices[0], lp = choice.logprobs || {};
    const chosen = (lp.tokens || [])[0];
    const alternatives = Object.entries((lp.top_logprobs || [])[0] || {}).filter(([, p]) => Number.isFinite(p)).sort((a, b) => b[1] - a[1]).slice(0, 20);
    if (!alternatives.length) throw new Error('Pro následující token nejsou dostupné logprobs (řetězec mohl skončit).');
    const entries = [];
    for (const [token, probability] of alternatives) {
      progress(`Větev ${entries.length + 1}/${alternatives.length} · databáze ${hits} · API ${calls}`);
      const record = token === chosen ? base : await get(parent.text + token);
      const c = record.choices[0], rlp = c.logprobs || {};
      const tokens = parent.tokens.map(t => ({...t}));
      let sum = parent.sum_logprob;
      const append = (text, logprob, tops) => {
        if (!Number.isFinite(logprob)) throw new Error('Pokračování obsahuje token bez logprob.');
        sum += logprob;
        tokens.push({token: text, logprob, cumulative: sum, prob: Math.exp(logprob), source: 'chosen', alternatives: Object.entries(tops || {}).map(([token, logprob]) => ({token, logprob}))});
      };
      // Score the selected token from the same distribution for every branch.
      append(token, probability, Object.fromEntries(alternatives));
      for (let i = token === chosen ? 1 : 0; i < (rlp.tokens || []).length; i++) append(rlp.tokens[i], (rlp.token_logprobs || [])[i], (rlp.top_logprobs || [])[i]);
      entries.push({rank: entries.length + 1, text: tokens.map(t => t.token).join(''), tokens, n: tokens.length, sum_logprob: sum, mean_logprob: sum / tokens.length, perplexity: Math.exp(-sum / tokens.length), id: record.id, end: c.finish_reason || 'length', finish_reason: c.finish_reason, cost: alternatives[0][1] - probability, deviation: {position: parent.tokens.length, original: chosen, alternative: token}, is_greedy: token === chosen});
    }
    return {entries, calls, hits};
  }

  root.GreedyBranches = {run, runNext, plan, fill};
})(typeof module === 'object' ? module.exports : window);
