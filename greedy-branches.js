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
    /* A failure that is not JSON must still be readable. An unhandled exception
       on the server sends an HTML error page, and parsing that threw a
       SyntaxError that reached the page instead of the reason -- which is how a
       DNS failure came out as "Unexpected token <" rather than "the API could
       not be reached". */
    let data = {};
    try { data = await response.json(); } catch { data = {}; }
    if (!response.ok) {
      const error = new Error(data.error
        || `HTTP ${response.status} ${response.statusText || ''}`.trim());
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

  /* One vocabulary for "what is happening", shared with every other program
     that can generate. Absent in node, where the test loads this file alone. */
  const report = (stage, detail) => {
    const r = root.CallReport || (typeof window !== 'undefined' && window.CallReport);
    if (r) r.report(stage, detail);
  };

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
  const policy = limits;
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
      // Not what gets deviated -- only how the string was generated, so the
      // continuations can be generated the same way.
      base_id: o.base_id || null,
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
    /* How the base was made, from the plan's own answer. Neutral values stand in
       for anything it does not carry -- 353 of the older records have no request
       block at all -- and they are spelled out in the request rather than left
       to the server, so the dialog shows every parameter that will be sent
       instead of some of them being filled in behind it. */
    const sampling = first.base_request || {};
    /* Setting first, then the base record, then the model: each level is more
       specific than the one before, and the setting is the only one somebody
       typed on purpose. */
    const length = policy().get().max_tokens
      || Number(sampling.max_tokens) || maxTokensFor(model);
    const fixedText = first.prompt !== undefined ? first.prompt : parts.prompt.join('');
    const promptFor = cell => (cell.prompt !== undefined ? cell.prompt
      : fixedText + parts.tokens.slice(0, cell.position).join('') + cell.alternative);
    /* The same string, in the pieces it is made of. fixedText is one opaque span
       when the base had a given prompt -- that part genuinely has no known
       tokenisation -- and the rest are real tokens. */
    const fixedTokens = first.prompt_tokens || parts.prompt;
    const tokensFor = cell => (cell.prompt !== undefined ? undefined
      : [...fixedTokens, ...parts.tokens.slice(0, cell.position), cell.alternative]);
    let calls = 0, skipped = 0, stopped = false, approvedAll = false;
    const failures = [];
    /* A run does not grind through hundreds of doomed calls. When the network
       goes while a batch is running, every remaining call fails the same way in
       milliseconds -- 245 of them, in the run that prompted this -- and the only
       thing that produces is a long wait and a wall of identical errors. */
    const GIVE_UP_AFTER = 5;
    let consecutive = 0;
    let stoppedReason = null;
    for (let i = 0; i < missing.length; i++) {
      const cell = missing[i];
      if (o.stopped && o.stopped()) { stopped = true; break; }

      /* Built first, shown, then posted -- the same object. Rebuilding it after
         the approval would mean the dialog and the call could differ, which
         would make the approval worthless.

         The sampling parameters are the ones the string being deviated was made
         with, so a deviation is that experiment with one token changed rather
         than a differently sampled string put beside it. What is NOT inherited:
         `logprobs`, because a base made with logprobs 0 has no alternatives and
         a continuation without them can be neither scored nor deviated in turn;
         and `max_tokens` when the length setting says otherwise, because that
         setting is an explicit instruction and this is a default. */
      const request = {
        prompt: promptFor(cell),
        /* The tokenisation is not a guess here: the fixed prompt, then the
           record's own generated tokens up to the position, then the
           alternative. Sending it is what lets the trie place the answer where
           this deviation actually is, instead of under a node of its own where
           nothing can reach it. */
        prompt_tokens: tokensFor(cell),
        model,
        temperature: 0, top_p: 1, frequency_penalty: 0, presence_penalty: 0,
        ...sampling,
        max_tokens: length,
        logprobs: logprobsFor(model),
        confirmed: true,
        /* This is one of many, so the record is written and fsynced but the
           store's indexes are not rebuilt for it. Rebuilding them took longer
           than the call itself -- about five seconds against one -- so a run of
           three hundred spent most of an hour reindexing the same history over
           and over. The re-plan at the end of the run reads the store, which is
           what settles it, once. Never sent to the API; the server drops it. */
        defer: true,
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
      report('calling', `${calls}/${missing.length} · pozice ${cell.position + 1} `
                      + `${JSON.stringify(cell.original)} → ${JSON.stringify(cell.alternative)}`);
      try {
        const answer = await post('/api/complete', request);
        consecutive = 0;
        report(answer && answer.from_cache ? 'cached' : 'saved',
               `${calls}/${missing.length}`);
        /* Hand the answer over as it arrives. The rows are only rebuilt when the
           whole run is done -- they come from a fresh plan, so that every row is
           scored by the same code -- which meant paying for a call and being
           asked about the next one without ever seeing what the last one said. */
        if (o.onResult) {
          try { o.onResult(answer, cell, calls, missing.length); } catch { /* display only */ }
        }
      } catch (error) {
        report(/nedosa|unreachable|getaddrinfo|Failed to fetch/i.test(error.message || '')
                 ? 'unreachable' : 'failed',
               `${calls}/${missing.length} · ${error.message || ''}`);
        // One refused or unsaveable call must not throw away the other 300.
        failures.push({cell, error});
        if (error.record) throw error;
        if (++consecutive >= GIVE_UP_AFTER) {
          stopped = true;
          stoppedReason = `${consecutive} volání za sebou selhalo: ${error.message || ''}`;
          report('failed', stoppedReason);
          break;
        }
      }
    }
    progress('Přepočítávám…');
    report('looking', 'přepočítávám plán');
    const again = await plan(parent, model, opts);
    /* A run where most calls failed is not "done": saying so was how a run of
       19 successes and 245 failures came out looking like a tidy finish. */
    const summary = `${calls} volání · ${failures.length ? failures.length + ' selhalo · ' : ''}`
      + `${skipped ? skipped + ' přeskočeno · ' : ''}${again.from_store} v tabulce`;
    report(failures.length ? 'failed' : stopped ? 'stopped' : 'saved',
           stoppedReason || summary);
    return {...again, calls, failures, skipped, stopped, stoppedReason,
            firstFailure: failures.length ? (failures[0].error.message || '') : null};
  }

  /* plan -> (optionally) buy -> plan again.

     `opts.onResult(record, cell, n, total)` is handed each answer as it comes
     back, because the rows are only rebuilt at the end of the run.

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
    report('looking', 'jednotokenové odchylky');
    const first = await plan(parent, model, o);
    if (first.error) throw new Error(first.error);
    if (!first.missing_count) {
      report('cached', `${first.from_store} odchylek z databáze, nic k dokoupení`);
      return {...first, calls: 0, failures: []};
    }
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
