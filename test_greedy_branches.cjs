const assert = require('node:assert/strict');
const {GreedyBranches} = require('./greedy-branches.js');

/* A 2-token parent. run() deviates INSIDE it, so the prompts are "b" (position 0)
   and "ay" (position 1) -- not "ax", which is what branching after the end would
   ask for. That distinction is the whole point of this file. */
const parent = {text: 'ax', sum_logprob: -0.3,
                tokens: [{token: 'a', logprob: -0.1, cumulative: -0.1},
                         {token: 'x', logprob: -0.2, cumulative: -0.3}]};

const row = (text, position, alternative) => ({
  text, n: 2, sum_logprob: -1, mean_logprob: -0.5, cost: 0.5,
  tokens: [], end: 'length', id: 'rec-' + text,
  deviation: {position, original: 'a', alternative},
});

const planWith = (entries, missing) => ({
  view: 'deviations', n: 2, planned_positions: 2,
  deviations: entries.length + missing.length,
  from_store: entries.length, missing, missing_count: missing.length,
  entries, count: entries.length,
});

(async () => {
  // ---- everything already in the store: no /api/complete at all -------------
  {
    const seen = [];
    const post = async (url, body) => {
      seen.push(url);
      assert.equal(url, '/api/deviations');
      assert.deepEqual(body.tokens, ['a', 'x']);
      return planWith([row('by', 0, 'b'), row('ay', 1, 'y')], []);
    };
    const r = await GreedyBranches.run(parent, 'gpt-3.5-turbo-instruct', () => {}, post);
    assert.deepEqual(r.entries.map(e => e.text), ['by', 'ay']);
    assert.equal(r.calls, 0);
    assert.deepEqual(seen, ['/api/deviations']);   // nothing was bought
  }

  // ---- something missing, and the confirm says no ---------------------------
  {
    let asked = 0, paid = 0;
    const post = async url => {
      if (url === '/api/complete') { paid++; return {}; }
      return planWith([row('by', 0, 'b')], [{position: 1, alternative: 'y', prompt: 'ay', cost: 1}]);
    };
    const r = await GreedyBranches.run(parent, 'gpt-3.5-turbo-instruct', () => {},
      {post, confirm: async p => { asked++; assert.equal(p.missing_count, 1); return false; }});
    assert.equal(asked, 1);
    assert.equal(paid, 0, 'a declined confirm must not spend anything');
    assert.equal(r.declined, true);
    assert.equal(r.entries.length, 1, 'what the store had is still returned');
  }

  // ---- confirmed: exactly the missing prompts are bought, then re-planned ----
  {
    const bought = [];
    let plans = 0;
    const post = async (url, body) => {
      if (url === '/api/complete') {
        assert.equal(body.confirmed, true);
        assert.equal(body.temperature, 0);
        assert.equal(body.max_tokens, 20);      // gpt: 20, as the sweeps used
        assert.equal(body.logprobs, 20);
        bought.push(body.prompt);
        return {id: 'new'};
      }
      plans++;
      return plans === 1
        ? planWith([row('by', 0, 'b')], [{position: 1, alternative: 'y', prompt: 'ay', cost: 1}])
        : planWith([row('by', 0, 'b'), row('ay', 1, 'y')], []);
    };
    const r = await GreedyBranches.run(parent, 'gpt-3.5-turbo-instruct', () => {},
      {post, confirm: async () => true});
    assert.deepEqual(bought, ['ay']);
    assert.equal(plans, 2, 'the rows come from a fresh plan, not assembled here');
    assert.equal(r.calls, 1);
    assert.deepEqual(r.entries.map(e => e.text), ['by', 'ay']);
    assert.equal(r.missing_count, 0);
  }

  // ---- the prompt is rebuilt from the tokens when the plan omits it ---------
  // (it always does now: sending it is quadratic in the length of the string)
  {
    const bought = [];
    let plans = 0;
    const post = async (url, body) => {
      if (url === '/api/complete') { bought.push(body.prompt); return {}; }
      plans++;
      return plans === 1
        ? planWith([], [{position: 0, alternative: 'b', cost: 1},
                        {position: 1, alternative: 'y', cost: 2}])
        : planWith([row('by', 0, 'b'), row('ay', 1, 'y')], []);
    };
    await GreedyBranches.run(parent, 'gpt-3.5-turbo-instruct', () => {},
      {post, confirm: async () => true});
    // parent is ['a','x']: position 0 -> "b", position 1 -> "a" + "y"
    assert.deepEqual(bought, ['b', 'ay']);
  }

  // ---- a fixed prompt is never deviated, and stays in front of every call ---
  {
    // 4 tokens, the first 2 of them the given prompt.
    const withPrompt = {text: 'PQax', sum_logprob: -0.3,
      tokens: [{token: 'P'}, {token: 'Q'}, {token: 'a'}, {token: 'x'}]};
    const sentTo = [];
    let plans = 0, planBody = null;
    const post = async (url, body) => {
      if (url === '/api/complete') { sentTo.push(body.prompt); return {}; }
      plans++;
      if (plans === 1) planBody = body;
      return {...planWith([], plans === 1
        ? [{position: 0, alternative: 'b', cost: 1}, {position: 1, alternative: 'y', cost: 2}]
        : []), prompt: 'PQ'};
    };
    await GreedyBranches.run(withPrompt, 'gpt-3.5-turbo-instruct', () => {},
      {post, fixed: 2, approve: async () => 'all'});
    // Only the generated tokens are offered for deviation...
    assert.deepEqual(planBody.tokens, ['a', 'x']);
    assert.deepEqual(planBody.prompt_tokens, ['P', 'Q']);
    // ...and the prompt is in front of every call, unchanged.
    assert.deepEqual(sentTo, ['PQb', 'PQay']);
  }

  // ---- the base record's own sampling is what a deviation is generated with -
  {
    const sent = [];
    let plans = 0;
    const post = async (url, body) => {
      if (url === '/api/complete') { sent.push(body); return {}; }
      plans++;
      return {
        ...planWith([], plans === 1 ? [{position: 0, alternative: 'b', cost: 1}] : []),
        // What the string being deviated was made with. logprobs is NOT here,
        // and must not be inherited even when the record carries it.
        base_request: {temperature: 0, top_p: 0.9, frequency_penalty: 0.5,
                       presence_penalty: -0.25, max_tokens: 50},
      };
    };
    await GreedyBranches.run(parent, 'gpt-3.5-turbo-instruct', () => {},
      {post, approve: async () => 'all'});
    assert.equal(sent[0].top_p, 0.9);
    assert.equal(sent[0].frequency_penalty, 0.5);
    assert.equal(sent[0].presence_penalty, -0.25);
    assert.equal(sent[0].temperature, 0);
    assert.equal(sent[0].max_tokens, 50, 'the base record wins over the per-model default');
    assert.equal(sent[0].logprobs, 20, 'never inherited: a base made with 0 would be useless');
  }

  // ---- with nothing to inherit, the neutral values are still spelled out -----
  {
    const sent = [];
    let plans = 0;
    const post = async (url, body) => {
      if (url === '/api/complete') { sent.push(body); return {}; }
      plans++;
      return planWith([], plans === 1 ? [{position: 0, alternative: 'b', cost: 1}] : []);
    };
    await GreedyBranches.run(parent, 'davinci-002', () => {}, {post, approve: async () => 'all'});
    assert.deepEqual(
      {temperature: sent[0].temperature, top_p: sent[0].top_p,
       frequency_penalty: sent[0].frequency_penalty, presence_penalty: sent[0].presence_penalty},
      {temperature: 0, top_p: 1, frequency_penalty: 0, presence_penalty: 0},
      'sent explicitly, so the dialog shows them instead of the server filling them in');
    assert.equal(sent[0].max_tokens, 5, 'per model when there is nothing to inherit');
  }

  // ---- per-call approval: the object shown is the object sent ---------------
  {
    const shown = [], sent = [];
    let plans = 0;
    const post = async (url, body) => {
      if (url === '/api/complete') { sent.push(body); return {}; }
      plans++;
      return plans === 1
        ? planWith([], [{position: 0, alternative: 'b', cost: 1},
                        {position: 1, alternative: 'y', cost: 2}])
        : planWith([row('by', 0, 'b'), row('ay', 1, 'y')], []);
    };
    const r = await GreedyBranches.run(parent, 'gpt-3.5-turbo-instruct', () => {},
      {post, approve: async (request, info) => { shown.push({request, info}); return 'yes'; }});
    assert.equal(shown.length, 2, 'asked once per call');
    assert.deepEqual(shown.map(s => s.info.index), [1, 2]);
    assert.deepEqual(shown.map(s => s.info.total), [2, 2]);
    assert.deepEqual(shown.map(s => s.request.prompt), ['b', 'ay']);
    assert.equal(shown[0].request.max_tokens, 20);
    assert.equal(shown[0].request.confirmed, true);
    // identity, not equality: what was approved is what went out
    assert.equal(sent[0], shown[0].request);
    assert.equal(sent[1], shown[1].request);
    assert.equal(r.calls, 2);
  }

  // ---- 'stop' at the first one sends nothing --------------------------------
  {
    let sent = 0, asked = 0;
    const post = async url => {
      if (url === '/api/complete') { sent++; return {}; }
      return planWith([row('by', 0, 'b')],
        [{position: 0, alternative: 'b', cost: 1}, {position: 1, alternative: 'y', cost: 2}]);
    };
    const r = await GreedyBranches.run(parent, 'gpt-3.5-turbo-instruct', () => {},
      {post, approve: async () => { asked++; return 'stop'; }});
    assert.equal(asked, 1, 'stopped, so the second was never offered');
    assert.equal(sent, 0);
    assert.equal(r.stopped, true);
    assert.equal(r.calls, 0);
  }

  // ---- 'skip' passes one over; 'all' stops the asking ----------------------
  {
    const sent = [];
    let asked = 0, plans = 0;
    const post = async (url, body) => {
      if (url === '/api/complete') { sent.push(body.prompt); return {}; }
      plans++;
      return plans === 1
        ? planWith([], [{position: 0, alternative: 'b', cost: 1},
                        {position: 0, alternative: 'c', cost: 2},
                        {position: 1, alternative: 'y', cost: 3}])
        : planWith([], []);
    };
    const r = await GreedyBranches.run(parent, 'gpt-3.5-turbo-instruct', () => {},
      {post, approve: async () => { asked++; return asked === 1 ? 'skip' : 'all'; }});
    assert.equal(asked, 2, 'after "all" nothing else is asked');
    assert.deepEqual(sent, ['c', 'ay']);
    assert.equal(r.skipped, 1);
    assert.equal(r.calls, 2);
  }

  // ---- a dismissed dialog is a stop, not a yes -----------------------------
  {
    let sent = 0;
    const post = async url => {
      if (url === '/api/complete') { sent++; return {}; }
      return planWith([], [{position: 0, alternative: 'b', cost: 1}]);
    };
    for (const verdict of [undefined, null, false, '', 'whatever']) {
      const r = await GreedyBranches.run(parent, 'gpt-3.5-turbo-instruct', () => {},
        {post, approve: async () => verdict});
      assert.equal(r.stopped, true, `verdict ${JSON.stringify(verdict)} must stop`);
    }
    assert.equal(sent, 0);
  }

  // ---- no gate at all: the plan, and not a single call ---------------------
  {
    let sent = 0;
    const post = async url => {
      if (url === '/api/complete') { sent++; return {}; }
      return planWith([row('by', 0, 'b')], [{position: 1, alternative: 'y', cost: 1}]);
    };
    const r = await GreedyBranches.run(parent, 'gpt-3.5-turbo-instruct', () => {}, {post});
    assert.equal(sent, 0, 'a forgotten option must not spend money');
    assert.equal(r.needs_approval, true);
    assert.equal(r.declined, true);
  }

  // ---- base models ask for 5 tokens and 5 alternatives ----------------------
  {
    const bodies = [];
    const post = async (url, body) => {
      if (url === '/api/complete') { bodies.push(body); return {}; }
      return planWith([], [{position: 0, alternative: 'q', prompt: 'q', cost: 1}]);
    };
    await GreedyBranches.run(parent, 'davinci-002', () => {}, {post, confirm: async () => true});
    assert.equal(bodies[0].max_tokens, 5);
    assert.equal(bodies[0].logprobs, 5);
  }

  // ---- one refused call does not throw away the rest ------------------------
  {
    let paid = 0;
    const post = async (url, body) => {
      if (url === '/api/complete') { paid++; throw new Error('rate limited'); }
      return planWith([row('by', 0, 'b')],
        [{position: 1, alternative: 'y', prompt: 'ay', cost: 1},
         {position: 1, alternative: 'z', prompt: 'az', cost: 2}]);
    };
    const r = await GreedyBranches.run(parent, 'gpt-3.5-turbo-instruct', () => {},
      {post, confirm: async () => true});
    assert.equal(paid, 2, 'both were attempted');
    assert.equal(r.failures.length, 2);
    assert.equal(r.entries.length, 1);
  }

  // ---- an unsaved paid answer still propagates, so it can be downloaded -----
  {
    const post = async url => {
      if (url === '/api/complete') {
        const e = new Error('saving failed'); e.record = {id: 'unsaved'}; throw e;
      }
      return planWith([], [{position: 0, alternative: 'b', prompt: 'b', cost: 1}]);
    };
    let caught = null;
    try {
      await GreedyBranches.run(parent, 'gpt-3.5-turbo-instruct', () => {},
        {post, confirm: async () => true});
    } catch (err) { caught = err; }
    assert.ok(caught && caught.record && caught.record.id === 'unsaved');
  }

  // ---- runNext keeps the old question: branch AFTER the string --------------
  {
    const p = {text: 'P', tokens: [{token: 'P', logprob: -1, cumulative: -1}], sum_logprob: -1};
    const base = {id: 'base', choices: [{finish_reason: 'length', logprobs: {tokens: ['a', 'x'], token_logprobs: [-.1, -.2], top_logprobs: [{a: -.1, b: -.8}, {}]}}]};
    const tail = {id: 'tail', choices: [{finish_reason: 'stop', logprobs: {tokens: ['y'], token_logprobs: [-.3], top_logprobs: [{}]}}]};
    const cached = async (url, body) => {
      assert.equal(url, '/api/lookup');
      return {hit: true, record: body.prompt === 'P' ? base : tail};
    };
    const r = await GreedyBranches.runNext(p, 'gpt-3.5-turbo-instruct', () => {}, cached);
    assert.deepEqual(r.entries.map(e => e.text), ['Pax', 'Pby']);
    assert.equal(r.calls, 0);
    assert.equal(r.entries[1].deviation.position, 1);

    let paid = 0;
    const missing = async (url, body) => {
      if (url === '/api/lookup') return {hit: body.prompt === 'P', record: base};
      paid++;
      assert.equal(body.confirmed, true);
      assert.equal(body.prompt, 'Pb');
      return tail;
    };
    // runNext pays too, so it needs the same gate.
    let ungated = null;
    try { await GreedyBranches.runNext(p, 'gpt-3.5-turbo-instruct', () => {}, missing); }
    catch (err) { ungated = err.message; }
    assert.match(String(ungated), /needs opts\.approve/);
    assert.equal(paid, 0, 'no gate, no call');

    const shownNext = [];
    const m = await GreedyBranches.runNext(p, 'gpt-3.5-turbo-instruct', () => {},
      {post: missing, approve: async req => { shownNext.push(req.prompt); return 'yes'; }});
    assert.equal(paid, 1);
    assert.equal(m.calls, 1);
    assert.deepEqual(shownNext, ['Pb'], 'the body was shown before it went out');

    let refused = null;
    try {
      await GreedyBranches.runNext(p, 'gpt-3.5-turbo-instruct', () => {},
        {post: missing, approve: async () => 'stop'});
    } catch (err) { refused = err.message; }
    assert.match(String(refused), /Zastaveno/);
    assert.equal(paid, 1, 'still 1: the refused one was never sent');

    let failed = false;
    try {
      await GreedyBranches.runNext(p, 'gpt-3.5-turbo-instruct', () => {},
        async () => { throw new Error('test failure'); });
    } catch { failed = true; }
    assert.ok(failed);
  }

  console.log('PASS deviations inside the string · approval per call, object identity · '
    + 'stop/skip/all · dismissal = stop · no gate = no calls · '
    + 'exact prompts · 20/5 per model · partial failure survives · unsaved answer '
    + 'propagates · runNext is gated too');
})();
