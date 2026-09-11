const assert = require('node:assert/strict');
/* `root` inside ask-policy.js is its own module.exports in node, so that is
   where a stub dialog has to be hung -- the test's own exports are a different
   object, and hanging it there is how this test first "failed". */
const policyHost = require('./ask-policy.js');
const {AskPolicy} = policyHost;

/* No localStorage and no fetch in node, so get() falls back to the default and
   set() only updates the cache -- which is exactly the surface worth testing:
   the decision rule and the fail-closed behaviour. */
const ask = (policy, request, info) => {
  AskPolicy.forget();
  AskPolicy.set(policy);
  return AskPolicy.shouldAsk(request, info);
};

const ONE = {max_tokens: 20};
const BIG = {max_tokens: 4096};

// ---- "každé": before every call that actually happens ----------------------
assert.equal(ask({always: true}, ONE, {total: 1}), true);
assert.equal(ask({always: true}, BIG, {total: 300}), true);

// ---- "od 4096 tokenů" -------------------------------------------------------
assert.equal(ask({big: true}, ONE, {total: 1}), false);
assert.equal(ask({big: true}, BIG, {total: 1}), true);
assert.equal(ask({big: true}, {max_tokens: 4095}, {total: 1}), false);
assert.equal(ask({big: true}, {max_tokens: 5000}, {total: 1}), true);

// ---- "víc než jedno volání" -------------------------------------------------
assert.equal(ask({batch: true}, ONE, {total: 1}), false);
assert.equal(ask({batch: true}, ONE, {total: 2}), true);
assert.equal(ask({batch: true}, ONE, {}), false, 'no total given means a single call');
assert.equal(ask({batch: true}, ONE, undefined), false);

// ---- the two conditions combine ---------------------------------------------
assert.equal(ask({big: true, batch: true}, ONE, {total: 1}), false);
assert.equal(ask({big: true, batch: true}, BIG, {total: 1}), true);
assert.equal(ask({big: true, batch: true}, ONE, {total: 9}), true);

// ---- "nikdy" = nothing ticked ----------------------------------------------
assert.equal(ask({}, BIG, {total: 900}), false);
assert.equal(ask({always: false, big: false, batch: false}, BIG, {total: 900}), false);

// ---- the default is "každé" -------------------------------------------------
AskPolicy.forget();
assert.deepEqual(AskPolicy.get(), {always: true, big: false, batch: false, max_tokens: 0});
assert.equal(AskPolicy.describe(), 'před každým voláním');
AskPolicy.set({big: true, batch: true});
assert.equal(AskPolicy.describe(), 'jen od 4096 tokenů nebo víc než jedno volání');
AskPolicy.set({});
assert.equal(AskPolicy.describe(), 'nikdy');

// ---- garbage in the stored policy is not a licence to spend -----------------
AskPolicy.forget();
AskPolicy.set({always: 'no', big: null, batch: 0});
assert.deepEqual(AskPolicy.get(), {always: true, big: false, batch: false, max_tokens: 0},
  'truthiness only: the strings a hand-edited file might hold must not mean "off"');

// ---- max_tokens: 0 means per model, a number means that number -------------
AskPolicy.forget();
AskPolicy.set({});
assert.equal(AskPolicy.maxTokensFor('gpt-3.5-turbo-instruct'), 20);
assert.equal(AskPolicy.maxTokensFor('davinci-002'), 5);
AskPolicy.set({max_tokens: 100});
assert.equal(AskPolicy.maxTokensFor('gpt-3.5-turbo-instruct'), 100);
assert.equal(AskPolicy.maxTokensFor('davinci-002'), 100, 'a set length applies to every model');
assert.equal(AskPolicy.logprobsFor('davinci-002'), 5, 'the alternative cap is the API, not a choice');
AskPolicy.set({max_tokens: 40000});
assert.equal(AskPolicy.maxTokensFor('gpt-3.5-turbo-instruct'), 4096, 'clamped, not obeyed');
for (const bad of [0, -5, 'twenty', null, undefined, NaN]) {
  AskPolicy.set({max_tokens: bad});
  assert.equal(AskPolicy.maxTokensFor('davinci-002'), 5, `${bad} falls back to per model`);
}
AskPolicy.set({});

// ---- guard(): asks, and fails closed with no dialog available ---------------
(async () => {
  AskPolicy.forget();
  AskPolicy.set({always: true});
  const quiet = console.error;
  console.error = () => {};
  assert.equal(await AskPolicy.guard(ONE, {total: 1}), 'stop',
    'the policy says ask and there is no dialog: refuse rather than send');
  console.error = quiet;

  // With a dialog, the verdict is whatever it returns, and the request reaches it.
  const seen = [];
  policyHost.ApproveRequest = {approve: async (req, info) => { seen.push({req, info}); return 'all'; }};
  assert.equal(await AskPolicy.guard(ONE, {total: 3}), 'all');
  assert.deepEqual(seen.map(s => s.info.total), [3]);
  assert.equal(seen[0].req, ONE, 'the object handed to the dialog is the caller’s own');

  // Policy says do not ask: no dialog, straight yes.
  seen.length = 0;
  AskPolicy.set({});
  assert.equal(await AskPolicy.guard(BIG, {total: 500}), 'yes');
  assert.equal(seen.length, 0, 'nothing was shown, because nothing was meant to be');

  // ---- the per-model limits, in their one home -----------------------------
  AskPolicy.set({});
  assert.equal(AskPolicy.maxTokensFor('gpt-3.5-turbo-instruct'), 20);
  assert.equal(AskPolicy.logprobsFor('gpt-3.5-turbo-instruct'), 20);
  for (const m of ['davinci-002', 'babbage-002', 'ada-002', 'curie-001']) {
    assert.equal(AskPolicy.maxTokensFor(m), 5, m);
    assert.equal(AskPolicy.logprobsFor(m), 5, m);
  }
  assert.equal(AskPolicy.maxTokensFor(''), 20, 'an unknown model is treated as gpt');

  console.log('PASS every/4096/batch/never · conditions combine · default is every · '
    + 'garbage is not "off" · guard fails closed without a dialog · one set of limits');
})();
