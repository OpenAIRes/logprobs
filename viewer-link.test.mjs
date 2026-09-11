import test from 'node:test';
import assert from 'node:assert/strict';
import { viewerEntries, viewerUrl, choicesInOrder } from './viewer-link.mjs';
import { promptRowsFromEvents } from './prompt-log.mjs';
const choice = (index, text, lp = true) => ({index, text, logprobs: lp ? {tokens:[text], token_logprobs:[-1], top_logprobs:[{[text]:-1}]} : null});
test('links resolve the exact choice despite unsorted and empty choices', () => {
  const event = {id:'event / 1', backend:'completions', response:{body:{id:'api-id', choices:[choice(2,'second'), choice(0,''), choice(1,'first',false)]}}, generatedPrompts:[{prompt:'first'},{prompt:'second'}]};
  const entries = viewerEntries([event]);
  const rows = promptRowsFromEvents([event]);
  assert.equal(rows[0].logprobsUrl, null);
  const id = new URL(rows[1].logprobsUrl, 'http://localhost').searchParams.get('id');
  assert.equal(entries.find(e=>e.id===id).choices[0].text, 'second');
  assert.equal(viewerUrl(event, choicesInOrder(event)[2]), rows[1].logprobsUrl);
});
test('no links for missing probabilities or responses; distinct events have distinct IDs', () => {
  const event = {id:'one', backend:'completions', response:{body:{id:'same', choices:[choice(0,'word')]}}};
  assert.equal(viewerUrl(event, choice(0,'word',false)), null);
  assert.equal(viewerUrl({...event,backend:'responses'},choice(0,'word')), null);
  assert.equal(new Set(viewerEntries([event,{...event,id:'two'}]).map(e=>e.id)).size,2);
});
