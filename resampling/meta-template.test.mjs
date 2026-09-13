import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { metaTemplateFromText, resolveMetaSource } from './meta-template.mjs';
import { buildResamplingPrompt } from './resample-prompt.mjs';
import { appendPromptLogEvent, readPromptLog, promptRowsFromEvents } from './prompt-log.mjs';
test('plain instruction becomes a self-resampling template without duplicate scaffolding',()=>{
 const t=metaTemplateFromText('Rewrite this instruction.');
 assert.equal(t,'Rewrite this instruction.\n\nInput: [INSTRUCTION]\nOutput:');
 assert.equal(metaTemplateFromText(t),t);
 assert.equal(metaTemplateFromText('```text\n'+t+'\n```'),t);
 assert.equal(buildResamplingPrompt(t,t),'Rewrite this instruction.\n\nInput: '+t+'\nOutput:');
 assert.throws(()=>metaTemplateFromText(''));
 assert.throws(()=>metaTemplateFromText('Rewrite.\nInput: text\nOutput:'));
});
test('source must identify the exact saved Meta row',()=>{
 const rows=[{id:'a',eventId:'event',mode:'meta',prompt:'Rewrite.'},{id:'b',mode:'custom',prompt:'Other'}];
 assert.deepEqual(resolveMetaSource(rows,'a'),{rowId:'a',eventId:'event',prompt:'Rewrite.'});
 assert.throws(()=>resolveMetaSource(rows,'b'));assert.throws(()=>resolveMetaSource(rows,'missing'));
});
test('ancestry survives append and reload while the parent remains intact',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'meta-lineage-'));
 try {
  const path=join(dir,'log.json');
  const first=await appendPromptLogEvent({mode:'meta',generatedPrompts:[{prompt:'Rewrite.'}]},path);
  const row=promptRowsFromEvents(first).at(-1);
  const source=resolveMetaSource([row],row.id);
  await appendPromptLogEvent({mode:'meta',template:metaTemplateFromText(row.prompt),templateSource:source,generatedPrompts:[{prompt:'Rephrase.'}]},path);
  const events=await readPromptLog(path);
  assert.deepEqual(events[1],JSON.parse(JSON.stringify(first[1])));
  assert.deepEqual(events.at(-1).templateSource,source);
  assert.deepEqual(promptRowsFromEvents(events).at(-1).templateSource,source);
 } finally {await rm(dir,{recursive:true,force:true});}
});
