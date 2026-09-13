export function metaTemplateFromText(text) {
  let instruction = String(text ?? '').trim();
  if (!instruction) throw new Error('The selected instruction is empty.');
  const fenced = instruction.match(/^```(?:text)?\s*\n([\s\S]*?)\n```$/);
  if (fenced) instruction = fenced[1].trim();
  if (instruction.includes('[INSTRUCTION]')) return instruction;
  if (/^\s*(Input|Output):/mi.test(instruction)) {
    throw new Error('This result has an incomplete Input/Output template. Add [INSTRUCTION] in the template editor before using it.');
  }
  return instruction + '\n\nInput: [INSTRUCTION]\nOutput:';
}

export function resolveMetaSource(rows, rowId) {
  const row = rows.find(row => row.id === rowId && row.mode === 'meta');
  if (!row || !row.prompt.trim()) throw new Error('The source Meta completion was not found. Refresh the history.');
  return { rowId: row.id, eventId: row.eventId, prompt: row.prompt };
}
