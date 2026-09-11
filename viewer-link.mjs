import { eventToEntries, hasLogprobs } from './export-logprobs.mjs';

// Match the backend's sorted output, including empty choices before log filtering.
export function choicesInOrder(event) {
  return [...(event.response?.body?.choices ?? [])].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
}

export function viewerUrl(event, choice) {
  if (event.backend !== 'completions' || !hasLogprobs(choice)) return null;
  return `/logprobs.html?id=${encodeURIComponent(`ape:${event.id}:${choice.index ?? 0}`)}`;
}

export function viewerEntries(events) {
  return events.flatMap(event => eventToEntries(event).map(entry => ({
    ...entry, id: `ape:${event.id}:${entry.meta.ape_choice_index}`,
  })));
}
