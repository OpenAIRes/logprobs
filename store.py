"""One record store for every view and every generator.

Until now each program owned its own copy of the data and its own answer to
"do we already have this call?": logprobs.html scanned IndexedDB, builder.html
kept a second database, sweep_alternatives.py replayed its JSONL ledger, and
dijkstra.html read frozen export files. Four implementations, and the cache
verdict was wrong in a different way in three of them.

The raw call log is the only primary artifact. A tree, a ranking, a sweep grid
and a greedy walk are all queries over it, so they live here, once.

Ranking is NOT reimplemented: build_trie/rank_by_sum and the entry builders come
from export_dijkstra_top, so a ranking served live is identical to the ranking a
regenerated file would hold. verify_store.py asserts exactly that.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

from best_avg_logprob_path import build_trie, record_model
from export_dijkstra_top import (
    apply_prefix,
    build_completion_entries,
    build_prefix_entries,
    index_sweeps,
    load_records,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

# Canonical inputs: every one is git-tracked. The uncompressed mirrors are gone,
# and the .jsonl files stay only as sweep_alternatives.py resume ledgers -- the
# same records already live in the .gz, so reading them here would just be slow.
SOURCES = [
    'completion_history.json',
    'sweep_history.json',
    'sweep2_history.json.gz',
    'sweep3_ihave.json.gz',
    'builder_history.json',
    'meta_resample_root.json',
]

VIEWS = ('prefixes', 'completions')

DEFAULTS = dict(top=200, min_n=1, prefix=None, model=None, chosen_only=False, max_alts=8)


class RecordStore:
    def __init__(self, sources: Optional[List[str]] = None, root: str = ROOT):
        self.root = root
        self.sources = list(sources or SOURCES)
        self.records: List[Dict] = []
        self.by_id: Dict[str, Dict] = {}
        self.by_prompt: Dict[str, List[Dict]] = {}
        self.sweep_by_prompt: Dict[tuple, Dict] = {}
        self._tries: Dict[Optional[str], object] = {}
        self._completions_cache: Dict[tuple, List[Dict]] = {}
        self.load_seconds = 0.0
        self.missing: List[str] = []

    # -- loading --------------------------------------------------------------

    def load(self) -> 'RecordStore':
        started = time.time()
        paths = []
        for name in self.sources:
            path = os.path.join(self.root, name)
            if os.path.exists(path):
                paths.append(path)
            else:
                self.missing.append(name)
        self.records = load_records(paths)
        self.by_id = {r['id']: r for r in self.records if r.get('id')}
        self.by_prompt = {}
        for rec in self.records:
            key = (rec.get('request') or {}).get('prompt')
            if isinstance(key, str):
                self.by_prompt.setdefault(key, []).append(rec)
        self.sweep_by_prompt = index_sweeps(self.records)
        self._tries.clear()
        self._completions_cache.clear()
        self.load_seconds = time.time() - started
        return self

    def trie(self, model: Optional[str] = None):
        if model not in self._tries:
            self._tries[model] = build_trie(self.records, model)
        return self._tries[model]

    # -- the one cache verdict ------------------------------------------------

    @staticmethod
    def _model_matches(rec: Dict, model: str) -> bool:
        """A request records the model it asked for; older records only carry the
        served model, which is versioned (gpt-3.5-turbo-instruct:20230824-v2).
        Falling back to a prefix match is what stops a davinci-002 record from
        being served for a gpt-3.5-turbo-instruct request, which is the bug this
        function exists to prevent."""
        asked = (rec.get('request') or {}).get('model')
        if isinstance(asked, str):
            return asked == model
        served = record_model(rec)
        return isinstance(served, str) and served.startswith(model)

    def lookup(self, prompt: str, model: Optional[str] = None,
               max_tokens: Optional[int] = None,
               temperature: Optional[float] = 0) -> Optional[Dict]:
        """The record that already answers this call, or None.

        A record may serve a request for FEWER tokens than it generated (the
        caller truncates), but never for more -- unless it stopped on its own,
        in which case there is nothing more to get.
        """
        for rec in self.by_prompt.get(prompt, []):
            req = rec.get('request') or {}
            if model and not self._model_matches(rec, model):
                continue
            if temperature is not None and req.get('temperature', 0) != temperature:
                continue
            if max_tokens is not None:
                choice = (rec.get('choices') or [{}])[0]
                have = len((choice.get('logprobs') or {}).get('tokens') or [])
                if have < max_tokens and choice.get('finish_reason') != 'stop':
                    continue
            return rec
        return None

    # -- views ----------------------------------------------------------------

    def _completions(self, args) -> List[Dict]:
        """Completions ranked, with prefix and top applied on a cached ranking.

        build_completion_entries walks every record's whole token path, which took
        ~5 s over 8.5k records -- far too slow to sit behind a filter box, and
        pure waste because that walk does not depend on prefix or top. So the
        unfiltered ranking is cached and only the tail of the pipeline reruns.
        Order of operations is unchanged: sort, then prefix, then top, exactly as
        build_completion_entries does it, which is what keeps verify_store green.
        """
        key = (args.model, bool(args.chosen_only), args.max_alts)
        if key not in self._completions_cache:
            full = SimpleNamespace(**{**vars(args), 'prefix': None, 'top': None})
            self._completions_cache[key] = build_completion_entries(
                self.trie(args.model), self.records, self.sweep_by_prompt, full)
        entries = apply_prefix(self._completions_cache[key], args.prefix)
        if args.top:
            entries = entries[:args.top]
        # Copy before stamping rank: the cached list is shared between requests,
        # and a filtered view must not renumber the entries another one is reading.
        return [{**e, 'rank': i} for i, e in enumerate(entries, 1)]

    def query(self, view: str = 'prefixes', **kwargs) -> Dict:
        if view not in VIEWS:
            raise ValueError(f'unknown view {view!r}; expected one of {VIEWS}')
        args = SimpleNamespace(**{**DEFAULTS,
                                  **{k: v for k, v in kwargs.items() if k in DEFAULTS}})
        root = self.trie(args.model)
        if view == 'completions':
            entries = self._completions(args)
            ranking = 'sum_logprob desc over whole strings (prompt + completion)'
        else:
            entries = build_prefix_entries(root, self.sweep_by_prompt, args)
            ranking = 'sum_logprob desc (best-first / Dijkstra over token trie)'
        return {
            'generated_from': ', '.join(self.sources),
            'model_filter': args.model,
            'chosen_only': args.chosen_only,
            'source_records': len(self.records),
            'ranking': ranking,
            'view': view,
            'prefix_filter': args.prefix,
            'count': len(entries),
            'entries': entries,
        }

    def sweep_grid(self, base_id: str) -> Dict:
        """position x alternative for one base -- the natural view of sweep data,
        which no program has offered so far."""
        base = self.by_id.get(base_id)
        cells: Dict[int, List[Dict]] = {}
        for rec in self.records:
            sw = (rec.get('meta') or {}).get('sweep') or {}
            if sw.get('base_id') != base_id:
                continue
            choice = (rec.get('choices') or [{}])[0]
            own = (choice.get('logprobs') or {}).get('token_logprobs') or []
            cells.setdefault(sw.get('position'), []).append({
                'alternative': sw.get('alternative'),
                'alternative_logprob': sw.get('alternative_logprob'),
                'is_greedy': sw.get('is_greedy'),
                'id': rec.get('id'),
                'generated_len': len(own),
                'generated_sum': sum(v for v in own if isinstance(v, (int, float))),
                'finish_reason': choice.get('finish_reason'),
                'text': choice.get('text'),
            })
        for row in cells.values():
            row.sort(key=lambda c: -(c['alternative_logprob'] or float('-inf')))
        base_tokens = []
        if base:
            base_tokens = ((base.get('choices') or [{}])[0].get('logprobs') or {}).get('tokens') or []
        return {
            'base_id': base_id,
            'base_found': base is not None,
            'base_tokens': list(base_tokens),
            'positions': [{'position': p, 'alternatives': cells[p]} for p in sorted(cells)],
            'cells': sum(len(v) for v in cells.values()),
        }

    def bases(self) -> List[Dict]:
        """Every record some sweep used as its base, busiest first."""
        counts: Dict[str, int] = {}
        for rec in self.records:
            sw = (rec.get('meta') or {}).get('sweep') or {}
            if sw.get('base_id'):
                counts[sw['base_id']] = counts.get(sw['base_id'], 0) + 1
        out = []
        for base_id, n in counts.items():
            rec = self.by_id.get(base_id)
            choice = ((rec or {}).get('choices') or [{}])[0]
            out.append({
                'base_id': base_id,
                'calls': n,
                'present': rec is not None,
                'model': record_model(rec) if rec else None,
                'text': choice.get('text'),
            })
        return sorted(out, key=lambda b: -b['calls'])

    def stats(self) -> Dict:
        models: Dict[str, int] = {}
        prompt_tokens = completion_tokens = 0
        for rec in self.records:
            name = record_model(rec)
            models[name] = models.get(name, 0) + 1
            usage = rec.get('usage') or {}
            prompt_tokens += usage.get('prompt_tokens', 0) or 0
            completion_tokens += usage.get('completion_tokens', 0) or 0
        return {
            'records': len(self.records),
            'distinct_prompts': len(self.by_prompt),
            'sweep_calls': len(self.sweep_by_prompt),
            'models': dict(sorted(models.items(), key=lambda kv: -kv[1])),
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'sources': self.sources,
            'missing_sources': self.missing,
            'load_seconds': round(self.load_seconds, 2),
        }


def _safe(text: str) -> str:
    """The Windows console is cp1250, and these records hold every byte the
    tokenizer can emit -- printing them raw raises UnicodeEncodeError."""
    import sys
    enc = (sys.stdout.encoding or 'utf-8')
    return text.encode(enc, errors='replace').decode(enc)


if __name__ == '__main__':
    import json
    store = RecordStore().load()
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    print('sweep bases:')
    for b in store.bases():
        print(_safe(f'  {b["base_id"]}  {b["calls"]:>5} calls  present={b["present"]}  '
                    f'{(b["text"] or "")[:46]!r}'))
