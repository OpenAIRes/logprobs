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
    walk_detail,
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

# How a completions ranking may be ordered. Σ logprob is the default and the only
# one the prefixes view can offer: there the ranking IS the search order, so
# reordering the n-best by something else would just be "the best mean among the
# n best by sum", which reads as an answer and is not one.
SORTS = {
    'sum': (lambda e: -e['sum_logprob'], 'sum_logprob desc'),
    'mean': (lambda e: -e['mean_logprob'], 'mean_logprob desc (length-independent)'),
    'length': (lambda e: (-e['n'], -e['sum_logprob']), 'length desc, then sum_logprob desc'),
}

DEFAULTS = dict(top=200, min_n=1, prefix=None, model=None, chosen_only=False,
                max_alts=8, sort='sum')


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
        self._alts_cache: Dict[tuple, tuple] = {}
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
        self._alts_cache.clear()
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

        entries = self._completions_cache[key]
        # Sorting must precede the cap, or "top 1 by mean" silently means "the best
        # mean among the top N by sum". That is not hypothetical: the 4096-token
        # root has the best mean logprob of all 8128 entries (-0.0127) and sits at
        # rank 7786 by sum, so a cap of 5000 would hide the very answer asked for.
        sort_key, _ = SORTS[args.sort if args.sort in SORTS else 'sum']
        if args.sort != 'sum':
            entries = sorted(entries, key=sort_key)
        entries = apply_prefix(entries, args.prefix)
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
            ranking = (SORTS.get(args.sort) or SORTS['sum'])[1]                 + ' over whole strings (prompt + completion)'
        else:
            # The heap search is by sum, so that is the only honest ranking here.
            args.sort = 'sum'
            entries = build_prefix_entries(root, self.sweep_by_prompt, args)
            ranking = 'sum_logprob desc (best-first / uniform-cost over token trie)'
        return {
            'generated_from': ', '.join(self.sources),
            'model_filter': args.model,
            'chosen_only': args.chosen_only,
            'source_records': len(self.records),
            'ranking': ranking,
            'sort': args.sort,
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
            'view': 'sweep',
            'base_id': base_id,
            'base_found': base is not None,
            'base_tokens': list(base_tokens),
            'positions': [{'position': p, 'alternatives': cells[p]} for p in sorted(cells)],
            'cells': sum(len(v) for v in cells.values()),
        }

    def greedy(self, prompt: str = '', model: str = 'gpt-3.5-turbo-instruct',
               max_steps: Optional[int] = None) -> Dict:
        """The greedy path: what temperature 0 produces from this prompt.

        This is the canonical string, and it is a PATH, not a ranking -- there is
        nothing to sort, because at every position there is exactly one argmax.
        Reaching it by ordering completions by length happens to work on this data
        and is the wrong question; it would break the moment a longer string
        existed that was not the greedy one.

        Records are chained: each hop's generated text is appended to the prompt
        and the store asked again, so a path can run past any single call. It
        stops where no record answers, and says which prompt it would need.

        The temperature-0 premise is not assumed, it is checked. Every generated
        token should be the argmax of its own top_logprobs; argmax_checked and
        argmax_ok report whether that held on this path. Across all 8492 records
        it holds for 179171 of 179171 tokens, but a -9999 sentinel (a sampled
        token missing from top_logprobs even at 20) is exactly the shape a
        violation would take, so the check earns its keep.
        """
        tokens: List[str] = []
        logprobs: List[float] = []
        tops: List[Dict] = []
        hops: List[Dict] = []
        argmax_checked = argmax_ok = 0
        text = prompt
        last_finish = None

        while max_steps is None or len(tokens) < max_steps:
            rec = self.lookup(text, model=model, max_tokens=None)
            if rec is None:
                break
            choice = (rec.get('choices') or [{}])[0]
            lp = choice.get('logprobs') or {}
            toks = list(lp.get('tokens') or [])
            if not toks:
                break
            tls = list(lp.get('token_logprobs') or [])
            tps = [dict(t or {}) for t in (lp.get('top_logprobs') or [])]

            # A hop can overshoot the budget -- one record holds 4096 tokens -- so
            # cut it here. Without this, max_steps=20 returned the whole 4096 and
            # the parameter silently meant nothing.
            if max_steps is not None:
                room = max_steps - len(tokens)
                toks, tls, tps = toks[:room], tls[:room], tps[:room]
                if not toks:
                    break

            for i, tok in enumerate(toks):
                if i < len(tps) and tps[i]:
                    argmax_checked += 1
                    if max(tps[i].items(), key=lambda kv: kv[1])[0] == tok:
                        argmax_ok += 1

            tokens += toks
            logprobs += tls
            tops += tps
            # If the hop was cut by max_steps, the record's own finish_reason no
            # longer describes where this path ends -- it ends because we said so.
            truncated_here = len(toks) < len(lp.get('tokens') or [])
            last_finish = 'max_steps' if truncated_here else choice.get('finish_reason')
            hops.append({'id': rec.get('id'), 'tokens': len(toks),
                         'finish_reason': last_finish,
                         'truncated': truncated_here,
                         'max_tokens': (rec.get('request') or {}).get('max_tokens')})
            text += ''.join(toks)
            if max_steps is not None and len(tokens) >= max_steps:
                break

        scored = [v for v in logprobs if isinstance(v, (int, float))]
        total = sum(scored)
        n = len(tokens)
        import math
        return {
            'view': 'greedy',
            'prompt': prompt,
            'model': model,
            'n': n,
            'sum_logprob': total,
            'mean_logprob': (total / n) if n else None,
            'perplexity': math.exp(-total / n) if n else None,
            'text': ''.join(tokens),
            'hops': hops,
            'first_id': hops[0]['id'] if hops else None,
            'argmax_checked': argmax_checked,
            'argmax_ok': argmax_ok,
            # A path that ended on `length` was cut by max_tokens, not by the
            # model; one that ended on `stop` is genuinely finished.
            'finish_reason': last_finish,
            'complete': last_finish == 'stop',
            'needs_call': None if last_finish == 'stop' else {
                'prompt': text, 'prompt_chars': len(text),
            },
        }

    def greedy_alternatives(self, prompt: str = '', model: str = 'gpt-3.5-turbo-instruct',
                            top: int = 20, sort: str = 'cost',
                            max_alts: int = 8) -> Dict:
        """The greedy path and its next-best siblings: one-token departures from it.

        "Second best by the greedy criterion" is the cheapest single-token
        deviation from the greedy path, followed by greedy decoding again. That is
        well defined and free to compute -- the cost of every departure is already
        in top_logprobs -- so this ranks all of them and marks which ones have a
        continuation on record.

        The ordering matters and is not one thing. Ranking by deviation cost is
        the greedy criterion literally read: which departure gives up the fewest
        nats at the point of departure. Ranking by the resulting string's Σ
        logprob asks which departure ends up most probable overall. These differ
        sharply, because the regenerated tail carries its own sum:

            by cost   #2 is  '9' -> '8'          cost 0.0352   Σ  -16.3878
            by sum    #2 is '\n' -> '\ufeffusing' cost 1.7802   Σ   -6.5815

        The second is more probable than the greedy path itself at the same length
        (greedy@21 = -15.4085) by about 8.8 nats, i.e. some 6800x. Which is the
        textbook point that greedy decoding is locally optimal and not MAP, here
        with a margin nobody could call marginal.
        """
        cache_key = (prompt, model, max_alts)
        if cache_key in self._alts_cache:
            greedy, candidates, missing = self._alts_cache[cache_key]
            return self._rank_alts(prompt, model, sort, top, greedy, candidates, missing)

        greedy = self.greedy(prompt=prompt, model=model)
        if not greedy['n'] or not greedy['first_id']:
            return {'view': 'greedy_alternatives', 'prompt': prompt, 'model': model,
                    'sort': sort, 'entries': [], 'count': 0, 'source_records': len(self.records),
                    'greedy': greedy, 'missing_continuation': 0}

        root = self.trie(None)
        base = self.by_id[greedy['first_id']]
        lp = (base.get('choices') or [{}])[0].get('logprobs') or {}
        toks = list(lp.get('tokens') or [])
        tls = list(lp.get('token_logprobs') or [])
        tops = [dict(t or {}) for t in (lp.get('top_logprobs') or [])]

        import itertools, math
        cum = list(itertools.accumulate(v if isinstance(v, (int, float)) else 0.0 for v in tls))

        def entry_for(tokens: List[str], detail_from: List[Dict]) -> Dict:
            n = len(tokens)
            total = detail_from[-1]['cumulative'] if detail_from else 0.0
            return {
                'n': n, 'sum_logprob': total, 'mean_logprob': total / n if n else 0.0,
                'perplexity': math.exp(-total / n) if n else None,
                'text': ''.join(tokens), 'tokens': detail_from,
            }

        # The greedy path itself is rank 1 with cost 0: it is the thing the others
        # depart from, so leaving it out of its own list would be odd.
        greedy_detail = walk_detail(root, toks, max_alts)
        entries = [{
            **entry_for(toks[:len(greedy_detail)], greedy_detail),
            'deviation': None, 'cost': 0.0, 'is_greedy': True,
            'id': base.get('id'), 'finish_reason': (base.get('choices') or [{}])[0].get('finish_reason'),
        }]

        missing = 0
        for i, tok in enumerate(toks):
            if i >= len(tops) or not tops[i] or not isinstance(tls[i], (int, float)):
                continue
            for alt, alt_lp in tops[i].items():
                if alt == tok or not isinstance(alt_lp, (int, float)):
                    continue
                cost = tls[i] - alt_lp
                prefix = ''.join(toks[:i]) + alt
                rec = self.lookup(prefix, model=model, max_tokens=None)
                if rec is None:
                    missing += 1
                    continue
                rlp = (rec.get('choices') or [{}])[0].get('logprobs') or {}
                path = toks[:i] + [alt] + list(rlp.get('tokens') or [])
                detail = walk_detail(root, path, max_alts)
                if len(detail) != len(path):
                    continue          # unscorable path: skip rather than half-score it
                entries.append({
                    **entry_for(path, detail),
                    'deviation': {'position': i, 'original': tok, 'alternative': alt,
                                  'original_logprob': tls[i], 'alternative_logprob': alt_lp},
                    'cost': cost, 'is_greedy': False,
                    'id': rec.get('id'),
                    'finish_reason': (rec.get('choices') or [{}])[0].get('finish_reason'),
                    # What the greedy path scores at this same length, so the row can
                    # be compared against it rather than only against its siblings.
                    'greedy_sum_at_n': cum[len(path) - 1] if len(path) <= len(cum) else None,
                })

        # Building the candidates costs ~3 s: one lookup per (position, alternative)
        # over a 4096-token path is ~82k probes, and only the swept positions have
        # a continuation. Sorting and capping are cheap, so only the build is cached.
        self._alts_cache[cache_key] = (greedy, entries, missing)
        return self._rank_alts(prompt, model, sort, top, greedy, entries, missing)

    ALT_SORTS = {
        'cost': (lambda e: (e['cost'], -e['sum_logprob']), 'deviation cost asc (the greedy criterion)'),
        'sum': (lambda e: -e['sum_logprob'], 'sum_logprob desc'),
        'mean': (lambda e: -e['mean_logprob'], 'mean_logprob desc'),
        'length': (lambda e: (-e['n'], -e['sum_logprob']), 'length desc'),
    }

    def _rank_alts(self, prompt, model, sort, top, greedy, candidates, missing) -> Dict:
        key, label = self.ALT_SORTS.get(sort) or self.ALT_SORTS['cost']
        entries = sorted(candidates, key=key)
        if top:
            entries = entries[:top]
        # Copy before stamping rank: the candidate list is shared between requests.
        entries = [{**e, 'rank': r} for r, e in enumerate(entries, 1)]
        return {
            'view': 'greedy_alternatives',
            'prompt': prompt,
            'model': model,
            'sort': sort if sort in self.ALT_SORTS else 'cost',
            'ranking': label,
            'source_records': len(self.records),
            'generated_from': ', '.join(self.sources),
            'greedy': {k: greedy[k] for k in
                       ('n', 'sum_logprob', 'mean_logprob', 'first_id', 'finish_reason')},
            # Every departure whose continuation is not on record; each would cost
            # one call. Most of these are positions no sweep ever covered, so the
            # number is large by design and not a sign of missing data.
            'departures_without_record': missing,
            'departures_ranked': len(candidates) - 1,
            'count': len(entries),
            'entries': entries,
        }

    def walk(self, base_id: Optional[str] = None, steps: int = 20, extend: int = 20,
             model: str = 'gpt-3.5-turbo-instruct', forward_only: bool = False) -> Dict:
        """Greedy walk: repeatedly apply the cheapest single-token deviation.

        Cost of deviating at position i is logprob(chosen_i) - logprob(alt). At
        temperature 0 the chosen token is the argmax, so the cost is >= 0 and the
        cheapest available deviation is well defined -- and it is free to compute,
        because one call already returns the top-20 alternatives at every
        position. Only the regeneration afterwards needs a call.

        Two guards, without which the naive version cannot work:

          * the position just deviated is frozen. Otherwise the very next
            cheapest move at that position is to restore the original argmax,
            whose cost is NEGATIVE -- it always wins, and the walk oscillates on
            step one.
          * prompts already walked are never revisited. Deviating at an earlier
            position unfreezes the later ones, so a longer cycle is reachable.

        forward_only additionally refuses positions before the last deviation,
        which keeps each string an extension of the previous one at the price of
        dearer steps.

        Calls nothing. When the next regeneration is not in the store, the walk
        stops and reports the prompt it would have to pay for.
        """
        if base_id is None:
            found = self.bases()
            base_id = found[0]['base_id'] if found else None
        base = self.by_id.get(base_id or '')
        if base is None:
            return {'view': 'walk', 'base_id': base_id, 'base_found': False,
                    'steps': [], 'needs_call': None}

        def node_of(prefix_tokens, prefix_logprobs, prefix_tops, record):
            lp = (record.get('choices') or [{}])[0].get('logprobs') or {}
            cut = lambda a: list(a or [])[:extend]
            return {
                'tokens': prefix_tokens + cut(lp.get('tokens')),
                'logprobs': prefix_logprobs + cut(lp.get('token_logprobs')),
                'tops': prefix_tops + [dict(t or {}) for t in cut(lp.get('top_logprobs'))],
            }

        def text_of(node):
            return ''.join(node['tokens'])

        node = node_of([], [], [], base)
        frozen: set = set()
        visited = {text_of({'tokens': []})}   # the base's own prompt
        visited.add((base.get('request') or {}).get('prompt') or '')
        out = {
            'view': 'walk',
            'base_id': base_id,
            'base_found': True,
            'model': model,
            'extend': extend,
            'forward_only': forward_only,
            'base': {'text': text_of(node), 'n': len(node['tokens']),
                     'sum': sum(v for v in node['logprobs'] if isinstance(v, (int, float)))},
            'steps': [],
            'needs_call': None,
        }

        last_pos = -1
        for _ in range(max(0, steps)):
            moves = []
            for i, tok in enumerate(node['tokens']):
                if i in frozen or (forward_only and i < last_pos):
                    continue
                chosen = node['logprobs'][i]
                if not isinstance(chosen, (int, float)):
                    continue
                for alt, alt_lp in (node['tops'][i] or {}).items():
                    if alt == tok or not isinstance(alt_lp, (int, float)):
                        continue
                    moves.append((chosen - alt_lp, i, alt, alt_lp))
            moves.sort(key=lambda m: (m[0], m[1], m[2]))

            picked = None
            for cost, i, alt, alt_lp in moves:
                if ''.join(node['tokens'][:i]) + alt not in visited:
                    picked = (cost, i, alt, alt_lp)
                    break
            if picked is None:
                break

            cost, i, alt, alt_lp = picked
            prompt = ''.join(node['tokens'][:i]) + alt
            rec = self.lookup(prompt, model=model, max_tokens=extend)
            if rec is None:
                out['needs_call'] = {'position': i, 'original': node['tokens'][i],
                                     'alternative': alt, 'cost': cost, 'prompt': prompt}
                break

            visited.add(prompt)
            replaced = node['tokens'][i]   # capture before the node is rebuilt
            node = node_of(node['tokens'][:i] + [alt],
                           node['logprobs'][:i] + [alt_lp],
                           node['tops'][:i] + [node['tops'][i]], rec)
            frozen = {p for p in frozen if p < i} | {i}
            last_pos = i
            out['steps'].append({
                'step': len(out['steps']) + 1,
                'position': i,
                'original': replaced,
                'alternative': alt,
                'cost': cost,
                'id': rec.get('id'),
                'n': len(node['tokens']),
                'sum': sum(v for v in node['logprobs'] if isinstance(v, (int, float))),
                'text': text_of(node),
            })
        return out

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
