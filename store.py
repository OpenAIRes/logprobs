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

import math
import os
import time
from collections import Counter
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

# What decided where a string ends -- the axis that says whether two rows' scores
# are comparable at all. Σ logprob over a prefix is a partial sum that can only
# get worse; over a string the model itself ended it is final. Mixing them in one
# ranking systematically favours the short prefixes, which is exactly what the
# default prefixes view does: 195 of its top 200 rows are 'open'.
#
# Only 'stop' is a finished string. The other two are both PREFIXES of something
# longer, differing only in whose cut it was -- nothing about the string itself
# separates them:
#
#   stop    the model emitted EOS. 219 of 8492 records, and the only ones whose
#           score is a finished quantity.
#   length  our max_tokens ran out. 8273 of 8492 -- 97.4% of the corpus. Always
#           the ceiling and nothing else: of 8232 distinct length-ended paths,
#           none stopped before its max_tokens.
#   open    our enumeration stopped. All 168,669 of these are interior nodes with
#           the trie continuing below; the trie only holds tokens some call
#           actually generated, so there are no open leaves. Prefix rows only.
#
# How close a 'length' string came to finishing cannot be recovered: top_logprobs
# is the distribution BEFORE each generated token, so the one after the last --
# the only place EOS could appear -- is never reported.
ENDS = ('stop', 'length', 'open')

# Per-token detail shipped for one row's extension. The row's totals always cover
# the whole thing; this only bounds the payload, since the single longest
# extension in the default prefixes top 200 adds 4095 tokens on its own.
DETAIL_CAP = 400

# How a completions ranking may be ordered. Σ logprob is the default and the only
# one the prefixes view can offer: there the ranking IS the search order, so
# reordering the n-best by something else would just be "the best mean among the
# n best by sum", which reads as an answer and is not one.
# perplexity = e^(-mean logprob) is a strictly monotone transform of the mean, so
# ordering by ascending perplexity is the SAME ordering as by descending mean --
# verified over all 363 greedy siblings, identical rank for rank. It is offered
# under its own name because that is the standard way to report the per-token
# measure, and 'mean' stays as an accepted alias so older links keep working;
# what is deliberately not done is shipping both as if they were two criteria.
SORTS = {
    'sum': (lambda e: -e['sum_logprob'], 'sum_logprob desc'),
    'ppl': (lambda e: e['perplexity'], 'perplexity asc (= mean_logprob desc)'),
    'mean': (lambda e: e['perplexity'], 'perplexity asc (= mean_logprob desc)'),
    'length': (lambda e: (-e['n'], -e['sum_logprob']), 'length desc, then sum_logprob desc'),
}

DEFAULTS = dict(top=200, min_n=1, prefix=None, model=None, chosen_only=False,
                max_alts=8, sort='sum', ends=None, extend=False)


class RecordStore:
    def __init__(self, sources: Optional[List[str]] = None, root: str = ROOT):
        self.root = root
        self.sources = list(sources or SOURCES)
        self.records: List[Dict] = []
        self.by_id: Dict[str, Dict] = {}
        self.by_prompt: Dict[str, List[Dict]] = {}
        self.sweep_by_prompt: Dict[tuple, Dict] = {}
        self.ends: Dict[tuple, str] = {}
        self._tries: Dict[Optional[str], object] = {}
        self._completions_cache: Dict[tuple, List[Dict]] = {}
        self._alts_cache: Dict[tuple, tuple] = {}
        self._counts_cache: Dict[tuple, Dict] = {}
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
        self.ends = self._index_ends(self.records)
        self._tries.clear()
        self._completions_cache.clear()
        self._alts_cache.clear()
        self._counts_cache.clear()
        self.load_seconds = time.time() - started
        return self

    @staticmethod
    def _index_ends(records: List[Dict]) -> Dict[tuple, str]:
        """Full token path -> how the call that produced it ended.

        Keyed by the token tuple rather than the text, because two different
        tokenisations can render the same characters and only the token path is
        what the trie is indexed by.

        'stop' wins a collision: if any call ended on EOS at exactly this path,
        then this path IS a place the model finishes, whatever some other call
        that was still cut off at max_tokens says.
        """
        ends: Dict[tuple, str] = {}
        for rec in records:
            choice = (rec.get('choices') or [{}])[0]
            prompt_tokens = ((rec.get('prompt') or {}).get('logprobs') or {}).get('tokens') or []
            own = (choice.get('logprobs') or {}).get('tokens') or []
            path = tuple(list(prompt_tokens) + list(own))
            if not path:
                continue
            if ends.get(path) == 'stop':
                continue
            ends[path] = choice.get('finish_reason') or 'length'
        return ends

    def _node_at(self, tokens, root):
        node = root
        for tok in tokens:
            node = node.children.get(tok)
            if node is None:
                return None
        return node

    def extend_tokens(self, tokens: List[str], root, limit: int = 4096) -> List[str]:
        """Follow the model's own argmax from here to the furthest end on record.

        The rule is the node's `top_logprobs` -- the distribution the API actually
        reported for the next token -- and NOT simply the highest-scoring child.
        The two agree at 168,979 of 168,992 nodes, but where they disagree the
        child is right about a different context: the trie is keyed by token path,
        and a path reached as someone's PROMPT is not the same context as the same
        path GENERATED from an empty prompt, because an empty prompt is served with
        an implicit <|endoftext|> the trie does not model. Exactly one node is
        reached both ways -- ('
',) -- and following its best child walks off into
        another record's Java, while following its top_logprobs continues the
        greedy path this tool shows everywhere else.
        """
        node = self._node_at(tokens, root)
        if node is None:
            return []
        out: List[str] = []
        while node.children and len(out) < limit:
            pick = None
            tl = node.top_logprobs or {}
            if tl:
                best = max(tl.items(), key=lambda kv: kv[1])[0]
                child = node.children.get(best)
                if child is not None and child.logprob is not None:
                    pick = best
            if pick is None:
                kids = [(t, c) for t, c in node.children.items() if c.logprob is not None]
                if not kids:
                    break
                pick = max(kids, key=lambda kv: kv[1].logprob)[0]
            out.append(pick)
            node = node.children[pick]
        return out

    def trie_counts(self, model: Optional[str] = None, chosen_only: bool = False,
                    ends=None) -> Dict:
        """How many prefixes exist, how many the search can reach, how many match.

        The list can never show them all -- 176,649 rows would be hundreds of
        megabytes -- so the honest thing is to say what fraction it is showing.
        Without this the default view looks like 200 answers when it is 200 of
        176,649, and a `results` cap of 5000 still only reaches 2.8%.

        `unreachable` is not a filter anyone chose. build_trie fills a node's
        logprob from a generated token or its parent's top_logprobs, so a sweep's
        deviated token sits unscored whenever the position it deviates at was only
        ever a prompt position -- and a path may not pass through it, because past
        an unscored node the running sum no longer counts from the root. 22 such
        nodes block 429 below them.
        """
        key = (model, bool(chosen_only), ends)
        if key in self._counts_cache:
            return self._counts_cache[key]

        root = self.trie(model)
        usable = (lambda n: n.logprob is not None and n.source == 'chosen')             if chosen_only else (lambda n: n.logprob is not None)

        total = reachable = 0
        stack = [(c, True) for c in root.children.values()]
        while stack:
            node, ok = stack.pop()
            total += 1
            ok = ok and usable(node)
            if ok:
                reachable += 1
            stack.extend((c, ok) for c in node.children.values())

        # Reachable end-paths, counted per reason by walking the 8,431 known ends
        # rather than carrying a path tuple for all 177,100 nodes.
        by_end = Counter()
        for path, reason in self.ends.items():
            node = root
            for tok in path:
                node = node.children.get(tok)
                if node is None or not usable(node):
                    node = None
                    break
            if node is not None:
                by_end[reason] += 1
        by_end['open'] = reachable - sum(by_end.values())

        out = {
            'total': total,
            'reachable': reachable,
            'unreachable': total - reachable,
            'by_end': dict(by_end),
            'matching': (reachable if not ends
                         else sum(v for k, v in by_end.items() if k in ends)),
        }
        self._counts_cache[key] = out
        return out

    def add_extensions(self, entries: List[Dict], root, max_alts: int = 8) -> int:
        """Attach each row's continuation, and report how many distinct strings result.

        The row itself is left alone. Extending changes the score -- over the
        default top 200 it reorders 11,325 of 19,900 pairs -- so the ranked object
        has to stay the prefix, or the list would claim a best-first order it no
        longer has. The extension rides along in its own field, with its own
        totals, and `same_as_rank` marks a row whose full string an earlier row
        already reached: 200 prefixes collapse onto 93 strings, one of them
        reached 22 times, and hiding that would read as 200 findings.
        """
        first_seen: Dict[str, int] = {}
        for e in entries:
            toks = [t['token'] for t in e['tokens']]
            added = self.extend_tokens(toks, root)
            if not added:
                # Already at an end on record. Re-walking a 4096-token path to
                # learn that cost 4.3 s over ten greedy siblings, all of which
                # add nothing.
                detail, full, total = e['tokens'], toks, e['sum_logprob']
            else:
                full = toks + added
                detail = walk_detail(root, full, max_alts)
                if len(detail) != len(full):
                    detail, full, added = e['tokens'], toks, []
                total = detail[-1]['cumulative'] if detail else 0.0
            n = len(full)
            text = ''.join(full)
            shown = detail[len(toks):]
            e['extension'] = {
                'added_n': len(added),
                'n': n,
                'sum_logprob': total,
                'mean_logprob': total / n if n else 0.0,
                'perplexity': math.exp(-total / n) if n else None,
                'text': text,
                # The totals above describe the WHOLE extension; only the per-token
                # detail is capped, because one row can add 4095 tokens and shipping
                # each with its top-k alternatives would run to megabytes per row.
                'tokens': shown[:DETAIL_CAP],
                'detail_truncated': len(shown) > DETAIL_CAP,
                'end': self.ends.get(tuple(full), 'open'),
                'same_as_rank': first_seen.get(text),
            }
            first_seen.setdefault(text, e.get('rank'))
        return len(first_seen)

    @staticmethod
    def _normalize_ends(value) -> Optional[frozenset]:
        """None / empty / everything -> no filter, so the fast path stays fast.

        An unchecked-everything box means the same thing as no filter, and saying
        so here keeps the search from paying for an accept() callback that always
        returns True.
        """
        if value is None:
            return None
        if isinstance(value, str):
            value = [p.strip() for p in value.split(',')]
        wanted = frozenset(v for v in value if v in ENDS)
        if not wanted:
            raise ValueError(f'ends must name at least one of {ENDS}')
        return None if wanted == frozenset(ENDS) else wanted

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

    def _completions(self, args, stats: Optional[Dict] = None) -> List[Dict]:
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
        # build_completion_entries already returned them sum-ordered, so only a
        # different key needs work.
        chosen = args.sort if args.sort in SORTS else 'sum'
        if chosen != 'sum':
            entries = sorted(entries, key=SORTS[chosen][0])
        # After the sort and before the cap, for the same reason the sort is:
        # filtering the top N would answer "the best finished string among the
        # top N by sum", which is a different question from the one asked.
        if args.ends:
            entries = [e for e in entries if e.get('end') in args.ends]
        entries = apply_prefix(entries, args.prefix)
        # Everything the cap was applied to, so the view can say what fraction of
        # it is on screen instead of looking like the whole answer.
        if stats is not None:
            stats['available'] = len(entries)
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
        args.ends = self._normalize_ends(args.ends)
        root = self.trie(args.model)
        counts = (self.trie_counts(args.model, args.chosen_only, args.ends)
                  if view == 'prefixes' else None)
        search: Dict = {}
        if view == 'completions':
            entries = self._completions(args, search)
            ranking = (SORTS.get(args.sort) or SORTS['sum'])[1]                 + ' over whole strings (prompt + completion)'
        else:
            # The heap search is by sum, so that is the only honest ranking here.
            args.sort = 'sum'
            entries = build_prefix_entries(root, self.sweep_by_prompt, args,
                                           ends_index=self.ends, stats=search)
            ranking = 'sum_logprob desc (best-first / uniform-cost over token trie)'
        # After the cap: only the rows actually shown need a continuation, and the
        # ranked object stays the prefix -- see add_extensions for why it must.
        distinct = self.add_extensions(entries, root, args.max_alts) if args.extend else None
        if args.extend:
            ranking += ' (rows extended for display; the RANKING is still the prefix)'
        return {
            'generated_from': ', '.join(self.sources),
            'model_filter': args.model,
            'chosen_only': args.chosen_only,
            'source_records': len(self.records),
            'ranking': ranking,
            'sort': args.sort,
            'view': view,
            'prefix_filter': args.prefix,
            # None means every kind of ending, which is the pre-existing behaviour.
            'ends': sorted(args.ends) if args.ends else None,
            'end_counts': dict(Counter(e.get('end') for e in entries)),
            # Only the trie search can run out of budget; a completions ranking is
            # a filter over a finite list and is always complete.
            'search_pops': search.get('pops'),
            'truncated_by_budget': bool(search.get('budget_exhausted')),
            'extend': bool(args.extend),
            'chosen_only_note': ('paths whose every token some call actually generated'
                                 if args.chosen_only else None),
            # What the cap was applied to, and what exists at all. A prefixes view
            # never shows more than a sliver, and it should say so rather than let
            # 200 rows read as 200 answers.
            'available': (search.get('available') if view == 'completions'
                          else counts['matching']),
            'trie_total': None if view == 'completions' else counts['total'],
            'unreachable': None if view == 'completions' else counts['unreachable'],
            # Split, because the two causes are not the same thing and one of them
            # is a setting. Calling 172,926 excluded-by-chosen_only nodes
            # "unscored" would be plainly false: they are scored, just recovered.
            'unreachable_unscored': (None if view == 'completions'
                                     else self.trie_counts(args.model, False, None)['unreachable']),
            # How many distinct strings the extended rows collapse onto. Far fewer
            # than the row count, and saying so is the point.
            'distinct_extended': distinct,
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
                            max_alts: int = 8, ends=None, extend: bool = False) -> Dict:
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
        # Not part of the cache key: the cached value is the unranked, unfiltered
        # candidate set, and sort/top/ends are all applied to a copy of it.
        ends = self._normalize_ends(ends)
        cache_key = (prompt, model, max_alts)
        if cache_key in self._alts_cache:
            greedy, candidates, missing = self._alts_cache[cache_key]
            return self._rank_alts(prompt, model, sort, top, greedy, candidates,
                                   missing, ends, extend, root_for_extend=self.trie(model),
                                   max_alts=max_alts)

        greedy = self.greedy(prompt=prompt, model=model)
        if not greedy['n'] or not greedy['first_id']:
            return {'view': 'greedy_alternatives', 'prompt': prompt, 'model': model,
                    'sort': sort, 'entries': [], 'count': 0, 'source_records': len(self.records),
                    'greedy': greedy, 'missing_continuation': 0}

        # trie(model), not trie(None). The unfiltered trie merges every model, and
        # a node keeps whatever record filled it first -- so the '8' at position 4
        # of the porchlight path was scored -1.6943 from a davinci-002 record
        # instead of -1.8337 from gpt-3.5-turbo-instruct's own top_logprobs, and
        # every sum downstream of it was wrong by 0.1394. The same string then had
        # two different Σ depending on which view computed it. lookup() in this
        # very function already filters by model; walk_detail bypassed it.
        root = self.trie(model)
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
            # Same field name every view uses. A sibling is always a real recorded
            # continuation, so it carries the API's verdict and is never 'open'.
            'end': (base.get('choices') or [{}])[0].get('finish_reason') or 'length',
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
                    'end': (rec.get('choices') or [{}])[0].get('finish_reason') or 'length',
                    # What the greedy path scores at this same length, so the row can
                    # be compared against it rather than only against its siblings.
                    'greedy_sum_at_n': cum[len(path) - 1] if len(path) <= len(cum) else None,
                })

        # Building the candidates costs ~3 s: one lookup per (position, alternative)
        # over a 4096-token path is ~82k probes, and only the swept positions have
        # a continuation. Sorting and capping are cheap, so only the build is cached.
        self._alts_cache[cache_key] = (greedy, entries, missing)
        return self._rank_alts(prompt, model, sort, top, greedy, entries, missing, ends,
                               extend, root_for_extend=root, max_alts=max_alts)

    # cost first, and it is the default: the greedy criterion is what makes a
    # sibling "second best", and the other keys answer a different question.
    ALT_SORTS = {
        'cost': (lambda e: (e['cost'], -e['sum_logprob']), 'deviation cost asc (the greedy criterion)'),
        'sum': (lambda e: -e['sum_logprob'], 'sum_logprob desc'),
        'ppl': (lambda e: e['perplexity'], 'perplexity asc (= mean_logprob desc)'),
        'mean': (lambda e: e['perplexity'], 'perplexity asc (= mean_logprob desc)'),
        'length': (lambda e: (-e['n'], -e['sum_logprob']), 'length desc'),
    }

    def _rank_alts(self, prompt, model, sort, top, greedy, candidates, missing,
                   ends=None, extend=False, root_for_extend=None, max_alts=8) -> Dict:
        key, label = self.ALT_SORTS.get(sort) or self.ALT_SORTS['cost']
        entries = sorted(candidates, key=key)
        # After the sort and before the cap, as everywhere else: filtering the
        # capped list would answer "the cheapest finished departure among the N
        # cheapest", which is not the question the checkbox asks.
        if ends:
            entries = [e for e in entries if e.get('end') in ends]
        if top:
            entries = entries[:top]
        # Copy before stamping rank: the candidate list is shared between requests.
        entries = [{**e, 'rank': r} for r, e in enumerate(entries, 1)]
        distinct = (self.add_extensions(entries, root_for_extend, max_alts)
                    if extend and root_for_extend is not None else None)
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
            'ends': sorted(ends) if ends else None,
            'end_counts': dict(Counter(e.get('end') for e in entries)),
            'extend': bool(extend),
            'distinct_extended': distinct,
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
