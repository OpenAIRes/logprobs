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
import json
import os
import tempfile
import threading
import time
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Dict, List, Optional

from best_avg_logprob_path import build_trie, model_matches, record_model
from export_dijkstra_top import (
    apply_prefix,
    ppl_from_mean,
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

# Which file a record came from, grouped. The six sources are two kinds of thing:
# the ordinary history of calls made by hand, and the sweeps -- exhaustive
# one-token perturbations of a few bases, which are 8080 of the 8492 records and
# so dominate any ranking they are in.
#
# The filter applies to what gets RANKED or COUNTED. It deliberately does not
# apply to lookup() or greedy(): the first answers "do we already have this, or
# must it be paid for", and narrowing that would make us buy a record we hold;
# the second reconstructs a path by chaining records, and dropping some of them
# would not filter the path, it would break it.
SOURCE_GROUPS = {
    'history': ['completion_history.json', 'builder_history.json',
                'meta_resample_root.json'],
    'sweep': ['sweep_history.json', 'sweep2_history.json.gz',
              'sweep3_ihave.json.gz'],
}
GROUP_OF = {f: g for g, files in SOURCE_GROUPS.items() for f in files}

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

# Whether the database holds anything past this string. A SEPARATE axis from how
# the string ended, and the two cross: every 'open' node has a continuation by
# construction, but 494 'length' paths do too -- a later call took that text as
# its prompt and carried on -- so "no continuation" is not "untick open".
#
#             has children   leaf
#   open           168,669       0
#   length             494   7,738
#   stop                 0     199
#
# Unlike the recovered/generated split, both sides here are populous, which is
# why this one is a pair of boxes and that one is a single box.
NODE_KINDS = ('continues', 'leaf')

# Per-token detail shipped for one row's extension. The row's totals always cover
# the whole thing; this only bounds the payload, since the single longest
# extension in the default prefixes top 200 adds 4095 tokens on its own.
DETAIL_CAP = 400

# How a completions ranking may be ordered. Σ logprob is the default and the only
# one the prefixes view can offer: there the ranking IS the search order, so
# reordering the n-best by something else would just be "the best mean among the
# n best by sum", which reads as an answer and is not one.
# Sigma logprob is the one that answers "what would the model actually say": a
# sample at temperature 1 comes out with probability exactly e^(sum), so the most
# frequently generated string is the highest-sum one, i.e. MAP. Perplexity is
# length-normalised and so rewards long predictable strings, which are the least
# likely to appear -- over the 197 completed strings on record the best by
# perplexity is 3e74 times less probable than the best by sum, and 194th of 197
# by sum. Neither is what greedy finds, which is only locally optimal.
#
# perplexity = e^(-mean logprob) is a strictly monotone transform of the mean, so
# ordering by ascending perplexity is the SAME ordering as by descending mean --
# verified over all 363 greedy siblings, identical rank for rank. It is offered
# under its own name because that is the standard way to report the per-token
# measure, and 'mean' stays as an accepted alias so older links keep working;
# what is deliberately not done is shipping both as if they were two criteria.
def _ppl_key(entry):
    """Ascending perplexity, with "no perplexity" last -- see ppl_from_mean."""
    value = entry.get('perplexity')
    return float('inf') if value is None else value


SORTS = {
    'sum': (lambda e: -e['sum_logprob'], 'sum_logprob desc'),
    'ppl': (lambda e: _ppl_key(e), 'perplexity asc (= mean_logprob desc)'),
    'mean': (lambda e: _ppl_key(e), 'perplexity asc (= mean_logprob desc)'),
    'length': (lambda e: (-e['n'], -e['sum_logprob']), 'length desc, then sum_logprob desc'),
}

DEFAULTS = dict(top=200, min_n=1, prefix=None, model=None, chosen_only=False,
                max_alts=8, sort='sum', ends=None, extend=False, nodes=None,
                sources=None)


@contextmanager
def history_lock(root):
    """Coordinate atomic history writes even between two server processes."""
    with open(os.path.join(root, '.history.lock'), 'a+b') as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell() == 0:
            fh.write(b'\0')
            fh.flush()
        fh.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fh.seek(0)
            if os.name == 'nt':
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class RecordStore:
    def __init__(self, sources: Optional[List[str]] = None, root: str = ROOT):
        self.root = root
        self.lock = threading.RLock()
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
        self._subsets: Dict[Optional[frozenset], List[Dict]] = {}
        self._subset_ends: Dict[Optional[frozenset], Dict[tuple, str]] = {}
        self._subset_sweeps: Dict[Optional[frozenset], Dict[tuple, Dict]] = {}
        self.load_seconds = 0.0
        self.missing: List[str] = []

    # -- loading --------------------------------------------------------------

    def load(self) -> 'RecordStore':
        started = time.time()
        self.missing = []
        paths = []
        for name in self.sources:
            path = os.path.join(self.root, name)
            if os.path.exists(path):
                paths.append(path)
            else:
                self.missing.append(name)
        self.records = load_records(paths)
        # Tagged at load, so a subset is a filter rather than a second read of the
        # files. load_records keeps the first copy of a duplicated id, and the
        # order of SOURCES decides which file that was -- the tag follows it.
        by_id_group = {}
        for path in paths:
            group = GROUP_OF.get(os.path.basename(path))
            for rec in load_records([path]):
                by_id_group.setdefault(rec.get('id'), group)
        for rec in self.records:
            rec['_group'] = by_id_group.get(rec.get('id'))
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
        self._subsets.clear()
        self._subset_ends.clear()
        self._subset_sweeps.clear()
        self.load_seconds = time.time() - started
        return self

    def save(self, record: Dict) -> Dict:
        """Commit one record. See save_many, which does the work."""
        return self.save_many([record])[0]

    def save_many(self, records: List[Dict]) -> List[Dict]:
        """Commit to the existing history before exposing the records to readers.

        The server shares this lock with queries and cache warming. The on-disk
        history is reread under the lock, never rebuilt from a filtered view.

        Taking a list rather than one record is not a convenience: writing the
        8.7 MB history and reloading the indexes costs about ten seconds, and
        doing that per record made importing eleven of them take two minutes,
        with a resample run of five variations hanging for the best part of a
        minute after the API had already answered. One read, n appends, one
        write, one reload.
        """
        with self.lock, history_lock(self.root):
            for record in records:
                if not isinstance(record.get('id'), str) or not record['id']:
                    raise ValueError('response has no record id')
            name = 'completion_history.json'
            if name not in self.sources:
                raise ValueError('completion_history.json must be a store source')
            path = os.path.join(self.root, name)
            history = []
            if os.path.exists(path):
                with open(path, encoding='utf-8') as fh:
                    history = json.load(fh)
                if isinstance(history, dict):
                    history = [history]
                if not isinstance(history, list) or not all(isinstance(r, dict) for r in history):
                    raise ValueError('invalid history; refusing to overwrite it')

            known = {r.get('id') for r in history}
            added = False
            for record in records:
                # A record already on disk wins: an id is a paid call's identity,
                # and the copy that got there first is the one every view has
                # been showing.
                if record['id'] in known:
                    continue
                history.append({k: v for k, v in record.items() if not k.startswith('_')})
                known.add(record['id'])
                added = True

            if added:
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                                     dir=self.root, delete=False,
                                                     prefix='.history-', suffix='.tmp') as fh:
                        temp_path = fh.name
                        json.dump(history, fh, ensure_ascii=False, allow_nan=False)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(temp_path, path)
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
                self.load()
            elif any(r['id'] not in self.by_id for r in records):
                # On disk but not in memory: another writer got there between our
                # last load and this call.
                self.load()
            return [self.by_id[r['id']] for r in records]

    def lookup_request(self, request: Dict) -> Optional[Dict]:
        """Exact Studio request match; analysis callers retain lookup semantics."""
        defaults = {'temperature': 0, 'top_p': 1, 'frequency_penalty': 0,
                    'presence_penalty': 0}
        for rec in reversed(self.by_prompt.get(request['prompt'], [])):
            req = rec.get('request') or {}
            if not self._model_matches(rec, request['model']):
                continue
            if any(req.get(k, default) != request[k] for k, default in defaults.items()):
                continue
            # Exact lengths avoid returning an untrimmed longer cached response.
            if req.get('max_tokens') != request['max_tokens']:
                continue
            if req.get('logprobs') != request['logprobs']:
                continue
            # Studio does not request stop sequences, suffixes or multiple choices.
            if req.get('stop') or req.get('suffix') or req.get('n', 1) != 1 or req.get('best_of', 1) != 1:
                continue
            return rec

        return None

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
                    ends=None, nodes=None, sources=None) -> Dict:
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
        key = (model, bool(chosen_only), ends, nodes, sources)
        if key in self._counts_cache:
            return self._counts_cache[key]

        root = self.trie(model, sources)
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

        # Reachable end-paths as an (ending x has-continuation) cross-tab, counted
        # by walking the 8,431 known ends rather than carrying a path tuple for all
        # 177,100 nodes. Every remaining reachable node is 'open', and an open node
        # always has children, so that cell takes the balance and open-leaf is 0.
        cells = Counter()
        for path, reason in self.ends_for(sources).items():
            node = root
            for tok in path:
                node = node.children.get(tok)
                if node is None or not usable(node):
                    node = None
                    break
            if node is not None:
                cells[(reason, 'continues' if node.children else 'leaf')] += 1
        cells[('open', 'continues')] = reachable - sum(cells.values())

        by_end = Counter()
        by_kind = Counter()
        for (reason, kind), v in cells.items():
            by_end[reason] += v
            by_kind[kind] += v

        out = {
            'total': total,
            'reachable': reachable,
            'unreachable': total - reachable,
            'by_end': dict(by_end),
            'by_kind': dict(by_kind),
            'matching': sum(v for (reason, kind), v in cells.items()
                            if (not ends or reason in ends)
                            and (not nodes or kind in nodes)),
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
                'perplexity': ppl_from_mean(total / n) if n else None,
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
    def _normalize_nodes(value) -> Optional[frozenset]:
        """None / empty / both -> no filter, so the search keeps its fast path."""
        if value is None:
            return None
        if isinstance(value, str):
            value = [p.strip() for p in value.split(',')]
        wanted = frozenset(v for v in value if v in NODE_KINDS)
        if not wanted:
            raise ValueError(f'nodes must name at least one of {NODE_KINDS}')
        return None if wanted == frozenset(NODE_KINDS) else wanted

    def node_kind(self, tokens, root) -> Optional[str]:
        node = self._node_at(tokens, root)
        if node is None:
            return None
        return 'continues' if node.children else 'leaf'

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

    @staticmethod
    def _normalize_sources(value) -> Optional[frozenset]:
        """None / empty / everything -> no filter, so the common case is the fast
        one and every existing link keeps its meaning."""
        if value is None:
            return None
        if isinstance(value, str):
            value = [p.strip() for p in value.split(',')]
        wanted = frozenset(v for v in value if v in SOURCE_GROUPS)
        if not wanted:
            raise ValueError(f'sources must name at least one of {tuple(SOURCE_GROUPS)}')
        return None if wanted == frozenset(SOURCE_GROUPS) else wanted

    def records_for(self, sources=None, model=None) -> List[Dict]:
        """The record list a query should see. The model filter belongs here as
        well as in the trie: build_completion_entries walks whatever list it is
        given, and leaving other models in it meant their entries were dropped
        only because they happened to be unscorable against the filtered trie --
        an accident, not a filter."""
        key = (sources, model)
        if key not in self._subsets:
            out = self.records
            if sources is not None:
                out = [r for r in out if r.get('_group') in sources]
            if model:
                out = [r for r in out if model_matches(r, model)]
            self._subsets[key] = out
        return self._subsets[key]

    def ends_for(self, sources=None) -> Dict[tuple, str]:
        if sources is None:
            return self.ends
        if sources not in self._subset_ends:
            self._subset_ends[sources] = self._index_ends(self.records_for(sources))
        return self._subset_ends[sources]

    def sweeps_for(self, sources=None) -> Dict[tuple, Dict]:
        if sources is None:
            return self.sweep_by_prompt
        if sources not in self._subset_sweeps:
            self._subset_sweeps[sources] = index_sweeps(self.records_for(sources))
        return self._subset_sweeps[sources]

    def trie(self, model: Optional[str] = None, sources=None):
        key = (model, sources)
        if key not in self._tries:
            self._tries[key] = build_trie(self.records_for(sources), model)
        return self._tries[key]

    # -- the one cache verdict ------------------------------------------------

    # One implementation, shared with build_trie. Two of them is how the trie came
    # to drop every davinci and babbage record while the cache verdict kept them.
    _model_matches = staticmethod(model_matches)

    def lookup(self, prompt: str, model: Optional[str] = None,
               max_tokens: Optional[int] = None,
               temperature: Optional[float] = 0,
               sources=None) -> Optional[Dict]:
        """The record that already answers this call, or None.

        A record may serve a request for FEWER tokens than it generated (the
        caller truncates), but never for more -- unless it stopped on its own,
        in which case there is nothing more to get.
        """
        for rec in self.by_prompt.get(prompt, []):
            req = rec.get('request') or {}
            # `sources` is for callers asking "is this in the databases I picked",
            # not for the cache verdict. The verdict must see everything: narrowing
            # it would report a miss for a record we hold and buy it again. Every
            # caller that answers "must this be paid for" leaves it None.
            if sources and rec.get('_group') not in sources:
                continue
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
        key = (args.model, bool(args.chosen_only), args.max_alts, args.sources)
        if key not in self._completions_cache:
            full = SimpleNamespace(**{**vars(args), 'prefix': None, 'top': None})
            self._completions_cache[key] = build_completion_entries(
                self.trie(args.model, args.sources),
                self.records_for(args.sources, args.model),
                self.sweeps_for(args.sources), full)

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
        if args.nodes:
            root = self.trie(args.model, args.sources)
            entries = [e for e in entries
                       if self.node_kind([t['token'] for t in e['tokens']], root) in args.nodes]
        entries = apply_prefix(entries, args.prefix)
        # Everything the cap was applied to, so the view can say what fraction of
        # it is on screen instead of looking like the whole answer.
        if stats is not None:
            stats['available'] = len(entries)
        if args.top:
            entries = entries[:args.top]
        # Copy before stamping rank: the cached list is shared between requests,
        # and a filtered view must not renumber the entries another one is reading.
        # node_kind is stamped here rather than in the cached ranking because it
        # is a question about the trie, and only the shown rows need the answer.
        trie = self.trie(args.model, args.sources)
        return [{**e, 'rank': i,
                 'node_kind': self.node_kind([t['token'] for t in e['tokens']], trie)}
                for i, e in enumerate(entries, 1)]

    def query(self, view: str = 'prefixes', **kwargs) -> Dict:
        if view not in VIEWS:
            raise ValueError(f'unknown view {view!r}; expected one of {VIEWS}')
        args = SimpleNamespace(**{**DEFAULTS,
                                  **{k: v for k, v in kwargs.items() if k in DEFAULTS}})
        args.ends = self._normalize_ends(args.ends)
        args.nodes = self._normalize_nodes(args.nodes)
        args.sources = self._normalize_sources(args.sources)
        root = self.trie(args.model, args.sources)
        counts = (self.trie_counts(args.model, args.chosen_only, args.ends, args.nodes,
                                   args.sources)
                  if view == 'prefixes' else None)
        search: Dict = {}
        if view == 'completions':
            entries = self._completions(args, search)
            ranking = (SORTS.get(args.sort) or SORTS['sum'])[1]                 + ' over whole strings (prompt + completion)'
        else:
            # The heap search is by sum, so that is the only honest ranking here.
            args.sort = 'sum'
            entries = build_prefix_entries(root, self.sweeps_for(args.sources), args,
                                           ends_index=self.ends_for(args.sources),
                                           stats=search)
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
            'source_records': len(self.records_for(args.sources, args.model)),
            'sources': sorted(args.sources) if args.sources else None,
            'ranking': ranking,
            'sort': args.sort,
            'view': view,
            'prefix_filter': args.prefix,
            # None means every kind of ending, which is the pre-existing behaviour.
            'ends': sorted(args.ends) if args.ends else None,
            'end_counts': dict(Counter(e.get('end') for e in entries)),
            'nodes': sorted(args.nodes) if args.nodes else None,
            'node_counts': dict(Counter(e.get('node_kind') for e in entries)),
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
                                     else self.trie_counts(args.model, False, None, None,
                                                           args.sources)['unreachable']),
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
            'perplexity': ppl_from_mean(total / n) if n else None,
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
                            max_alts: int = 8, ends=None, extend: bool = False,
                            nodes=None, sources=None) -> Dict:
        """Greedy II: branch only at the first generated token after the prompt.

        One string per recorded top-20 next token, including the original greedy
        string. Later token positions never contribute additional candidates.
        """
        # Not part of the cache key: the cached value is the unranked, unfiltered
        # candidate set, and sort/top/ends are all applied to a copy of it.
        ends = self._normalize_ends(ends)
        nodes = self._normalize_nodes(nodes)
        sources = self._normalize_sources(sources)
        cache_key = (prompt, model, max_alts, sources)
        if cache_key in self._alts_cache:
            greedy, candidates, missing = self._alts_cache[cache_key]
            return self._rank_alts(prompt, model, sort, top, greedy, candidates,
                                   missing, ends, extend,
                                   root_for_extend=self.trie(model, sources),
                                   max_alts=max_alts, nodes=nodes, sources=sources)

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
        root = self.trie(model, sources)
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
                'perplexity': ppl_from_mean(total / n) if n else None,
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
            'node_kind': self.node_kind(toks[:len(greedy_detail)], root),
        }]

        missing = 0
        for i, tok in enumerate(toks[:1]):
            if i >= len(tops) or not tops[i] or not isinstance(tls[i], (int, float)):
                continue
            for alt, alt_lp in sorted(tops[i].items(), key=lambda item: -item[1])[:20]:
                if alt == tok or not isinstance(alt_lp, (int, float)):
                    continue
                cost = tls[i] - alt_lp
                prefix = ''.join(toks[:i]) + alt
                # Restricted to the chosen databases: the candidates are what
                # those databases contain, while the base path below still comes
                # from everything, because a path with records removed is not a
                # filtered path, it is a broken one.
                rec = self.lookup(prefix, model=model, max_tokens=None, sources=sources)
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
                    'node_kind': self.node_kind(path, root),
                    # What the greedy path scores at this same length, so the row can
                    # be compared against it rather than only against its siblings.
                    'greedy_sum_at_n': cum[len(path) - 1] if len(path) <= len(cum) else None,
                })

        # Cache the first-token branches; display sorting and filters are applied later.
        self._alts_cache[cache_key] = (greedy, entries, missing)
        return self._rank_alts(prompt, model, sort, top, greedy, entries, missing, ends,
                               extend, root_for_extend=root, max_alts=max_alts,
                               nodes=nodes, sources=sources)

    # A 16-token string has ~300 deviations; the 4096-token record has ~78,000,
    # and the greedy path IS that record. Planning them all means 78,000 store
    # lookups and a response no browser should be handed, for a set nobody would
    # buy anyway. So the plan stops at a number of cells and says that it did.
    DEVIATION_CELLS = 2000

    # Deviating to end-of-text is not a prompt. The string simply stops there, so
    # there is no continuation to look up and none to buy -- sending
    # "...<|endoftext|>" to the API would just ask about the literal 13
    # characters. It is not rare either: EOS is in top_logprobs at 22.7% of
    # positions, so left in the plan it would be about one call in nineteen,
    # every one of them meaningless. Counted separately instead.
    EOS_TOKEN = '<|endoftext|>'

    def deviations(self, tokens, model: str = 'gpt-3.5-turbo-instruct',
                   alts: int = 20, max_alts: int = 8, sources=None,
                   ends=None, nodes=None, sort: str = 'cost',
                   max_cells: Optional[int] = None,
                   prompt_tokens=None) -> Dict:
        """Every one-token deviation of ONE given string, with real continuations.

        For each position i of the string and each alternative the model recorded
        at i, the deviated prefix tokens[:i] + [alt] is a prompt, and the answer
        is whatever the model generated from that prompt: the record the store
        already holds, or -- for the ones it does not hold -- a prompt handed back
        in `missing` for the caller to buy. The tail is therefore never the
        original tail; it is what the model really continues with.

        `prompt_tokens` is the part that CANNOT be deviated and is not optional
        bookkeeping. A given prompt has no logprobs -- the API returns them only
        for tokens it generated -- so there are no alternatives at those positions
        and no question to ask about them; they stay in front of every deviated
        prompt exactly as they are.

        They also decide where in the trie the walk starts, which is what makes
        this correct rather than merely tidy. build_trie hangs a record's
        generated tokens under its prompt tokens, and gives a prompt node no
        logprob, so walking from the root along the generated tokens of a prompted
        record either stops at once or -- worse -- follows some other record's
        branch that happens to begin with the same token and scores the string
        against the wrong context. 8487 of 8493 records have a non-empty prompt,
        so before this argument existed this function answered 0 deviations for
        all but six of them.

        Deliberately not what greedy_alternatives does, and not what
        single_token_variants does:

          greedy_alternatives  branches at ONE position, the first token after the
                               prompt, and asks what comes next instead.
          variants_for         deviates at every position but KEEPS the original
                               tail, so it needs no model and answers a textual
                               question about strings the model never produced.
          this                 deviates at every position AND regenerates, so a
                               16-token string yields ~16 x 19 real strings. It is
                               the sweep, done for one base on demand.
        """
        ends = self._normalize_ends(ends)
        nodes = self._normalize_nodes(nodes)
        sources = self._normalize_sources(sources)
        root = self.trie(model, sources)
        tokens = [t for t in tokens if isinstance(t, str)]
        prompt_tokens = [t for t in (prompt_tokens or []) if isinstance(t, str)]
        prompt_text = ''.join(prompt_tokens)
        start = self._node_at(prompt_tokens, root) if prompt_tokens else root
        if start is None:
            return {'view': 'deviations', 'model': model,
                    'error': 'this prompt is not in the chosen databases, '
                             'so there is nothing recorded to deviate from',
                    'prompt': prompt_text, 'prompt_tokens': prompt_tokens,
                    'n': len(tokens), 'planned_positions': 0,
                    'positions_in_trie': 0, 'deviations': 0, 'from_store': 0,
                    'missing': [], 'missing_count': 0, 'entries': [], 'count': 0}

        # alts (20) is the deviation plan; max_alts (8) is only how many
        # alternatives each row carries for display. Two different numbers, and
        # conflating them is how a plan silently loses more than half its cells.
        plan = walk_detail(start, tokens, alts)

        entries: List[Dict] = []
        missing: List[Dict] = []
        seen = set()
        unscorable = duplicates = cells = eos = 0
        cap = self.DEVIATION_CELLS if max_cells is None else max(1, max_cells)
        planned = 0
        for i, step in enumerate(plan):
            if cells >= cap:
                break
            planned = i + 1
            # The fixed prompt first, always: the deviation is inside the part
            # that was generated, and the call has to be made in the same context
            # the original was.
            prefix = prompt_text + ''.join(tokens[:i])
            for a in step['alternatives']:
                alt = a['token']
                if alt == tokens[i] or not isinstance(a['logprob'], (int, float)):
                    continue
                if alt == self.EOS_TOKEN:
                    eos += 1
                    continue
                cells += 1
                deviation = {'position': i, 'original': tokens[i],
                             'alternative': alt,
                             'original_logprob': step['logprob'],
                             'alternative_logprob': a['logprob']}
                cost = step['logprob'] - a['logprob']
                rec = self.lookup(prefix + alt, model=model, max_tokens=None,
                                  sources=sources)
                if rec is None:
                    # No 'prompt' field on purpose: it is tokens[:i] + alt, so
                    # for position i it is i tokens long and the whole list of
                    # them is quadratic in the length of the string -- hundreds
                    # of megabytes for the 4096-token record. The caller has the
                    # tokens and joins them itself.
                    missing.append({**deviation, 'cost': cost})
                    continue
                choice = (rec.get('choices') or [{}])[0]
                path = tokens[:i] + [alt] + list(
                    (choice.get('logprobs') or {}).get('tokens') or [])
                key = tuple(path)
                if key in seen:
                    duplicates += 1
                    continue
                detail = walk_detail(start, path, max_alts)
                if not detail or len(detail) != len(path):
                    # Scored half-way is worse than absent -- the same rule the
                    # rankings use. Counted, so the total still adds up.
                    unscorable += 1
                    continue
                seen.add(key)
                total = detail[-1]['cumulative']
                end = choice.get('finish_reason') or 'length'
                entries.append({
                    'n': len(path), 'text': ''.join(path), 'tokens': detail,
                    'sum_logprob': total, 'mean_logprob': total / len(path),
                    'perplexity': ppl_from_mean(total / len(path)),
                    'cost': cost, 'deviation': deviation, 'is_greedy': False,
                    'id': rec.get('id'), 'end': end, 'finish_reason': end,
                    'node_kind': self.node_kind(path, start),
                })

        # Cheapest departure first, and the same for the ones that have to be
        # bought: the plan is walked in position order, but whoever stops after
        # thirty calls wants the thirty likeliest deviations, not every
        # alternative at the first two positions. Cost is the only criterion
        # available here -- the others need the string, which is what is missing.
        missing.sort(key=lambda m: m['cost'])

        visible = [e for e in entries
                   if (not ends or e.get('end') in ends)
                   and (not nodes or e.get('node_kind') in nodes)]
        visible.sort(key=(self.ALT_SORTS.get(sort) or self.ALT_SORTS['cost'])[0])
        visible = [{**e, 'rank': i} for i, e in enumerate(visible, 1)]
        return {
            'view': 'deviations', 'model': model, 'sort': sort,
            'ranking': (self.ALT_SORTS.get(sort) or self.ALT_SORTS['cost'])[1],
            'base_tokens': tokens, 'base_text': ''.join(tokens),
            # Reported so a caller can say which part is immovable, and so the
            # strings in `entries` (the generated part only) are not mistaken for
            # whole prompts.
            'prompt': prompt_text, 'prompt_tokens': prompt_tokens,
            'fixed': len(prompt_tokens),
            # planned < n means the string leaves the trie part-way: nothing is
            # recorded past that point, so there is nothing to deviate from.
            'n': len(tokens), 'planned_positions': planned,
            # planned < len(plan) means the cell cap stopped it; len(plan) < n
            # means the string leaves the trie, and there is nothing recorded
            # past that point to deviate from. Two different partial answers.
            'positions_in_trie': len(plan), 'cell_cap': cap,
            'capped': planned < len(plan),
            'deviations': len(entries) + len(missing) + duplicates + unscorable,
            'from_store': len(entries), 'duplicates': duplicates,
            'unscorable': unscorable, 'eos_deviations': eos,
            'missing': missing, 'missing_count': len(missing),
            'entries': visible, 'count': len(visible),
            'available': len(entries),
            'ends': sorted(ends) if ends else None,
            'nodes': sorted(nodes) if nodes else None,
            'end_counts': dict(Counter(e['end'] for e in visible)),
            'node_counts': dict(Counter(e['node_kind'] for e in visible)),
        }

    def deviations_for_id(self, base_id: str, **kw) -> Dict:
        """The same, for a record named by id -- its own generated tokens."""
        rec = self.by_id.get(base_id)
        if rec is None:
            return {'view': 'deviations', 'error': 'no record with that id',
                    'entries': [], 'count': 0, 'missing': [], 'missing_count': 0}
        lp = (rec.get('choices') or [{}])[0].get('logprobs') or {}
        # The record's own split: its prompt is fixed, its generated tokens are
        # what can be deviated.
        out = self.deviations(
            list(lp.get('tokens') or []),
            prompt_tokens=list(((rec.get('prompt') or {}).get('logprobs') or {}).get('tokens') or []),
            **kw)
        out['base_id'] = base_id
        return out

    def greedy_i(self, prompt='', model='gpt-3.5-turbo-instruct', top=20,
                 sort='cost', max_alts=8, ends=None, extend=False, nodes=None, sources=None):
        """Best departure prefix among all accepted paths, using stored continuations."""
        import heapq
        import itertools
        sources = self._normalize_sources(sources)
        ends = self._normalize_ends(ends)
        nodes = self._normalize_nodes(nodes)
        root = self.trie(model, sources)
        baseline = self.greedy_alternatives(prompt=prompt, model=model, top=1,
                                             max_alts=max_alts, sources=sources)
        seed = next((e for e in self._alts_cache.get((prompt, model, max_alts, sources),
                    (None, [], None))[1] if e.get('is_greedy')), None)
        if seed is None:
            return {**baseline, 'view': 'greedy_i', 'entries': [], 'count': 0}
        accepted = [{**seed, 'discovery_rank': 1, 'cost': 0.0}]
        seen_paths = {tuple(t['token'] for t in seed['tokens'])}
        seen_departures = set()
        queue = []
        serial = itertools.count()
        missing = 0

        def expand(parent):
            nonlocal missing
            path = [t['token'] for t in parent['tokens']]
            node = root
            prefix_sum = 0.0
            prefix_text = ''
            for i, tok in enumerate(path):
                for alt, lp in (node.top_logprobs or {}).items():
                    if alt == tok or not isinstance(lp, (int, float)) or not math.isfinite(lp):
                        continue
                    departure_key = (id(node), alt)
                    if departure_key in seen_departures:
                        continue
                    seen_departures.add(departure_key)
                    rec = self.lookup(prefix_text + alt, model=model, max_tokens=None, sources=sources)
                    if rec is None:
                        missing += 1
                        continue
                    choice = (rec.get('choices') or [{}])[0]
                    full = tuple(path[:i] + [alt]) + tuple((choice.get('logprobs') or {}).get('tokens') or [])
                    if full in seen_paths:
                        continue
                    score = prefix_sum + lp
                    heapq.heappush(queue, (-score, next(serial), full, rec,
                        {'position': i, 'original': tok, 'alternative': alt,
                         'original_logprob': parent['tokens'][i]['logprob'],
                         'alternative_logprob': lp, 'parent_rank': parent['discovery_rank']}))
                prefix_text += tok
                prefix_sum += parent['tokens'][i]['logprob']
                node = node.children[tok]

        def visible(e):
            return (not ends or e.get('end') in ends) and (not nodes or e.get('node_kind') in nodes)

        count = int(visible(accepted[0]))
        expand(accepted[0])
        while queue and (not top or count < top):
            cost, _, path, rec, deviation = heapq.heappop(queue)
            if path in seen_paths:
                continue
            detail = walk_detail(root, list(path), max_alts)
            if len(detail) != len(path) or not detail:
                continue
            seen_paths.add(path)
            total = detail[-1]['cumulative']
            end = (rec.get('choices') or [{}])[0].get('finish_reason') or 'length'
            entry = {'n': len(path), 'text': ''.join(path), 'tokens': detail,
                     'sum_logprob': total, 'mean_logprob': total / len(path),
                     'perplexity': ppl_from_mean(total / len(path)), 'cost': cost,
                     'departure_logprob': -cost, 'deviation': deviation,
                     'is_greedy': False, 'id': rec.get('id'), 'end': end,
                     'finish_reason': end, 'node_kind': self.node_kind(list(path), root),
                     'discovery_rank': len(accepted) + 1}
            accepted.append(entry)
            count += int(visible(entry))
            expand(entry)
        entries = [e for e in accepted if visible(e)]
        if sort != 'cost':
            entries.sort(key=(self.ALT_SORTS.get(sort) or self.ALT_SORTS['sum'])[0])
        entries = [{**e, 'rank': i} for i, e in enumerate(entries, 1)]
        distinct = self.add_extensions(entries, root, max_alts) if extend else None
        return {**baseline, 'view': 'greedy_i', 'sort': sort,
                'ranking': 'discovery order; maximize departure prefix sum_logprob',
                'entries': entries, 'count': len(entries), 'available': None,
                'departures_ranked': len(accepted) - 1, 'departures_without_record': missing,
                'extend': bool(extend), 'distinct_extended': distinct,
                'ends': sorted(ends) if ends else None, 'nodes': sorted(nodes) if nodes else None,
                'end_counts': dict(Counter(e['end'] for e in entries)),
                'node_counts': dict(Counter(e['node_kind'] for e in entries))}

    # cost first, and it is the default: the greedy criterion is what makes a
    # sibling "second best", and the other keys answer a different question.
    ALT_SORTS = {
        'cost': (lambda e: (e['cost'], -e['sum_logprob']), 'deviation cost asc (the greedy criterion)'),
        'sum': (lambda e: -e['sum_logprob'], 'sum_logprob desc'),
        'ppl': (lambda e: _ppl_key(e), 'perplexity asc (= mean_logprob desc)'),
        'mean': (lambda e: _ppl_key(e), 'perplexity asc (= mean_logprob desc)'),
        'length': (lambda e: (-e['n'], -e['sum_logprob']), 'length desc'),
    }

    def _rank_alts(self, prompt, model, sort, top, greedy, candidates, missing,
                   ends=None, extend=False, root_for_extend=None, max_alts=8,
                   nodes=None, sources=None) -> Dict:
        key, label = self.ALT_SORTS.get(sort) or self.ALT_SORTS['cost']
        entries = sorted(candidates, key=key)
        # After the sort and before the cap, as everywhere else: filtering the
        # capped list would answer "the cheapest finished departure among the N
        # cheapest", which is not the question the checkbox asks.
        if ends:
            entries = [e for e in entries if e.get('end') in ends]
        if nodes:
            entries = [e for e in entries if e.get('node_kind') in nodes]
        # What the cap was applied to, so the page can say "20 of N" here as well
        # as in the two rankings.
        available = len(entries)
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
            'source_records': len(self.records_for(sources)),
            'sources': sorted(sources) if sources else None,
            'generated_from': ', '.join(self.sources),
            'greedy': {k: greedy[k] for k in
                       ('n', 'sum_logprob', 'mean_logprob', 'first_id', 'finish_reason')},
            # Every departure whose continuation is not on record; each would cost
            # one call. Most of these are positions no sweep ever covered, so the
            # number is large by design and not a sign of missing data.
            'departures_without_record': missing,
            'departures_ranked': len(candidates) - 1,
            'available': available,
            'ends': sorted(ends) if ends else None,
            'end_counts': dict(Counter(e.get('end') for e in entries)),
            'nodes': sorted(nodes) if nodes else None,
            'node_counts': dict(Counter(e.get('node_kind') for e in entries)),
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
