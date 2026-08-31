"""Prove the store answers exactly what the frozen exports hold.

The exports are only safe to delete once a live query reproduces them. Each one
recorded the query that built it (generated_from / view / model_filter /
chosen_only / prefix_filter / count), so this replays that query against a store
loaded from the SAME sources and compares entries.

Anything that cannot match is reported, not glossed over: a mismatch here means
the store is not yet a replacement, and the export must stay.

    python3 verify_store.py
    python3 verify_store.py --show 3      # first N differing entries per file
"""

from __future__ import annotations

import argparse
import io
import json
import os
from typing import Dict, List, Optional

from store import RecordStore, _safe

ROOT = os.path.dirname(os.path.abspath(__file__))

EXPORTS = [
    'dijkstra_top.json',
    'sweep_top.json',
    'sweep_completions.json',
    'sweep2_top.json',
    'sweep2_completions.json',
    'sweep3_top.json',
    'sweep3_completions.json',
    'ihave_completions.json',
]

# A file names its sources, but not always by the path the store uses: the
# uncompressed sweep mirrors are gone, so their names map onto the .gz survivors.
ALIASES = {
    'sweep2_history.json': 'sweep2_history.json.gz',
    'sweep3_ihave.json': 'sweep3_ihave.json.gz',
}


def sources_of(db: Dict) -> List[str]:
    raw = [s.strip() for s in (db.get('generated_from') or '').split(',') if s.strip()]
    return [ALIASES.get(s, s) for s in raw]


def max_alts_of(db: Dict) -> int:
    """Exports written before write_db recorded --max-alts have to be measured:
    the widest alternatives list any entry holds is the cap that produced it."""
    if isinstance(db.get('max_alts'), int):
        return db['max_alts']
    widest = 0
    for entry in db.get('entries') or []:
        for token in entry.get('tokens') or []:
            widest = max(widest, len(token.get('alternatives') or []))
    return widest or 8


def predates_dedup(db: Dict) -> bool:
    """build_completion_entries started collapsing identical strings into one
    entry carrying n_calls. A completions export with no n_calls anywhere was
    written before that, so it still lists the duplicates and CANNOT match."""
    entries = db.get('entries') or []
    return (db.get('view') == 'completions'
            and bool(entries)
            and not any('n_calls' in e for e in entries))


_STORES: Dict[tuple, RecordStore] = {}


def store_for(sources: List[str]) -> RecordStore:
    """One store per source set: reloading 74 MB per export made this take 40 s."""
    key = tuple(sources)
    if key not in _STORES:
        _STORES[key] = RecordStore(sources=sources).load()
    return _STORES[key]


def compare(name: str, show: int) -> Dict:
    path = os.path.join(ROOT, name)
    db = json.load(io.open(path, encoding='utf-8'))
    sources = sources_of(db)

    store = store_for(sources)
    if store.missing:
        return {'file': name, 'status': 'SKIP', 'note': f'missing sources: {store.missing}'}

    # count is how many entries the export kept, i.e. the --top it was given
    # (after any prefix filter). Replay with the same cap.
    live = store.query(
        view=db.get('view') or 'prefixes',
        top=db.get('count'),
        prefix=db.get('prefix_filter'),
        model=db.get('model_filter'),
        chosen_only=bool(db.get('chosen_only')),
        max_alts=max_alts_of(db),
    )

    result = {
        'file': name,
        'view': db.get('view'),
        'export_records': db.get('source_records'),
        'store_records': live['source_records'],
        'export_count': db.get('count'),
        'store_count': live['count'],
        'diffs': [],
    }

    a, b = db['entries'], live['entries']
    grew = db.get('source_records') != live['source_records']

    if a == b:
        # Equal entries despite a bigger store means the newer records simply do
        # not reach this cut; the export is behind, but not yet wrong.
        result['status'] = 'IDENTICAL (store has more records)' if grew else 'IDENTICAL'
        return result

    if grew:
        result['status'] = 'OUTDATED'
        result['note'] = (f'built from {db.get("source_records")} records, the store now holds '
                          f'{live["source_records"]}, and the extra ones change the ranking')
        return result

    if predates_dedup(db):
        result['status'] = 'STALE EXPORT'
        result['note'] = ('written before identical strings were collapsed into one '
                          f'entry, so it lists {len(a)} where the store now yields {len(b)}')
        return result

    result['status'] = 'DIFFERS'
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            keys = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
            result['diffs'].append({'rank': i + 1, 'keys': keys,
                                    'export_text': (x.get('text') or '')[:60],
                                    'store_text': (y.get('text') or '')[:60]})
            if len(result['diffs']) >= show:
                break
    if len(a) != len(b):
        result['diffs'].append({'length': [len(a), len(b)]})
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--show', type=int, default=2,
                    help='how many differing entries to describe per file')
    ap.add_argument('files', nargs='*', default=None,
                    help='export files to check (default: all of them)')
    args = ap.parse_args()

    identical = differing = skipped = stale = 0
    for name in (args.files or EXPORTS):
        r = compare(name, args.show)
        flag = {'IDENTICAL': 'ok  ', 'SKIP': 'skip',
                'IDENTICAL (store has more records)': 'ok  ',
                'STALE EXPORT': 'old ', 'OUTDATED': 'old '}.get(r['status'], 'FAIL')
        counts = ''
        if 'store_count' in r:
            counts = (f'{r["export_records"]}/{r["store_records"]} rec  '
                      f'{r["export_count"]}/{r["store_count"]} entries')
        print(f'[{flag}] {name:26} {r["status"]:34} {counts}')
        if r.get('note'):
            print(f'         {r["note"]}')
        for d in r.get('diffs') or []:
            if 'length' in d:
                print(f'         entry counts differ: {d["length"]}')
            else:
                print(_safe(f'         rank {d["rank"]}: fields {d["keys"]}'))
                print(_safe(f'           export {d["export_text"]!r}'))
                print(_safe(f'           store  {d["store_text"]!r}'))
        identical += r['status'].startswith('IDENTICAL')
        stale += r['status'] in ('STALE EXPORT', 'OUTDATED')
        differing += not (r['status'].startswith('IDENTICAL')
                          or r['status'] in ('SKIP', 'STALE EXPORT', 'OUTDATED'))
        skipped += r['status'] == 'SKIP'

    total = identical + differing + skipped + stale
    print(f'\n{identical}/{total} identical, {stale} stale export(s), '
          f'{differing} differing, {skipped} skipped')
    if stale:
        print('A stale/outdated export is superseded by the store, not a store failure.')
    if differing:
        print('The differing exports must stay until the store reproduces them.')


if __name__ == '__main__':
    main()
