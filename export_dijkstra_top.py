"""Export the top-N prefixes by SUM of logprobs into a standalone database file.

Ranking comes from best_sum_logprob_list.rank_by_sum() — the best-first /
Dijkstra-style search over the token trie (correct because every token's
logprob is <= 0, so extending a prefix can only lower its sum).

Unlike the console script, this writes a self-contained JSON file that carries
per-token detail for each ranked prefix, so a viewer can show where the
probability mass actually went without re-reading the full history export:

    {
      "generated_from": "completion_history.json",
      "model_filter": null,
      "chosen_only": false,
      "count": 20,
      "entries": [
        {
          "rank": 1,
          "n": 1,
          "sum_logprob": -2.243145,
          "mean_logprob": -2.243145,
          "perplexity": 9.42...,
          "text": "\n",
          "tokens": [
            {"token": "\n", "logprob": -2.243145, "cumulative": -2.243145,
             "prob": 0.106..., "source": "chosen",
             "alternatives": [{"token": "\n\n", "logprob": -2.631}, ...]}
          ]
        }, ...
      ]
    }

`source` is "chosen" when some API call actually emitted that token, or "alt"
when its logprob was recovered from the parent's top_logprobs table (a token
picked as a tooltip replacement, which the app never persisted a logprob for).

The viewer (dijkstra.html) chooses how many of these entries to display, but it
can only ever show what this file holds — so export a generous pool.

Usage:
    python3 export_dijkstra_top.py completion_history.json
    python3 export_dijkstra_top.py completion_history.json --top 500
    python3 export_dijkstra_top.py completion_history.json --model davinci-002
"""

import argparse
import gzip
import json
import math
from typing import Dict, List, Optional


def open_text(path):
    """Transparently read .gz — the raw sweep histories are tens of MB, so they
    are stored compressed."""
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, encoding='utf-8')

from best_avg_logprob_path import TrieNode, build_trie
from best_sum_logprob_list import rank_by_sum


def walk_detail(root: TrieNode, tokens: List[str], max_alts: int = 8) -> List[Dict]:
    """Re-walk the trie along `tokens`, collecting per-token detail."""
    out: List[Dict] = []
    node = root
    cumulative = 0.0
    for tok in tokens:
        parent = node
        child = node.children.get(tok)
        if child is None:  # shouldn't happen: the path came from this same trie
            break
        if child.logprob is None:
            # build_trie fills a node's logprob only from a generated token or
            # from its parent's top_logprobs, so a prompt token that no call ever
            # generated stays unscored. Stop here: the caller compares len(detail)
            # against len(tokens) and skips a string it cannot score. rank_by_sum
            # already refuses these via _usable(); this path did not, and a record
            # with a custom prompt (the APE meta prompt) made it raise TypeError.
            break
        cumulative += child.logprob
        alts = []
        if parent.top_logprobs:
            ranked = sorted(parent.top_logprobs.items(), key=lambda kv: -kv[1])
            alts = [{"token": t, "logprob": lp} for t, lp in ranked[:max_alts]]
        out.append({
            "token": tok,
            "logprob": child.logprob,
            "cumulative": cumulative,
            "prob": math.exp(child.logprob),
            "source": child.source,
            "alternatives": alts,
        })
        node = child
    return out


def build_completion_entries(root, records, sweep_by_prompt, args) -> List[Dict]:
    """One entry per recorded call: the whole string that call produced.

    A "completion" here is prompt + generated tokens, i.e. exactly what the
    call returned in context — not a prefix of the trie. Note the strings are
    NOT all the same length: the generated part is max_tokens, but the prompt
    grows with the deviation position, so a position-0 branch is ~21 tokens
    and a position-19 branch ~40. Ranking by cumulative logprob therefore
    favours the shorter ones; mean_logprob is reported alongside so that the
    length effect stays visible.
    """
    entries = []
    by_text: Dict[str, Dict] = {}
    for rec in records:
        lp = (rec.get("choices") or [{}])[0].get("logprobs") or {}
        own = lp.get("tokens") or []
        prompt_tokens = ((rec.get("prompt") or {}).get("logprobs") or {}).get("tokens") or []
        tokens = list(prompt_tokens) + list(own)
        if not tokens:
            continue
        detail = walk_detail(root, tokens, args.max_alts)
        if len(detail) != len(tokens):
            # a token on this path has no known logprob, so the string has no
            # comparable score — skip rather than report a truncated sum
            continue
        total = detail[-1]["cumulative"]
        n = len(detail)
        text = "".join(tokens)
        # Different calls routinely produce the identical string: an on-path
        # deviation at position i yields base[:i] + base[i] + the greedy
        # continuation, which at temperature 0 is the same path again, so all
        # on-path sweeps of one base converge — the more so when that base
        # stopped early. Listing the same string many times is noise, so keep
        # one entry and count how many calls produced it.
        prev = by_text.get(text)
        if prev is not None:
            prev["n_calls"] += 1
            continue
        entry = {
            "n": n,
            "sum_logprob": total,
            "mean_logprob": total / n,
            "perplexity": math.exp(-total / n),
            "text": "".join(tokens),
            "tokens": detail,
            "prompt_len": len(prompt_tokens),
            "generated_len": len(own),
            "finish_reason": (rec.get("choices") or [{}])[0].get("finish_reason"),
            # Same name in every view, so one column can show it and one filter
            # can select on it. A completion is a recorded call, so its end is
            # always the API's verdict -- "open" can only happen to a prefix.
            "end": (rec.get("choices") or [{}])[0].get("finish_reason") or "length",
            "id": rec.get("id"),
            "n_calls": 1,
            **({"sweep": (rec.get("meta") or {}).get("sweep")}
               if (rec.get("meta") or {}).get("sweep") else {}),
        }
        by_text[text] = entry
        entries.append(entry)

    entries.sort(key=lambda e: -e["sum_logprob"])
    entries = apply_prefix(entries, args.prefix)
    if args.top:
        entries = entries[:args.top]
    for i, e in enumerate(entries, 1):
        e["rank"] = i
    return entries


def load_records(paths: List[str]) -> List[Dict]:
    """Read one or more record files, keeping the first copy of each id.

    Extracted from main() so a long-lived store can hold the same records the
    exporter would have read, rather than re-deriving them a second way.
    """
    records: List[Dict] = []
    seen_ids = set()
    for path in paths:
        with open_text(path) as fh:
            data = json.load(fh)
        for r in (data if isinstance(data, list) else [data]):
            rid = r.get("id")
            if rid and rid in seen_ids:      # same record in two files — keep one
                continue
            if rid:
                seen_ids.add(rid)
            records.append(r)
    return records


def index_sweeps(records: List[Dict]) -> Dict[tuple, Dict]:
    """Which (position, alternative) each call deviated at, keyed by prompt path,
    so a ranked prefix can be traced back to the branch it came from."""
    sweep_by_prompt: Dict[tuple, Dict] = {}
    for r in records:
        sw = (r.get("meta") or {}).get("sweep")
        if sw:
            key = tuple(((r.get("prompt") or {}).get("logprobs") or {}).get("tokens") or [])
            sweep_by_prompt[key] = sw
    return sweep_by_prompt


# A filtered best-first search can walk most of the trie before it finds enough
# qualifying strings, so it is bounded. 1M pops covers the whole trie today
# (exhausting it for the strictest filter takes well under that); the bound
# exists so a future, larger trie degrades into "incomplete list, and it says so"
# rather than into a hung request.
MAX_POPS = 1_000_000


def build_prefix_entries(root, sweep_by_prompt, args, ends_index=None,
                         stats=None) -> List[Dict]:
    """Best-first ranking over the trie, one entry per prefix.

    Extracted from main() unchanged so the exporter and the store cannot drift:
    both call this, so a ranking served live is the ranking a file would hold.

    `ends_index` maps a full token path to how the call that produced it ended,
    which is what lets a prefix say whether it is a real endpoint or just a place
    the enumeration stopped. Without it every prefix reads as "open", which is
    the honest answer when nothing recorded ends there.
    """
    ends_index = ends_index or {}
    allow = getattr(args, "ends", None)
    kinds = getattr(args, "nodes", None)
    accept = None
    if allow or kinds:
        def accept(toks, node):
            if allow and ends_index.get(toks, "open") not in allow:
                return False
            # "Do we have a continuation?" is a property of the NODE, not of how
            # the string ended: 494 length-ended paths were later continued past
            # by another call, so unticking `open` alone still lists them.
            if kinds and ("leaf" if not node.children else "continues") not in kinds:
                return False
            return True
    ranked = rank_by_sum(root, args.top, args.min_n, args.chosen_only,
                         accept=accept, max_pops=MAX_POPS, stats=stats)

    entries = []
    for i, r in enumerate(ranked, 1):
        detail = walk_detail(root, r["tokens"], args.max_alts)
        entry = {
            "rank": i,
            "n": r["n"],
            "sum_logprob": r["sum"],
            "mean_logprob": r["mean"],
            "perplexity": math.exp(-r["mean"]),
            "text": "".join(r["tokens"]),
            "tokens": detail,
            "end": ends_index.get(tuple(r["tokens"]), "open"),
            "node_kind": r.get("node_kind"),
        }
        # deepest prompt path that is a prefix of this entry and was itself a
        # sweep call — that is the branch point this prefix descends from
        toks = tuple(r["tokens"])
        for cut in range(len(toks), 0, -1):
            sw = sweep_by_prompt.get(toks[:cut])
            if sw:
                entry["sweep"] = sw
                break
        entries.append(entry)

    entries = apply_prefix(entries, args.prefix)
    for i, e in enumerate(entries, 1):
        e["rank"] = i
    return entries


def apply_prefix(entries: List[Dict], prefix: Optional[str]) -> List[Dict]:
    """Filter by leading text. Applied before the --top cap, so a subtree that
    ranks far down the list is still fully represented."""
    if not prefix:
        return entries
    return [e for e in entries
            if e["text"].startswith(prefix) or e["text"].lstrip().startswith(prefix)]


def write_db(entries, records, args, ranking, view) -> None:
    db = {
        "generated_from": ", ".join(args.file) if isinstance(args.file, list) else args.file,
        "model_filter": args.model,
        "chosen_only": args.chosen_only,
        "source_records": len(records),
        "ranking": ranking,
        "view": view,
        "prefix_filter": args.prefix,
        # Recorded because it changes every entry's token detail: without it,
        # reproducing an export means guessing the width from its own contents.
        "max_alts": args.max_alts,
        "count": len(entries),
        "entries": entries,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}: {len(entries)} entries from {len(records)} records [{view}]")
    for e in entries[:20]:
        print(f"  #{e['rank']:<3} n={e['n']:<4} sum={e['sum_logprob']:9.4f}  {e['text'][:60]!r}")
    if len(entries) > 20:
        print(f"  ... and {len(entries) - 20} more (see {args.out})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", nargs="+",
                    help="one or more record files. Pass several to combine levels: a "
                         "level-2 sweep has no empty-prompt record, so on its own the "
                         "first token has no logprob and the trie comes out headless.")
    ap.add_argument("-o", "--out", default="dijkstra_top.json", help="output database file")
    ap.add_argument("--top", type=int, default=200,
                    help="how many ranked prefixes to export. Export generously: the viewer "
                         "picks how many of these to show, and can only ever show what the "
                         "file holds, so a bigger pool means no re-export to see further down.")
    ap.add_argument("--min-n", type=int, default=1, help="skip prefixes shorter than this")
    ap.add_argument("--model", help="substring filter on model, e.g. davinci-002")
    ap.add_argument("--chosen-only", action="store_true",
                    help="ignore top_logprobs-recovered alternative tokens")
    ap.add_argument("--max-alts", type=int, default=8,
                    help="how many sibling alternatives to record per token")
    ap.add_argument("--prefix",
                    help="keep only strings starting with this text (leading whitespace is "
                         "ignored, as most strings begin with a newline). Use it to build a "
                         "focused dataset for one subtree: the --top cap is applied AFTER "
                         "filtering, so matches deep in the ranking still come through.")
    ap.add_argument("--mode", choices=["prefixes", "completions"], default="prefixes",
                    help="prefixes: rank every prefix in the trie (best-first order). "
                         "completions: one entry per recorded call — the whole string it "
                         "produced (prompt + completion) — ranked by cumulative logprob.")
    args = ap.parse_args()

    records = load_records(args.file)
    sweep_by_prompt = index_sweeps(records)
    root = build_trie(records, args.model)

    if args.mode == "completions":
        entries = build_completion_entries(root, records, sweep_by_prompt, args)
        write_db(entries, records, args,
                 ranking="sum_logprob desc over whole strings (prompt + completion)",
                 view="completions")
        return

    entries = build_prefix_entries(root, sweep_by_prompt, args)

    write_db(entries, records, args,
             ranking="sum_logprob desc (best-first / Dijkstra over token trie)",
             view="prefixes")


if __name__ == "__main__":
    main()
