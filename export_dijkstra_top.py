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
import json
import math
from typing import Dict, List, Optional

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", help="completion_history.json (array) or a single response object")
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
    args = ap.parse_args()

    data = json.load(open(args.file, encoding="utf-8"))
    records = data if isinstance(data, list) else [data]

    root = build_trie(records, args.model)
    ranked = rank_by_sum(root, args.top, args.min_n, args.chosen_only)

    entries = []
    for i, r in enumerate(ranked, 1):
        detail = walk_detail(root, r["tokens"], args.max_alts)
        entries.append({
            "rank": i,
            "n": r["n"],
            "sum_logprob": r["sum"],
            "mean_logprob": r["mean"],
            "perplexity": math.exp(-r["mean"]),
            "text": "".join(r["tokens"]),
            "tokens": detail,
        })

    db = {
        "generated_from": args.file,
        "model_filter": args.model,
        "chosen_only": args.chosen_only,
        "source_records": len(records),
        "ranking": "sum_logprob desc (best-first / Dijkstra over token trie)",
        "count": len(entries),
        "entries": entries,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print(f"wrote {args.out}: {len(entries)} entries from {len(records)} records")
    for e in entries[:20]:
        print(f"  #{e['rank']:<3} n={e['n']:<4} sum={e['sum_logprob']:9.4f}  {e['text']!r}")
    if len(entries) > 20:
        print(f"  ... and {len(entries) - 20} more (see {args.out})")


if __name__ == "__main__":
    main()
