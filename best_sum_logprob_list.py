"""Rank prefixes (from the very root of the tree) by SUM of logprobs, best
(closest to zero, i.e. lowest negative-log-likelihood / "loss") first —
across a completion_history.json export.

Every token's logprob is <= 0, so extending any string by one more token can
only keep its sum the same or push it lower — never higher. That monotone
non-increasing property is exactly what makes a best-first / Dijkstra-style
search correct here: seed a max-heap with every root child (all length-1
candidates), repeatedly pop the current best, emit it, and push its own
children (one token longer, necessarily <= its parent's sum). The pop order
is then guaranteed globally correct without ever visiting the whole tree —
same reasoning as Dijkstra's algorithm with non-negative edge weights (here:
non-positive "gains", equivalently non-negative "losses" of -logprob).

Concretely: item #1 is always some length-1 string (the single most likely
first token anywhere in the tree). Item #2 is then whichever is better of
(a) the next-best length-1 string, or (b) item #1 extended by one token —
they compete on equal footing, exactly as either could come next.

Reuses build_trie() from best_avg_logprob_path.py, including its recovery of
logprobs for tokens that were only ever seen as a top_logprobs alternative
(picked as a tooltip replacement) rather than as someone's chosen output.

Usage:
    python3 best_sum_logprob_list.py completion_history.json --top 20
    python3 best_sum_logprob_list.py completion_history.json --model davinci-002 --top 10
    python3 best_sum_logprob_list.py completion_history.json --min-n 2 --top 10
    python3 best_sum_logprob_list.py completion_history.json --chosen-only --top 10
"""

import argparse
import heapq
import json
from typing import Dict, List, Optional, Tuple

from best_avg_logprob_path import TrieNode, build_trie


def _usable(node: TrieNode, chosen_only: bool) -> bool:
    if node.logprob is None:
        return False
    if chosen_only and node.source != "chosen":
        return False
    return True


def rank_by_sum(root: TrieNode, top: int, min_n: int = 1, chosen_only: bool = False) -> List[Dict]:
    heap: List[Tuple[float, int, TrieNode, Tuple[str, ...]]] = []
    counter = 0  # tie-breaker so heapq never has to compare TrieNode/tuples directly

    def push(node: TrieNode, tokens: Tuple[str, ...], cum_sum: float):
        nonlocal counter
        counter += 1
        heapq.heappush(heap, (-cum_sum, counter, node, tokens))

    for tok, child in root.children.items():
        if _usable(child, chosen_only):
            push(child, (child.token,), child.logprob)

    results: List[Dict] = []
    while heap and len(results) < top:
        neg_sum, _, node, tokens = heapq.heappop(heap)
        cum_sum = -neg_sum
        n = len(tokens)

        if n >= min_n:
            results.append({"tokens": list(tokens), "sum": cum_sum, "n": n, "mean": cum_sum / n})

        for tok, child in node.children.items():
            if _usable(child, chosen_only):
                push(child, tokens + (child.token,), cum_sum + child.logprob)

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", help="completion_history.json (array) or a single response object")
    ap.add_argument("--model", help="substring filter on model, e.g. davinci-002")
    ap.add_argument("--min-n", type=int, default=1, help="skip prefixes shorter than this")
    ap.add_argument("--top", type=int, default=10, help="how many ranked entries to list")
    ap.add_argument("--chosen-only", action="store_true",
                     help="ignore top_logprobs-recovered alternative tokens; only count tokens "
                          "that were actually the chosen output of some API call")
    args = ap.parse_args()

    data = json.load(open(args.file, encoding="utf-8"))
    records = data if isinstance(data, list) else [data]

    root = build_trie(records, args.model)
    results = rank_by_sum(root, args.top, args.min_n, args.chosen_only)

    if not results:
        raise SystemExit("No scored prefixes found (empty file, or --model matched nothing).")

    for i, r in enumerate(results, 1):
        print(f"#{i:<3} n={r['n']:<4} sum logP={r['sum']:9.4f}  mean logP={r['mean']:8.4f}  "
              f"text={''.join(r['tokens'])!r}")


if __name__ == "__main__":
    main()
