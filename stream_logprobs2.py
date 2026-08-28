"""Stream tokens+logprobs from OpenAI responses, legacy or chat format.

Legacy (`object: text_completion`) stores parallel arrays:
    choices[0].logprobs = {tokens[], token_logprobs[], top_logprobs[], text_offset[]}
Chat (`object: chat.completion`) stores a list of objects:
    choices[0].logprobs.content = [{token, logprob, top_logprobs:[{token,logprob}]}]

Both are normalised to the same record:
    {token, logprob, offset, top: {tok: logprob}, source}

Usage:
    python3 stream_logprobs2.py logprobs.json --limit 5
    python3 stream_logprobs2.py 'completion_history (33).json' --record 3
    python3 stream_logprobs2.py logprobs.json --format tsv
"""

import argparse
import json
from typing import Dict, Iterator, List, Optional


def _norm_legacy(lp: Dict) -> List[Dict]:
    toks = lp.get("tokens") or []
    lps = lp.get("token_logprobs") or []
    tops = lp.get("top_logprobs") or []
    offs = lp.get("text_offset") or []
    out = []
    for i, t in enumerate(toks):
        out.append({
            "token": t,
            "logprob": lps[i] if i < len(lps) else None,
            "offset": offs[i] if i < len(offs) else None,
            "top": dict(tops[i]) if i < len(tops) and tops[i] else {},
            "source": "legacy",
        })
    return out


def _norm_chat(content: List[Dict]) -> List[Dict]:
    out = []
    cursor = 0
    for c in content:
        tok = c.get("token", "")
        out.append({
            "token": tok,
            "logprob": c.get("logprob"),
            # chat format carries no text_offset; reconstruct it
            "offset": cursor,
            "top": {d["token"]: d["logprob"] for d in (c.get("top_logprobs") or [])},
            "source": "chat",
        })
        cursor += len(tok)
    return out


def tokens_from_choice(choice: Dict) -> List[Dict]:
    lp = choice.get("logprobs")
    if not lp:
        return []
    if isinstance(lp, dict) and "content" in lp and lp["content"] is not None:
        return _norm_chat(lp["content"])
    if isinstance(lp, dict) and "tokens" in lp:
        return _norm_legacy(lp)
    return []


def iter_tokens(path: str, record: Optional[int] = None) -> Iterator[Dict]:
    """Yield normalised token dicts from a response file.

    Handles a single response object or a list of them (completion_history).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data if isinstance(data, list) else [data]
    if record is not None:
        records = [records[record]]

    for ri, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        for choice in rec.get("choices") or []:
            for tok in tokens_from_choice(choice):
                tok["record"] = ri if record is None else record
                tok["id"] = rec.get("id")
                yield tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file")
    ap.add_argument("--limit", type=int, help="print only the first N tokens")
    ap.add_argument("--record", type=int, help="index into a history list")
    ap.add_argument("--format", choices=["repr", "tsv", "json"], default="repr")
    ap.add_argument("--top", type=int, default=0, help="also show N alternatives")
    args = ap.parse_args()

    for i, tok in enumerate(iter_tokens(args.file, args.record)):
        if args.format == "json":
            print(json.dumps(tok, ensure_ascii=False))
        elif args.format == "tsv":
            lp = "" if tok["logprob"] is None else f"{tok['logprob']:.4f}"
            print(f"{tok['offset']}\t{lp}\t{tok['token']!r}")
        else:
            lp = "None" if tok["logprob"] is None else f"{tok['logprob']:.3f}"
            print(f"[{tok['offset']:>5}] {lp:>9}  {tok['token']!r}")
            if args.top:
                alts = sorted(tok["top"].items(), key=lambda z: -z[1])[:args.top]
                for t, v in alts:
                    print(f"          {v:9.3f}    {t!r}")
        if args.limit and i + 1 >= args.limit:
            break


if __name__ == "__main__":
    main()
