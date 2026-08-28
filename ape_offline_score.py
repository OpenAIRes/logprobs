"""APE-style log-likelihood scoring, offline, on saved completion objects.

APE (Zhou et al. 2022) scores a prompt by log P(answer | instruction, input).
It obtains that with echo=True + logprobs, then slices the returned token
array down to the character range covering the answer, using text_offset.

get_token_indices below is a verbatim port of the function in
automatic_prompt_engineer/llm.py. The rest replaces the API call with a
lookup into a saved response, so the scoring can be exercised with no
network and no live model.
"""

import argparse
import json
from typing import Dict, List, Sequence, Tuple


def get_token_indices(offsets: Sequence[int],
                      log_prob_range: Tuple[int, int]) -> Tuple[int, int]:
    """Verbatim from APE: llm.py::GPT_Forward.get_token_indices."""
    lower_index = 0
    for i in range(len(offsets)):
        if offsets[i] <= log_prob_range[0]:
            lower_index = i
        else:
            break

    upper_index = len(offsets)
    for i in range(len(offsets)):
        if offsets[i] >= log_prob_range[1]:
            upper_index = i
            break

    return lower_index, upper_index


def load_choice(record: Dict) -> Dict:
    """Pull the legacy logprobs arrays out of a saved completion object."""
    lp = record["choices"][0]["logprobs"]
    return {
        "tokens": lp["tokens"],
        "logprobs": lp["token_logprobs"],
        "offsets": lp["text_offset"],
        "text": record["choices"][0]["text"],
        "prompt_text": record.get("prompt_text", ""),
    }


def score_range(ch: Dict, lo: int, hi: int) -> Dict:
    """Score the character range [lo, hi) the way APE scores an answer."""
    li, ui = get_token_indices(ch["offsets"], (lo, hi))
    toks = ch["tokens"][li:ui]
    lps = [p for p in ch["logprobs"][li:ui] if p is not None]
    total = sum(lps)
    return {
        "token_span": (li, ui),
        "tokens": toks,
        "sum_logprob": total,
        "mean_logprob": total / len(lps) if lps else float("nan"),
        "perplexity": pow(2.718281828459045, -total / len(lps)) if lps else float("nan"),
        "n_tokens": len(lps),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", help="response json or completion_history list")
    ap.add_argument("--record", type=int, default=0)
    ap.add_argument("--substring", help="score the span of this substring")
    ap.add_argument("--range", nargs=2, type=int, metavar=("LO", "HI"))
    args = ap.parse_args()

    data = json.load(open(args.file, encoding="utf-8"))
    rec = data[args.record] if isinstance(data, list) else data
    ch = load_choice(rec)

    # text_offset is authoritative for where the completion starts in the
    # string the server saw. prompt_text is NOT: when the request carried no
    # prompt the API substitutes "<|endoftext|>" (13 chars, 1 token), so
    # prompt_text == "" while offsets[0] == 13. Trusting prompt_text here
    # shifts every span by 13 characters.
    base = ch["offsets"][0]
    if base != len(ch["prompt_text"]):
        print(f"note: prompt_text is {len(ch['prompt_text'])} chars but the "
              f"completion starts at offset {base}; using the offset.")

    if args.substring:
        pos = ch["text"].find(args.substring)
        if pos < 0:
            raise SystemExit(f"substring not found in: {ch['text'][:120]!r}")
        lo, hi = base + pos, base + pos + len(args.substring)
    elif args.range:
        lo, hi = args.range
    else:
        # default: score the completion, i.e. everything after the prompt
        lo, hi = ch["offsets"][0], ch["offsets"][-1] + len(ch["tokens"][-1])

    r = score_range(ch, lo, hi)
    print(f"model      : {rec.get('model')}")
    print(f"prompt_text: {ch['prompt_text']!r}  (offset base {base})")
    print(f"range      : [{lo}, {hi})  -> tokens [{r['token_span'][0]}:{r['token_span'][1]}]")
    print(f"tokens     : {r['tokens']}")
    print(f"sum logP   : {r['sum_logprob']:.4f}   over {r['n_tokens']} tokens")
    print(f"mean logP  : {r['mean_logprob']:.4f}")
    print(f"perplexity : {r['perplexity']:.3f}")


if __name__ == "__main__":
    main()
