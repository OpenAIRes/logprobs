#!/usr/bin/env python3
"""Simple parser for OpenAI logprobs JSON output.

Usage::
    python parse_logprobs.py [path] [--top N]

It prints each token with its log probability and N alternative tokens.
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print tokens and log probabilities from OpenAI logprobs output")
    parser.add_argument("path", nargs="?", default="logprobs.json", help="Path to the JSON file")
    parser.add_argument("--top", type=int, default=3, help="Number of alternative tokens to show")
    args = parser.parse_args()

    data = json.loads(Path(args.path).read_text())
    content = data["choices"][0]["logprobs"]["content"]

    for idx, entry in enumerate(content):
        print(f"{idx}: {entry['token']}\t(logprob {entry['logprob']:.4f})")
        for alt in entry["top_logprobs"][: args.top]:
            print(f"    {alt['token']}\t{alt['logprob']:.4f}")
        print()


if __name__ == "__main__":
    main()
