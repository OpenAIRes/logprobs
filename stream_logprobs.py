import json
import argparse
from typing import Iterator, Dict

def iter_tokens(path: str) -> Iterator[Dict]:
    """Yield token objects from a potentially large logprobs.json file."""
    decoder = json.JSONDecoder()
    with open(path, 'r', encoding='utf-8') as f:
        buf = ''
        # Find the beginning of the tokens array
        while True:
            chunk = f.read(8192)
            if not chunk:
                return
            buf += chunk
            idx = buf.find('"content"')
            if idx != -1:
                # Seek to the '[' that opens the token list
                idx = buf.find('[', idx)
                if idx != -1:
                    buf = buf[idx+1:]
                    break
        # Stream individual token objects
        while True:
            # Read until we can decode one object
            while True:
                try:
                    obj, end = decoder.raw_decode(buf)
                    yield obj
                    buf = buf[end:].lstrip()
                    if buf.startswith(','):
                        buf = buf[1:].lstrip()
                    elif buf.startswith(']'):
                        return
                except json.JSONDecodeError:
                    chunk = f.read(8192)
                    if not chunk:
                        return
                    buf += chunk


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream tokens from logprobs.json")
    ap.add_argument('file', help='path to logprobs.json')
    ap.add_argument('--limit', type=int, help='print only the first N tokens')
    args = ap.parse_args()
    for i, token in enumerate(iter_tokens(args.file)):
        print(token)
        if args.limit and i + 1 >= args.limit:
            break

if __name__ == '__main__':
    main()
