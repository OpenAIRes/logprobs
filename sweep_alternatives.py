"""Systematic one-token-deviation sweep around a base completion.

Takes a base completion of N tokens and, for every position i and every
alternative token t that the API reported at that position, calls the model
with the prompt

    base_tokens[0:i] + [t]

so the whole grid of single-token deviations from the base path gets its own
continuation. With N=20 positions and 20 alternatives each that is 400 calls,
of which 20 are the greedy ones (t == base_tokens[i]) that stay on the base
path and simply extend it deeper — they are kept for completeness and flagged
via meta.sweep.is_greedy.

Records are written in the same shape builder.html and the other scripts read,
so the output feeds straight into export_dijkstra_top.py and dijkstra.html.

Logging matches what builder.html captures, plus the headers a browser cannot
see: run from a shell there is no CORS, so openai-processing-ms and the
rate-limit headers come through as well as x-request-id.

Paid data is protected the same way: each response is appended to a JSONL file
the moment it arrives, before anything else happens, and a rerun skips prompts
already present — so an interrupted run never pays for the same call twice.

Usage:
    set OPENAI_API_KEY, then
    python sweep_alternatives.py --base builder_history.json
    python sweep_alternatives.py --base builder_history.json --dry-run
    python sweep_alternatives.py --max-tokens 20 --limit 5
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = 'https://api.openai.com/v1/completions'
RETRY_DELAYS = [1.0, 4.0, 10.0]          # only for 429 / 5xx


def load_base(path):
    """Pick the base completion: the record with an empty prompt."""
    data = json.load(open(path, encoding='utf-8'))
    recs = data if isinstance(data, list) else [data]
    roots = [r for r in recs
             if not ((r.get('prompt') or {}).get('logprobs') or {}).get('tokens')]
    if not roots:
        raise SystemExit(f'{path}: nenalezen zaznam s prazdnym promptem')
    roots.sort(key=lambda r: r.get('created', 0))
    return roots[-1]


def call_api(key, body, log):
    """POST once, with retries only on 429/5xx. Returns (json, http_meta)."""
    payload = json.dumps(body).encode('utf-8')   # no BOM: json.dumps is plain UTF-8
    last = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        if attempt:
            wait = RETRY_DELAYS[attempt - 1]
            log(f'    opakuji za {wait}s (pokus {attempt + 1}): {last}')
            time.sleep(wait)
        req = urllib.request.Request(URL, data=payload, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f'Bearer {key}')
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as res:
                latency = int((time.time() - t0) * 1000)
                body_txt = res.read().decode('utf-8')
                h = res.headers
                meta = {
                    'status': res.status,
                    'request_id': h.get('x-request-id'),
                    'cf_ray': h.get('cf-ray'),
                    # these two a browser can never read (not CORS-exposed)
                    'processing_ms': h.get('openai-processing-ms'),
                    'ratelimit_remaining_requests': h.get('x-ratelimit-remaining-requests'),
                    'ratelimit_remaining_tokens': h.get('x-ratelimit-remaining-tokens'),
                    'latency_ms': latency,
                    'attempts': attempt + 1,
                }
                try:
                    return json.loads(body_txt), meta
                except json.JSONDecodeError:
                    last = f'HTTP {res.status}: odpoved neni JSON'
                    if res.status < 500:
                        raise SystemExit(last)
                    continue
        except urllib.error.HTTPError as e:
            latency = int((time.time() - t0) * 1000)
            txt = e.read().decode('utf-8', 'replace')[:200]
            last = f'HTTP {e.code}: {txt}'
            if e.code == 429 or e.code >= 500:
                continue
            raise SystemExit(f'neopakovatelna chyba {last}')
        except urllib.error.URLError as e:
            last = f'sitova chyba: {e.reason}'
            continue
    raise SystemExit(f'vycerpany pokusy: {last}')


def build_record(raw, request, prompt_tokens, prompt_text, prev_id, max_tokens, http, sweep):
    return {
        'id': raw['id'],
        'object': raw.get('object', 'text_completion'),
        'created': raw.get('created', 0),
        'model': raw.get('model'),
        'previous_response_id': prev_id,
        'prompt': {
            'text': prompt_text,
            'index': 0,
            'logprobs': {'tokens': prompt_tokens, 'token_logprobs': [], 'top_logprobs': []},
        },
        'choices': raw['choices'],
        'usage': raw.get('usage'),
        'max_tokens': len(raw['choices'][0]['logprobs']['tokens']),
        'temperature': 0,
        'seed': 0,
        'meta': {
            'target_max_tokens': max_tokens,
            'source': 'sweep',
            'http': http,
            'sweep': sweep,
        },
        'request': request,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base', default='builder_history.json',
                    help='soubor se zakladni completion (zaznam s prazdnym promptem)')
    ap.add_argument('--out', default='sweep_history.jsonl', help='vystupni JSONL (prubezne)')
    ap.add_argument('--json-out', default='sweep_history.json', help='vystupni JSON pole')
    ap.add_argument('--log', default='sweep_log.txt')
    ap.add_argument('--max-tokens', type=int, default=20)
    ap.add_argument('--limit', type=int, help='jen prvnich N volani (pro zkousku)')
    ap.add_argument('--dry-run', action='store_true', help='nic nevolat, jen vypsat plan')
    args = ap.parse_args()

    base = load_base(args.base)
    blp = base['choices'][0]['logprobs']
    base_tokens = blp['tokens']
    base_tops = blp['top_logprobs']
    model_alias = (base.get('request') or {}).get('model') or base.get('model')

    logf = None if args.dry_run else open(args.log, 'a', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        if logf:
            logf.write(msg + '\n')
            logf.flush()

    log('=' * 70)
    log(f'sweep zaklad: {base["id"]}  model={base.get("model")}  alias={model_alias}')
    log(f'zakladni tokeny ({len(base_tokens)}): {base_tokens}')

    # plan: (pozice, alternativa, logprob alternativy)
    plan = []
    for i, top in enumerate(base_tops):
        if not top:
            continue
        for tok, lp in sorted(top.items(), key=lambda kv: -kv[1]):
            plan.append((i, tok, lp))
    greedy = sum(1 for i, t, _ in plan if t == base_tokens[i])
    log(f'plan: {len(plan)} volani ({greedy} on-path, {len(plan) - greedy} odbocek)')

    # co uz je hotove -> nezaplatit dvakrat
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    sw = (r.get('meta') or {}).get('sweep') or {}
                    done.add((sw.get('position'), sw.get('alternative')))
                except json.JSONDecodeError:
                    continue
        log(f'v {args.out} uz je {len(done)} hotovych volani — preskakuji je')

    todo = [(i, t, lp) for i, t, lp in plan if (i, t) not in done]
    if args.limit:
        todo = todo[:args.limit]
    log(f'k provedeni: {len(todo)}')

    if args.dry_run:
        for i, t, lp in todo[:10]:
            prefix = base_tokens[:i] + [t]
            log(f'  pos {i:2d} alt {t!r:14} logP={lp:8.4f}  prompt={"".join(prefix)!r}')
        log(f'  ... celkem {len(todo)}')
        # odhad nakladu
        pt = sum(i + 1 for i, _, _ in todo)
        ct = len(todo) * args.max_tokens
        log(f'odhad: ~{pt} prompt + {ct} completion tokenu '
            f'=> ${pt/1e6*1.5 + ct/1e6*2.0:.4f}')
        return

    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise SystemExit('chybi OPENAI_API_KEY')

    outf = open(args.out, 'a', encoding='utf-8')
    t_start = time.time()
    tot_pt = tot_ct = 0

    for n, (i, tok, lp) in enumerate(todo, 1):
        prompt_tokens = base_tokens[:i] + [tok]
        prompt_text = ''.join(prompt_tokens)
        body = {'model': model_alias, 'prompt': prompt_text, 'logprobs': 20,
                'max_tokens': args.max_tokens, 'temperature': 0, 'seed': 0}

        raw, http = call_api(key, body, log)
        rec = build_record(
            raw, body, prompt_tokens, prompt_text, base['id'], args.max_tokens, http,
            {'base_id': base['id'], 'position': i, 'alternative': tok,
             'alternative_logprob': lp, 'is_greedy': tok == base_tokens[i]},
        )
        # zapis PRED cimkoli dalsim — zaplacene volani nesmi zmizet
        outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
        outf.flush()
        os.fsync(outf.fileno())

        u = raw.get('usage') or {}
        tot_pt += u.get('prompt_tokens', 0)
        tot_ct += u.get('completion_tokens', 0)
        got = len(raw['choices'][0]['logprobs']['tokens'])
        log(f'[{n}/{len(todo)}] pos={i:2d} alt={tok!r:14} logP={lp:8.4f} '
            f'-> {raw["id"]} req={http["request_id"]} {http["latency_ms"]}ms '
            f'proc={http["processing_ms"]}ms tok={got} fin={raw["choices"][0]["finish_reason"]}'
            + ('  [on-path]' if tok == base_tokens[i] else ''))

    outf.close()
    elapsed = time.time() - t_start
    cost = tot_pt / 1e6 * 1.5 + tot_ct / 1e6 * 2.0
    log(f'hotovo: {len(todo)} volani za {elapsed:.0f}s, '
        f'{tot_pt} prompt + {tot_ct} completion tokenu, ${cost:.6f}')

    write_json(args, base, log)


def write_json(args, base, log):
    """JSONL -> JSON pole pro dijkstra/export skripty.

    Zaklad se prida taky: bez zaznamu s prazdnym promptem nema prvni token
    odkud vzit logprob (jeho hodnota je jen v top_logprobs korene), takze
    strom by byl bezhlavy a razeni podle souctu by nevratilo nic.
    """
    recs = []
    with open(args.out, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not any(r.get('id') == base.get('id') for r in recs):
        recs.insert(0, base)
    json.dump(recs, open(args.json_out, 'w', encoding='utf-8'), ensure_ascii=False)
    log(f'zapsano {args.json_out}: {len(recs)} zaznamu (vcetne zakladu {base["id"]})')


if __name__ == '__main__':
    main()
