"""Systematic one-token-deviation sweep around one or more base completions.

For a base string and every position i in it that the API reported
alternatives for, calls the model once per alternative t with the prompt

    base_tokens[0:i] + [t]

so the whole grid of single-token deviations gets its own continuation. For a
20-position base with 20 alternatives each that is 400 calls, 20 of them the
greedy ones that stay on the base path; those are kept for completeness and
flagged as meta.sweep.is_greedy.

Levels
------
--level1 (default) sweeps the empty-prompt record: positions 0..19 of its own
20 generated tokens.

--level2 sweeps every first-token deviation found in the input, i.e. each
record whose sweep position is 0 and which is not the greedy one — 19 of them
for a 20-alternative root. Each such record's full string is prompt (the
deviating first token) + 20 generated tokens; only the generated positions are
swept, since the first token's own alternatives were already covered at level
1. That is 19 x 400 = 7600 calls.

Positions recorded in meta.sweep.position are absolute in the base's full
token string, so a level-2 sweep reports 1..20 rather than 0..19.

Records are written in the same shape builder.html and the other scripts read,
so the output feeds into export_dijkstra_top.py and strings.html.

Logging matches builder.html plus the headers a browser cannot see: run from a
shell there is no CORS, so openai-processing-ms and the rate-limit headers come
through as well as x-request-id.

Paid data is protected: each response is appended to a JSONL file and fsynced
the moment it arrives, before anything else, and a rerun skips
(base_id, position, alternative) triples already present — so an interrupted
run never pays for the same call twice. The base id is part of that key
because (position, alternative) alone repeats across different bases.

Usage:
    python sweep_alternatives.py --base builder_history.json
    python sweep_alternatives.py --base sweep_history.json --level2 \
        --out sweep2_history.jsonl --json-out sweep2_history.json --concurrency 8
    python sweep_alternatives.py --base sweep_history.json --level2 --dry-run
"""

import argparse
import gzip
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = 'https://api.openai.com/v1/completions'
RETRY_DELAYS = [1.0, 4.0, 10.0]          # only for 429 / 5xx

PRICE_IN, PRICE_OUT = 1.50, 2.00         # $ / 1M tokens, gpt-3.5-turbo-instruct


def open_text(path):
    """Transparently read .gz — raw sweep histories run to tens of MB and are
    therefore stored compressed."""
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, encoding='utf-8')


def load_records(path):
    with open_text(path) as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else [data]


def pick_bases(recs, level2, base_id):
    """Return [(base_record, offset)] — offset = index of the first swept position."""
    if base_id:
        r = next((x for x in recs if x.get('id') == base_id), None)
        if not r:
            raise SystemExit(f'zaznam {base_id} nenalezen')
        off = len(((r.get('prompt') or {}).get('logprobs') or {}).get('tokens') or [])
        return [(r, off)]

    if level2:
        out = []
        for r in recs:
            sw = (r.get('meta') or {}).get('sweep') or {}
            if sw.get('position') == 0 and not sw.get('is_greedy'):
                # full string = the deviating first token + its 20 generated tokens;
                # only the generated part is swept, so the offset is the prompt length
                out.append((r, len(r['prompt']['logprobs']['tokens'])))
        if not out:
            raise SystemExit('nenalezeny zadne odbocky na pozici 0 — je vstup sweep vystup?')
        out.sort(key=lambda p: -(p[0]['meta']['sweep']['alternative_logprob']))
        return out

    roots = [r for r in recs
             if not ((r.get('prompt') or {}).get('logprobs') or {}).get('tokens')]
    if not roots:
        raise SystemExit('nenalezen zaznam s prazdnym promptem')
    roots.sort(key=lambda r: r.get('created', 0))
    return [(roots[-1], 0)]


def base_tokens_of(rec):
    """Full token string of a base: its prompt path plus what it generated."""
    prompt = ((rec.get('prompt') or {}).get('logprobs') or {}).get('tokens') or []
    own = rec['choices'][0]['logprobs']['tokens']
    return list(prompt) + list(own)


def plan_for(rec, offset):
    """[(absolute_position, alternative_token, its_logprob)] for one base."""
    lp = rec['choices'][0]['logprobs']
    plan = []
    for j, top in enumerate(lp['top_logprobs'] or []):
        if not top:
            continue
        for tok, val in sorted(top.items(), key=lambda kv: -kv[1]):
            plan.append((offset + j, tok, val))
    return plan


def call_api(key, body, log):
    """POST once, with retries only on 429/5xx. Returns (json, http_meta)."""
    payload = json.dumps(body).encode('utf-8')     # plain UTF-8, no BOM
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
            with urllib.request.urlopen(req, timeout=180) as res:
                latency = int((time.time() - t0) * 1000)
                txt = res.read().decode('utf-8')
                h = res.headers
                meta = {
                    'status': res.status,
                    'request_id': h.get('x-request-id'),
                    'cf_ray': h.get('cf-ray'),
                    # not CORS-exposed, so a browser can never record these
                    'processing_ms': h.get('openai-processing-ms'),
                    'ratelimit_remaining_requests': h.get('x-ratelimit-remaining-requests'),
                    'ratelimit_remaining_tokens': h.get('x-ratelimit-remaining-tokens'),
                    'latency_ms': latency,
                    'attempts': attempt + 1,
                }
                try:
                    return json.loads(txt), meta
                except json.JSONDecodeError:
                    last = f'HTTP {res.status}: odpoved neni JSON'
                    if res.status < 500:
                        raise SystemExit(last)
                    continue
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode('utf-8', 'replace')[:200]
            last = f'HTTP {e.code}: {body_txt}'
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
                    help='soubor se zaznamy, z nichz se vybere zaklad(y)')
    ap.add_argument('--base-id', help='sweepovat jen tento konkretni zaznam')
    ap.add_argument('--level2', action='store_true',
                    help='sweepovat kazdou odbocku na pozici 0 (19 zakladu)')
    ap.add_argument('--level', type=int,
                    help='cim oznacit vznikle zaznamy v meta.sweep.level; '
                         'jinak 2 pri --level2, jinak 1')
    ap.add_argument('--out', default='sweep_history.jsonl')
    ap.add_argument('--json-out', default='sweep_history.json')
    ap.add_argument('--log', default='sweep_log.txt')
    ap.add_argument('--max-tokens', type=int, default=20)
    ap.add_argument('--limit', type=int, help='jen prvnich N volani (pro zkousku)')
    ap.add_argument('--concurrency', type=int, default=1,
                    help='paralelnich volani; 7600 volani sekvencne trva hodiny')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--include-bases', action='store_true',
                    help='do --json-out pridat i zaznamy ze --base (aby byl soubor samonosny)')
    args = ap.parse_args()

    recs = load_records(args.base)
    bases = pick_bases(recs, args.level2, args.base_id)

    logf = None if args.dry_run else open(args.log, 'a', encoding='utf-8')
    log_lock = threading.Lock()

    def log(msg):
        with log_lock:
            print(msg, flush=True)
            if logf:
                logf.write(msg + '\n')
                logf.flush()

    log('=' * 70)
    log(f'zakladu: {len(bases)}  max_tokens={args.max_tokens}  concurrency={args.concurrency}')

    # (base_id, position, alternative) -> uz hotovo
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sw = (json.loads(line).get('meta') or {}).get('sweep') or {}
                    done.add((sw.get('base_id'), sw.get('position'), sw.get('alternative')))
                except json.JSONDecodeError:
                    continue
        log(f'v {args.out} uz je {len(done)} hotovych volani — preskakuji je')

    jobs = []
    for rec, offset in bases:
        btoks = base_tokens_of(rec)
        for pos, tok, val in plan_for(rec, offset):
            if (rec['id'], pos, tok) in done:
                continue
            jobs.append({'base': rec, 'btoks': btoks, 'pos': pos, 'tok': tok, 'val': val,
                         'greedy': pos < len(btoks) and btoks[pos] == tok})
    if args.limit:
        jobs = jobs[:args.limit]

    greedy_n = sum(1 for j in jobs if j['greedy'])
    log(f'k provedeni: {len(jobs)} volani ({greedy_n} on-path, {len(jobs) - greedy_n} odbocek)')

    if args.dry_run:
        for b, off in bases[:4]:
            sw = (b.get('meta') or {}).get('sweep') or {}
            label = repr(sw.get('alternative')) if sw else '(zaklad s prazdnym promptem)'
            log(f'  zaklad {b["id"]} offset={off} prvni token={label} '
                f'delka={len(base_tokens_of(b))}')
        if len(bases) > 4:
            log(f'  ... a dalsich {len(bases) - 4} zakladu')
        pt = sum(j['pos'] + 1 for j in jobs)
        ct = len(jobs) * args.max_tokens
        log(f'odhad: ~{pt} prompt + {ct} completion tokenu '
            f'=> ${pt/1e6*PRICE_IN + ct/1e6*PRICE_OUT:.4f}')
        secs = len(jobs) * 1.1 / max(1, args.concurrency)
        log(f'odhad casu pri concurrency={args.concurrency}: ~{secs/60:.0f} min')
        return

    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise SystemExit('chybi OPENAI_API_KEY')

    outf = open(args.out, 'a', encoding='utf-8')
    write_lock = threading.Lock()
    counter = {'n': 0, 'pt': 0, 'ct': 0}
    t_start = time.time()

    def run_job(job):
        btoks, pos, tok = job['btoks'], job['pos'], job['tok']
        prompt_tokens = btoks[:pos] + [tok]
        prompt_text = ''.join(prompt_tokens)
        body = {'model': (job['base'].get('request') or {}).get('model')
                         or job['base'].get('model'),
                'prompt': prompt_text, 'logprobs': 20,
                'max_tokens': args.max_tokens, 'temperature': 0, 'seed': 0}

        raw, http = call_api(key, body, log)
        rec = build_record(
            raw, body, prompt_tokens, prompt_text, job['base']['id'], args.max_tokens, http,
            {'base_id': job['base']['id'], 'position': pos, 'alternative': tok,
             'alternative_logprob': job['val'], 'is_greedy': job['greedy'],
             'level': args.level if args.level else (2 if args.level2 else 1)},
        )
        # zapis PRED cimkoli dalsim — zaplacene volani nesmi zmizet
        with write_lock:
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
            outf.flush()
            os.fsync(outf.fileno())
            counter['n'] += 1
            u = raw.get('usage') or {}
            counter['pt'] += u.get('prompt_tokens', 0)
            counter['ct'] += u.get('completion_tokens', 0)
            n = counter['n']
        got = len(raw['choices'][0]['logprobs']['tokens'])
        if n % 25 == 0 or got != args.max_tokens:
            rate = n / max(0.001, time.time() - t_start)
            eta = (len(jobs) - n) / max(1e-9, rate) / 60
            log(f'[{n}/{len(jobs)}] pos={pos:2d} alt={tok!r:14} -> {raw["id"]} '
                f'req={http["request_id"]} {http["latency_ms"]}ms tok={got} '
                f'fin={raw["choices"][0]["finish_reason"]} '
                f'| {rate:.1f}/s ETA {eta:.0f}min')

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        list(ex.map(run_job, jobs))

    outf.close()
    elapsed = time.time() - t_start
    cost = counter['pt'] / 1e6 * PRICE_IN + counter['ct'] / 1e6 * PRICE_OUT
    log(f'hotovo: {counter["n"]} volani za {elapsed:.0f}s, '
        f'{counter["pt"]} prompt + {counter["ct"]} completion tokenu, ${cost:.6f}')

    write_json(args, recs, log)


def write_json(args, base_recs, log):
    """JSONL -> JSON pole pro export/viewer skripty.

    U level 1 se prida i zaznam s prazdnym promptem: bez nej nema prvni token
    odkud vzit logprob (ta hodnota je jen v top_logprobs korene), takze strom
    by byl bezhlavy. U level 2 se na tohle spolehat neda, protoze zaklady
    samy jsou az level-1 zaznamy — proto --include-bases, nebo se pri exportu
    predaji oba soubory zaraz.
    """
    out = []
    with open(args.out, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))

    if args.include_bases:
        have = {r.get('id') for r in out}
        out = [r for r in base_recs if r.get('id') not in have] + out
        log(f'pridano {len(out) - sum(1 for _ in open(args.out, encoding="utf-8") if _.strip())}'
            f' zaznamu ze --base')
    elif not args.level2:
        roots = [r for r in base_recs
                 if not ((r.get('prompt') or {}).get('logprobs') or {}).get('tokens')]
        if roots and roots[-1]['id'] not in {r.get('id') for r in out}:
            out.insert(0, roots[-1])

    json.dump(out, open(args.json_out, 'w', encoding='utf-8'), ensure_ascii=False)
    log(f'zapsano {args.json_out}: {len(out)} zaznamu')


if __name__ == '__main__':
    main()
