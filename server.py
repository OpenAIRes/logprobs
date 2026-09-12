"""Local server for the logprobs tooling: static files plus live queries.

Replaces `python -m http.server 8899`. Two things it does that the plain static
server could not:

  * answers queries from store.py, so a viewer reads the current records instead
    of a frozen export file that silently falls behind them;
  * binds 127.0.0.1 only. The static server listened on every interface, and
    local_api_key.json sits in this directory -- anyone on the same network
    could fetch the API key. logprobs.html needs that file, so it stays
    readable, but now only from this machine.

It also sends no-store on HTML, which is why editing a viewer needed a
Ctrl+Shift+R to take effect.

Confirmed Studio requests call the API and commit to completion_history.json.

    python3 server.py                 # http://127.0.0.1:8899
    python3 server.py --port 8900     # alongside the old one, to compare
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import threading
import urllib.parse
import tempfile
import urllib.request
import urllib.error
from functools import wraps
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from store import RecordStore
from completion import normalize_request, build_record
from single_token_variants import variants_for

ROOT = os.path.dirname(os.path.abspath(__file__))

MAX_TOP = 5000          # a query is cheap, but shipping 8k entries to a tab is not


def _int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value, default=False):
    if value is None:
        return default
    return str(value).lower() in ('1', 'true', 'yes', 'on')


# A bound is not optional here. The whole variant set is len(tokens) x
# alternatives strings, each as long as the original -- for the 4096-token record
# that is ~82,000 strings of 16 kB, well over a gigabyte of JSON from one GET.
# So the answer is bounded by total text rather than by a count: a short
# completion (20 tokens, 400 variants, 32 kB) never reaches the budget and comes
# back complete, which is the case that matters, and the giant record answers
# with its first few positions and says that is what it did.
VARIANT_CHAR_BUDGET = 4_000_000
VARIANT_MAX = 5000


def bounded_variants(rec, k=None, limit=None, budget=VARIANT_CHAR_BUDGET):
    """variants_for over a trimmed copy of the record.

    Trimming top_logprobs is what bounds the work: it decides which positions get
    deviated and how many alternatives each one contributes, while the tokens
    list stays whole -- so every variant returned is still a full-length string.
    Nothing is trimmed unless the budget forces it.
    """
    lp = (rec.get('choices') or [{}])[0].get('logprobs') or {}
    tokens = lp.get('tokens') or []
    tops = lp.get('top_logprobs') or []
    widest = max((len(t or {}) for t in tops), default=0)
    if not tokens or not widest:
        return {**variants_for(rec), 'positions': 0, 'positions_total': len(tops),
                'k': 0, 'k_total': widest, 'total_substitutions': 0,
                'truncated': False}

    k = max(1, min(k or widest, widest))

    def topk(top):
        return [tok for tok, _ in sorted((top or {}).items(), key=lambda kv: -kv[1])[:k]]

    # A substitution whose token is the one actually generated changes nothing,
    # so it is not a variant. Counted before trimming: this is the size of the
    # complete answer, against which a partial one is reported.
    total = 0
    for i, top in enumerate(tops):
        if i >= len(tokens):
            break
        names = topk(top)
        total += len(names) - (1 if tokens[i] in names else 0)

    per_variant = max(1, len(''.join(tokens)))
    cap = min(VARIANT_MAX, limit or VARIANT_MAX)
    room = min(budget // (per_variant * k), cap // k)
    positions = max(1, min(len(tops), room))

    cut = [{tok: (top or {})[tok] for tok in topk(top)} for top in tops[:positions]]
    choice = dict((rec.get('choices') or [{}])[0])
    choice['logprobs'] = {**lp, 'top_logprobs': cut}
    trimmed = {**rec, 'choices': [choice] + list((rec.get('choices') or [])[1:])}

    out = variants_for(trimmed)
    out.update({'positions': positions, 'positions_total': len(tops),
                'k': k, 'k_total': widest, 'total_substitutions': total,
                'truncated': positions < len(tops) or k < widest})
    return out


ASK_POLICY_FILE = 'ask_policy.json'
ASK_POLICY_KEYS = ('always', 'big', 'batch')
# 0 means "per model": 20 for gpt-3.5-turbo-instruct, 5 for the base models.
# Anything else is that many tokens for every call the viewers make.
ASK_POLICY_DEFAULT = {'always': True, 'big': False, 'batch': False, 'max_tokens': 0}


def _clamp_max_tokens(value):
    """0 (or anything unusable) means the per-model default; otherwise 1..4096.

    Clamped rather than rejected: this is a number typed into a box, and a
    typo should not be able to ask for a 40,000-token completion.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return 0 if n <= 0 else min(n, 4096)


def read_ask_policy():
    """When the program asks before calling the API.

    Kept on the server rather than in localStorage because the viewers are served
    from :8899 and prompt-studio from :8787 -- different origins, so different
    localStorage. A setting that is meant to cover every part of the program has
    to live somewhere every part can read.
    """
    path = os.path.join(ROOT, ASK_POLICY_FILE)
    try:
        with open(path, encoding='utf-8') as fh:
            saved = json.load(fh)
        out = {k: bool(saved.get(k, ASK_POLICY_DEFAULT[k])) for k in ASK_POLICY_KEYS}
        out['max_tokens'] = _clamp_max_tokens(saved.get('max_tokens'))
        return out
    except (OSError, ValueError, AttributeError):
        return dict(ASK_POLICY_DEFAULT)


def write_ask_policy(policy):
    """Atomic, like every other write here: a half-written policy file would come
    back as the default on the next read, which is the strictest option and so
    harmless -- but a truncated file that still parses is not, so do not risk it."""
    clean = {k: bool((policy or {}).get(k)) for k in ASK_POLICY_KEYS}
    clean['max_tokens'] = _clamp_max_tokens((policy or {}).get('max_tokens'))
    path = os.path.join(ROOT, ASK_POLICY_FILE)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False,
                                         dir=ROOT, prefix='.ask_policy-',
                                         suffix='.tmp') as fh:
            json.dump(clean, fh, ensure_ascii=False, indent=2)
            temp = fh.name
        os.replace(temp, path)
        temp = None
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)
    return clean


def locked(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.store.lock:
            return method(self, *args, **kwargs)
    return wrapped


class Handler(SimpleHTTPRequestHandler):
    store: RecordStore = None       # set by serve()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    # -- plumbing -------------------------------------------------------------

    def log_message(self, fmt, *args):
        # One line per request, without the noisy default timestamp block.
        print(f'  {self.command} {self.path.split("?")[0]} -> {args[1] if len(args) > 1 else ""}')

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # A viewer edited on disk must not be served from the browser cache.
        if self.path.split('?')[0].endswith(('.html', '.js', '.css')):
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.end_headers()

    # -- routing --------------------------------------------------------------

    @locked
    def do_POST(self):
        """POST /api/lookup, because a prompt can be 4096 tokens.

        The GET form is fine for poking at by hand, but a long prompt in a query
        string runs into the request-line limit, and this is the one call that
        must never fail for a reason unrelated to the data -- a false miss costs
        real money.
        """
        route = posixpath.normpath(urllib.parse.urlparse(self.path).path)
        # Persist a record another program has already paid for. No API call and
        # no key: this only writes. The resampling program calls OpenAI itself
        # from node, and without this its results lived in its own log where none
        # of the viewers could see them -- so the same string had to be read in a
        # second, weaker copy of the viewer. Saving through here puts it in the
        # one store, which is what makes the lists, the greedy views and the
        # deviations work on it.
        #
        # store.save is idempotent by id and reloads the indexes, so a record
        # offered twice comes back rather than being duplicated, and a viewer
        # open in another tab sees it on its next request.
        if route == '/api/save':
            try:
                length = _int(self.headers.get('Content-Length'), 0) or 0
                body = json.loads(self.rfile.read(length) or b'{}')
                # One record or a list of them. The list is the one that matters:
                # each write reloads the indexes, so saving a run of variations
                # one at a time costs ten seconds apiece for no reason.
                if isinstance(body, dict) and 'records' in body:
                    records = body['records']
                elif isinstance(body, dict) and 'record' in body:
                    records = [body['record']]
                elif isinstance(body, list):
                    records = body
                else:
                    records = [body]
                if not isinstance(records, list) or not records or not all(
                        isinstance(r, dict) and isinstance(r.get('id'), str) and r.get('choices')
                        for r in records):
                    return self.send_json(
                        {'error': 'a record (or a list of records) with an id and choices is required'}, 400)
                saved = self.store.save_many(records)
                return self.send_json({'saved': True, 'ids': [r.get('id') for r in saved],
                                       'id': saved[0].get('id'),
                                       'records': len(self.store.records)})
            except (ValueError, json.JSONDecodeError) as exc:
                return self.send_json({'error': str(exc)}, 400)
            except OSError as exc:
                return self.send_json({'error': f'could not write the history: {exc}',
                                       'saved': False}, 507)

        if route == '/api/ask_policy':
            try:
                length = _int(self.headers.get('Content-Length'), 0) or 0
                body = json.loads(self.rfile.read(length) or b'{}')
                return self.send_json({'policy': write_ask_policy(body)})
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                return self.send_json({'error': str(exc)}, 400)

        if route == '/api/deviations':
            # POST, because the string being deviated can be 4096 tokens and a
            # row's path is not always one record's own tokens -- so it is sent
            # rather than named. GET /api/deviations?base_id= covers the case
            # where a record's own tokens are what is wanted.
            try:
                length = _int(self.headers.get('Content-Length'), 0) or 0
                body = json.loads(self.rfile.read(length) or b'{}')
                if not isinstance(body, dict) or not isinstance(body.get('tokens'), list):
                    return self.send_json({'error': 'tokens (a list of strings) is required'}, 400)
                return self.send_json(self.store.deviations(
                    body['tokens'],
                    # Only to read how that record was generated; the tokens
                    # above are what actually gets deviated.
                    base_request=(self.store.by_id.get(body.get('base_id') or '')
                                  or {}).get('request'),
                    # The immovable part: a given prompt has no logprobs, so no
                    # alternatives, and it also says where in the trie to start.
                    prompt_tokens=body.get('prompt_tokens') or None,
                    model=body.get('model') or 'gpt-3.5-turbo-instruct',
                    alts=min(_int(body.get('alts'), 20) or 20, 20),
                    sort=body.get('sort') or 'cost',
                    ends=body.get('ends') or None,
                    nodes=body.get('nodes') or None,
                    sources=body.get('sources') or None))
            except (ValueError, json.JSONDecodeError) as exc:
                return self.send_json({'error': str(exc)}, 400)
        if route not in ('/api/lookup', '/api/complete'):
            return self.send_json({'error': f'unknown route {route}'}, 404)
        try:
            length = _int(self.headers.get('Content-Length'), 0) or 0
            body = json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(body, dict) or not isinstance(body.get('prompt'), str):
                return self.send_json({'error': 'prompt is required'}, 400)
            if route == '/api/complete':
                if body.get('confirmed') is not True:
                    return self.send_json({'error': 'explicit confirmation is required'}, 400)
                request_body = normalize_request(body)
                cached = self.store.lookup_request(request_body) if request_body['temperature'] == 0 else None
                if cached is not None:
                    return self.send_json({**cached, 'saved': True, 'from_cache': True})
                key = os.environ.get('OPENAI_API_KEY', '').strip()
                if not key:
                    return self.send_json({'error': 'OPENAI_API_KEY is not available to the local server'}, 503)
                req = urllib.request.Request(
                    'https://api.openai.com/v1/completions',
                    data=json.dumps(request_body).encode('utf-8'),
                    headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
                    method='POST',
                )
                try:
                    with urllib.request.urlopen(req, timeout=120) as response:
                        raw = json.loads(response.read().decode('utf-8'))
                    try:
                        # The caller may know how the prompt is tokenised -- a
                        # deviation built it out of recorded tokens. Refused
                        # unless the parts join back to the prompt that was sent.
                        record = build_record(raw, request_body, body.get('prompt_tokens'))
                    except ValueError as exc:
                        return self.send_json({'error': str(exc), 'saved': False,
                                               'record': raw, 'api_completed': True}, 502)
                    try:
                        record = self.store.save(record)
                    except Exception:
                        return self.send_json({'error': 'API odpovědělo, ale uložení selhalo. Stáhněte záznam; neopakujte placené volání.',
                                               'saved': False, 'record': record,
                                               'api_completed': True}, 507)
                    return self.send_json({**record, 'saved': True, 'from_cache': False})
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode('utf-8', errors='replace')[:500]
                    return self.send_json({'error': f'OpenAI HTTP {exc.code}', 'detail': detail}, exc.code)
                except (urllib.error.URLError, OSError) as exc:
                    # Logged, not only answered. Making this readable in the
                    # browser and invisible here meant 245 failed calls in one
                    # run left no trace at all, and the reason could not be
                    # recovered afterwards.
                    print(f'  upstream unreachable: {type(exc).__name__}: '
                          f'{getattr(exc, "reason", exc)}', flush=True)
                    # The request never left the machine: DNS, no route, a
                    # refused connection. Only HTTPError was handled, so this
                    # escaped as a 500 with an HTML body, and the page could not
                    # even read why. Saying "nothing was spent" is the first
                    # thing anyone wants to know when a paid call fails.
                    reason = getattr(exc, 'reason', exc)
                    return self.send_json(
                        {'error': f'API nebylo dosaženo: {reason}. '
                                  f'Požadavek neodešel, takže nic nebylo účtováno.',
                         'saved': False, 'api_completed': False,
                         'unreachable': True}, 503)
            rec = self.store.lookup_request(normalize_request(body)) if body.get('exact') else self.store.lookup(
                prompt=body['prompt'],
                model=body.get('model') or None,
                max_tokens=_int(body.get('max_tokens')),
                temperature=body.get('temperature', 0),
            )
            return self.send_json({'hit': rec is not None,
                                   'exact': bool(body.get('exact')),
                                   'id': (rec or {}).get('id'),
                                   'record': rec})
        except (ValueError, json.JSONDecodeError) as exc:
            return self.send_json({'error': str(exc)}, 400)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return self.send_json({'error': f'{type(exc).__name__}: {exc}'}, 500)

    def landing(self) -> str:
        """Where `/` sends you: the result, not a page of choices.

        Greedy at temperature 0 from the empty prompt, which is the 4096-token
        record we have in full -- so the first thing on screen is a string with
        its logprobs rather than a form. Derived, not hardcoded: if the store
        gains a longer argmax chain, this follows it.
        """
        try:
            first = self.store.greedy(prompt='', model='gpt-3.5-turbo-instruct').get('first_id')
        except Exception:
            first = None
        if not first:
            return '/index.html'
        # The settings that produced it, spelled out. Sending only `id` let the
        # bar fall back to whatever was last used, so the landing showed the
        # greedy string while the bar said `completions` -- a bar that
        # misdescribes what is on screen is worse than no bar.
        # defaults=1 tells the bar to ignore what was last used. Without it a
        # remembered filter leaked into the landing and the bar described the
        # string as filtered when it was not.
        return ('/logprobs.html?view=greedy&top=1&defaults=1'
                f'&id={urllib.parse.quote(first)}')

    @locked
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        route = posixpath.normpath(parsed.path)
        # `/` is the landing, and the landing is a result. /index.html is still
        # the settings page, reachable from the bar; this only changes what the
        # bare root does.
        if route == '/' and not parsed.query:
            self.send_response(302)
            self.send_header('Location', self.landing())
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            return
        if not route.startswith('/api/'):
            return super().do_GET()

        # keep_blank_values matters: the root call has prompt="", and dropping
        # blanks turned the single most important lookup into "prompt is required".
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        one = lambda key, default=None: (query.get(key) or [default])[0]

        try:
            if route == '/api/stats':
                return self.send_json(self.store.stats())

            if route == '/api/records':
                limit = max(1, min(_int(one('limit'), 200) or 200, 500))
                offset = max(0, _int(one('offset'), 0) or 0)
                records = sorted(self.store.records, key=lambda r: r.get('created') or 0, reverse=True)
                search = one('q', '').casefold()
                if search:
                    records = [r for r in records if search in json.dumps(
                        {k: r.get(k) for k in ('id', 'model', 'request', 'choices')},
                        ensure_ascii=False).casefold()]
                return self.send_json({'records': records[offset:offset + limit], 'total': len(records)})

            if route == '/api/bases':
                return self.send_json({'bases': self.store.bases()})

            if route == '/api/query':
                top = min(_int(one('top'), 200) or 200, MAX_TOP)
                return self.send_json(self.store.query(
                    view=one('view', 'prefixes'),
                    top=top,
                    min_n=_int(one('min_n'), 1),
                    prefix=one('prefix') or None,
                    model=one('model') or None,
                    # chosen_only is not the mild filter it sounds like: a path must
                    # be chosen ALL the way down, and the root has one chosen child
                    # out of 21, so it takes the trie from 176,649 reachable to 4,174.
                    chosen_only=_bool(one('chosen_only')),
                    max_alts=_int(one('max_alts'), 8),
                    sort=one('sort', 'sum'),
                    # ends=stop,length,open -- which kinds of string ending fall
                    # into the scope. Absent means all of them, which is what
                    # every existing link means, so none of them change.
                    ends=one('ends') or None,
                    extend=_bool(one('extend')),
                    # nodes=continues,leaf -- whether the database holds anything
                    # past the string. Crosses with ends: 494 length-ended paths
                    # were later continued, so this is not "untick open".
                    nodes=one('nodes') or None,
                    # sources=history,sweep -- which databases get ranked. Absent
                    # means all of them, so every existing link is unchanged.
                    sources=one('sources') or None,
                ))

            if route == '/api/ask_policy':
                return self.send_json({'policy': read_ask_policy()})

            if route == '/api/deviations':
                return self.send_json(self.store.deviations_for_id(
                    one('base_id') or one('id') or '',
                    model=one('model', 'gpt-3.5-turbo-instruct'),
                    alts=min(_int(one('alts'), 20) or 20, 20),
                    sort=one('sort', 'cost'),
                    # Same shape as every other route: one comma-separated
                    # value, normalised in the store. A list of one
                    # "stop,length" string would fail the member test there.
                    ends=one('ends') or None, nodes=one('nodes') or None,
                    sources=one('sources') or None))

            if route == '/api/single-token-variants':
                rec = self.store.by_id.get(one('id') or '')
                if rec is None:
                    return self.send_json({'error': 'no record with that id'}, 404)
                if not ((rec.get('choices') or [{}])[0].get('logprobs') or {}).get('tokens'):
                    return self.send_json({'error': 'that record has no token logprobs'}, 400)
                return self.send_json(bounded_variants(
                    rec, k=_int(one('k')), limit=_int(one('limit'))))

            if route == '/api/record':
                rec = self.store.by_id.get(one('id') or '')
                if rec is None:
                    return self.send_json({'error': 'no record with that id'}, 404)
                return self.send_json(rec)

            if route == '/api/greedy':
                return self.send_json(self.store.greedy(
                    prompt=(query.get('prompt') or [''])[0],
                    model=one('model', 'gpt-3.5-turbo-instruct'),
                    max_steps=_int(one('max_steps')),
                ))

            if route in ('/api/greedy_alternatives', '/api/greedy_i'):
                return self.send_json((self.store.greedy_i if route == '/api/greedy_i' else self.store.greedy_alternatives)(
                    prompt=(query.get('prompt') or [''])[0],
                    model=one('model', 'gpt-3.5-turbo-instruct'),
                    top=min(_int(one('top'), 20) or 20, MAX_TOP),
                    sort=one('sort', 'cost'),
                    max_alts=_int(one('max_alts'), 8),
                    ends=one('ends') or None,
                    extend=_bool(one('extend')),
                    nodes=one('nodes') or None,
                    sources=one('sources') or None,
                ))

            if route == '/api/walk':
                return self.send_json(self.store.walk(
                    base_id=one('base_id') or None,
                    steps=min(_int(one('steps'), 20) or 20, 200),
                    extend=min(_int(one('extend'), 20) or 20, 200),
                    model=one('model', 'gpt-3.5-turbo-instruct'),
                    forward_only=_bool(one('forward_only')),
                ))

            if route == '/api/sweep':
                base_id = one('base_id') or ''
                if not base_id:
                    return self.send_json({'error': 'base_id is required'}, 400)
                return self.send_json(self.store.sweep_grid(base_id))

            if route == '/api/lookup':
                # The one cache verdict. `prompt` is sent verbatim, including
                # leading and trailing whitespace -- trimming it was the cause of
                # every false miss that led to a needless paid call.
                if 'prompt' not in query:
                    return self.send_json({'error': 'prompt is required'}, 400)
                rec = self.store.lookup(
                    prompt=query['prompt'][0],
                    model=one('model') or None,
                    max_tokens=_int(one('max_tokens')),
                    temperature=float(one('temperature', 0) or 0),
                )
                return self.send_json({'hit': rec is not None,
                                       'id': (rec or {}).get('id'),
                                       'record': rec})

            return self.send_json({'error': f'unknown route {route}'}, 404)
        except ValueError as exc:
            return self.send_json({'error': str(exc)}, 400)
        except Exception as exc:
            # Without this the connection just closed and curl showed an empty
            # body, which is how a TypeError in walk_detail looked like "the
            # endpoint returns nothing". Say what broke, and keep serving.
            import traceback
            traceback.print_exc()
            return self.send_json({'error': f'{type(exc).__name__}: {exc}'}, 500)


def serve(port: int, host: str = '127.0.0.1') -> None:
    store = RecordStore().load()
    stats = store.stats()
    print(f'store: {stats["records"]} records, {stats["distinct_prompts"]} distinct prompts, '
          f'loaded in {stats["load_seconds"]}s')
    if stats['missing_sources']:
        print(f'  missing sources (skipped): {stats["missing_sources"]}')

    # The completions ranking walks every record's token path, ~4 s the first
    # time. Warm it in the background so the port opens immediately and the cache
    # is ready before anyone switches to that view.
    def warm_unlocked():
        import time as _t
        started = _t.time()
        store.query(view='completions', top=1)
        print(f'  completions ranking warmed in {_t.time() - started:.1f}s')
        # ~82k lookups over a 4096-token path, so the same treatment.
        started = _t.time()
        store.greedy_alternatives(top=1)
        print(f'  greedy siblings warmed in {_t.time() - started:.1f}s')
        # One walk over 177k nodes; cached, but 1.4 s is too long to sit behind
        # the first page load.
        started = _t.time()
        store.trie_counts()
        print(f'  prefix counts warmed in {_t.time() - started:.1f}s')

    def warm():
        with store.lock:
            warm_unlocked()

    threading.Thread(target=warm, daemon=True).start()

    handler = type('BoundHandler', (Handler,), {'store': store})
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f'serving {ROOT}')
    print(f'  http://{host}:{port}/logprobs.html')
    print(f'  http://{host}:{port}/dijkstra.html')
    print(f'  http://{host}:{port}/api/stats')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--port', type=int, default=8899)
    ap.add_argument('--host', default='127.0.0.1',
                    help='127.0.0.1 by default: this directory holds the API key file')
    serve(**vars(ap.parse_args()))
