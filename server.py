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

Nothing here calls the OpenAI API. Generators come later; this is read-only.

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
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from store import RecordStore

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
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # A viewer edited on disk must not be served from the browser cache.
        if self.path.split('?')[0].endswith(('.html', '.js', '.css')):
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    # -- routing --------------------------------------------------------------

    def do_POST(self):
        """POST /api/lookup, because a prompt can be 4096 tokens.

        The GET form is fine for poking at by hand, but a long prompt in a query
        string runs into the request-line limit, and this is the one call that
        must never fail for a reason unrelated to the data -- a false miss costs
        real money.
        """
        route = posixpath.normpath(urllib.parse.urlparse(self.path).path)
        if route != '/api/lookup':
            return self.send_json({'error': f'unknown route {route}'}, 404)
        try:
            length = _int(self.headers.get('Content-Length'), 0) or 0
            body = json.loads(self.rfile.read(length) or b'{}')
            if 'prompt' not in body:
                return self.send_json({'error': 'prompt is required'}, 400)
            rec = self.store.lookup(
                prompt=body['prompt'],
                model=body.get('model') or None,
                max_tokens=_int(body.get('max_tokens')),
                temperature=body.get('temperature', 0),
            )
            return self.send_json({'hit': rec is not None,
                                   'id': (rec or {}).get('id'),
                                   'record': rec})
        except (ValueError, json.JSONDecodeError) as exc:
            return self.send_json({'error': str(exc)}, 400)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return self.send_json({'error': f'{type(exc).__name__}: {exc}'}, 500)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        route = posixpath.normpath(parsed.path)
        if not route.startswith('/api/'):
            return super().do_GET()

        # keep_blank_values matters: the root call has prompt="", and dropping
        # blanks turned the single most important lookup into "prompt is required".
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        one = lambda key, default=None: (query.get(key) or [default])[0]

        try:
            if route == '/api/stats':
                return self.send_json(self.store.stats())

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
                ))

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

            if route == '/api/greedy_alternatives':
                return self.send_json(self.store.greedy_alternatives(
                    prompt=(query.get('prompt') or [''])[0],
                    model=one('model', 'gpt-3.5-turbo-instruct'),
                    top=min(_int(one('top'), 20) or 20, MAX_TOP),
                    sort=one('sort', 'cost'),
                    max_alts=_int(one('max_alts'), 8),
                    ends=one('ends') or None,
                    extend=_bool(one('extend')),
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
    def warm():
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
