"""Offline persistence and HTTP regression tests. Never calls an external API."""

import copy
import io
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
import urllib.error
from unittest.mock import patch

from completion import build_record, normalize_request
from server import Handler
from store import RecordStore


def fixture(rid='test-1', prompt='Hi'):
    request = normalize_request({'prompt': prompt, 'max_tokens': 1, 'logprobs': 0})
    raw = {'id': rid, 'created': 1, 'model': request['model'],
           'choices': [{'text': '!', 'index': 0, 'finish_reason': 'length',
                        'logprobs': {'tokens': ['!'],
                                     'token_logprobs': [-.1],
                                     'top_logprobs': [{}],
                                     'text_offset': [len(prompt)]}}],
           'usage': {'prompt_tokens': int(bool(prompt)), 'completion_tokens': 1, 'total_tokens': 2}}
    return request, raw


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = RecordStore(sources=['completion_history.json'], root=self.temp.name).load()


class PersistenceTests(StoreTestCase):
    def test_save_reload_lookup_and_views(self):
        self.store.query(view='completions', top=10)
        req, raw = fixture(prompt='')
        record = build_record(raw, req)
        self.store.save(record)
        self.assertEqual(self.store.stats()['records'], 1)
        self.assertEqual(self.store.query(view='completions', top=10)['count'], 1)
        self.assertEqual(self.store.greedy(prompt='')['first_id'], raw['id'])
        restarted = RecordStore(sources=['completion_history.json'], root=self.temp.name).load()
        self.assertEqual(restarted.lookup_request(req)['raw_response'], raw)
        self.assertEqual(restarted.missing, [])
        self.assertEqual(restarted.by_id[raw['id']]['choices'][0]['text'], '!')

    def test_deferred_save_is_durable_and_findable_before_the_reindex(self):
        """A batch pays for the indexes once, at the end, not once per record."""
        self.store.query(view='completions', top=10)   # warm the cached view
        req, raw = fixture(prompt='')
        saved = self.store.save(build_record(raw, req), defer=True)

        # On disk first, whatever happens next: a paid call is never left only
        # in memory, which is the whole reason deferring the reindex is safe.
        history = json.loads(
            (Path(self.temp.name) / 'completion_history.json').read_text('utf-8'))
        self.assertEqual([r['id'] for r in history], [raw['id']])

        # Findable by id and by prompt, so the next call in the batch sees it as
        # already bought rather than paying for the same string twice.
        self.assertIs(self.store.by_id[raw['id']], saved)
        self.assertIsNotNone(self.store.lookup_request(req))

        # The ranked views are deliberately stale until something reads them.
        self.assertTrue(self.store._dirty)
        self.assertEqual(self.store.query(view='completions', top=10)['count'], 0)
        self.store.settle()
        self.assertFalse(self.store._dirty)
        self.assertEqual(self.store.query(view='completions', top=10)['count'], 1)
        self.assertEqual(self.store.greedy(prompt='')['first_id'], raw['id'])

    def test_settle_does_nothing_when_no_save_was_deferred(self):
        with patch.object(RecordStore, 'load') as reload:
            self.store.settle()
        reload.assert_not_called()

    def test_prompt_split_and_unicode(self):
        req, raw = fixture(prompt=' Ahoj 🐈\n')
        record = build_record(raw, req)
        self.assertEqual(record['prompt']['logprobs']['tokens'], [req['prompt']])
        self.assertEqual(record['choices'][0]['logprobs']['tokens'], ['!'])
        self.assertEqual(record['raw_response'], raw)

    def test_concurrent_writes_and_duplicate_id(self):
        def save(i):
            req, raw = fixture(f'test-{i}')
            return self.store.save(build_record(raw, req))
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(save, range(18)))
        save(0)
        data = json.loads(Path(self.temp.name, 'completion_history.json').read_text(encoding='utf-8'))
        self.assertEqual(len(data), 18)
        self.assertEqual(len({r['id'] for r in data}), 18)

    def test_two_store_instances_preserve_each_others_writes(self):
        other = RecordStore(sources=['completion_history.json'], root=self.temp.name).load()
        def save(i):
            req, raw = fixture(f'shared-{i}')
            return (self.store if i % 2 else other).save(build_record(raw, req))
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(save, range(8)))
        other.load()
        self.assertEqual(other.stats()['records'], 8)

    def test_failed_write_preserves_disk_and_memory(self):
        req, raw = fixture()
        self.store.save(build_record(raw, req))
        path = Path(self.temp.name, 'completion_history.json')
        before = path.read_bytes()
        raw['id'] = 'new'
        with patch('store.os.replace', side_effect=OSError('disk error')):
            with self.assertRaises(OSError):
                self.store.save(build_record(raw, req))
        self.assertEqual(path.read_bytes(), before)
        self.assertNotIn('new', self.store.by_id)
        self.assertFalse(list(Path(self.temp.name).glob('.history-*.tmp')))

    def test_corrupt_history_not_overwritten(self):
        path = Path(self.temp.name, 'completion_history.json')
        path.write_text('{broken', encoding='utf-8')
        req, raw = fixture()
        with self.assertRaises(ValueError):
            self.store.save(build_record(raw, req))
        self.assertEqual(path.read_text(), '{broken')

    def test_exact_parameters_and_zero_values(self):
        req, raw = fixture()
        req['top_p'] = 0
        self.store.save(build_record(raw, req))
        self.assertEqual(normalize_request(req)['top_p'], 0)
        self.assertEqual(normalize_request(req)['logprobs'], 0)
        self.assertIsNotNone(self.store.lookup_request(req))
        for field, other in [('top_p', 1), ('temperature', .5), ('frequency_penalty', 1),
                             ('presence_penalty', 1), ('max_tokens', 2), ('logprobs', 1),
                             ('prompt', 'Hi '), ('model', 'davinci-002')]:
            self.assertIsNone(self.store.lookup_request({**req, field: other}), field)
        for field, invalid in [('top_p', float('nan')), ('logprobs', -1),
                               ('max_tokens', 1.5), ('temperature', True)]:
            with self.assertRaises(ValueError):
                normalize_request({**req, field: invalid})


class AnnotationTests(StoreTestCase):
    def annotated(self, rid='test-1', prompt=''):
        req, raw = fixture(rid=rid, prompt=prompt)
        self.store.save(build_record(raw, req))
        return raw['id']

    def test_metadata_is_an_overlay_the_record_files_never_see(self):
        """The label must not rewrite the record: most of them live in gzipped
        sweep files, and one typed word is not worth rewriting one of those."""
        rid = self.annotated()
        before = (Path(self.temp.name) / 'completion_history.json').read_text('utf-8')
        self.store.annotate(rid, 'group', 'keepers')
        after = (Path(self.temp.name) / 'completion_history.json').read_text('utf-8')
        self.assertEqual(before, after, 'annotating rewrote the history file')

        overlay = json.loads((Path(self.temp.name) / 'annotations.json').read_text('utf-8'))
        self.assertEqual(overlay, {rid: {'group': 'keepers'}})
        self.assertEqual(self.store.by_id[rid]['metadata'], {'group': 'keepers'})

    def test_the_label_survives_a_reload_and_reaches_the_record(self):
        rid = self.annotated()
        self.store.annotate(rid, 'group', 'keepers')
        restarted = RecordStore(sources=['completion_history.json'], root=self.temp.name).load()
        self.assertEqual(restarted.by_id[rid]['metadata'], {'group': 'keepers'})

    def test_an_empty_value_removes_the_key_and_then_the_record(self):
        rid = self.annotated()
        self.store.annotate(rid, 'group', 'keepers')
        self.assertEqual(self.store.annotate(rid, 'group', ''), {})
        self.assertNotIn('metadata', self.store.by_id[rid])
        # The whole entry goes, rather than leaving {} behind for every record
        # that was ever labelled and then unlabelled.
        overlay = json.loads((Path(self.temp.name) / 'annotations.json').read_text('utf-8'))
        self.assertEqual(overlay, {})

    def test_annotating_does_not_reindex(self):
        """Metadata is in no trie, ranking or cache, so nothing computed goes
        stale and a label must not cost a reload."""
        rid = self.annotated()
        with patch.object(RecordStore, 'load') as reload:
            self.store.annotate(rid, 'group', 'keepers')
        reload.assert_not_called()
        self.assertFalse(self.store._dirty)

    def test_refuses_what_it_cannot_keep(self):
        rid = self.annotated()
        with self.assertRaises(ValueError):
            self.store.annotate('no-such-record', 'group', 'x')
        with self.assertRaises(ValueError):
            self.store.annotate(rid, '   ', 'x')
        with self.assertRaises(ValueError):
            self.store.annotate(rid, 'g' * 65, 'x')
        with self.assertRaises(ValueError):
            self.store.annotate(rid, 'group', 'v' * 513)
        for n in range(16):
            self.store.annotate(rid, f'k{n}', 'v')
        with self.assertRaises(ValueError):
            self.store.annotate(rid, 'one-too-many', 'v')

    def test_a_corrupt_overlay_does_not_take_the_store_down(self):
        self.annotated()
        (Path(self.temp.name) / 'annotations.json').write_text('{ not json', encoding='utf-8')
        restarted = RecordStore(sources=['completion_history.json'], root=self.temp.name).load()
        self.assertEqual(len(restarted.records), 1)
        self.assertEqual(restarted.annotations, {})


class HttpTests(StoreTestCase):
    def setUp(self):
        super().setUp()
        handler = type('TestHandler', (Handler,), {'store': self.store,
                                                  'log_message': lambda *args: None})
        self.http = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        # Default denies every upstream request, including accidental test calls.
        self.upstream = patch('server.urllib.request.urlopen', side_effect=AssertionError('external API forbidden')).start()
        self.addCleanup(patch.stopall)

    def stop_server(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()

    def call(self, path, body=None):
        conn = HTTPConnection('127.0.0.1', self.http.server_port, timeout=10)
        try:
            conn.request('GET' if body is None else 'POST', path,
                         body=None if body is None else json.dumps(body),
                         headers={'Content-Type': 'application/json'})
            res = conn.getresponse()
            return res.status, json.loads(res.read())
        finally:
            conn.close()

    def test_full_flow_and_cache_prevents_second_call(self):
        req, raw = fixture()
        def fake_response(request, **kwargs):
            sent = json.loads(request.data)
            self.assertNotIn('echo', sent)
            self.assertEqual(sent['logprobs'], 0)
            return io.BytesIO(json.dumps(raw).encode())
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'offline-test-placeholder'}):
            self.upstream.side_effect = fake_response
            status, result = self.call('/api/complete', {**req, 'confirmed': True})
        self.assertEqual(status, 200)
        self.assertTrue(result['saved'])
        self.assertEqual(result['choices'][0]['text'], '!')
        self.assertEqual(self.call('/api/records')[1]['records'][0]['id'], raw['id'])
        self.assertEqual(self.call('/api/record?id=test-1')[0], 200)
        self.assertTrue(self.call('/api/lookup', {**req, 'exact': True})[1]['hit'])
        self.assertTrue(self.call('/api/lookup', {**req, 'exact': True})[1]['exact'])
        self.assertTrue(self.call('/api/complete', {**req, 'confirmed': True})[1]['from_cache'])
        self.assertEqual(self.upstream.call_count, 1)

    def test_defer_reindexes_once_for_the_whole_batch(self):
        """Two paid calls, one rebuild, and both rows there when it is read.

        Reindexing after every record is what made a run of three hundred
        deviations take an hour: the rebuild cost several times the API call.
        """
        reindexes = []
        real_load = RecordStore.load

        def counted(store):
            reindexes.append(store)
            return real_load(store)

        with patch.object(RecordStore, 'load', counted), \
             patch.dict('os.environ', {'OPENAI_API_KEY': 'offline-test-placeholder'}):
            for n, prompt in enumerate(('a', 'b'), start=1):
                req, raw = fixture(rid=f'test-{n}', prompt=prompt)
                self.upstream.side_effect = (
                    lambda request, _raw=raw, **kwargs: io.BytesIO(json.dumps(_raw).encode()))
                status, result = self.call(
                    '/api/complete', {**req, 'confirmed': True, 'defer': True})
                self.assertEqual(status, 200)
                self.assertTrue(result['saved'])
                self.assertEqual(result['id'], raw['id'])
            self.assertEqual(reindexes, [], 'the batch rebuilt the indexes')
            listed = self.call('/api/records')[1]
            self.assertEqual(len(reindexes), 1, 'the read did not settle the store')

        self.assertEqual({r['id'] for r in listed['records']}, {'test-1', 'test-2'})
        self.assertEqual(self.upstream.call_count, 2)

    def test_defer_still_answers_a_repeat_from_history(self):
        """Deferring must not turn into paying twice for the same prompt."""
        req, raw = fixture()
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'offline-test-placeholder'}):
            self.upstream.side_effect = (
                lambda request, **kwargs: io.BytesIO(json.dumps(raw).encode()))
            first = self.call('/api/complete', {**req, 'confirmed': True, 'defer': True})[1]
            again = self.call('/api/complete', {**req, 'confirmed': True, 'defer': True})[1]
        self.assertTrue(first['saved'])
        self.assertTrue(again['from_cache'])
        self.assertEqual(self.upstream.call_count, 1)

    def test_annotate_over_http_and_read_the_overlay_back(self):
        req, raw = fixture()
        self.store.save(build_record(raw, req))
        status, result = self.call('/api/annotate', {'id': raw['id'], 'key': 'group', 'value': 'keepers'})
        self.assertEqual(status, 200)
        self.assertEqual(result['metadata'], {'group': 'keepers'})
        self.assertEqual(self.call('/api/annotations')[1]['annotations'],
                         {raw['id']: {'group': 'keepers'}})
        # The record answers with it too, which is what the token browser reads.
        self.assertEqual(self.call(f'/api/record?id={raw["id"]}')[1]['metadata'],
                         {'group': 'keepers'})
        self.assertEqual(self.call('/api/annotate', {'id': 'nope', 'key': 'group', 'value': 'x'})[0], 400)

    def test_positive_temperature_skips_history(self):
        for temperature in (.01, 1, 2):
            with self.subTest(temperature=temperature):
                req, raw = fixture(f'old-{temperature}')
                req['temperature'] = temperature
                self.store.save(build_record(raw, req))
                raw['id'] = f'new-{temperature}'
                self.upstream.side_effect = lambda *a, **kw: io.BytesIO(json.dumps(raw).encode())
                with patch.dict('os.environ', {'OPENAI_API_KEY': 'offline-test-placeholder'}), patch.object(self.store, 'lookup_request', side_effect=AssertionError('history must be skipped')):
                    status, result = self.call('/api/complete', {**req, 'confirmed': True})
                self.assertEqual(status, 200)
                self.assertFalse(result['from_cache'])
                self.assertTrue(result['saved'])
                self.assertEqual(result['id'], raw['id'])
        self.assertEqual(self.upstream.call_count, 3)

    def test_failed_save_returns_recoverable_response(self):
        req, raw = fixture()
        self.upstream.side_effect = lambda *a, **kw: io.BytesIO(json.dumps(raw).encode())
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'offline-test-placeholder'}), patch('store.os.replace', side_effect=OSError('full')):
            status, result = self.call('/api/complete', {**req, 'confirmed': True})
        self.assertEqual(status, 507)
        self.assertFalse(result['saved'])
        self.assertEqual(result['record']['raw_response'], raw)
        self.assertTrue(result['api_completed'])
        self.assertEqual(self.store.stats()['records'], 0)

    def test_unreachable_api_is_a_readable_refusal(self):
        """DNS down, no route, connection refused: the request never left.

        Only HTTPError was handled, so this escaped the handler as a 500 with an
        HTML body, and the page could not even parse the reason -- a network
        outage came out as "Unexpected token <". The one thing anyone wants to
        know when a paid call fails is whether it was paid for.
        """
        req, _ = fixture()
        import socket
        for failure in (urllib.error.URLError(socket.gaierror(11001, 'getaddrinfo failed')),
                        urllib.error.URLError('timed out'),
                        OSError('connection refused')):
            with self.subTest(failure=type(failure).__name__):
                self.upstream.side_effect = failure
                with patch.dict('os.environ', {'OPENAI_API_KEY': 'offline-test-placeholder'}):
                    status, result = self.call('/api/complete', {**req, 'confirmed': True})
                self.assertEqual(status, 503)
                self.assertTrue(result['unreachable'])
                self.assertFalse(result['saved'])
                self.assertFalse(result['api_completed'])
                self.assertIn('nic nebylo', result['error'])
                self.assertEqual(self.store.stats()['records'], 0)

    def test_bad_request_never_calls_api(self):
        req, _ = fixture()
        for body in [{**req, 'confirmed': 'yes'}, {**req, 'confirmed': True, 'top_p': 5}, []]:
            self.assertEqual(self.call('/api/complete', body)[0], 400)
        self.upstream.assert_not_called()



class PromptTokenizationTest(unittest.TestCase):
    """A caller that knows how the prompt is tokenised may say so.

    It is the difference between a bought deviation that every view can reach
    and one that sits under a node of its own, found by text lookup and dropped
    by every walk.
    """

    RAW = {'id': 'cmpl-x', 'created': 1, 'model': 'gpt-3.5-turbo-instruct',
           'choices': [{'text': ' b', 'index': 0, 'finish_reason': 'stop',
                        'logprobs': {'tokens': [' b'], 'token_logprobs': [-0.5],
                                     'top_logprobs': [{' b': -0.5}]}}],
           'usage': {}}

    def request(self, prompt):
        return normalize_request({'prompt': prompt, 'model': 'gpt-3.5-turbo-instruct'})

    def test_known_tokenization_is_kept(self):
        req = self.request('AB')
        rec = build_record(self.RAW, req, ['A', 'B'])
        self.assertEqual(rec['prompt']['logprobs']['tokens'], ['A', 'B'])
        self.assertEqual(rec['meta']['prompt_tokenization'], 'known')

    def test_opaque_span_without_it(self):
        req = self.request('AB')
        rec = build_record(self.RAW, req)
        self.assertEqual(rec['prompt']['logprobs']['tokens'], ['AB'])
        self.assertEqual(rec['meta']['prompt_tokenization'], 'opaque')

    def test_parts_that_do_not_join_are_refused(self):
        req = self.request('AB')
        for bad in (['A', 'C'], ['A'], ['A', 'B', ''], 'AB', [1, 2], ['AB', '']):
            with self.assertRaises(ValueError):
                build_record(self.RAW, req, bad)


if __name__ == '__main__':
    unittest.main()
