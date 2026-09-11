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

    def test_bad_request_never_calls_api(self):
        req, _ = fixture()
        for body in [{**req, 'confirmed': 'yes'}, {**req, 'confirmed': True, 'top_p': 5}, []]:
            self.assertEqual(self.call('/api/complete', body)[0], 400)
        self.upstream.assert_not_called()


if __name__ == '__main__':
    unittest.main()
