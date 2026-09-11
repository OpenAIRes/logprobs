"""Offline regression tests for the recursive greedy departure search."""
import unittest
from types import SimpleNamespace
from store import RecordStore
from export_dijkstra_top import walk_detail


class GreedyITest(unittest.TestCase):
    def fixture(self):
        def node(lp=None, tops=None):
            return SimpleNamespace(logprob=lp, top_logprobs=tops or {}, children={}, source=None)
        root = node(tops={'a': -0.1, 'b': -0.8, 'c': -1.3})
        for token, lp in root.top_logprobs.items():
            root.children[token] = node(lp)
        root.children['a'].top_logprobs = {'x': -0.1, 'y': -2.0}
        root.children['b'].top_logprobs = {'x': -0.1, 'y': -0.3}
        for token in ('a', 'b'):
            for tail, lp in root.children[token].top_logprobs.items():
                root.children[token].children[tail] = node(lp)
        root.children['c'].children['x'] = node(-0.1)
        detail = walk_detail(root, ['a', 'x'])
        seed = dict(tokens=detail, text='ax', is_greedy=True, cost=0, deviation=None,
                    n=2, sum_logprob=-0.2, mean_logprob=-0.1, perplexity=1.1,
                    end='stop', node_kind='leaf')
        store = RecordStore()
        store._alts_cache[('', 'test', 8, None)] = (None, [seed], 0)
        store.trie = lambda *args: root
        store.greedy_alternatives = lambda **kwargs: {}
        store.node_kind = lambda *args: 'leaf'
        def record(prompt, tokens):
            return {'id': prompt, 'choices': [{'finish_reason': 'stop', 'logprobs': {'tokens': tokens}}]}
        records = {'b': record('b', ['x']), 'c': record('c', ['x']),
                   'ay': record('ay', []), 'by': record('by', []),
                   'a': record('a', ['x']), 'bx': record('bx', [])}
        store.lookup = lambda prompt, **kwargs: records.get(prompt)
        return store

    def test_prefix_score_and_recursive_parent(self):
        result = self.fixture().greedy_i(model='test', top=4)
        self.assertEqual([e['text'] for e in result['entries']], ['ax', 'bx', 'by', 'cx'])
        self.assertEqual(result['entries'][2]['deviation']['parent_rank'], 2)
        self.assertAlmostEqual(result['entries'][2]['departure_logprob'], -1.1)

    def test_cycles_duplicates_and_exhaustion(self):
        result = self.fixture().greedy_i(model='test', top=20)
        self.assertEqual([e['text'] for e in result['entries']], ['ax', 'bx', 'by', 'cx', 'ay'])

    def test_first_string_is_fixed(self):
        self.assertEqual([e['text'] for e in self.fixture().greedy_i(model='test', top=1)['entries']], ['ax'])


if __name__ == '__main__':
    unittest.main()
