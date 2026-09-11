import unittest
from single_token_variants import variants_for

class VariantsTest(unittest.TestCase):
    def test_exactly_one_change_and_escaping(self):
        record = {'id':'test','choices':[{'logprobs':{'tokens':['a',' b','\n'], 'top_logprobs':[{'a':-1,'<x>':-2},{' b':-1,' c':-2},{'\n':-1,'!':-2}]}}]}
        result = variants_for(record)
        self.assertEqual({v['text'] for v in result['variants']},{'<x> b\n','a c\n','a b!'})
        self.assertEqual(result['count'],3)
        self.assertEqual(record['choices'][0]['logprobs']['tokens'],['a',' b','\n'])
    def test_no_alternatives(self):
        self.assertEqual(variants_for({'id':'empty','choices':[{'logprobs':{'tokens':['a'],'top_logprobs':[]}}]})['count'],0)
    def test_deduplicates_strings_but_keeps_provenance(self):
        r={'id':'duplicate','choices':[{'logprobs':{'tokens':['a','a'],'top_logprobs':[{'':-2},{'':-3}]}}]}
        v=variants_for(r)
        self.assertEqual(v['count'],1)
        self.assertEqual(len(v['variants'][0]['changes']),2)

if __name__=='__main__': unittest.main()
