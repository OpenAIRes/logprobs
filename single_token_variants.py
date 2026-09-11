from sweep_alternatives import plan_for
import json
import sys


def variants_for(record):
    lp = record['choices'][0]['logprobs']
    tokens = lp.get('tokens') or []
    original = ''.join(tokens)
    grouped = {}
    for position, token, logprob in plan_for(record, 0):
        if position >= len(tokens) or token == tokens[position]:
            continue
        text = ''.join(tokens[:position] + [token] + tokens[position + 1:])
        change = {'position': position + 1, 'original': tokens[position],
                  'replacement': token, 'logprob': logprob}
        if text not in grouped:
            grouped[text] = {'text': text, 'changes': []}
        grouped[text]['changes'].append(change)
    return {'id': record['id'], 'original': original,
            'count': len(grouped), 'variants': list(grouped.values())}


if __name__ == '__main__':
    json.dump(variants_for(json.load(sys.stdin)), sys.stdout, ensure_ascii=True)
