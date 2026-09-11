"""Validation and lossless conversion of Studio completions; no network I/O."""

import copy
import math

MODELS = {'gpt-3.5-turbo-instruct': 20, 'davinci-002': 5, 'babbage-002': 5}


def normalize_request(body):
    if not isinstance(body, dict) or not isinstance(body.get('prompt'), str):
        raise ValueError('prompt must be a string')
    model = body.get('model', 'gpt-3.5-turbo-instruct')
    if not isinstance(model, str) or model not in MODELS:
        raise ValueError('unsupported model')
    out = {'model': model, 'prompt': body['prompt']}
    for name, default, low, high, integer in (
        ('max_tokens', 20, 1, 4096, True),
        ('temperature', 0, 0, 2, False),
        ('top_p', 1, 0, 1, False),
        ('frequency_penalty', 0, -2, 2, False),
        ('presence_penalty', 0, -2, 2, False),
        ('logprobs', MODELS[model], 0, MODELS[model], True),
    ):
        value = body.get(name, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not low <= value <= high
                or (integer and int(value) != value)):
            raise ValueError(f'invalid {name}: expected {low}..{high}')
        out[name] = int(value) if integer else value
    return out


def build_record(raw, request):
    """Keep the response intact; prompts use the original viewer's opaque span."""
    if not isinstance(raw, dict) or not raw.get('id') or not raw.get('choices'):
        raise ValueError('invalid completion response')
    record = copy.deepcopy(raw)
    record['raw_response'] = copy.deepcopy(raw)
    record['request'] = dict(request)
    prompt = request['prompt']
    record['prompt'] = {'text': prompt, 'index': 0,
                        'logprobs': {'tokens': [prompt] if prompt else [],
                                     'token_logprobs': [], 'top_logprobs': []}}
    record['meta'] = {**(record.get('meta') or {}),
                      'prompt_tokenization': 'opaque' if prompt else 'empty'}
    return record
