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


def build_record(raw, request, prompt_tokens=None):
    """Keep the response intact; prompts use the original viewer's opaque span.

    Unless the caller knows better. A deviation is bought with a prompt it built
    itself -- the base record's prompt, then that record's own generated tokens
    up to the position, then the alternative -- so its tokenisation is not a
    guess. Storing it as one opaque span instead put the record under a node of
    its own in the trie, where nothing could reach it: the lookup found it by
    text, the walk could not score a single token of it, and the deviation was
    dropped as unscorable. A record that had been paid for looked, to every
    view, like one that had never been made.

    Accepted only when the parts join back to exactly the prompt that was sent.
    """
    if prompt_tokens is not None:
        # Non-empty, because an empty string is not a token and would put an
        # empty child in the trie that every walk then has to step over.
        if (not isinstance(prompt_tokens, list)
                or not all(isinstance(t, str) and t for t in prompt_tokens)
                or ''.join(prompt_tokens) != request['prompt']):
            raise ValueError('prompt_tokens must be non-empty strings that join to the prompt')
    if not isinstance(raw, dict) or not raw.get('id') or not raw.get('choices'):
        raise ValueError('invalid completion response')
    record = copy.deepcopy(raw)
    record['raw_response'] = copy.deepcopy(raw)
    record['request'] = dict(request)
    prompt = request['prompt']
    tokens = prompt_tokens if prompt_tokens is not None else ([prompt] if prompt else [])
    record['prompt'] = {'text': prompt, 'index': 0,
                        'logprobs': {'tokens': tokens,
                                     'token_logprobs': [], 'top_logprobs': []}}
    record['meta'] = {**(record.get('meta') or {}),
                      'prompt_tokenization': ('known' if prompt_tokens is not None
                                              else 'opaque' if prompt else 'empty')}
    return record
