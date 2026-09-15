"""Write metadata.children on every resampling record: 1 or >1.

The question the string asks: does it tell the model to produce ONE rewriting or
SEVERAL? That is the number of the output noun -- "a different version" against
"different versions" -- and nothing else. A plural in the INPUT does not count:
"a new version of the provided instructions" still asks for one version.

The rule is a list of plural output nouns plus one determiner-less plural that no
noun list catches ("Create new instructions based on the given input"). It is
checked against a count, so if the population changes and the rule stops matching
what it matched when it was written, this stops rather than mislabelling.
"""
import io, json, re, sys, urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = r'C:\Users\Jan\Desktop\gpt\extracted'
API = 'http://127.0.0.1:8899'

PLURAL_OUTPUT = re.compile(
    r'\b(versions|variations|variants|paraphrases|rewordings|alternatives'
    r'|renditions|restatements|rephrasings|phrasings|iterations)\b', re.I)
# "Create new instructions based on the given input." -- plural, no determiner,
# and "instructions" is the input noun everywhere else, so it cannot go in the
# list above without catching the 11 strings where it is the input.
BARE_PLURAL = re.compile(r'^\s*\w+\s+new\s+instructions\b', re.I)
# A set of instructions is more than one instruction, whichever side of the
# sentence it is on. Asking for "a new set of instructions" produces several;
# so does asking for "a new version of the provided set of instructions", because
# the version of a set is a set. The head noun is singular in both, which is
# exactly why the grammatical rule above misses them.
SET_OF = re.compile(r'\bset of instructions\b', re.I)
# The same thing said without the word "set": rewriting several instructions
# leaves you with several instructions. The multiplicity is in what was handed
# in rather than in what is asked for -- "a new version of the provided
# instructions" -- so every rule above misses it, and the output is plural all
# the same. The singular forms are untouched: \binstructions\b does not match
# "instruction", which is what the other 380-odd strings say.
PLURAL_INPUT = re.compile(
    r'\bversions?\s+of\b[\w\s,]{0,40}?\b(instructions|directions|guidelines|steps)\b', re.I)


def many(text):
    return bool(PLURAL_OUTPUT.search(text) or BARE_PLURAL.match(text)
                or SET_OF.search(text) or PLURAL_INPUT.search(text))


def is_resampling(rec):
    meta = rec.get('meta') or {}
    return (str(rec.get('id', '')).startswith('ape:')
            or meta.get('prompt_tokenization') == 'known')


def string_of(rec):
    """The string, as the deviations table means it: the given prompt is the
    first token and is not part of it; everything after it is."""
    tokens = ((rec.get('prompt') or {}).get('logprobs') or {}).get('tokens') or []
    return (''.join(tokens[1:]) + ((rec.get('choices') or [{}])[0].get('text') or '')).strip()


history = json.load(io.open(f'{ROOT}/completion_history.json', encoding='utf-8'))
records = [r for r in history if is_resampling(r)]
rows = [(r['id'], string_of(r)) for r in records]

plural_strings = sorted({t for _, t in rows if many(t)})
assert len(plural_strings) == 34, f'the rule now matches {len(plural_strings)} strings, not 34'
print(f'{len(rows)} resampling records, {len({t for _, t in rows})} distinct strings')
print(f'{len(plural_strings)} of them ask for more than one:\n')
for t in plural_strings:
    print('  >1  ', t)

if '--write' not in sys.argv:
    print('\n(dry run; pass --write to save)')
    raise SystemExit

sent = 0
for rid, text in rows:
    body = json.dumps({'id': rid, 'key': 'children', 'value': '>1' if many(text) else '1'}).encode()
    req = urllib.request.Request(f'{API}/api/annotate', data=body,
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=30) as response:
        json.load(response)
    sent += 1
print(f'\nwrote children on {sent} records')
