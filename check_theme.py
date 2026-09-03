"""Assert the two dark palettes in app.css say the same thing.

The dark tokens are listed twice on purpose -- once under
`@media (prefers-color-scheme: dark)` for the system setting, once under
`:root[data-theme="dark"]` for an explicit choice from the bar. Plain CSS has no
way to share one block between a media query and an attribute selector, so the
duplication is unavoidable; what is avoidable is the two drifting apart, which
would show as a palette that changes depending on how you arrived at dark.

    python3 check_theme.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BLOCK = re.compile(
    r'(?::root:not\(\[data-theme="light"\]\)|:root\[data-theme="dark"\]) \{\n(.*?)\n\}',
    re.S)
TOKEN = re.compile(r'(--[\w-]+):\s*([^;]+);')


def main() -> int:
    css = open(os.path.join(ROOT, 'app.css'), encoding='utf-8').read()
    blocks = BLOCK.findall(css)
    if len(blocks) != 2:
        print(f'FAIL: expected 2 dark palettes in app.css, found {len(blocks)}')
        return 1

    system, explicit = (dict(TOKEN.findall(b)) for b in blocks)
    if system == explicit:
        print(f'ok: both dark palettes list the same {len(system)} tokens')
        return 0

    print('FAIL: the dark palettes have drifted')
    for key in sorted(set(system) | set(explicit)):
        a, b = system.get(key), explicit.get(key)
        if a != b:
            print(f'  {key}: system={a!r}  explicit={b!r}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
