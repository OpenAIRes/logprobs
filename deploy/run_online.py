"""Run the private RecordStore service and the public resampling UI together.

The Python store stays bound to loopback. Only the Node UI is exposed by the
container, so the root repository (history files, local state, credentials) is
not directly available as static content on the public port.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_PORT = os.environ.get("PORT", "8080")
STORE_PORT = os.environ.get("LOGPROBS_STORE_PORT", "8899")


def start(*args, env=None):
    return subprocess.Popen(args, cwd=ROOT, env=env or os.environ.copy())


def main():
    store = start(sys.executable, "server.py", "--host", "127.0.0.1", "--port", STORE_PORT)

    node_env = os.environ.copy()
    node_env["PORT"] = PUBLIC_PORT
    node_env["HOST"] = "0.0.0.0"
    node_env["LOGPROBS_STORE_ORIGIN"] = f"http://127.0.0.1:{STORE_PORT}"
    node_env["LOGPROBS_VIEWER_DIR"] = ROOT
    ui = start("node", "resampling/server.mjs", env=node_env)

    children = [store, ui]

    def stop(*_):
        for child in children:
            if child.poll() is None:
                child.terminate()
        deadline = time.time() + 8
        for child in children:
            try:
                child.wait(max(0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                child.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while True:
        for child in children:
            code = child.poll()
            if code is not None:
                stop()
                return code
        time.sleep(0.5)


if __name__ == "__main__":
    main()
