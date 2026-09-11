# Prompt Studio + GPT data

The Prompt Studio frontend reads the existing GPT record store from
`http://127.0.0.1:8899`.

## Start locally

1. In `..`, run `python server.py`.
2. In this directory (`prompt-studio`), run `npm install` once and then `npm run dev`.
3. Open the local URL printed by Prompt Studio.

The application checks the local record store only at temperature zero. Paid
API calls are disabled by default and require both enabling the switch in the
UI and confirming the individual request. The local server reads
`OPENAI_API_KEY` from its process environment; the key is never sent to the
browser or stored in this repository.

## Connected data

The Logs view uses `/api/stats` and `/api/records` (the latest 200 full records).
The Playground uses
`/api/lookup` before offering a new request only at temperature zero. The Explorer links to the original
greedy, completions, prefixes, and alternatives views.

## Persistence and recovery

Restart `server.py` after updating it, and reload Prompt Studio. The page requires
the server's exact-lookup acknowledgement at temperature zero before allowing a paid request.

Confirmed `/api/complete` requests check the history again on the server only at temperature zero. Matching
uses the exact prompt (including whitespace), model, token limit, temperature,
top P, penalties and logprobs. Analysis endpoints keep their existing lookup rules.

Successful responses are atomically appended to the existing
`../completion_history.json`. A file lock coordinates server writers,
and a shared in-process lock protects queries, indexes and cache warming. Existing
records are retained; saves reread the disk history under the lock. Other tools
that write this file must use the same locking protocol or run while the server
is stopped. Each server instance should be restarted to observe another process's
changes; use one server for normal operation.

The saved record includes exact request settings, all response fields, usage,
completion token data, the exact prompt text, and the unmodified `raw_response`.
Requests omit `echo`. As in the original viewer, a nonempty prompt is an opaque,
unscored span, not an API tokenization. Its token probabilities are unknown. New
records are visible immediately after saving, and survive server restarts.

On a save or response-conversion failure, the page retains the returned response,
offers a JSON download and blocks another send until the user acknowledges it.
It never automatically retries a paid request. Downloading is a recovery copy,
not confirmation that the record has been imported into the shared history.

Offline verification: run `python -m unittest test_completion -v` from the package root.
Tests use temporary histories and stub every upstream API request. No real key
or paid call is used, and the existing data files are not modified.
