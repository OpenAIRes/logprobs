# Logprobs — model-output exploration toolkit

This repository started as a small HTML viewer for token log probabilities. It has grown into a local toolkit for **recording, inspecting, ranking, perturbing, and resampling language-model outputs**.

The raw model calls are the primary data. Token trees, ranked strings, sweeps, deviations, greedy walks, and browser views are derived from the same persistent record store, so experiments can reuse observations that have already been collected instead of treating every view as a separate dataset.

## What is here

The repository has three closely related layers:

1. **Token/logprob viewers** — inspect generated tokens, their log probabilities, and alternative tokens at each position.
2. **Persistent search and ranking tools** — combine recorded calls into token tries, rank complete strings and prefixes, inspect deviations, and query previously observed continuations.
3. **Prompt resampling experiments** — generate variations of instructions, record the complete API interaction, inspect their probabilities when available, and feed interesting results back into later resampling runs.

A useful mental model is:

```text
model API calls
      |
      v
 recorded responses
      |
      v
  RecordStore
   /   |    \
 trie ranking deviations
   \   |    /
    browser views
         |
         v
   select / resample
         |
         +----> model API calls
```

The result is an empirical, reusable map of parts of the model's output space: observations can be browsed directly, compared by probability-derived scores, deliberately branched at token alternatives, and extended with additional calls when needed.

## Running the main tooling

Use the local server rather than a generic static server:

```bash
python3 server.py
```

Then open:

```text
http://127.0.0.1:8899/
```

`server.py` serves the browser tools and live queries from `store.py`. It binds to `127.0.0.1` deliberately because local API credentials may be present in the repository directory.

The original standalone logprob viewer is still available as `logprobs.html`.

## RecordStore: one source of truth

`store.py` is the central data layer. Instead of each viewer or experiment maintaining its own cache, the store loads the tracked call histories and provides shared indexes and queries over them.

The canonical sources currently include:

```text
completion_history.json
sweep_history.json
sweep2_history.json.gz
sweep3_ihave.json.gz
builder_history.json
meta_resample_root.json
```

From these records the store builds or serves:

- lookup by record ID and prompt;
- token tries;
- completion and prefix rankings;
- sweep indexes;
- token alternatives and deviations;
- greedy-path reconstruction;
- filtering by model and source group;
- cached query results for the browser views.

The history files are the primary artifacts. Rankings, trees, lists, and views should be treated as projections of those records rather than independent truth.

## Strings, prefixes, and ranking

The toolkit distinguishes between strings that genuinely finished and strings that are only prefixes of something longer.

A recorded path can end because:

- **`stop`** — the model ended the completion;
- **`length`** — generation hit the requested token limit;
- **`open`** — enumeration stopped at an internal trie node.

That distinction matters when interpreting scores. A sum of token log probabilities over a prefix is only a partial probability; comparing it as though it were a completed string systematically favors shorter prefixes.

For completed strings, sum log probability corresponds to sequence probability:

```text
P(string) = exp(sum_logprob)
```

The tools also expose length-normalized measures such as mean log probability / perplexity. These answer a different question from sequence probability: a long, individually predictable sequence can have excellent perplexity while still being much less probable as a complete string than a shorter alternative.

## Deviations and sweeps

When top-token alternatives are available, the tooling can branch from an observed generation by substituting alternatives at individual token positions and examining the resulting strings.

Sweeps are exhaustive or semi-exhaustive perturbation experiments around selected base generations. They are kept distinct from ordinary history because a large sweep can contribute thousands of closely related records and otherwise dominate rankings.

The server bounds variant generation by payload size. Enumerating every alternative of a long completion can create tens of thousands of full strings, so large requests are deliberately truncated rather than allowed to produce unbounded JSON responses.

## Prompt resampling

`resampling/` contains the prompt-resampling experiment. It began as a minimal implementation of the resampling step from *Large Language Models Are Human-Level Prompt Engineers* (Automatic Prompt Engineer / APE).

The basic operation is:

```text
Generate a variation of the following instruction while keeping the semantic meaning.

Input: [INSTRUCTION]
Output:
```

The subsystem now includes a web UI, CLI, backend abstraction, append-only prompt history, tests, logprob export, and meta-template support.

Run it with:

```bash
cd resampling
node server.mjs
```

Then open:

```text
http://127.0.0.1:8787/
```

See [`resampling/README.md`](resampling/README.md) for backend-specific parameters and detailed usage.

### Meta resampling

A generated result can be promoted into the template/instruction for a later run. This turns resampling into an iterative process rather than a single generation step:

```text
instruction
    |
    v
resample
  / | \
 A  B  C
    |
 inspect / rank
    |
 select a result
    |
 use as meta template
    |
 resample again
```

The prompt log records ancestry so later runs can be traced back to the template or result that produced them.

## Original logprobs viewer

`logprobs.html` displays token-level log probabilities and top alternatives from recorded model output. The repository supports both the newer chat-style logprob representation and legacy completion-style data used by the resampling experiments.

For large JSON datasets, `stream_logprobs.py` can process records incrementally instead of loading the complete file into memory:

```bash
python3 stream_logprobs.py logprobs.json --limit 5
```

## Repository orientation

Some useful entry points:

| Path | Purpose |
| --- | --- |
| `server.py` | Local HTTP/API server for the main viewers and live store queries |
| `store.py` | Canonical record store, indexes, filtering, and ranking queries |
| `logprobs.html` | Token/logprob viewer |
| `strings.html` | Unified string-list/ranking/deviations viewer |
| `best_avg_logprob_path.py` | Trie/search utilities used by ranking code |
| `export_dijkstra_top.py` | Ranking and entry-building logic used by the live store |
| `single_token_variants.py` | One-token deviation generation |
| `builder.html` | Interactive generation/builder tooling |
| `resampling/` | APE-style prompt resampling and meta-template experiments |

## Design principles

A few principles have emerged as the project has grown:

- **Recorded calls are valuable experimental data.** Persist them before building derived views.
- **Do not pay twice for an observation already in the store.** Prompt lookup is shared across generators and views.
- **Keep derived representations reproducible.** Rankings and tries should be rebuildable from tracked histories.
- **Do not confuse prefixes with completed strings.** Their probability scores have different meanings.
- **Keep sweep data identifiable.** Exhaustive perturbations should not silently overwhelm ordinary-history rankings.
- **Keep API-spending actions explicit.** Browser tooling distinguishes inspection from operations that make new model calls.

## Status

This is active experimental tooling rather than a packaged library. Interfaces and data formats may change as the experiments evolve. The root README is intended to describe the overall architecture; subsystem-specific details belong close to the relevant code, especially in `resampling/README.md` and source comments.
