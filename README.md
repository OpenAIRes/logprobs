# Logprobs Viewer

This repository provides a small HTML viewer for exploring token log probabilities returned by OpenAI's Chat Completion API.

## Opening the viewer

Serve the directory with any static file server so the service worker can load:

```bash
python3 -m http.server
```

Then open [http://localhost:8000/logprobs.html](http://localhost:8000/logprobs.html) in your browser. The page fetches `logprobs.json` and displays each token along with its top alternatives.

You can also open `logprobs.html` directly from the filesystem, but the service worker will be disabled.

## What the data represents

`logprobs.json` is the raw response from an OpenAI chat completion request made with log probability support enabled. Each entry in `choices[0].logprobs.content` corresponds to a token of the assistant reply. For every token you can inspect its log probability and the probabilities of the highest ranked alternative tokens.

## Generating your own JSON

Use the OpenAI Python library to request a chat completion with logprobs turned on and save the result:

```python
import json
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hi"}],
    logprobs=True,
    top_logprobs=20,
    temperature=0,
)

with open("logprobs.json", "w") as f:
    json.dump(response, f)
```

Place the generated `logprobs.json` next to `logprobs.html` and reload the page to inspect your own data.
