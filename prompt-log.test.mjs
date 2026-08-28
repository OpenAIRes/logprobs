import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  appendPromptLogEvent,
  promptRowsFromEvents,
  readPromptLog,
} from "./prompt-log.mjs";
import { BASE_RESAMPLING_PROMPT } from "./resample-prompt.mjs";

async function tempLogPath() {
  const dir = await mkdtemp(join(tmpdir(), "ape-log-"));
  return join(dir, "prompt-log.json");
}

test("creates a seeded prompt log on first read", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ape-log-"));
  const logPath = join(dir, "prompt-log.json");

  const entries = await readPromptLog(logPath);

  assert.equal(entries.length, 1);
  assert.equal(entries[0].id, "seed-resampling-prompt");
  assert.equal(entries[0].type, "seed");
  assert.deepEqual(entries[0].run, {
    id: "seed-run",
    requestedCount: 1,
    index: 1,
    total: 1,
  });
  assert.equal(entries[0].generatedPrompts[0].prompt, BASE_RESAMPLING_PROMPT);
  assert.equal(entries[0].parentPrompt, null);
});

test("keeps the seed prompt first when an existing log is read", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ape-log-"));
  const logPath = join(dir, "prompt-log.json");

  await appendPromptLogEvent({
    mode: "custom",
    model: "test-model",
    parentPrompt: "Parent prompt.",
    parentInstruction: "Parent instruction.",
    request: {
      url: "https://api.example.test/v1/responses",
      body: { model: "test-model", input: "Parent prompt." },
    },
    response: {
      status: 200,
      body: { output_text: "First generated prompt." },
    },
    generatedPrompts: [
      {
        prompt: "First generated prompt.",
        parentPrompt: "Parent prompt.",
        parentInstruction: "Parent instruction.",
      },
    ],
  }, logPath);

  const entries = await readPromptLog(logPath);

  assert.equal(entries[0].id, "seed-resampling-prompt");
  assert.equal(entries[0].generatedPrompts[0].prompt, BASE_RESAMPLING_PROMPT);
  assert.equal(typeof entries[1].run.id, "string");
  assert.deepEqual({
    requestedCount: entries[1].run.requestedCount,
    index: entries[1].run.index,
    total: entries[1].run.total,
  }, {
    requestedCount: 1,
    index: 1,
    total: 1,
  });
});

test("appends full OpenAI events with request and parsed response JSON", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ape-log-"));
  const logPath = join(dir, "prompt-log.json");

  const responseBody = {
    id: "resp_123",
    output: [
      {
        content: [
          {
            type: "output_text",
            text: "Rewrite the instruction without changing the task.",
          },
        ],
      },
    ],
  };

  const entries = await appendPromptLogEvent({
    type: "openai_resample",
    mode: "custom",
    model: "test-model",
    parentPrompt: "Generate a variation...\n\nInput: original\nOutput:",
    parentInstruction: "original",
    run: {
      id: "run-123",
      requestedCount: 5,
      index: 3,
      total: 5,
    },
    request: {
      url: "https://api.example.test/v1/responses",
      body: {
        model: "test-model",
        input: "Generate a variation...\n\nInput: original\nOutput:",
      },
    },
    response: {
      status: 200,
      body: responseBody,
    },
    generatedPrompts: [
      {
        prompt: "Rewrite the instruction without changing the task.",
        parentPrompt: "Generate a variation...\n\nInput: original\nOutput:",
        parentInstruction: "original",
      },
    ],
  }, logPath);

  assert.equal(entries.length, 2);
  assert.equal(entries[1].mode, "custom");
  assert.equal(entries[1].model, "test-model");
  assert.deepEqual(entries[1].request.body, {
    model: "test-model",
    input: "Generate a variation...\n\nInput: original\nOutput:",
  });
  assert.deepEqual(entries[1].response.body, responseBody);
  assert.deepEqual(entries[1].run, {
    id: "run-123",
    requestedCount: 5,
    index: 3,
    total: 5,
  });
  assert.equal(entries[1].parentInstruction, "original");
  assert.match(entries[1].parentPrompt, /Input: original/);
});

test("derives prompt table rows from complete log events", async () => {
  const events = [
    {
      id: "event-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      mode: "custom",
      model: "test-model",
      request: {
        url: "https://api.example.test/v1/responses",
        body: {
          model: "test-model",
          input: "Parent prompt",
        },
      },
      generatedPrompts: [
        {
          prompt: "Generated prompt",
          parentPrompt: "Parent prompt",
          parentInstruction: "Parent instruction",
        },
      ],
    },
  ];

  assert.deepEqual(promptRowsFromEvents(events), [
    {
      id: "event-1:0:Generated prompt",
      eventId: "event-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      mode: "custom",
      backend: null,
      model: "test-model",
      request: {
        url: "https://api.example.test/v1/responses",
        body: {
          model: "test-model",
          input: "Parent prompt",
        },
      },
      response: null,
      variationCount: 1,
      prompt: "Generated prompt",
      rawPrompt: "Generated prompt",
      finishReason: null,
      parentPrompt: "Parent prompt",
      parentInstruction: "Parent instruction",
    },
  ]);
});

test("rows carry the response and how many variations shared it", () => {
  const rows = promptRowsFromEvents([
    {
      id: "event-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      mode: "custom",
      backend: "completions",
      model: "gpt-3.5-turbo-instruct",
      request: null,
      response: {
        status: 200,
        body: { usage: { prompt_tokens: 25, completion_tokens: 21, total_tokens: 46 } },
      },
      generatedPrompts: [
        { prompt: "First", finishReason: "stop" },
        { prompt: "Second", finishReason: "length" },
      ],
    },
  ]);

  assert.equal(rows.length, 2);
  // Usage is per request, so both rows point at the same response object.
  assert.equal(rows[0].response.body.usage.completion_tokens, 21);
  assert.deepEqual(rows.map((r) => r.variationCount), [2, 2]);
  assert.deepEqual(rows.map((r) => r.finishReason), ["stop", "length"]);
});

test("row ids stay unique when one completions request returns duplicates", () => {
  // The completions backend returns n variations per event, and n=30 at
  // temperature 0.9 does produce exact duplicates.
  const rows = promptRowsFromEvents([
    {
      id: "event-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      mode: "custom",
      backend: "completions",
      model: "gpt-3.5-turbo-instruct",
      request: null,
      generatedPrompts: [
        { prompt: "Same variation", parentPrompt: null, parentInstruction: null },
        { prompt: "Same variation", parentPrompt: null, parentInstruction: null },
      ],
    },
  ]);

  assert.equal(rows.length, 2);
  assert.notEqual(rows[0].id, rows[1].id);
  assert.equal(rows[0].backend, "completions");
});

test("events written before the backend split are marked as responses", async () => {
  const logPath = await tempLogPath();
  const entries = await appendPromptLogEvent({
    type: "openai_resample",
    mode: "custom",
    model: "gpt-5.5",
    generatedPrompts: [{ prompt: "A variation" }],
  }, logPath);

  // Strip the field the way the pre-split log had it, then read it back.
  const withoutBackend = entries.map(({ backend, ...rest }) => rest);
  await writeFile(logPath, `${JSON.stringify(withoutBackend, null, 2)}\n`, "utf8");

  const reread = await readPromptLog(logPath);
  assert.equal(reread[1].backend, "responses");
  assert.equal(reread[0].backend, null, "the seed predates any API call");
});
