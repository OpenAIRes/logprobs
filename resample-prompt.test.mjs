import assert from "node:assert/strict";
import test from "node:test";
import {
  BASE_RESAMPLING_PROMPT,
  buildOpenAIRequestBody,
  buildResamplingPrompt,
  extractVariations,
  getInstructionForMode,
  planRequests,
  resolveBackendParams,
  unsupportedParams,
} from "./resample-prompt.mjs";

test("builds the APE resampling prompt from the paper", () => {
  assert.equal(
    buildResamplingPrompt("write the antonym of the word."),
    [
      "Generate a variation of the following instruction while keeping the semantic meaning.",
      "",
      "Input: write the antonym of the word.",
      "Output:",
    ].join("\n"),
  );
});

test("rejects an empty instruction", () => {
  assert.throws(() => buildResamplingPrompt("  "), /Instruction is empty/);
});

test("meta mode uses the full base resampling prompt as INSTRUCTION", () => {
  assert.equal(
    getInstructionForMode({ instruction: "ignored", mode: "meta" }),
    BASE_RESAMPLING_PROMPT,
  );
});

test("builds the literal meta resampling prompt", () => {
  assert.equal(
    buildResamplingPrompt(BASE_RESAMPLING_PROMPT),
    [
      "Generate a variation of the following instruction while keeping the semantic meaning.",
      "",
      "Input: Generate a variation of the following instruction while keeping the semantic meaning.",
      "",
      "Input: [INSTRUCTION]",
      "Output:",
      "Output:",
    ].join("\n"),
  );
});

test("the prompt sent to the API has no trailing whitespace after Output:", () => {
  // llm.py strips every prompt before sending, so `Output:` ends the string.
  const prompt = buildResamplingPrompt("write the antonym of the word.");
  assert.equal(prompt, prompt.trimEnd());
  assert.ok(prompt.endsWith("Output:"));
});

test("completions backend defaults come from instruction_induction.yaml", () => {
  assert.deepEqual(resolveBackendParams("completions"), {
    temperature: 0.9,
    topP: 0.9,
    maxTokens: 50,
    frequencyPenalty: 0,
    presencePenalty: 0,
  });
});

test("responses backend sends nothing unless asked", () => {
  assert.deepEqual(resolveBackendParams("responses"), {});
});

test("explicit values override backend defaults, null forces omission", () => {
  assert.deepEqual(
    resolveBackendParams("completions", { temperature: 0, maxTokens: null }),
    {
      temperature: 0,
      topP: 0.9,
      frequencyPenalty: 0,
      presencePenalty: 0,
    },
  );
});

test("builds the faithful legacy completions request body", () => {
  assert.deepEqual(
    buildOpenAIRequestBody({
      backend: "completions",
      prompt: "Prompt",
      n: 30,
      params: resolveBackendParams("completions"),
    }),
    {
      model: "gpt-3.5-turbo-instruct",
      prompt: "Prompt",
      n: 30,
      temperature: 0.9,
      top_p: 0.9,
      max_tokens: 50,
      frequency_penalty: 0,
      presence_penalty: 0,
    },
  );
});

test("omits n when only one completion is requested", () => {
  const body = buildOpenAIRequestBody({
    backend: "completions",
    prompt: "Prompt",
    n: 1,
    params: {},
  });
  assert.equal("n" in body, false);
});

test("builds a bare Responses request body", () => {
  assert.deepEqual(
    buildOpenAIRequestBody({
      backend: "responses",
      prompt: "Prompt",
      params: resolveBackendParams("responses"),
    }),
    {
      model: "gpt-5.5",
      input: "Prompt",
    },
  );
});

test("Responses backend maps maxTokens and drops the penalties", () => {
  assert.deepEqual(
    buildOpenAIRequestBody({
      backend: "responses",
      prompt: "Prompt",
      model: "gpt-5.5",
      params: { temperature: 0.9, maxTokens: 50, frequencyPenalty: 0, presencePenalty: 0 },
    }),
    {
      model: "gpt-5.5",
      input: "Prompt",
      temperature: 0.9,
      max_output_tokens: 50,
    },
  );
});

test("reports which parameters a backend cannot send", () => {
  assert.deepEqual(
    unsupportedParams("responses", resolveBackendParams("completions")),
    ["frequencyPenalty", "presencePenalty"],
  );
  assert.deepEqual(unsupportedParams("completions", resolveBackendParams("completions")), []);
});

test("completions asks for every variation in one request, Responses loops", () => {
  assert.deepEqual(planRequests("completions", 5), [5]);
  assert.deepEqual(planRequests("responses", 5), [1, 1, 1, 1, 1]);
  assert.deepEqual(planRequests("completions", 1), [1]);
});

test("rejects an unknown backend", () => {
  assert.throws(() => planRequests("insert", 1), /Unknown backend/);
});

test("keeps completion text verbatim and orders choices by index", () => {
  const variations = extractVariations("completions", {
    choices: [
      { index: 1, text: " second variation.", finish_reason: "length" },
      { index: 0, text: " first variation.", finish_reason: "stop" },
    ],
  });

  assert.deepEqual(variations, [
    { raw: " first variation.", text: "first variation.", finishReason: "stop" },
    { raw: " second variation.", text: "second variation.", finishReason: "length" },
  ]);
});

test("reads a Responses payload from output_text or the output array", () => {
  assert.deepEqual(
    extractVariations("responses", { output_text: " a variation. ", status: "completed" }),
    [{ raw: " a variation. ", text: "a variation.", finishReason: "completed" }],
  );

  assert.deepEqual(
    extractVariations("responses", {
      output: [{ content: [{ type: "output_text", text: "a " }, { type: "output_text", text: "variation." }] }],
    }),
    [{ raw: "a variation.", text: "a variation.", finishReason: null }],
  );
});
