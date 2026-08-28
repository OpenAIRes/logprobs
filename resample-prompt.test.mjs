import assert from "node:assert/strict";
import test from "node:test";
import {
  BASE_RESAMPLING_PROMPT,
  buildOpenAIRequestBody,
  buildResamplingPrompt,
  extractVariations,
  getInstructionForMode,
  maxLogprobs,
  minMaxTokens,
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

test("completions sampling defaults come from instruction_induction.yaml", () => {
  // logprobs is ours, not the paper's, so it is checked separately below.
  const { logprobs, ...fromPaper } = resolveBackendParams("completions");
  assert.deepEqual(fromPaper, {
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
      logprobs: 5,
    },
  );
});

test("builds the faithful legacy completions request body", () => {
  // logprobs off gives a body literally identical to the paper's config.
  assert.deepEqual(
    buildOpenAIRequestBody({
      backend: "completions",
      prompt: "Prompt",
      n: 30,
      params: resolveBackendParams("completions", { logprobs: null }),
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

test("the default body adds only logprobs on top of the paper's config", () => {
  const body = buildOpenAIRequestBody({
    backend: "completions",
    prompt: "Prompt",
    n: 30,
    params: resolveBackendParams("completions"),
  });
  assert.equal(body.logprobs, 5);
  // Observational only: it cannot change which tokens get generated.
  assert.equal(body.temperature, 0.9);
  assert.equal(body.top_p, 0.9);
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

test("Responses backend renames maxTokens to max_output_tokens", () => {
  assert.deepEqual(
    buildOpenAIRequestBody({
      backend: "responses",
      prompt: "Prompt",
      model: "gpt-5.5",
      params: { maxTokens: 200 },
    }),
    {
      model: "gpt-5.5",
      input: "Prompt",
      max_output_tokens: 200,
    },
  );
});

test("reports which parameters a backend cannot send", () => {
  // Feeding the completions config to the modern backend leaves only max_tokens.
  assert.deepEqual(
    unsupportedParams("responses", resolveBackendParams("completions")),
    ["temperature", "topP", "frequencyPenalty", "presencePenalty", "logprobs"],
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
    { raw: " first variation.", text: "first variation.", finishReason: "stop", logprob: null, probability: null },
    { raw: " second variation.", text: "second variation.", finishReason: "length", logprob: null, probability: null },
  ]);
});

test("reads a Responses payload from output_text or the output array", () => {
  assert.deepEqual(
    extractVariations("responses", { output_text: " a variation. ", status: "completed" }),
    [{ raw: " a variation. ", text: "a variation.", finishReason: "completed", logprob: null, probability: null }],
  );

  assert.deepEqual(
    extractVariations("responses", {
      output: [{ content: [{ type: "output_text", text: "a " }, { type: "output_text", text: "variation." }] }],
    }),
    [{ raw: "a variation.", text: "a variation.", finishReason: null, logprob: null, probability: null }],
  );
});

test("the Responses backend advertises only max_output_tokens", () => {
  // Verified against gpt-5.5: temperature and top_p return HTTP 400
  // "Unsupported parameter", so they must not be offered.
  assert.deepEqual(
    unsupportedParams("responses", { temperature: 0.9, topP: 0.9, maxTokens: 200 }),
    ["temperature", "topP"],
  );

  assert.deepEqual(
    buildOpenAIRequestBody({
      backend: "responses",
      prompt: "Prompt",
      params: { temperature: 0.9, topP: 0.9, maxTokens: 200 },
    }),
    { model: "gpt-5.5", input: "Prompt", max_output_tokens: 200 },
  );
});

test("exposes the endpoint's max_output_tokens floor", () => {
  assert.equal(minMaxTokens("responses"), 16);
  assert.equal(minMaxTokens("completions"), 1);
});

test("completions asks for logprobs by default, Responses cannot", () => {
  assert.equal(resolveBackendParams("completions").logprobs, 5);
  assert.equal(resolveBackendParams("responses").logprobs, undefined);
  assert.equal(maxLogprobs("completions"), 20);
  assert.equal(maxLogprobs("responses"), 0);

  // Verified against gpt-5.5: "logprobs are not supported with reasoning models."
  assert.deepEqual(unsupportedParams("responses", { logprobs: 5 }), ["logprobs"]);
  assert.equal(
    "logprobs" in buildOpenAIRequestBody({ backend: "responses", prompt: "P", params: { logprobs: 5 } }),
    false,
  );
});

test("logprobs 0 is sent, not treated as absent", () => {
  // 0 still returns the chosen tokens' probabilities, which is what P(sequence)
  // needs, so it must survive the falsy checks.
  const body = buildOpenAIRequestBody({
    backend: "completions",
    prompt: "P",
    params: resolveBackendParams("completions", { logprobs: 0 }),
  });
  assert.equal(body.logprobs, 0);

  const off = buildOpenAIRequestBody({
    backend: "completions",
    prompt: "P",
    params: resolveBackendParams("completions", { logprobs: null }),
  });
  assert.equal("logprobs" in off, false);
});

test("sums token logprobs into an exact sequence probability", () => {
  const [variation] = extractVariations("completions", {
    choices: [{
      index: 0,
      text: " provide the opposite.",
      finish_reason: "stop",
      logprobs: { token_logprobs: [-1.5, -0.5, -0.25] },
    }],
  });

  assert.equal(variation.logprob, -2.25);
  assert.ok(Math.abs(variation.probability - Math.exp(-2.25)) < 1e-12);
});

test("reports no probability when logprobs were not requested", () => {
  const [none] = extractVariations("completions", {
    choices: [{ index: 0, text: " x", finish_reason: "stop", logprobs: null }],
  });
  assert.equal(none.logprob, null);
  assert.equal(none.probability, null);

  // echo mode can return a null for the first token; refuse to sum that.
  const [partial] = extractVariations("completions", {
    choices: [{ index: 0, text: " x", logprobs: { token_logprobs: [null, -0.5] } }],
  });
  assert.equal(partial.logprob, null);
});
