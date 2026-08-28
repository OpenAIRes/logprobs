/**
 * Two ways of sending the APE resampling prompt to OpenAI.
 *
 * `completions` reproduces the mechanics of the reference implementation
 * (`automatic_prompt_engineer/llm.py`, `GPT_Forward.__generate_text`): a single
 * `openai.Completion.create` call carrying `n`, with the sampling parameters
 * taken literally from `experiments/configs/instruction_induction.yaml`.
 * The endpoint retires on 2026-09-28.
 *
 * `responses` is the modern path. Reasoning models reject several of the
 * parameters the paper uses, so its defaults are empty and the model decides.
 */

export const COMPLETIONS_DEADLINE = "2026-09-28";

export const BACKENDS = {
  completions: {
    id: "completions",
    label: "Legacy completions (faithful)",
    apiUrl: "https://api.openai.com/v1/completions",
    model: "gpt-3.5-turbo-instruct",
    promptField: "prompt",
    // instruction_induction.yaml, `generation.model.gpt_config`
    defaults: {
      temperature: 0.9,
      topP: 0.9,
      maxTokens: 50,
      frequencyPenalty: 0,
      presencePenalty: 0,
    },
    supported: ["temperature", "topP", "maxTokens", "frequencyPenalty", "presencePenalty"],
    // One request returns n completions, exactly like the reference code.
    supportsN: true,
    retiresOn: COMPLETIONS_DEADLINE,
  },
  responses: {
    id: "responses",
    label: "Responses API (modern)",
    apiUrl: "https://api.openai.com/v1/responses",
    model: "gpt-5.5",
    promptField: "input",
    defaults: {},
    // Verified 2026-08-28 against gpt-5.5: temperature and top_p are refused
    // outright ("Unsupported parameter"), not ignored — only the default
    // temperature of 1 is accepted, which makes sending it pointless.
    supported: ["maxTokens"],
    // No `n`; variations need one request each.
    supportsN: false,
    minMaxTokens: 16,
    caveats: [
      "Reasoning tokens count against max_output_tokens and are spent first:"
      + " 50 leaves about 11 tokens for the text, and 16 leaves none at all"
      + " (empty result, status incomplete). Leave it empty unless you have a reason.",
    ],
    retiresOn: null,
  },
};

export const DEFAULT_BACKEND = "completions";

/** Wire names per backend, so a param maps to the right key or is dropped. */
const WIRE_NAMES = {
  completions: {
    temperature: "temperature",
    topP: "top_p",
    maxTokens: "max_tokens",
    frequencyPenalty: "frequency_penalty",
    presencePenalty: "presence_penalty",
  },
  responses: {
    temperature: "temperature",
    topP: "top_p",
    maxTokens: "max_output_tokens",
  },
};

export const PARAM_NAMES = Object.keys(WIRE_NAMES.completions);

export function getBackend(id = DEFAULT_BACKEND) {
  const backend = BACKENDS[id];
  if (!backend) {
    throw new Error(`Unknown backend: ${id}. Use one of: ${Object.keys(BACKENDS).join(", ")}.`);
  }
  return backend;
}

/**
 * Explicit values win, then the backend defaults, then nothing (the API
 * decides). An explicit `null` forces omission even where a default exists.
 */
export function resolveBackendParams(backendId, overrides = {}) {
  const backend = getBackend(backendId);
  const resolved = {};

  for (const name of PARAM_NAMES) {
    const override = overrides[name];
    if (override === null) {
      continue;
    }
    const value = override === undefined ? backend.defaults[name] : override;
    if (value !== undefined) {
      resolved[name] = value;
    }
  }

  return resolved;
}

/** Params that carry a value but that this backend cannot send. */
export function unsupportedParams(backendId, params = {}) {
  const backend = getBackend(backendId);
  return Object.keys(params)
    .filter((name) => params[name] !== undefined)
    .filter((name) => !backend.supported.includes(name));
}

/**
 * How many requests, and with what `n`. The reference implementation asks for
 * all completions in one call; the Responses API has to loop.
 */
export function planRequests(backendId, count) {
  const backend = getBackend(backendId);
  if (!Number.isInteger(count) || count < 1) {
    throw new Error("Count must be a positive integer.");
  }
  return backend.supportsN ? [count] : Array.from({ length: count }, () => 1);
}

/** Lower bound the endpoint enforces, if any. */
export function minMaxTokens(backendId) {
  return getBackend(backendId).minMaxTokens ?? 1;
}

export function buildOpenAIRequestBody({ backend: backendId = DEFAULT_BACKEND, prompt, model, n = 1, params = {} } = {}) {
  const backend = getBackend(backendId);
  const wire = WIRE_NAMES[backend.id];
  const body = {
    model: model || backend.model,
    [backend.promptField]: prompt,
  };

  if (backend.supportsN && n > 1) {
    body.n = n;
  }

  for (const name of PARAM_NAMES) {
    const value = params[name];
    if (value === undefined || !wire[name] || !backend.supported.includes(name)) {
      continue;
    }
    body[wire[name]] = value;
  }

  return body;
}

function extractResponsesText(responseJson) {
  if (typeof responseJson.output_text === "string") {
    return responseJson.output_text;
  }

  const chunks = [];
  for (const item of responseJson.output ?? []) {
    for (const content of item.content ?? []) {
      if (content.type === "output_text" && typeof content.text === "string") {
        chunks.push(content.text);
      }
    }
  }

  return chunks.join("");
}

/**
 * `raw` is what the API returned verbatim. The reference implementation does
 * not strip completions, so leading whitespace after `Output:` is part of the
 * record; `text` is the trimmed form used for display.
 */
export function extractVariations(backendId, responseJson) {
  const backend = getBackend(backendId);

  if (backend.id === "completions") {
    const choices = [...(responseJson.choices ?? [])]
      .sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    return choices.map((choice) => {
      const raw = typeof choice.text === "string" ? choice.text : "";
      return { raw, text: raw.trim(), finishReason: choice.finish_reason ?? null };
    });
  }

  const raw = extractResponsesText(responseJson);
  return [{ raw, text: raw.trim(), finishReason: responseJson.status ?? null }];
}
