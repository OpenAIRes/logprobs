#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import {
  BACKENDS,
  buildOpenAIRequestBody,
  COMPLETIONS_DEADLINE,
  DEFAULT_BACKEND,
  extractVariations,
  getBackend,
  maxLogprobs,
  minMaxTokens,
  PARAM_NAMES,
  planRequests,
  resolveBackendParams,
  unsupportedParams,
} from "./backends.mjs";

export {
  BACKENDS,
  buildOpenAIRequestBody,
  COMPLETIONS_DEADLINE,
  DEFAULT_BACKEND,
  extractVariations,
  getBackend,
  maxLogprobs,
  minMaxTokens,
  planRequests,
  resolveBackendParams,
  unsupportedParams,
};

export const RESAMPLING_INSTRUCTION =
  "Generate a variation of the following instruction while keeping the semantic meaning.";

/**
 * The slot a resampling template puts the instruction into. It used to appear
 * only as a literal inside BASE_RESAMPLING_PROMPT, while buildResamplingPrompt
 * pasted `Input: ${instruction}` together by hand — so nothing ever checked that
 * a template had a slot at all. That starts to matter once the template is not
 * hard-coded: meta mode comes back with the paraphrased instruction sentence and
 * drops the `Input:`/`Output:` scaffolding, and such a template would build a
 * prompt that never states what to resample.
 */
export const INSTRUCTION_PLACEHOLDER = "[INSTRUCTION]";

export const BASE_RESAMPLING_PROMPT = [
  RESAMPLING_INSTRUCTION,
  "",
  `Input: ${INSTRUCTION_PLACEHOLDER}`,
  "Output:",
].join("\n");

/** A fatal defect: the template cannot be used at all. Null means usable. */
export function templateError(template) {
  const text = String(template ?? "");
  if (!text.trim()) {
    return "The resampling template is empty.";
  }
  if (!text.includes(INSTRUCTION_PLACEHOLDER)) {
    return `The resampling template has no ${INSTRUCTION_PLACEHOLDER} slot, so the instruction would never reach the prompt.`;
  }
  return null;
}

/**
 * Defects worth saying out loud but not worth refusing over: a template may
 * legitimately end in some cue other than the paper's.
 */
export function templateWarnings(template) {
  const text = String(template ?? "");
  const warnings = [];
  if (text.trim() && !text.trimEnd().endsWith("Output:")) {
    warnings.push('The template does not end with "Output:", the cue the model completes after.');
  }
  if (text && text !== text.trimEnd()) {
    warnings.push("The template ends with whitespace; the reference implementation strips prompts before sending.");
  }
  return warnings;
}

export function buildResamplingPrompt(instruction, template = BASE_RESAMPLING_PROMPT) {
  const cleaned = String(instruction ?? "").trim();
  if (!cleaned) {
    throw new Error("Instruction is empty.");
  }

  const problem = templateError(template);
  if (problem) {
    throw new Error(problem);
  }

  // The replacer function keeps `$&` and friends in the instruction literal, and
  // stops a meta-mode instruction that itself contains the placeholder from being
  // rescanned: it is inserted once, verbatim.
  return String(template).replaceAll(INSTRUCTION_PLACEHOLDER, () => cleaned);
}

/**
 * Meta mode resamples the template it is about to use, so a caller that swapped
 * in a generated template keeps iterating on that one, not on the paper's.
 */
export function getInstructionForMode({
  instruction,
  mode = "custom",
  template = BASE_RESAMPLING_PROMPT,
} = {}) {
  if (mode === "meta") {
    return template;
  }
  return instruction;
}

function printHelp() {
  const backendList = Object.values(BACKENDS)
    .map((backend) => `    ${backend.id.padEnd(12)} ${backend.label} — ${backend.model}`)
    .join("\n");

  console.log(`Usage:
  node resample-prompt.mjs --instruction "write the antonym of the word."
  node resample-prompt.mjs --instruction "..." --backend responses
  node resample-prompt.mjs --input-file instruction.txt --count 5 --dry-run
  node resample-prompt.mjs -i "..." --template-file my-template.txt --dry-run

Backends (--backend, default: ${DEFAULT_BACKEND}):
${backendList}

  The completions endpoint retires on ${COMPLETIONS_DEADLINE}.

Options:
  --instruction, -i        Source instruction to resample
  --input-file, -f         Read source instruction from a UTF-8 text file
  --template               Resampling template; must contain [INSTRUCTION]
  --template-file          Read the template from a UTF-8 text file
  --count, -n              Number of variations to request (default: 1)
  --backend                completions | responses
  --model                  Override the backend's default model
  --temperature            Sampling temperature ("off" omits it)
  --top-p                  Nucleus sampling ("off" omits it)
  --max-tokens             Token limit per variation ("off" omits it)
  --frequency-penalty      Frequency penalty ("off" omits it)
  --presence-penalty       Presence penalty ("off" omits it)
  --logprobs               Alternatives per token, 0-20 ("off" omits it).
                           0 still returns the chosen tokens' probabilities,
                           which is all P(sequence) needs. completions only —
                           reasoning models refuse logprobs outright.
  --api-url                Override the backend's endpoint
  --dry-run                Print the prompt and request body without calling the API
  --json                   Print a JSON result
  --help, -h               Show this help

Environment:
  OPENAI_API_KEY           Required unless --dry-run is used
  OPENAI_BACKEND           Optional backend override
  OPENAI_MODEL             Optional model override
`);
}

/** `off` means "do not send this parameter at all", vs. omitting the flag. */
function parseParamValue(arg, value, { integer = false } = {}) {
  if (value === "off" || value === "none") {
    return null;
  }
  const parsed = integer ? Number.parseInt(value, 10) : Number.parseFloat(value);
  if (!Number.isFinite(parsed) || (integer && !Number.isInteger(parsed))) {
    throw new Error(`${arg} must be a number or "off".`);
  }
  return parsed;
}

function parseArgs(argv) {
  const options = {
    count: 1,
    backend: process.env.OPENAI_BACKEND || DEFAULT_BACKEND,
    model: process.env.OPENAI_MODEL,
    apiUrl: process.env.OPENAI_API_URL,
    overrides: {},
    dryRun: false,
    json: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      i += 1;
      if (i >= argv.length) {
        throw new Error(`Missing value for ${arg}.`);
      }
      return argv[i];
    };

    switch (arg) {
      case "--instruction":
      case "-i":
        options.instruction = next();
        break;
      case "--input-file":
      case "-f":
        options.inputFile = next();
        break;
      case "--template":
        options.template = next();
        break;
      case "--template-file":
        options.templateFile = next();
        break;
      case "--count":
      case "-n":
        options.count = Number.parseInt(next(), 10);
        break;
      case "--backend":
        options.backend = next();
        break;
      case "--model":
        options.model = next();
        break;
      case "--temperature":
        options.overrides.temperature = parseParamValue(arg, next());
        break;
      case "--top-p":
        options.overrides.topP = parseParamValue(arg, next());
        break;
      case "--max-tokens":
      case "--max-output-tokens":
        options.overrides.maxTokens = parseParamValue(arg, next(), { integer: true });
        break;
      case "--frequency-penalty":
        options.overrides.frequencyPenalty = parseParamValue(arg, next());
        break;
      case "--presence-penalty":
        options.overrides.presencePenalty = parseParamValue(arg, next());
        break;
      case "--logprobs":
        options.overrides.logprobs = parseParamValue(arg, next(), { integer: true });
        break;
      case "--api-url":
        options.apiUrl = next();
        break;
      case "--dry-run":
        options.dryRun = true;
        break;
      case "--json":
        options.json = true;
        break;
      case "--help":
      case "-h":
        options.help = true;
        break;
      default:
        throw new Error(`Unknown option: ${arg}`);
    }
  }

  if (!Number.isInteger(options.count) || options.count < 1) {
    throw new Error("--count must be a positive integer.");
  }
  getBackend(options.backend);

  return options;
}

async function getTemplate(options) {
  if (options.template && options.templateFile) {
    throw new Error("Use either --template or --template-file, not both.");
  }
  const template = options.templateFile
    ? await readFile(options.templateFile, "utf8")
    : options.template;
  if (template === undefined) {
    return BASE_RESAMPLING_PROMPT;
  }

  // A template read from a file usually carries a trailing newline; the prompt
  // must end at `Output:`, so drop it before the checks run.
  const trimmed = template.trimEnd();
  const problem = templateError(trimmed);
  if (problem) {
    throw new Error(problem);
  }
  return trimmed;
}

async function getInstruction(options) {
  if (options.instruction && options.inputFile) {
    throw new Error("Use either --instruction or --input-file, not both.");
  }
  if (options.inputFile) {
    return readFile(options.inputFile, "utf8");
  }
  if (options.instruction) {
    return options.instruction;
  }
  throw new Error("Provide --instruction or --input-file.");
}

/**
 * One API request. For the completions backend this returns all `n`
 * variations; the Responses backend always returns exactly one.
 */
export async function callOpenAI({ prompt, backend = DEFAULT_BACKEND, model, n = 1, params = {}, apiUrl }) {
  const resolvedBackend = getBackend(backend);
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not set. Use --dry-run to preview the prompt without an API call.");
  }

  const url = apiUrl || resolvedBackend.apiUrl;
  const requestBody = buildOpenAIRequestBody({
    backend: resolvedBackend.id,
    prompt,
    model,
    n,
    params,
  });

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestBody),
  });

  const bodyText = await response.text();
  let body;
  try {
    body = JSON.parse(bodyText);
  } catch {
    body = { raw: bodyText };
  }

  if (!response.ok) {
    const message = body?.error?.message || bodyText || response.statusText;
    const error = new Error(`OpenAI API request failed (${response.status}): ${message}`);
    error.openAIRequest = { url, body: requestBody, backend: resolvedBackend.id };
    error.openAIResponse = { status: response.status, body };
    throw error;
  }

  return {
    backend: resolvedBackend.id,
    variations: extractVariations(resolvedBackend.id, body),
    request: { url, body: requestBody, backend: resolvedBackend.id },
    response: { status: response.status, body },
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  const backend = getBackend(options.backend);
  const params = resolveBackendParams(backend.id, options.overrides);
  const ignored = unsupportedParams(backend.id, params);
  const template = await getTemplate(options);
  const instruction = await getInstruction(options);
  const prompt = buildResamplingPrompt(instruction, template);
  const plan = planRequests(backend.id, options.count);

  if (ignored.length) {
    console.error(`Warning: ${backend.id} does not support ${ignored.join(", ")}; not sent.`);
  }
  for (const warning of templateWarnings(template)) {
    console.error(`Warning: ${warning}`);
  }

  if (options.dryRun) {
    const preview = {
      backend: backend.id,
      model: options.model || backend.model,
      apiUrl: options.apiUrl || backend.apiUrl,
      requests: plan.length,
      template,
      prompt,
      requestBody: buildOpenAIRequestBody({
        backend: backend.id,
        prompt,
        model: options.model,
        n: plan[0],
        params,
      }),
    };
    if (options.json) {
      console.log(JSON.stringify(preview, null, 2));
    } else {
      console.log(prompt);
      console.log();
      console.log(`--- ${backend.id} request body (${plan.length} request(s)) ---`);
      console.log(JSON.stringify(preview.requestBody, null, 2));
    }
    return;
  }

  const variations = [];
  for (const n of plan) {
    const result = await callOpenAI({
      prompt,
      backend: backend.id,
      model: options.model,
      n,
      params,
      apiUrl: options.apiUrl,
    });
    variations.push(...result.variations);
  }

  if (options.json) {
    console.log(JSON.stringify({
      backend: backend.id,
      model: options.model || backend.model,
      requests: plan.length,
      template,
      prompt,
      variations,
    }, null, 2));
    return;
  }

  variations.forEach((variation, index) => {
    const score = variation.logprob === null || variation.logprob === undefined
      ? ""
      : `  [logprob ${variation.logprob.toFixed(4)}, p=${variation.probability.toExponential(3)}]`;
    console.log(`${index + 1}. ${variation.text}${score}`);
  });
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    console.error(`Error: ${error.message}`);
    process.exitCode = 1;
  });
}
