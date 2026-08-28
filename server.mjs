#!/usr/bin/env node

import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { extname, join, normalize } from "node:path";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { appendPromptLogEvent, promptRowsFromEvents, readPromptLog } from "./prompt-log.mjs";
import {
  BACKENDS,
  buildOpenAIRequestBody,
  buildResamplingPrompt,
  callOpenAI,
  COMPLETIONS_DEADLINE,
  DEFAULT_BACKEND,
  getBackend,
  getInstructionForMode,
  maxLogprobs,
  minMaxTokens,
  planRequests,
  RESAMPLING_INSTRUCTION,
  resolveBackendParams,
  unsupportedParams,
} from "./resample-prompt.mjs";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const PUBLIC_DIR = join(ROOT, "public");
const PORT = Number.parseInt(process.env.PORT || "8787", 10);

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml; charset=utf-8",
};

function sendJson(response, statusCode, payload) {
  response.writeHead(statusCode, { "Content-Type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

async function readJsonBody(request) {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(chunk);
  }
  const text = Buffer.concat(chunks).toString("utf8");
  return text ? JSON.parse(text) : {};
}

const PARAM_LIMITS = {
  temperature: { min: 0, max: 2, integer: false, label: "Temperature" },
  topP: { min: 0, max: 1, integer: false, label: "Top P" },
  maxTokens: { min: 1, max: 1000, integer: true, label: "Max tokens" },
  frequencyPenalty: { min: -2, max: 2, integer: false, label: "Frequency penalty" },
  presencePenalty: { min: -2, max: 2, integer: false, label: "Presence penalty" },
  logprobs: { min: 0, max: 20, integer: true, label: "Logprobs" },
};

/**
 * An absent field means "use the backend default"; the literal string "off"
 * means "send nothing", which is how the modern backend gets a bare request.
 */
function readParamOverride(raw, name) {
  if (raw === undefined || raw === null || raw === "") {
    return undefined;
  }
  if (raw === "off") {
    return null;
  }

  const limits = PARAM_LIMITS[name];
  const value = limits.integer ? Number.parseInt(raw, 10) : Number.parseFloat(raw);
  if (!Number.isFinite(value) || (limits.integer && !Number.isInteger(value))) {
    throw new Error(`${limits.label} must be a number or "off".`);
  }
  if (value < limits.min || value > limits.max) {
    throw new Error(`${limits.label} must be between ${limits.min} and ${limits.max}.`);
  }
  return value;
}

function normalizeOptions(body) {
  const mode = body.mode === "meta" ? "meta" : "custom";
  const instruction = getInstructionForMode({
    instruction: body.instruction,
    mode,
  });
  const count = Number.parseInt(body.count ?? "1", 10);

  if (!Number.isInteger(count) || count < 1 || count > 20) {
    throw new Error("Count must be an integer from 1 to 20.");
  }

  const backend = getBackend(body.backend || process.env.OPENAI_BACKEND || DEFAULT_BACKEND);

  const overrides = {};
  for (const name of Object.keys(PARAM_LIMITS)) {
    const override = readParamOverride(body[name], name);
    if (override !== undefined) {
      overrides[name] = override;
    }
  }

  const params = resolveBackendParams(backend.id, overrides);

  const floor = minMaxTokens(backend.id);
  if (params.maxTokens !== undefined && params.maxTokens < floor) {
    throw new Error(`Max tokens must be at least ${floor} for the ${backend.id} backend.`);
  }

  // The endpoint clamps above its ceiling instead of erroring, so refuse the
  // request rather than silently returning fewer alternatives than asked for.
  // A backend that cannot do logprobs at all is left to the unsupported-param
  // report instead, which says so without pretending a ceiling of 0 is a limit.
  const ceiling = maxLogprobs(backend.id);
  if (ceiling > 0 && params.logprobs !== undefined && params.logprobs > ceiling) {
    throw new Error(`Logprobs must be at most ${ceiling} for the ${backend.id} backend.`);
  }

  return {
    mode,
    instruction,
    count,
    backend: backend.id,
    params,
    ignoredParams: unsupportedParams(backend.id, params),
    model: String(body.model || process.env.OPENAI_MODEL || backend.model).trim(),
    apiUrl: String(body.apiUrl || process.env.OPENAI_API_URL || backend.apiUrl).trim(),
  };
}

async function handleApi(request, response) {
  try {
    if (request.method === "GET" && request.url === "/api/log") {
      const events = await readPromptLog();
      sendJson(response, 200, {
        events,
        promptRows: promptRowsFromEvents(events),
      });
      return;
    }

    if (request.method === "GET" && request.url === "/api/backends") {
      sendJson(response, 200, {
        defaultBackend: DEFAULT_BACKEND,
        completionsDeadline: COMPLETIONS_DEADLINE,
        backends: Object.values(BACKENDS).map((backend) => ({
          id: backend.id,
          label: backend.label,
          model: backend.model,
          apiUrl: backend.apiUrl,
          defaults: backend.defaults,
          supported: backend.supported,
          supportsN: backend.supportsN,
          minMaxTokens: minMaxTokens(backend.id),
          maxLogprobs: maxLogprobs(backend.id),
          caveats: backend.caveats ?? [],
          retiresOn: backend.retiresOn,
        })),
      });
      return;
    }

    const body = await readJsonBody(request);
    const options = normalizeOptions(body);
    const prompt = buildResamplingPrompt(options.instruction);

    const plan = planRequests(options.backend, options.count);

    if (request.url === "/api/preview") {
      sendJson(response, 200, {
        mode: options.mode,
        backend: options.backend,
        model: options.model,
        instruction: options.instruction,
        prompt,
        requests: plan.length,
        params: options.params,
        ignoredParams: options.ignoredParams,
        requestBody: buildOpenAIRequestBody({
          backend: options.backend,
          prompt,
          model: options.model,
          n: plan[0],
          params: options.params,
        }),
        resamplingInstruction: RESAMPLING_INSTRUCTION,
      });
      return;
    }

    if (request.url === "/api/resample") {
      const variations = [];
      const runId = randomUUID();

      for (let i = 0; i < plan.length; i += 1) {
        const run = {
          id: runId,
          requestedCount: options.count,
          index: i + 1,
          total: plan.length,
          backend: options.backend,
          n: plan[i],
        };

        let result;
        try {
          result = await callOpenAI({
            prompt,
            backend: options.backend,
            model: options.model,
            n: plan[i],
            params: options.params,
            apiUrl: options.apiUrl,
          });
        } catch (error) {
          if (error.openAIRequest || error.openAIResponse) {
            await appendPromptLogEvent({
              type: "openai_resample_error",
              mode: options.mode,
              backend: options.backend,
              model: options.model,
              parentInstruction: options.instruction,
              parentPrompt: prompt,
              run,
              request: error.openAIRequest || null,
              response: error.openAIResponse || null,
              generatedPrompts: [],
            });
          }
          throw error;
        }

        variations.push(...result.variations);

        await appendPromptLogEvent({
          type: "openai_resample",
          mode: options.mode,
          backend: options.backend,
          model: options.model,
          parentInstruction: options.instruction,
          parentPrompt: prompt,
          run,
          request: result.request,
          response: result.response,
          generatedPrompts: result.variations.map((variation) => ({
            prompt: variation.text,
            rawPrompt: variation.raw,
            finishReason: variation.finishReason,
            logprob: variation.logprob ?? null,
            probability: variation.probability ?? null,
            parentPrompt: prompt,
            parentInstruction: options.instruction,
          })),
        });
      }

      sendJson(response, 200, {
        mode: options.mode,
        backend: options.backend,
        model: options.model,
        instruction: options.instruction,
        prompt,
        requests: plan.length,
        params: options.params,
        ignoredParams: options.ignoredParams,
        variations,
      });
      return;
    }

    sendJson(response, 404, { error: "Unknown API endpoint." });
  } catch (error) {
    sendJson(response, 400, { error: error.message });
  }
}

async function serveStatic(request, response) {
  const requestPath = request.url === "/" ? "/index.html" : request.url;
  const decodedPath = decodeURIComponent(requestPath.split("?")[0]);
  const safePath = normalize(decodedPath).replace(/^(\.\.[/\\])+/, "");
  const filePath = join(PUBLIC_DIR, safePath);

  if (!filePath.startsWith(PUBLIC_DIR)) {
    response.writeHead(403);
    response.end("Forbidden");
    return;
  }

  try {
    const content = await readFile(filePath);
    response.writeHead(200, {
      "Content-Type": MIME_TYPES[extname(filePath)] || "application/octet-stream",
    });
    response.end(content);
  } catch {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("Not found");
  }
}

const server = createServer(async (request, response) => {
  if ((request.method === "POST" || request.method === "GET") && request.url?.startsWith("/api/")) {
    await handleApi(request, response);
    return;
  }
  if (request.method === "GET") {
    await serveStatic(request, response);
    return;
  }
  response.writeHead(405, { "Content-Type": "text/plain; charset=utf-8" });
  response.end("Method not allowed");
});

server.listen(PORT, () => {
  console.log(`APE Resampling UI running at http://localhost:${PORT}`);
});
