#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { extname, join, normalize } from "node:path";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { choicesInOrder, viewerUrl, viewerEntries } from "./viewer-link.mjs";
import { appendPromptLogEvent, promptRowsFromEvents, readPromptLog } from "./prompt-log.mjs";
import {
  BACKENDS,
  BASE_RESAMPLING_PROMPT,
  buildOpenAIRequestBody,
  buildResamplingPrompt,
  callOpenAI,
  COMPLETIONS_DEADLINE,
  DEFAULT_BACKEND,
  getBackend,
  getInstructionForMode,
  INSTRUCTION_PLACEHOLDER,
  maxLogprobs,
  minMaxTokens,
  planRequests,
  RESAMPLING_INSTRUCTION,
  resolveBackendParams,
  templateError,
  templateWarnings,
  unsupportedParams,
} from "./resample-prompt.mjs";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const PUBLIC_DIR = join(ROOT, "public");
// The viewer package is the parent directory now: this program used to live in
// a repository of its own beside it, which meant a clone of either one was
// half a program. LOGPROBS_VIEWER_DIR still overrides it.
const VIEWER_DIR = process.env.LOGPROBS_VIEWER_DIR || join(ROOT, '..');
/* The viewer package's own server. Everything shared lives there: the record
   store every view reads, the ask-before-calling policy, and the deviation
   engine behind the branch button. This program keeps its own prompt log -- that
   is its subject -- but a result nobody can open in the real viewer may as well
   not exist, so each one is offered to the store as well. */
const STORE_ORIGIN = process.env.LOGPROBS_STORE_ORIGIN || 'http://127.0.0.1:8899';
// logprobs.html loads ask-policy.js and approve-request.js -- one policy and
// one dialog for asking before a paid call, shared with the viewer package.
// Without them here the page 404s on both and every paid path throws.
const VIEWER_FILES = new Set(['/logprobs.html', '/app.css', '/theme.js', '/bar.js', '/single-token-variants.html', '/logprobs.json', '/ask-policy.js', '/approve-request.js', '/call-report.js', '/greedy-branches.js', '/deviations.html', '/help.html']);
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

/**
 * An absent or blank template means "use the paper's", so a client that predates
 * the field keeps working. Anything else is checked before it can reach the API:
 * a template without the placeholder would send a prompt that never names the
 * instruction, which reads as a successful call and returns nonsense.
 */
function readTemplate(raw) {
  if (raw === undefined || raw === null || String(raw).trim() === "") {
    return BASE_RESAMPLING_PROMPT;
  }
  const template = String(raw).trimEnd();
  const problem = templateError(template);
  if (problem) {
    throw new Error(problem);
  }
  return template;
}

function normalizeOptions(body) {
  const mode = body.mode === "meta" ? "meta" : "custom";
  const template = readTemplate(body.template);
  const instruction = getInstructionForMode({
    instruction: body.instruction,
    mode,
    template,
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
    template,
    isDefaultTemplate: template === BASE_RESAMPLING_PROMPT,
    templateWarnings: templateWarnings(template),
    count,
    backend: backend.id,
    params,
    ignoredParams: unsupportedParams(backend.id, params),
    model: String(body.model || process.env.OPENAI_MODEL || backend.model).trim(),
    apiUrl: String(body.apiUrl || process.env.OPENAI_API_URL || backend.apiUrl).trim(),
  };
}

/* First of these that exists on disk; `py` is the Windows launcher and is
   assumed to be on PATH rather than checked for. */
function defaultPython() {
  const home = process.env.USERPROFILE || process.env.HOME || '';
  const bundled = home
    ? join(home, '.cache', 'codex-runtimes', 'codex-primary-runtime',
           'dependencies', 'python', 'python.exe')
    : '';
  if (bundled && existsSync(bundled)) return bundled;
  return process.platform === 'win32' ? 'py' : 'python3';
}

async function singleTokenVariants(record) {
  /* The path to the bundled runtime was written relative to the old location,
     two levels up from a sibling directory; from here that resolves somewhere
     else entirely. It is spelled from the home directory instead, and if it is
     not there the launcher is tried -- `python` alone is a Microsoft Store stub
     on this machine and fails in a way that reads like a code error. */
  const python = process.env.LOGPROBS_PYTHON || defaultPython();
  return new Promise((resolve, reject) => {
    const child = spawn(python, [join(VIEWER_DIR, 'single_token_variants.py')], { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    let output = '', errors = '';
    const timeout = setTimeout(() => { child.kill(); reject(new Error('Variant generation timed out.')); }, 30000);
    child.stdout.on('data', data => { output += data; });
    child.stderr.on('data', data => { errors += data; });
    child.on('error', error => { clearTimeout(timeout); reject(error); });
    child.stdin.on('error', error => { clearTimeout(timeout); reject(error); });
    child.on('close', code => {
      clearTimeout(timeout);
      if (code !== 0) return reject(new Error('Could not generate variants: ' + errors));
      try { resolve(JSON.parse(output)); } catch (error) { reject(error); }
    });
    child.stdin.end(JSON.stringify(record));
  });
}

/* One setting for the whole package, and it is kept by the store server. Proxied
   rather than copied so this program cannot drift into a second opinion about
   when to ask; with the store down the client falls back to its own strictest
   default, which is to ask every time. */
async function askPolicy(method, body) {
  const response = await fetch(`${STORE_ORIGIN}/api/ask_policy`, {
    method,
    headers: method === 'POST' ? { 'Content-Type': 'application/json' } : undefined,
    body: method === 'POST' ? JSON.stringify(body ?? {}) : undefined,
  });
  return response.json();
}

/* Offer a finished record to the shared store. Best effort on purpose: the call
   has already been paid for and logged here, so a store that is down must not
   turn a successful resample into an error. What is lost is only the link into
   the full viewer, and the answer says so. */
async function offerToStore(entries) {
  if (!entries.length) return [];
  try {
    // One request for the whole run: the store rewrites its history and reloads
    // its indexes per call, so five variations sent one by one would leave the
    // page waiting the best part of a minute after the API had answered.
    const response = await fetch(`${STORE_ORIGIN}/api/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ records: entries }),
    });
    const json = await response.json();
    return response.ok && json.saved ? (json.ids ?? []) : [];
  } catch {
    return [];   // store not running: the call is logged here either way
  }
}

async function handleApi(request, response) {
  try {
    const url = new URL(request.url, 'http://localhost');
    if (url.pathname === '/api/ask_policy') {
      try {
        const body = request.method === 'POST' ? await readJsonBody(request) : null;
        sendJson(response, 200, await askPolicy(request.method, body));
      } catch {
        sendJson(response, 503, { error: 'the store server is not answering', policy: null });
      }
      return;
    }
    if (request.method === 'GET' && url.pathname === '/api/single-token-variants') {
      const entry = viewerEntries(await readPromptLog()).find(entry => entry.id === url.searchParams.get('id'));
      if (!entry) { sendJson(response, 404, { error: 'Completion not found or has no logprobs.' }); return; }
      sendJson(response, 200, await singleTokenVariants(entry));
      return;
    }
    if (request.method === 'GET' && url.pathname === '/api/record') {
      const entry = viewerEntries(await readPromptLog()).find(entry => entry.id === url.searchParams.get('id'));
      sendJson(response, entry ? 200 : 404, entry || { error: 'Completion not found or has no logprobs.' });
      return;
    }
    if (request.method === 'GET' && url.pathname === '/api/stats') {
      sendJson(response, 200, { records: viewerEntries(await readPromptLog()).length });
      return;
    }
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
        // Where the shared store and the deviation engine live, so the page can
        // link a result into the real viewer instead of the half-powered copy
        // this server can offer on its own.
        storeOrigin: STORE_ORIGIN,
        // The prompt used to be duplicated in app.js and index.html, so editing
        // it here left the UI previewing a prompt the server no longer sends.
        resamplingInstruction: RESAMPLING_INSTRUCTION,
        baseResamplingPrompt: BASE_RESAMPLING_PROMPT,
        instructionPlaceholder: INSTRUCTION_PLACEHOLDER,
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
    const prompt = buildResamplingPrompt(options.instruction, options.template);

    const plan = planRequests(options.backend, options.count);

    if (request.url === "/api/preview") {
      sendJson(response, 200, {
        mode: options.mode,
        backend: options.backend,
        model: options.model,
        instruction: options.instruction,
        template: options.template,
        isDefaultTemplate: options.isDefaultTemplate,
        templateWarnings: options.templateWarnings,
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
      /* The same rule the rest of the package follows: nothing is bought until
         somebody has seen the request and said yes. The call happens here, in
         node, so the dialog cannot be here -- the page shows it and sends the
         acknowledgement, exactly as the store server's own /api/complete
         requires. Refusing without it is what makes a stray POST, a replayed
         curl or a forgotten caller cost nothing. */
      if (body.confirmed !== true) {
        sendJson(response, 400, { error: 'explicit confirmation is required before a paid call' });
        return;
      }
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
              template: options.template,
              run,
              request: error.openAIRequest || null,
              response: error.openAIResponse || null,
              generatedPrompts: [],
            });
          }
          throw error;
        }

        const savedEvents = await appendPromptLogEvent({
          type: "openai_resample",
          mode: options.mode,
          backend: options.backend,
          model: options.model,
          parentInstruction: options.instruction,
          parentPrompt: prompt,
          template: options.template,
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
        const saved = savedEvents.at(-1);
        const choices = choicesInOrder(saved);
        /* Into the shared store, so the string can be read in the real viewer --
           token by token, in a ranked list beside everything else, and with the
           branch button for its one-token deviations. Where that worked the link
           points at the store server; where it did not, at the local copy, which
           can still show the single string. */
        const entries = viewerEntries([saved]);
        const inStore = new Set(await offerToStore(entries));
        variations.push(...result.variations.map((variation, index) => {
          const local = viewerUrl(saved, choices[index]);
          const id = `ape:${saved.id}:${choices[index]?.index ?? index}`;
          return {
            ...variation,
            logprobsUrl: local && inStore.has(id)
              ? `${STORE_ORIGIN}/logprobs.html?id=${encodeURIComponent(id)}`
              : local,
            inStore: inStore.has(id),
          };
        }));
      }

      sendJson(response, 200, {
        mode: options.mode,
        backend: options.backend,
        model: options.model,
        instruction: options.instruction,
        template: options.template,
        isDefaultTemplate: options.isDefaultTemplate,
        templateWarnings: options.templateWarnings,
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
  const pathname = new URL(request.url, 'http://localhost').pathname;
  if (pathname === '/completion_history.json') {
    sendJson(response, 200, viewerEntries(await readPromptLog()));
    return;
  }
  if (VIEWER_FILES.has(pathname)) {
    try {
      const content = await readFile(join(VIEWER_DIR, pathname.slice(1)));
      response.writeHead(200, { 'Content-Type': MIME_TYPES[extname(pathname)], 'Cache-Control': 'no-cache' });
      response.end(content);
    } catch {
      response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      response.end('Logprobs Viewer not found. Configure LOGPROBS_VIEWER_DIR.');
    }
    return;
  }
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
