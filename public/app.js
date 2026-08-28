const form = document.querySelector("#prompt-form");
const instructionInput = document.querySelector("#instruction");
const previewButton = document.querySelector("#preview-button");
const promptPreview = document.querySelector("#prompt-preview");
const requestPreview = document.querySelector("#request-preview");
const modelInput = document.querySelector("#model");
const backendNote = document.querySelector("#backend-note");
const results = document.querySelector("#results");
const status = document.querySelector("#status");
const copyPrompt = document.querySelector("#copy-prompt");
const refreshLogButton = document.querySelector("#refresh-log");
const logTable = document.querySelector("#log-table");
const submitButton = form.querySelector("button[type='submit']");

const PARAM_NAMES = ["temperature", "topP", "maxTokens", "frequencyPenalty", "presencePenalty"];

/** Filled from /api/backends so the server stays the single source of truth. */
let backends = {};

const META_INSTRUCTION = "Generate a variation of the following instruction while keeping the semantic meaning.";
const BASE_RESAMPLING_PROMPT = [
  META_INSTRUCTION,
  "",
  "Input: [INSTRUCTION]",
  "Output:",
].join("\n");

function selectedMode() {
  return new FormData(form).get("mode");
}

function selectedBackend() {
  return new FormData(form).get("backend") || "completions";
}

/**
 * Placeholders show the backend default, and unsupported parameters are greyed
 * out rather than hidden, so the difference between the two paths stays visible.
 */
function applyBackend() {
  const backend = backends[selectedBackend()];
  if (!backend) {
    return;
  }

  modelInput.placeholder = backend.model;
  const parts = [`${backend.label} — ${backend.apiUrl}`];
  parts.push(backend.supportsN
    ? "Count is sent as n in a single request, as in the reference implementation."
    : "No n parameter: one request per variation.");
  if (backend.retiresOn) {
    parts.push(`Endpoint retires on ${backend.retiresOn}.`);
  }
  parts.push(...(backend.caveats ?? []));
  backendNote.textContent = parts.join(" ");

  for (const name of PARAM_NAMES) {
    const field = form.querySelector(`[data-param="${name}"]`);
    const input = form.querySelector(`[name="${name}"]`);
    if (!field || !input) {
      continue;
    }
    const supported = backend.supported.includes(name);
    input.disabled = !supported;
    field.classList.toggle("unsupported", !supported);
    const fallback = backend.defaults[name];
    input.placeholder = supported
      ? (fallback === undefined ? "API default" : String(fallback))
      : "rejected by the model";
    if (!supported) {
      input.value = "";
    }
  }
}

async function loadBackends() {
  try {
    const json = await getJson("/api/backends");
    backends = Object.fromEntries(json.backends.map((backend) => [backend.id, backend]));
    const preferred = form.querySelector(`[name="backend"][value="${json.defaultBackend}"]`);
    if (preferred) {
      preferred.checked = true;
    }
    applyBackend();
  } catch (error) {
    backendNote.textContent = `Could not load backend list: ${error.message}`;
  }
}

function setStatus(text, isError = false) {
  status.textContent = text;
  status.classList.toggle("error", isError);
}

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  previewButton.disabled = isBusy;
}

function payload() {
  const data = new FormData(form);
  const body = {
    mode: data.get("mode"),
    backend: data.get("backend"),
    instruction: data.get("instruction"),
    model: String(data.get("model") || "").trim(),
    count: data.get("count"),
  };

  if (!body.model) {
    delete body.model;
  }

  for (const name of PARAM_NAMES) {
    const input = form.querySelector(`[name="${name}"]`);
    if (input?.disabled) {
      continue;
    }
    const value = String(data.get(name) || "").trim();
    if (value) {
      body[name] = value;
    }
  }

  return body;
}

function localPreview() {
  const instruction = selectedMode() === "meta" ? BASE_RESAMPLING_PROMPT : instructionInput.value.trim();
  if (!instruction) {
    promptPreview.textContent = "";
    return;
  }

  promptPreview.textContent = [
    META_INSTRUCTION,
    "",
    `Input: ${instruction}`,
    "Output:",
  ].join("\n");
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const json = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(json.error || "Request failed.");
  }
  return json;
}

async function getJson(url) {
  const response = await fetch(url);
  const json = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(json.error || "Request failed.");
  }
  return json;
}

async function readJsonResponse(response) {
  const text = await response.text();
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    if (response.status === 404 && text === "Not found") {
      throw new Error("The running server does not expose this API route yet. Restart node server.mjs and reload the page.");
    }
    throw new Error(`Expected JSON from ${response.url}, but received: ${text || response.statusText}`);
  }
}

function renderRequestPreview(json) {
  const lines = [];
  if (json.ignoredParams?.length) {
    lines.push(`// ${json.backend} ignores: ${json.ignoredParams.join(", ")}`);
  }
  lines.push(`// ${json.requests} request(s) to ${json.backend}`);
  lines.push(JSON.stringify(json.requestBody, null, 2));
  requestPreview.textContent = lines.join("\n");
}

async function refreshPreview() {
  setBusy(true);
  setStatus("Preview");
  try {
    const json = await postJson("/api/preview", payload());
    promptPreview.textContent = json.prompt;
    renderRequestPreview(json);
    setStatus(json.mode === "meta" ? "Meta" : "Ready");
  } catch (error) {
    setStatus("Error", true);
    results.innerHTML = `<article class="result-empty">${escapeHtml(error.message)}</article>`;
  } finally {
    setBusy(false);
  }
}

function renderResults(variations) {
  if (!variations.length) {
    results.innerHTML = '<article class="result-empty">The API did not return any variations.</article>';
    return;
  }

  results.innerHTML = variations.map((variation, index) => {
    const text = typeof variation === "string" ? variation : variation.text;
    const truncated = typeof variation === "object" && variation.finishReason === "length";
    return `
      <article class="result-item">
        <strong>Variation ${index + 1}${truncated ? ' <span class="truncated">truncated at max_tokens</span>' : ""}</strong>
        <div>${escapeHtml(text)}</div>
      </article>
    `;
  }).join("");
}

function shortText(value) {
  const text = String(value ?? "");
  return text.length > 260 ? `${text.slice(0, 260)}...` : text;
}

/**
 * One glanceable line per response. Token usage covers the whole request, so
 * when several variations came back together the line says so.
 */
function responseSummary(entry) {
  const body = entry.response?.body ?? {};
  const parts = [`HTTP ${entry.response.status}`];

  if (body.error) {
    parts.push(body.error.code || body.error.type || "error");
    return parts.join(" · ");
  }

  const usage = body.usage ?? {};
  const inTok = usage.prompt_tokens ?? usage.input_tokens;
  const outTok = usage.completion_tokens ?? usage.output_tokens;
  if (inTok !== undefined || outTok !== undefined) {
    const reasoning = usage.output_tokens_details?.reasoning_tokens;
    parts.push(`${inTok ?? "?"}→${outTok ?? "?"} tok${reasoning ? ` (${reasoning} reasoning)` : ""}`);
  }

  if (entry.finishReason) {
    parts.push(entry.finishReason);
  }
  if (entry.variationCount > 1) {
    parts.push(`request of ${entry.variationCount}`);
  }

  return parts.join(" · ");
}

function renderLog(entries) {
  logTable.innerHTML = entries.map((entry) => `
    <tr>
      <td>
        <details ${entry.mode === "seed" ? "open" : ""}>
          <summary>${escapeHtml(shortText(entry.prompt))}</summary>
          <pre class="table-pre">${escapeHtml(entry.prompt)}</pre>
        </details>
      </td>
      <td>${entry.parentPrompt ? `
        <details>
          <summary>${escapeHtml(shortText(entry.parentInstruction || entry.parentPrompt))}</summary>
          <pre class="table-pre">${escapeHtml(entry.parentPrompt)}</pre>
        </details>
      ` : '<span class="muted">Default resampling prompt</span>'}</td>
      <td>${entry.request ? `
        <details>
          <summary>${escapeHtml(entry.request.body?.model || entry.model || "View request")}</summary>
          <pre class="table-pre">${escapeHtml(JSON.stringify(entry.request, null, 2))}</pre>
        </details>
      ` : '<span class="muted">No API request</span>'}</td>
      <td>${entry.response ? `
        <details>
          <summary>${escapeHtml(responseSummary(entry))}</summary>
          <pre class="table-pre">${escapeHtml(JSON.stringify(entry.response, null, 2))}</pre>
        </details>
      ` : '<span class="muted">No API response</span>'}</td>
      <td>${entry.backend ? `<span class="mode-chip">${escapeHtml(entry.backend)}</span>` : '<span class="muted">—</span>'}</td>
      <td><span class="mode-chip">${escapeHtml(entry.mode)}</span></td>
      <td>${entry.createdAt === "1970-01-01T00:00:00.000Z" ? '<span class="muted">seed</span>' : escapeHtml(new Date(entry.createdAt).toLocaleString())}</td>
    </tr>
  `).join("");
}

async function loadLog() {
  try {
    const json = await getJson("/api/log");
    renderLog(json.promptRows || json.entries || []);
  } catch (error) {
    logTable.innerHTML = `<tr><td colspan="7">${escapeHtml(error.message)}</td></tr>`;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

form.addEventListener("input", (event) => {
  if (event.target.name === "mode") {
    instructionInput.disabled = selectedMode() === "meta";
  }
  if (event.target.name === "backend") {
    applyBackend();
  }
  localPreview();
});

previewButton.addEventListener("click", refreshPreview);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(true);
  setStatus("Generating");
  results.innerHTML = '<article class="result-empty">Waiting for the API...</article>';

  try {
    const json = await postJson("/api/resample", payload());
    promptPreview.textContent = json.prompt;
    renderResults(json.variations);
    await loadLog();
    setStatus(`Done — ${json.backend}`);
  } catch (error) {
    setStatus("Error", true);
    results.innerHTML = `<article class="result-empty">${escapeHtml(error.message)}</article>`;
  } finally {
    setBusy(false);
  }
});

copyPrompt.addEventListener("click", async () => {
  await navigator.clipboard.writeText(promptPreview.textContent);
  setStatus("Copied");
});

refreshLogButton.addEventListener("click", loadLog);

instructionInput.disabled = selectedMode() === "meta";
localPreview();
loadBackends();
loadLog();
