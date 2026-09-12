const form = document.querySelector("#prompt-form");
const instructionInput = document.querySelector("#instruction");
const templateInput = document.querySelector("#template");
const templateNote = document.querySelector("#template-note");
const resetTemplateButton = document.querySelector("#reset-template");
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
const logFilterNote = document.querySelector("#log-filter-note");
const submitButton = form.querySelector("button[type='submit']");

const PARAM_NAMES = ["temperature", "topP", "maxTokens", "frequencyPenalty", "presencePenalty", "logprobs"];

/** Filled from /api/backends so the server stays the single source of truth. */
let backends = {};

/** Every row from the last /api/log fetch; the table renders a filtered view. */
let logRows = [];
let customInstruction = instructionInput.value;
let displayedMode = 'custom';

function syncInstructionField() {
  const mode = selectedMode();
  if (mode === 'meta') {
    if (displayedMode !== 'meta') customInstruction = instructionInput.value;
    instructionInput.value = currentTemplate();
  } else if (displayedMode === 'meta') {
    instructionInput.value = customInstruction;
  }
  instructionInput.disabled = false;
  instructionInput.readOnly = mode === 'meta';
  instructionInput.setAttribute('aria-readonly', String(mode === 'meta'));
  displayedMode = mode;
}

/**
 * Also from /api/backends. The instruction and the template used to be copied
 * into this file and into index.html, so editing resample-prompt.mjs left the UI
 * previewing a prompt the server no longer sent. Nothing here hard-codes them.
 */
let promptConfig = {
  resamplingInstruction: "",
  baseResamplingPrompt: "",
  instructionPlaceholder: "[INSTRUCTION]",
  // Where the shared store lives; the server says, so it is configurable there.
  storeOrigin: "",
};

/** A blank box means "the default", so clearing it cannot break the prompt. */
function currentTemplate() {
  return templateInput.value.trimEnd() || promptConfig.baseResamplingPrompt;
}

function setTemplateNote(text, isError = false) {
  templateNote.textContent = text;
  templateNote.classList.toggle("error", isError);
}

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
    promptConfig = {
      resamplingInstruction: json.resamplingInstruction ?? "",
      baseResamplingPrompt: json.baseResamplingPrompt ?? "",
      instructionPlaceholder: json.instructionPlaceholder ?? "[INSTRUCTION]",
      storeOrigin: json.storeOrigin ?? "",
    };
    if (!templateInput.value.trim()) {
      templateInput.value = promptConfig.baseResamplingPrompt;
    }
    localPreview();
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
    template: currentTemplate(),
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

/**
 * Mirrors buildResamplingPrompt() without restating the prompt: whatever is in
 * the template box is the only source. Press Preview for the server's own answer.
 */
function localPreview() {
  syncInstructionField();
  const template = currentTemplate();
  const placeholder = promptConfig.instructionPlaceholder;

  if (!template) {
    promptPreview.textContent = "";
    setTemplateNote("Loading the resampling template...");
    return;
  }

  if (!template.includes(placeholder)) {
    promptPreview.textContent = "";
    setTemplateNote(`No ${placeholder} slot, so the instruction would never reach the prompt. Resampling in Meta mode returns just the instruction sentence and drops the Input:/Output: scaffolding, which lands exactly here.`, true);
    return;
  }

  const notes = [];
  if (template !== promptConfig.baseResamplingPrompt) {
    notes.push("Custom template, not the paper's.");
  }
  if (!template.endsWith("Output:")) {
    notes.push('Does not end with "Output:", the cue the model completes after.');
  }
  setTemplateNote(notes.join(" ") || "The resampling template from the paper.");

  // Meta mode resamples the template itself, so it follows the box.
  const instruction = selectedMode() === "meta" ? template : instructionInput.value.trim();
  if (!instruction) {
    promptPreview.textContent = "";
    return;
  }

  promptPreview.textContent = template.replaceAll(placeholder, () => instruction);
}

/* The policy and the dialog are the package's, not this program's: one rule and
   one dialog everywhere, so a resampling run is shown the same way a token click
   is. Missing scripts mean no dialog, and no dialog means no call -- the same
   fail-closed rule the rest of the package follows. */
async function approveRequest(plan) {
  const policy = window.AskPolicy;
  if (!policy) {
    throw new Error("ask-policy.js se nenačetlo, takže se není čím zeptat — placené volání se neprovede.");
  }
  const request = plan.requestBody ?? {};
  window.CallReport.report("asking", `${plan.model} · ${plan.requests}× · ${plan.backend}`);
  const verdict = await policy.guard(request, { total: Number(plan.requests) || 1 });
  if (verdict !== "yes" && verdict !== "all" && verdict !== true) {
    window.CallReport.report("declined");
  }
  return verdict;
}

/* The deviations of a string: every position x every recorded alternative as a
   new prompt, with the continuation the model really produces. It has to point
   at the store server, because that is where the records and the engine are --
   this server can serve the page but has nothing to answer it with.

   It used to point at single-token-variants.html, which keeps the original tail
   and never calls a model. That is a different question and a useful one, but it
   is not the one this link is for; the two pages now link to each other, so
   picking the other one is one click from either. */
function storeUrl(url, page) {
  const origin = promptConfig.storeOrigin || "";
  const path = String(url).replace("/logprobs.html", page);
  return /^https?:/i.test(path) ? path : origin + path;
}
const deviationsUrl = url => storeUrl(url, "/deviations.html");
/* Both links lead to the store, not to the copy of the viewer this server can
   put up: that copy has the page but not the store behind it, so its settings
   bar, its lists and its branch button have nothing to answer them. Every
   result is pushed to the store as it is made, and the older ones were
   imported, so there is a record there to open. */
const viewerUrlInStore = url => storeUrl(url, "/logprobs.html");

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

function renderTemplateStatus(json) {
  const notes = [];
  if (json.isDefaultTemplate === false) {
    notes.push("Custom template, not the paper's.");
  }
  notes.push(...(json.templateWarnings ?? []));
  setTemplateNote(notes.join(" ") || "The resampling template from the paper.");
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
    renderTemplateStatus(json);
    renderRequestPreview(json);
    setStatus(json.mode === "meta" ? "Meta" : "Ready");
  } catch (error) {
    setStatus("Error", true);
    results.innerHTML = `<article class="result-empty">${escapeHtml(error.message)}</article>`;
  } finally {
    setBusy(false);
  }
}

function logprobsLink(entry) {
  return entry?.logprobsUrl
    ? `<a class="logprobs-link" href="${escapeHtml(viewerUrlInStore(entry.logprobsUrl))}" target="_blank" rel="noopener">View logprobs ↗</a> <a class="logprobs-link" href="${escapeHtml(deviationsUrl(entry.logprobsUrl))}" target="_blank" rel="noopener">One-token deviations ↗</a>`
    : "";
}

function renderResults(variations) {
  if (!variations.length) {
    results.innerHTML = '<article class="result-empty">The API did not return any variations.</article>';
    return;
  }

  results.innerHTML = variations.map((variation, index) => {
    const text = typeof variation === "string" ? variation : variation.text;
    const truncated = typeof variation === "object" && variation.finishReason === "length";
    const logprob = typeof variation === "object" && typeof variation.logprob === "number"
      ? ` <span class="score">logprob ${variation.logprob.toFixed(4)} · p=${variation.probability.toExponential(3)}</span>`
      : "";
    return `
      <article class="result-item">
        <strong>Variation ${index + 1}${truncated ? ' <span class="truncated">truncated at max_tokens</span>' : ""}${logprob}</strong>
        <div>${escapeHtml(text)}</div>
        ${logprobsLink(variation)}
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
  if (typeof entry.logprob === "number") {
    parts.push(`logprob ${entry.logprob.toFixed(2)} · p=${entry.probability.toExponential(2)}`);
  }
  if (entry.variationCount > 1) {
    parts.push(`request of ${entry.variationCount}`);
  }

  return parts.join(" · ");
}

/**
 * The table follows the mode switch: in Meta mode only meta runs, in Custom
 * mode only custom ones. The seed stays in both — it is the shared template
 * every run descends from, not a run of either mode.
 */
function visibleLogRows() {
  const mode = selectedMode();
  return logRows.filter((row) => row.mode === mode || row.mode === "seed");
}

function renderLog() {
  const mode = selectedMode();
  const entries = visibleLogRows();
  const runs = entries.filter((row) => row.mode !== "seed").length;
  const hidden = logRows.filter((row) => row.mode !== mode && row.mode !== "seed").length;

  logFilterNote.textContent = hidden
    ? `${runs} run${runs === 1 ? "" : "s"} in ${mode} mode · ${hidden} hidden from the other mode`
    : `${runs} run${runs === 1 ? "" : "s"} in ${mode} mode`;

  const withLogprobs = entries.filter(row => row.logprobsUrl).length;
  const otherLogprobs = logRows.filter(row => row.mode !== mode && row.logprobsUrl).length;
  logFilterNote.textContent += withLogprobs
    ? ' · ' + withLogprobs + ' variants with logprobs (links below).'
    : ' · No saved logprobs in this mode.';
  if (otherLogprobs) logFilterNote.textContent += ' Switch to ' + (mode === 'meta' ? 'Custom' : 'Meta') + ' for ' + otherLogprobs + ' variants with logprobs.';

  if (!entries.length) {
    logTable.innerHTML = '<tr><td colspan="7"><span class="muted">Nothing logged yet.</span></td></tr>';
    return;
  }

  logTable.innerHTML = entries.map((entry) => `
    <tr>
      <td>
        <details ${entry.mode === "seed" ? "open" : ""}>
          <summary>${escapeHtml(shortText(entry.prompt))}</summary>
          <pre class="table-pre">${escapeHtml(entry.prompt)}</pre>
        </details>
        ${logprobsLink(entry)}
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
    logRows = json.promptRows || json.entries || [];
    renderLog();
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
    syncInstructionField();
    renderLog();
  }
  if (event.target.name === "backend") {
    applyBackend();
  }
  localPreview();
});

// The template box sits in the preview panel, outside the form, so it needs its
// own listener rather than riding the form's input event.
templateInput.addEventListener("input", localPreview);

previewButton.addEventListener("click", refreshPreview);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(true);
  setStatus("Generating");
  results.innerHTML = '<article class="result-empty">Waiting for the API...</article>';

  try {
    /* Nothing is bought before somebody has seen it. The preview is what the
       server would send -- the same builder produces the body for the real call
       -- so it is the honest thing to show, and the policy decides whether to
       show it at all. `confirmed` then travels with the request; without it the
       server refuses. */
    const plan = await postJson("/api/preview", payload());
    const verdict = await approveRequest(plan);
    if (verdict !== "yes" && verdict !== "all" && verdict !== true) {
      setStatus("Nevolalo se");
      results.innerHTML = '<article class="result-empty">Volání zrušeno — nic se neposlalo.</article>';
      return;
    }
    window.CallReport.report("calling", `${plan.model} · ${plan.requests}×`);
    const json = await postJson("/api/resample", { ...payload(), confirmed: true });
    window.CallReport.report("saved", `${json.variations.length} variant`);
    promptPreview.textContent = json.prompt;
    renderTemplateStatus(json);
    renderResults(json.variations);
    await loadLog();
    setStatus(`Done — ${json.backend}`);
  } catch (error) {
    window.CallReport.report(
      /nedosa|unreachable|getaddrinfo|Failed to fetch/i.test(error.message || "")
        ? "unreachable" : "failed", error.message);
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

resetTemplateButton.addEventListener("click", () => {
  templateInput.value = promptConfig.baseResamplingPrompt;
  localPreview();
});

localPreview();
loadBackends();
loadLog();
