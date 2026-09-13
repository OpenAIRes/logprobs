import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { BASE_RESAMPLING_PROMPT } from "./resample-prompt.mjs";
import { choicesInOrder, viewerUrl } from "./viewer-link.mjs";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
export const DEFAULT_LOG_PATH = join(ROOT, "data", "prompt-log.json");

export function seedPromptLog() {
  return {
    id: "seed-resampling-prompt",
    type: "seed",
    createdAt: new Date(0).toISOString(),
    mode: "seed",
    backend: null,
    model: null,
    parentInstruction: null,
    parentPrompt: null,
    template: BASE_RESAMPLING_PROMPT,
    run: {
      id: "seed-run",
      requestedCount: 1,
      index: 1,
      total: 1,
    },
    request: null,
    response: null,
    generatedPrompts: [
      {
        prompt: BASE_RESAMPLING_PROMPT,
        rawPrompt: BASE_RESAMPLING_PROMPT,
        finishReason: null,
        logprob: null,
        probability: null,
        parentPrompt: null,
        parentInstruction: null,
      },
    ],
  };
}

function legacyEntryToEvent(entry) {
  if (entry.type && Array.isArray(entry.generatedPrompts)) {
    // Events written before the backend split predate the Responses/completions
    // choice, and all of them went through the Responses API.
    return entry.backend === undefined && entry.type !== "seed"
      ? { ...entry, backend: entry.model ? "responses" : null }
      : entry;
  }

  return {
    id: entry.id || randomUUID(),
    type: entry.id === "seed-resampling-prompt" || entry.mode === "seed" ? "seed" : "legacy_prompt",
    createdAt: entry.createdAt || new Date(0).toISOString(),
    mode: entry.mode || "custom",
    backend: entry.backend ?? (entry.model ? "responses" : null),
    model: entry.model ?? null,
    parentInstruction: entry.parentInstruction || null,
    parentPrompt: entry.parentPrompt || null,
    run: entry.run || {
      id: entry.id ? `legacy-${entry.id}` : randomUUID(),
      requestedCount: 1,
      index: 1,
      total: 1,
    },
    request: null,
    response: null,
    generatedPrompts: [
      {
        prompt: String(entry.prompt ?? "").trim(),
        rawPrompt: entry.rawPrompt ?? String(entry.prompt ?? ""),
        finishReason: entry.finishReason ?? null,
        logprob: entry.logprob ?? null,
        probability: entry.probability ?? null,
        parentPrompt: entry.parentPrompt || null,
        parentInstruction: entry.parentInstruction || null,
      },
    ].filter((prompt) => prompt.prompt),
  };
}

function ensureSeed(entries) {
  const seed = seedPromptLog();
  const normalized = entries.map(legacyEntryToEvent).map((entry) => ({
    ...entry,
    run: entry.run || {
      id: `legacy-${entry.id || randomUUID()}`,
      requestedCount: 1,
      index: 1,
      total: 1,
    },
  }));
  const withoutSeed = normalized.filter((entry) => entry.id !== seed.id);
  return [seed, ...withoutSeed];
}

async function writeLog(entries, logPath) {
  await mkdir(dirname(logPath), { recursive: true });
  await writeFile(logPath, `${JSON.stringify(entries, null, 2)}\n`, "utf8");
}

export async function readPromptLog(logPath = DEFAULT_LOG_PATH) {
  try {
    const text = await readFile(logPath, "utf8");
    const parsed = JSON.parse(text);
    if (!Array.isArray(parsed) || parsed.length === 0) {
      const seeded = ensureSeed([]);
      await writeLog(seeded, logPath);
      return seeded;
    }
    const seeded = ensureSeed(parsed);
    if (
      JSON.stringify(seeded) !== JSON.stringify(parsed)
      || seeded[0].id !== "seed-resampling-prompt"
    ) {
      await writeLog(seeded, logPath);
    }
    return seeded;
  } catch (error) {
    if (error.code !== "ENOENT") {
      throw error;
    }
    const seeded = ensureSeed([]);
    await writeLog(seeded, logPath);
    return seeded;
  }
}

export function promptRowsFromEvents(events) {
  return events.flatMap((event) => (event.generatedPrompts ?? []).map((generatedPrompt, index) => ({
    id: `${event.id}:${index}:${generatedPrompt.prompt.slice(0, 24)}`,
    eventId: event.id,
    logprobsUrl: viewerUrl(event, choicesInOrder(event).filter(choice => String(choice.text ?? '').trim())[index]),
    createdAt: event.createdAt,
    mode: event.mode,
    backend: event.backend ?? null,
    model: event.model,
    request: event.request,
    response: event.response ?? null,
    // Usage in the response covers the whole request, so a row needs to say
    // how many variations shared it.
    variationCount: (event.generatedPrompts ?? []).length,
    prompt: generatedPrompt.prompt,
    rawPrompt: generatedPrompt.rawPrompt ?? generatedPrompt.prompt,
    finishReason: generatedPrompt.finishReason ?? null,
    logprob: generatedPrompt.logprob ?? null,
    probability: generatedPrompt.probability ?? null,
    parentPrompt: generatedPrompt.parentPrompt,
    parentInstruction: generatedPrompt.parentInstruction,
    template: event.template ?? null,
    ...(event.templateSource ? { templateSource: event.templateSource } : {}),
  })));
}

export async function appendPromptLogEvent(event, logPath = DEFAULT_LOG_PATH) {
  const current = await readPromptLog(logPath);
  const normalized = {
    id: randomUUID(),
    type: event.type || "openai_resample",
    createdAt: new Date().toISOString(),
    mode: event.mode,
    backend: event.backend ?? null,
    model: event.model,
    parentInstruction: event.parentInstruction || null,
    parentPrompt: event.parentPrompt || null,
    // Null on entries written before templates could be swapped; those all used
    // the paper's, but saying so here would be a guess dressed as a record.
    template: event.template ?? null,
    ...(event.templateSource ? { templateSource: event.templateSource } : {}),
    run: event.run || {
      id: randomUUID(),
      requestedCount: 1,
      index: 1,
      total: 1,
    },
    request: event.request || null,
    response: event.response || null,
    generatedPrompts: (event.generatedPrompts ?? []).map((generatedPrompt) => ({
      prompt: String(generatedPrompt.prompt ?? "").trim(),
      // Kept verbatim: the reference implementation does not strip completions,
      // so the leading space after `Output:` is part of the record.
      rawPrompt: String(generatedPrompt.rawPrompt ?? generatedPrompt.prompt ?? ""),
      finishReason: generatedPrompt.finishReason ?? null,
      logprob: generatedPrompt.logprob ?? null,
      probability: generatedPrompt.probability ?? null,
      parentPrompt: generatedPrompt.parentPrompt || event.parentPrompt || null,
      parentInstruction: generatedPrompt.parentInstruction || event.parentInstruction || null,
    })).filter((generatedPrompt) => generatedPrompt.prompt),
  };

  const next = [...current, normalized];
  await writeLog(next, logPath);
  return next;
}
