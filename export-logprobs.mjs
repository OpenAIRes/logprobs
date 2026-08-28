#!/usr/bin/env node

/**
 * Converts resampling runs from data/prompt-log.json into the entry shape the
 * Logprobs Viewer (OpenAIRes/logprobs) reads, so the per-token probabilities
 * from this study can be inspected there.
 *
 * Only the completions backend produces legacy logprobs, so responses-backend
 * events are skipped — reasoning models refuse logprobs outright.
 *
 * The viewer reads choices[0] everywhere, so a request with n>1 becomes one
 * entry per variation rather than one entry holding all of them; otherwise only
 * the first variation would ever be visible.
 *
 * Writing merges by id instead of replacing the file, so an existing history
 * (the viewer's own 390-odd entries) survives.
 */

import { readFile, writeFile, copyFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const DEFAULT_LOG = join(ROOT, "data", "prompt-log.json");
const DEFAULT_OUT = join(ROOT, "logprobs", "completion_history.json");

function printHelp() {
  console.log(`Usage:
  node export-logprobs.mjs
  node export-logprobs.mjs --out logprobs/ape_only.json --no-merge

Options:
  --log <path>     Source event log (default: data/prompt-log.json)
  --out <path>     Target history file (default: logprobs/completion_history.json)
  --no-merge       Overwrite the target instead of merging into it
  --no-backup      Skip the .bak copy taken before an in-place merge
  --pretty         Indent the output. Off by default: the viewer's history is
                   compact single-line JSON, and indenting a 6 MB file doubles
                   its size and turns any diff into hundreds of thousands of
                   reformatting lines.
  --dry-run        Report what would be written without touching anything
  --help, -h       Show this help

The viewer fetches completion_history.json next to logprobs.html, so the
default target is the one it will pick up. Serve that directory and open
logprobs.html to see the entries.
`);
}

function parseArgs(argv) {
  const options = {
    logPath: DEFAULT_LOG,
    outPath: DEFAULT_OUT,
    merge: true,
    backup: true,
    pretty: false,
    dryRun: false,
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
      case "--log":
        options.logPath = next();
        break;
      case "--out":
        options.outPath = next();
        break;
      case "--no-merge":
        options.merge = false;
        break;
      case "--no-backup":
        options.backup = false;
        break;
      case "--pretty":
        options.pretty = true;
        break;
      case "--dry-run":
        options.dryRun = true;
        break;
      case "--help":
      case "-h":
        options.help = true;
        break;
      default:
        throw new Error(`Unknown option: ${arg}`);
    }
  }

  return options;
}

/** A choice carries usable logprobs only if the probabilities are really there. */
export function hasLogprobs(choice) {
  return Array.isArray(choice?.logprobs?.token_logprobs)
    && choice.logprobs.token_logprobs.length > 0;
}

/**
 * One log event becomes one entry per choice. `index` is preserved so a
 * variation can still be traced back to its position in the original request.
 */
export function eventToEntries(event) {
  if (event?.backend !== "completions") {
    return [];
  }

  const body = event.response?.body;
  const choices = body?.choices;
  if (!Array.isArray(choices) || !choices.length) {
    return [];
  }

  const promptText = event.parentPrompt ?? event.request?.body?.prompt ?? "";
  const requestBody = event.request?.body ?? {};
  const single = choices.length === 1;

  return choices.filter(hasLogprobs).map((choice) => {
    const index = choice.index ?? 0;
    return {
      // Unique per variation: the API returns one id for the whole request.
      id: single ? body.id : `${body.id}-${index}`,
      object: "text_completion",
      created: body.created ?? Math.floor(new Date(event.createdAt).getTime() / 1000),
      model: body.model,
      previous_response_id: null,
      prompt: {
        text: promptText,
        index: 0,
        // Single-element array is the viewer's own convention for a prompt it
        // has no token breakdown for; extractPromptTextSafe reads prompt.text
        // first either way.
        logprobs: { tokens: promptText ? [promptText] : [], token_logprobs: [], top_logprobs: [] },
      },
      choices: [{
        text: choice.text,
        index: 0,
        logprobs: choice.logprobs,
        finish_reason: choice.finish_reason ?? null,
      }],
      usage: body.usage,
      max_tokens: requestBody.max_tokens ?? null,
      temperature: requestBody.temperature ?? null,
      meta: {
        target_max_tokens: requestBody.max_tokens ?? null,
        // Provenance, so an entry can be told apart from ones the viewer made.
        ape_source: "resampling-study",
        ape_event_id: event.id,
        ape_mode: event.mode,
        ape_choice_index: index,
        ape_variations_in_request: choices.length,
      },
      request: requestBody,
      prompt_text: promptText.trim(),
    };
  });
}

export function mergeById(existing, added) {
  const byId = new Map(existing.map((entry) => [entry.id, entry]));
  let replaced = 0;
  for (const entry of added) {
    if (byId.has(entry.id)) {
      replaced += 1;
    }
    byId.set(entry.id, entry);
  }
  return { entries: [...byId.values()], replaced };
}

async function readJsonArray(path) {
  try {
    const parsed = JSON.parse(await readFile(path, "utf8"));
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    if (error.code === "ENOENT") {
      return [];
    }
    throw error;
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  const events = await readJsonArray(options.logPath);
  if (!events.length) {
    throw new Error(`No events found in ${options.logPath}.`);
  }

  const added = events.flatMap(eventToEntries);
  const skipped = events.length - new Set(added.map((e) => e.meta.ape_event_id)).size;

  const existing = options.merge ? await readJsonArray(options.outPath) : [];
  const { entries, replaced } = mergeById(existing, added);

  console.log(`Source:   ${options.logPath}`);
  console.log(`  ${events.length} events -> ${added.length} viewer entries (${skipped} events had no legacy logprobs)`);
  console.log(`Target:   ${options.outPath}`);
  console.log(`  ${existing.length} existing + ${added.length - replaced} new${replaced ? ` (${replaced} updated in place)` : ""} = ${entries.length} entries`);

  if (options.dryRun) {
    console.log("\nDry run: nothing written.");
    return;
  }

  if (options.merge && options.backup && existing.length) {
    await copyFile(options.outPath, `${options.outPath}.bak`);
    console.log(`Backup:   ${options.outPath}.bak`);
  }

  const json = options.pretty
    ? `${JSON.stringify(entries, null, 2)}\n`
    : JSON.stringify(entries);
  await writeFile(options.outPath, json, "utf8");
  console.log(`\nWritten (${(json.length / 1e6).toFixed(1)} MB${options.pretty ? ", indented" : ""}).`);
  console.log(`Serve ${dirname(options.outPath)} and open logprobs.html.`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    console.error(`Error: ${error.message}`);
    process.exitCode = 1;
  });
}
