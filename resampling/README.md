# APE Resampling Prompt MVP

Minimalni CLI podle resampling casti paperu [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/pdf/2211.01910) a jejich repozitare [keirp/automatic_prompt_engineer](https://github.com/keirp/automatic_prompt_engineer).

Program sklada prompt podle Figure 3:

```text
Generate a variation of the following instruction while keeping the semantic meaning.

Input: [INSTRUCTION]
Output:
```

V paperu je `Output: <COMPLETE>` znazorneni mista, kde ma language model doplnit novou instrukci. Pri volani moderniho API se proto posila `Output:` jako completion cue.

Tato sablona v referencni implementaci **neni** — `generate.py` umi jen forward
generation z input-output demonstraci. Zneni z clanku je jediny zdroj pravdy,
viz `CLAUDE.md` §1.

## Dva backendy

| | `completions` (vychozi) | `responses` |
|---|---|---|
| Endpoint | `/v1/completions` | `/v1/responses` |
| Vychozi model | `gpt-3.5-turbo-instruct` | `gpt-5.5` |
| Pole s promptem | `prompt` | `input` |
| `n` variant | jeden request s `n=N` | N requestu |
| `temperature`, `top_p` | 0.9 / 0.9 | **odmitnuto modelem** |
| `max_tokens` | 50 | `max_output_tokens`, min. 16, model default |
| `frequency_penalty`, `presence_penalty` | 0.0 / 0.0 | nepodporovano |
| `logprobs` | 20 (= strop) | **odmitnuto modelem** |
| Konec zivota | **2026-09-28** | — |

Na `gpt-5.5` neni `temperature` nastavitelna: jina hodnota nez vychozi 1 vraci
`400 Unsupported parameter`. Totez `top_p`. Z parametru clanku projde na
reasoning modelu jen `max_output_tokens` — a reasoning tokeny se do nej pocitaji
a beru se prvni, takze hodnota 50 z clanku text ureze a 16 vrati prazdny
vysledek. Nechavej to pole prazdne. Detaily v `CLAUDE.md` §4.

`completions` reprodukuje mechaniku `GPT_Forward.__generate_text` z `llm.py`
vcetne `n` v jednom requestu a hodnot z `experiments/configs/instruction_induction.yaml`.
`responses` je moderni cesta; reasoning modely cast tech parametru odmitaji,
takze jeho vychozi stav je prazdny request a rozhoduje model.

Prazdne pole = backend default. Hodnota `off` = parametr se neposle vubec.

Log uklada `backend` u kazde udalosti a `rawPrompt` s nestripnutym textem
z API, protoze referencni implementace completions nestripuje.

## Logprobs

`logprobs` neni z clanku — je to jediny parametr navic, ktery `completions`
backend posila. Je ciste observacni, nemuze zmenit vygenerovane tokeny, takze
replika tim netrpi. Pro telo requestu doslova identicke s
`instruction_induction.yaml` staci `--logprobs off`.

Z `token_logprobs` se scita `logprob` cele sekvence a `probability = exp(logprob)`.
Oboji se uklada do logu u kazde varianty a zobrazuje u vysledku:

```
1. enter the opposite of the term.  [logprob -4.5203, p=1.089e-2]
2. find the opposite of the word.   [logprob -6.1868, p=2.056e-3]
3. type the opposite of the term.   [logprob -3.7906, p=2.258e-2]
```

Tri veci, ktere je dobre vedet:

- Logprobs jsou **surove** — nezkreslene `temperature` ani `top_p`. Vraci
  distribuci modelu, ne toho, co ti nastaveni nasamplovalo.
- Strop je 20 a endpoint nad nim **tise kappuje**, nehlasi chybu. Server to
  proto validuje sam.
- `logprobs: 0` uz vraci `token_logprobs`, tedy presnou `P(sekvence)`, jen bez
  alternativ na pozici. Nejlevnejsi varianta co do objemu logu.
- Nekdy je `logprob` u varianty `null`. To neni chyba: endpoint vraci pro
  nektere nasamplovane tokeny sentinel `-9999` misto pravdepodobnosti, a jeden
  neznamy faktor dela cely soucin neznamym. UI v takovem pripade skore vynecha.

Vychozich 20 je strop endpointu a plati se objemem logu, ne tokeny:

| `logprobs` | odpoved (n=1) | ~ na request pri n=30 |
|---|---|---|
| vypnuto | 316 B | 9 kB |
| 0 | 602 B | 18 kB |
| 5 | 1 448 B | 42 kB |
| 20 | 3 661 B | **107 kB** |

Pri delsim experimentovani s `n=30` roste `data/prompt-log.json` rychle. Kdyz
alternativy na pozici nepotrebujes, `--logprobs 0` da tez presne `P(sekvence)`
za sestinu objemu.

Reasoning modely logprobs odmitaji ve vsech formach, takze na `responses`
backendu je to pole neaktivni. Detaily v `CLAUDE.md` §4.

### Prohlizeni v Logprobs Vieweru

[OpenAIRes/logprobs](https://github.com/OpenAIRes/logprobs) je HTML prohlizec
tokenovych logprobs. Uz umi legacy format, tedy presne ten nas tvar
(`tokens`, `token_logprobs`, `top_logprobs`, `text_offset`).

```bash
git clone https://github.com/OpenAIRes/logprobs.git logprobs
```

Prevod nasich behu do jeho formatu:

```powershell
node .\export-logprobs.mjs --dry-run
node .\export-logprobs.mjs
```

Pak naservirovat `logprobs/` a otevrit `logprobs.html`.

Dve veci, ktere prevodnik resi:

- Viewer cte vsude `choices[0]`, takze **jeden request s `n>1` se rozpada na
  jednu polozku za variantu** (`id` dostane suffix `-index`). Jinak by byla
  videt jen prvni varianta.
- Zapis **sluceje podle `id`**, neprepisuje soubor, takze existujici historie
  vieweru zustava. Pred zapisem se dela `.bak`.

Nase polozky maji v `meta` znacku `ape_source: "resampling-study"` plus
`ape_event_id`, `ape_mode` a `ape_choice_index`, aby se odlisily od tech, ktere
si viewer vygeneroval sam.

Slozka `logprobs/` je v `.gitignore` — je to samostatny repozitar.

## Pouziti

Webove UI:

```powershell
$env:OPENAI_API_KEY="sk-..."
node .\server.mjs
```

Pak otevri `http://localhost:8787`.

V UI jsou dva mody:

- `Vlastni`: do `INSTRUCTION` jde text z textarea.
- `Meta`: do `INSTRUCTION` jde samotna resampling instrukce z paperu.

Pod nimi je prepinac backendu. Prepnuti prepise placeholdery parametru na
hodnoty daneho backendu a nepodporovana pole zneaktivni. `Request body`
v pravem panelu ukaze presne telo requestu, ktere se posle.

UI contains a `Prompt History` table. The complete append-only event log is stored in:

```text
data/prompt-log.json
```

On first run, the log is created with a seed event: the default Resampling Prompt with the `[INSTRUCTION]` placeholder. Each successful OpenAI call is then stored as a complete event with the request body, parsed response JSON, and generated prompt. The UI table is derived from `generatedPrompts` inside those events.

Tabulka `Prompt History` ma sloupce `API Request` a `Response`, oba rozbalitelne
na cele telo. Souhrn u odpovedi ukazuje HTTP status, tokeny (u reasoning modelu
i z toho reasoning cast), `finish_reason` a pocet variant, ktere ten jeden
request vratil. Token usage plati na cely request, takze pri `n>1` maji vsechny
radky teze cislo — proto ten stitek `request of N`.

Tabulka se ridi prepinacem modu: v modu `Meta` vidis jen meta behy, v modu
`Vlastni` jen custom. Seed radek je v obou, protoze to je sablona, ze ktere
oba mody vychazeji, ne beh jednoho z nich. Pocet skrytych radku je pod
nadpisem. Filtruje se na klientovi z jednoho fetche, prepnuti tedy neposila
novy dotaz.

Nahled promptu a request body bez API volani:

```powershell
node .\resample-prompt.mjs --instruction "write the antonym of the word." --dry-run
```

Volani OpenAI API:

```powershell
$env:OPENAI_API_KEY="sk-..."
node .\resample-prompt.mjs --instruction "write the antonym of the word."
```

Vice variant (jeden request s `n=5`):

```powershell
node .\resample-prompt.mjs --instruction "write the antonym of the word." --count 5
```

Moderni backend:

```powershell
node .\resample-prompt.mjs --instruction "write the antonym of the word." --backend responses
```

Vypnuti jednoho parametru:

```powershell
node .\resample-prompt.mjs --instruction "..." --temperature off
```

JSON vystup:

```powershell
node .\resample-prompt.mjs --instruction "write the antonym of the word." --count 3 --json
```

Instrukce ze souboru:

```powershell
node .\resample-prompt.mjs --input-file .\instruction.txt --count 5
```

## Konfigurace

CLI bere `OPENAI_API_KEY` z prostredi. Backend, model a endpoint lze prepsat
argumentem nebo promennou:

```powershell
node .\resample-prompt.mjs --instruction "..." --backend responses --model gpt-5.5
```

- `OPENAI_BACKEND` — `completions` (vychozi) nebo `responses`
- `OPENAI_MODEL` — prepise model daneho backendu
- `OPENAI_API_URL` — prepise endpoint

Vychozi hodnoty parametru jsou vzdy hodnoty vybraneho backendu, viz tabulka vyse.
`node .\resample-prompt.mjs --help` je vypise.

## Overeni

```powershell
node --test .\resample-prompt.test.mjs .\prompt-log.test.mjs
```

## Otevreni logprobs z resamplingu

U vysledku a radku historie s ulozenymi tokenovymi logprobs je odkaz
`Zobrazit logprobs`. Otevre konkretni variantu v nove zalozce bez volani API.
Funguje i pro vice variants v jednom requestu a pro existujici historii.

Server pouziva primo viewer z nadrazene slozky `..` -- tenhle program je od
zari 2026 podslozkou `resampling/` v repozitari logprobs, drive samostatne repo
vedle nej (verze z 5. 9. 2026),
ktery je shodny s `Documents/ChatGPT/chat 2/logprobs.html`. Kopie v
`AI/logprobs` byla starsi (28. 8. 2026). Nevytvari se dalsi kopie vieweru.
Jine umisteni lze nastavit promennou `LOGPROBS_VIEWER_DIR` pred spustenim.

Na portu 8787 se viewer napojuje na historii resamplingu; nejde o cely
record-store server z projektu gpt. Server zpristupnuje HTML vieweru a jeho
sdilene skripty (app.css, theme.js, bar.js, ask-policy.js, approve-request.js,
greedy-branches.js, strings.html). Historii pro viewer sklada za behu
z prompt logu.

To uz ale neni jedina cesta: kazdy vysledek s logprobs se **nabidne i sdilenemu
store** (POST /api/save na 8899), takze completion_history.json se dnes doplnuje
-- drive to tenhle soubor vyslovne nedelal. Bez toho zustaval resamplingovy
retezec mimo dosah zebricku, greedy pohledu i jednotokenovych odchylek, protoze
vsechno tri umi store, ne tenhle server. Odkazy u vysledku proto vedou na 8899,
ne do zdejsi polovicni kopie vieweru. Kdyz store nebezi, volani se jen zaloguje
sem a odkaz zustane lokalni.

Testy propojeni: `node --test viewer-link.test.mjs`.

## Shared viewer and single-token variants

The sole maintained Logprobs Viewer HTML lives one directory up, in the package
root (`../logprobs.html`).
The old `AI/logprobs/logprobs.html` and `Documents/ChatGPT/chat 2/logprobs.html`
are forwarding pages, preserving query parameters and fragment. They contain no viewer implementation.
Historical JSON datasets stay in their existing locations.

Every resampling completion with logprobs offers **View logprobs** and
**One-token deviations**, both pointing at the store server.

The deviations are the question this program is usually asking: every position
of the string x every alternative recorded there becomes a new prompt, and the
row is what the model actually generates from it, so the tail after the change
is real. The store answers from its own records where it can and asks before it
buys the rest.

The neighbouring page, `single-token-variants.html`, calls the shared
`extracted/single_token_variants.py`, which reuses `plan_for` from
`sweep_alternatives.py`. It enumerates the recorded alternatives at each
completion position, changes exactly one token, **preserves the suffix**,
excludes unchanged text, and groups identical output strings. No API calls are
made and it costs nothing -- but the strings are ones the model never produced,
and the replacement logprobs describe the original prefix, not a score for the
modified string. A logprobs=0 result can have no alternatives. The two pages
link to each other; the link from here goes to the deviations, because that is
the one that was meant.

## Asking before a paid call

`/api/resample` refuses a request without `confirmed: true`. The call happens in
node, so the dialog cannot: the page shows the exact request body first -- the
same one `/api/preview` returns -- and sends the acknowledgement with it. The
policy behind that ("every call", "from 4096 tokens", "more than one call",
"never") is the viewer package's one setting, kept by the store server and
proxied here at `/api/ask_policy`, so one switch governs every part of the
program. With the store unreachable the client falls back to asking every time,
and a page whose dialog did not load refuses to call at all.

Python defaults to the bundled Codex runtime. Override with `LOGPROBS_PYTHON`;
override the shared source folder with `LOGPROBS_VIEWER_DIR`.
The separate API-backed continuation sweep remains in `sweep_alternatives.py`.

## Reuse a Meta completion

Use **Use as meta template** on a Meta result or history row. Plain instructions get the Input/Output scaffold; existing templates with [INSTRUCTION] are retained. The action prepares a Meta preview only. Generate starts the next run. Saved events retain templateSource (source row, event and original completion), including when you edit the derived template. Reset to the paper clears this ancestry.
