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
