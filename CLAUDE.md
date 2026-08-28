# APE — kontext k článku a jeho reprodukci

Předávka z chatové konverzace (srpen 2026). Týká se článku
**"Large Language Models Are Human-Level Prompt Engineers"** (Zhou et al.,
arXiv:2211.01910) a jeho referenční implementace
`github.com/keirp/automatic_prompt_engineer`.

Vše níže je ověřené proti naklonovanému repu a proti aktuální dokumentaci
OpenAI, ne z paměti.

---

## 1. Šablony

### Znění v článku (Appendix, Table 5)

`[APE]` / `<INSERT>` = místo pro generování, `<COMPLETE>` = konec promptu.

**Forward generation** (návrh instrukcí, Fig. 2):

```
I gave a friend an instruction and five inputs. The friend read the
instruction and wrote an output for every one of the inputs.
Here are the input-output pairs:

Input: [Q1]
Output: [A1]
...

The instruction was <COMPLETE>
```

**Reverse (insert) generation, template 1** — instruction induction:

```
I instructed my friend to <INSERT>. The friend read the instruction and
wrote an output for every one of the inputs.
Here are the input-output pairs:

Input: [Q1]
Output: [A1]
...
```

**Reverse generation, template 2** — TruthfulQA:

```
Professor Smith was given the following instructions: <INSERT>
Here are the Professor's responses:

Q: [Q1]
A: [A1]
Q: [Q2]
A: [A2]
```

**Resample** (iterative APE, Table 5 „Resample Instruction" + Fig. 3):

```
Generate a variation of the following instruction while keeping the semantic meaning.

Input: [INSTRUCTION]
Output:<COMPLETE>
```

Ověřeno 28. 8. 2026 proti PDF (`arxiv.org/pdf/2211.01910`, `pdftotext -layout`,
Table 5 je na str. 19). Detaily k té verifikaci níže v §1.1.

**Vyhodnocení** (zero-shot; few-shot přidá před testovací vstup další `Input:/Output:` páry):

```
Instruction: [INSTRUCTION]
Input: [Q_test]
Output: <COMPLETE>
```

**Zero-shot-CoT hledání** — model doplňuje jen pokračování za „Let's",
odtud výsledné „Let's work this out in a step by step way to be sure we
have the right answer.":

```
Instruction: Answer the following question.
Q: [INPUT]
A: Let's <INSERT>. [OUTPUT]
```

### 1.1 Verifikace resample šablony proti PDF (28. 8. 2026)

Šablona v repu není (viz níže), takže jediný zdroj pravdy je článek. Ověřeno
tedy přímo proti PDF, ne proti téhle předávce.

**Jak je Table 5 vytištěná** (`pdftotext -layout`, str. 19):

```
Resample Instruction  Generate a variation of the following instruction while keeping the semantic
                      meaning.

                      Input: [INSTRUCTION]\nOutput:<COMPLETE>
```

První dva řádky jsou jen zalomení jedné věty. Po dekódování a odebrání
`<COMPLETE>` (značka místa doplnění, ne text promptu):

```
"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: [INSTRUCTION]\nOutput:"
```

`buildResamplingPrompt('[INSTRUCTION]')` v `resample-prompt.mjs` vrací **bajt
za bajt totéž**. Věta má 85 znaků v obou.

**Pozor: Table 5 zapisuje newlines nekonzistentně.** `\n` uvnitř řádku píše
explicitně, ale zlomy mezi bloky nechává typograficky. Nedá se z ní tedy přímo
vyčíst, jestli je mezi „meaning." a „Input:" jeden newline nebo dva. Kalibrace
přes řádek, u kterého skutečnou odpověď známe z kódu:

| Řádek Table 5 | jak je vytištěný | skutečný kód |
|---|---|---|
| Zero-shot Evaluation | `Instruction: [INSTRUCTION]` ⏎ `Input: [ ]\nOutput:<COMPLETE>` | `Instruction: [PROMPT]\n\nInput: [INPUT]\nOutput: [OUTPUT]` |

Table 5 tedy zobrazuje `\n\n` jako pouhý zlom řádku — newlines **podreprezentuje**,
ne nadreprezentuje. Když je u resample řádku vytištěný celý prázdný řádek, je to
nejméně `\n\n`. Potvrzuje to i Fig. 3, kde je v rámečku promptu mezi „meaning."
a „Input: write the antonym of the word." taky mezera. `\n\n` je tedy dobře
podložené čtení, ne odhad.

**Rozpor uvnitř článku, o jednu mezeru:**

- Table 5: `Output:<COMPLETE>` — bez mezery
- Fig. 3: `Output: <COMPLETE>` — s mezerou

Table 5 je v tom konzistentní napříč všemi řádky (`Output:<COMPLETE>`,
`The instruction was<COMPLETE>`, `to<INSERT>`), takže výjimka je Fig. 3.
Ostatní šablony výše v této sekci mají v předávce zapsané `Output: <COMPLETE>`
s mezerou — to je znění Fig. 3, ne Table 5.

Na kód to nemá vliv v žádném čtení. Posílá se `Output:` a tím prompt končí. Podle
Table 5 je to přesná shoda; podle Fig. 3 by prompt končil `Output: ` s mezerou,
kterou ale `llm.py:156` (`.strip()`) stejně odstraní. Obě čtení se sbíhají.

Na okraj: Fig. 3 používá jako příklad `write the antonym of the word.`, což je
default v UI i v README.

### Znění v kódu (liší se od článku!)

`experiments/run_instruction_induction.py`:

```python
eval_template       = "Instruction: [PROMPT]\n\nInput: [INPUT]\nOutput: [OUTPUT]"
prompt_gen_template = ("I gave a friend a instruction. Based on the instruction they produced "
                       "the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]")
demos_template      = "Input: [INPUT]\nOutput: [OUTPUT]"
```

Ano, včetně překlepu „a instruction". Demonstrace se spojují `\n\n`.

`experiments/run_truthful_qa.py` (insert model):

```python
eval_template       = "Instructions: [PROMPT].\n\nQ: [INPUT]\nA: [OUTPUT]"
prompt_gen_template = "You are given the following instructions: [APE]. Now please answer the following questions.\n\n[full_DEMO]"
demos_template      = "Q: [INPUT]\nA: [OUTPUT]"
```

### Resample šablona v kódu vůbec není (ověřeno 28. 8. 2026)

Grep na `variation|resample|semantic` přes všechny `*.py`, `*.yaml`, `*.md`
a `demo.ipynb` v naklonovaném repu: **nula výskytů.** `generate.py` umí jen
forward generation z input-output demonstrací (`generate_prompts` →
`get_query` → `prompt_gen_template.fill(...)`). Iterativní APE / Monte Carlo
resampling z Fig. 3 v referenční implementaci není.

Důsledky:

- Pro resample šablonu **neexistuje verze „v kódu"**, se kterou by se znění
  z článku mohlo rozcházet. Text článku je jediný zdroj pravdy.
- Neexistuje ani doložený API config pro resample krok. Použití generačního
  configu (`instruction_induction.yaml`: temp 0.9, top_p 0.9, max_tokens 50,
  penalties 0.0) je **odvození**. Jak přesně, viz níže.

#### Jak je ten config odvozený

Fakta z repa. Každý YAML má **tři** bloky `gpt_config` — `generation`,
`evaluation`, `demo` — ale jen **dvě** různé sady hodnot:

| Sada | Hodnoty | Bloky | Kdo ji volá |
|---|---|---|---|
| navrhování | temp 0.9, top_p 0.9, max_tokens 50 | `generation` | `generate.generate_prompts` (`ape.py:143`) |
| vykonávání | temp 0.7, top_p 1.0, max_tokens 200 | `evaluation`, `demo` | scoring a `demo_function` (`ape.py:157`) |

Odvození: resample krok **vyrábí instrukci**, patří tedy do první sady.
Tři argumenty, od nejsilnějšího:

1. `ape.py:204` čte `conf['generation']['model']['gpt_config']['max_tokens']`
   do proměnné pojmenované **`max_prompt_len`**. Autoři tím explicitně říkají,
   že `generation.max_tokens` znamená „maximální délka vygenerované instrukce".
   Resamplovaná instrukce je instrukce, takže platí stejná mez.
2. `max_tokens` 50 vs. 200 — 50 je dimenzované na instrukci, 200 na odpověď.
3. temp 0.9 + top_p 0.9 vs. 0.7 + top_p 1.0. Resampling je v článku rámovaný
   jako Monte Carlo *explorace* okolo dobrého kandidáta, chce tedy tu
   diverzifikovanější sadu.

Co zůstává neznámé:

- **Článek nespecifikuje sampling hyperparametry pro žádný krok**, ani pro
  původní navrhování. Nemůže tedy nic rozhodnout — proto ty hodnoty pocházejí
  z YAML, ne z textu.
- Blok `resample:` v žádném z `default.yaml`, `bandits.yaml`,
  `instruction_induction.yaml`, `truthful_qa.yaml` ani v `config.py` není.
  Pokud měl nezveřejněný iterativní kód vlastní config, nezůstala po něm stopa.
- Tedy: **příslušnost do sady „navrhování" je dobře podložená, konkrétní
  hodnoty jsou zděděné předpokladem.**

Empiricky (28. 8. 2026, `gpt-3.5-turbo-instruct`, `n=3`): krátká instrukce
i celý meta-prompt se resamplují bez ořezu, všechny `finish_reason: stop`,
~13–15 tokenů na variantu. Zděděná mez 50 tedy v praxi nepřekáží; naráželo by
to teprve u instrukcí blížících se 50 tokenům.

Jediné číslo, které k iterativnímu APE článek uvádí, je v §5.3: kvalita se
stabilizuje **po třech kolech** resamplingu („we observe diminishing returns
to further selection rounds"). Počet variant na kolo neuvádí.

Mechanika, kterou má smysl kopírovat z `llm.py:148-169` (`__generate_text`):

1. `prompt[i].replace('[APE]', '').strip()` — prompt se před odesláním
   **stripuje**, takže končí `Output:` bez mezery na konci.
2. Návrat je `response['choices'][i]['text']` **bez `.strip()`** — úvodní
   mezera z completion je součástí záznamu.
3. `config['n'] = n`, jeden `Completion.create` na celý batch. Ne smyčka.
4. `ape.py:147` dělá `list(set(prompts))` — deduplikace a ztráta pořadí.
5. Retry: nekonečná smyčka s `time.sleep(5)`; `auto_reduce_n` půlí `n` při
   překročení batch limitu.

---

## 2. Parametry API

Vše přes legacy `openai.Completion.create(**gpt_config, prompt=…)`.
`gpt_config` se bere doslova z YAML, doplní se jen `n`.

**Generování kandidátů** (`experiments/configs/instruction_induction.yaml`):

```yaml
model: text-davinci-002
temperature: 0.9
max_tokens: 50
top_p: 0.9
frequency_penalty: 0.0
presence_penalty: 0.0
```

- `n = num_prompts_per_subsample` (30 v experimentu, 50 v `default.yaml`)
- `batch_size: 500` — `prompt` je **list** stringů, jeden request = až 500 promptů × n completions
- limit 50 tokenů je důvod, proč menší modely ořezávaly instrukce (Fig. 12)

**Vyhodnocení, execution accuracy:**

```yaml
model: text-davinci-002
temperature: 0.7
max_tokens: 200
top_p: 1.0
batch_size: 20
```

Volá se `generate_text(queries, n=1)`. Pozor: exekuce instrukce **neběží na
temperature 0**, ale 0.7. Pro vlastní měření dát 0, ta 0.7 jen přidává šum.

**Vyhodnocení, log-probability:** tentýž config, ale v `__log_probs` se přepíše:

```python
config['logprobs']   = 1
config['echo']       = True
config['max_tokens'] = 0
```

Před text se přidá `\n` (a offsety se pak o 1 posouvají zpět), aby existoval
token, ke kterému lze přiřadit logprob. Skóruje se jen znakový rozsah odpovídající
výstupu `A`: do promptu se dočasně vloží sentinel `[[[[OUTPUT]]]]`, najde se jeho
pozice a přes `text_offset` se vyberou příslušné tokeny (`get_token_indices`
v `automatic_prompt_engineer/llm.py`).

**Insert mód:** prompt se rozřízne na `[APE]` a posílá se
`openai.Completion.create(**config, prompt=prefix, suffix=suffix)`
s vynuceným `batch_size == 1`.

---

## 3. Rozpočet vzorků

- Instruction induction: `num_subsamples=3` × `num_demos=5` ×
  `num_prompts_per_subsample=30` → **90 kandidátů**.
  `default.yaml` má 5 × 5 × 50 → **250** (to je číslo z analýz v článku).
- Výběr: UCB bandit (`configs/bandits.yaml`) — `rounds: 5`,
  `num_prompts_per_round: 50`, `c = 1.0`, base metrika likelihood, 50 vzorků na kolo.
- Data: `prompt_gen_size = min(0.5 × |induce|, 100)`; během hledání se skóruje na
  `min(20, |eval|)` vzorcích, finální test na `min(100, |test|)`.
- TruthfulQA: 100 z 817 otázek jako trénink, 200 kandidátů, výběr top 10.

---

## 4. Co z toho jde spustit dnes (srpen 2026)

### Stav modelů

| Model | Role | Stav |
|---|---|---|
| `text-davinci-002` | **hlavní model článku** | mrtvý od 4. 1. 2024 |
| `code-davinci-002` | base pod ním, insert mód | mrtvý 2023 |

| `gpt-3.5-turbo-instruct` | nástupce InstructGPT řady | **končí 28. 9. 2026** |
| `davinci-002` | base, analog `davinci` | **končí 28. 9. 2026** |
| `babbage-002` | base, analog `babbage` | **končí 28. 9. 2026** |
| `gpt-3.5-turbo-completions` ap. | chat model za completions fasádou | končí 23. 10. 2026 |

Náhrada u všech je `gpt-5.6-terra`, ale ta už není na completions endpointu.

**Co API vrátí na mrtvý model** (ověřeno 28. 8. 2026, resampling prompt
s configem z článku, `n=30`). `HTTP 404 Not Found`, tělo:

```json
{
    "error": {
        "message": "The model `text-davinci-002` has been deprecated, learn more here: https://platform.openai.com/docs/deprecations",
        "type": "invalid_request_error",
        "param": null,
        "code": "model_not_found"
    }
}
```

Totožné pro `code-davinci-002`. Endpoint tedy request neodmítne kvůli
parametrům — celý `gpt_config` z článku včetně `n`, `top_p` a obou penalties
projde validací bez připomínky a padne teprve na neexistujícím modelu.
Mechanika legacy completions je pořád živá, chybí jen ty váhy.

**Pozor na záměnu:** `davinci-002` ≠ `text-davinci-002`. První je base model
netrénovaný na following instrukcí (náhrada za `curie`/`davinci`), druhý byl
instruct model. Sufix `-002` je u obou náhoda. OpenAI mapovalo náhrady podle
linie: base → base, instruct → instruct, proto všech šest `text-*-00X` šlo na
`gpt-3.5-turbo-instruct`.

### Dvě věci, které reprodukci blokují

1. **Likelihood scoring je mrtvý.** APE počítá `log P(A | [ρ; Q])` přes
   `echo=True, max_tokens=0, logprobs=1` — tedy logprob **vstupních** tokenů.
   Moderní API (Chat Completions i Responses) vrací logprobs jen pro tokeny,
   které model **vygeneroval**. Teacher forcing přes vstup nejde nikde.

   **Nezaměňovat s `logprobs` samotným, které živé je.** Ověřeno 28. 8. 2026:

   | | `completions` / `gpt-3.5-turbo-instruct` | `responses` / `gpt-5.5` |
   |---|---|---|
   | `echo` + `logprobs` | 400 `Setting 'echo' and 'logprobs' at the same time is not supported for this model.` | ❌ |
   | `logprobs` samo | ✅ strop **20** | 400 `logprobs are not supported with reasoning models.` |

   Reasoning modely odmítají všechny tři formy — `top_logprobs`,
   `include: ["message.output_text.logprobs"]` i obojí naráz.

   Nad 20 completions endpoint **tiše kappuje** na 20, nehlásí chybu
   (`logprobs: 21` → 20 klíčů, HTTP 200). Proto to server validuje sám.

   `logprobs: 0` už vrací `token_logprobs` a `text_offset`, jen bez
   `top_logprobs`. Na přesnou `P(sekvence)` to stačí a je to nejlevnější
   varianta co do objemu dat.

   **Logprobs jsou surové, nezkreslené `temperature` ani `top_p`.** Distribuce
   prvního tokenu je identická na 4 desetinná místa při temp 0 / 0.9 / 1 / 2
   a top_p 0.1 / 0.9 / 1. Vrací tedy model, ne tvoje nastavení. Kdo chce
   pravděpodobnost pod samplovacím configem článku, musí si obě transformace
   dopočítat sám — obojí je deterministická funkce surové distribuce.

   Praktický důsledek: `P(sekvence)` je součet `token_logprobs`, tedy exaktní
   z jednoho requestu, místo odhadu z opakovaného samplování. Naopak celou
   distribuci nad instrukcemi to nedá — prostor je kombinatorický a top-20
   na pozici pokryje jen část masy (u resample promptu ~85 % na prvním tokenu).

   Vedlejší zjištění k duplikátům: `top_p 0.9` po `temperature 0.9` nechá
   z 20 kandidátů na prvním tokenu jen **10** s nenulovou šancí. To je
   analytické vysvětlení duplikátů, které jinak vycházejí ze samplování.
2. **Reverse mód neexistuje.** Insert-trénované modely zmizely v lednu 2024.
   Padá „Reverse Generation 1/2" i celý TruthfulQA experiment.

### Mapování parametrů

| Parametr APE | legacy completions | Chat Completions | Responses API |
|---|---|---|---|
| `prompt` (list) | ✅ batch až 500 | ❌ | ❌ |
| `n` | ✅ | ✅ | ❌ paralelní requesty |
| `temperature`, `top_p` | ✅ | ✅, reasoning modely ignorují | totéž |
| `max_tokens` | ✅ | `max_completion_tokens` | `max_output_tokens` |
| `frequency_penalty`, `presence_penalty` | ✅ | ✅ legacy | ❌ |
| `logprobs` (int) | ✅ | `logprobs: true` + `top_logprobs` 0–5 | `include: [...logprobs]` |
| `echo` + `max_tokens: 0` | ✅ | ❌ | ❌ |
| `suffix` (insert) | endpoint bere, žádný model netrénovaný na FIM | ❌ | ❌ |

**Oprava k řádku `temperature`, `top_p` — ověřeno 28. 8. 2026 na `gpt-5.5`
přes Responses API.** Reasoning modely ty parametry **neignorují, ale request
odmítnou**:

```
temperature: 1     -> 200 OK   (jediná přijatá hodnota = default, tedy no-op)
temperature: 0.9   -> 400  Unsupported parameter: 'temperature' is not supported with this model.
temperature: 0     -> 400  Unsupported parameter: 'temperature' is not supported with this model.
top_p: 0.9         -> 400  Unsupported parameter: 'top_p' is not supported with this model.
```

Na reasoning modelu tedy z celé tabulky parametrů článku projde jediný:
`max_output_tokens`. A i ten má dvě pasti:

- minimum je **16**, níž vrací 400;
- **reasoning tokeny se do něj počítají a berou se první.** Při hodnotě 50
  z článku spotřebuje uvažování ~39 tokenů a text se uřízne v půli věty; při
  16 nezbyde nic a vrátí se prázdný string se statusem `incomplete`. Žádná
  chyba, jen prázdný výsledek — tichá past. Bez limitu to funguje.

Pozor při vlastním testování: `output_text` je pomocné pole SDK. Surový REST
ho nevrací, text je v `output[]` jako item typu `message`. Kdo si napíše
vlastní skript proti REST API a čte `output_text`, uvidí `undefined`
a bude si myslet, že model nic nevrátil.

### Doporučené cesty

**A. Doslovná replika, do 28. 9. 2026.** Jediný rozumný kandidát je
`gpt-3.5-turbo-instruct` ve forward módu — jediný zbylý model InstructGPT linie
na completions endpointu. Base modely (`davinci-002`, `babbage-002`) splňují
mechaniku, ale ne zero-shot following, a článek sám měří, že nesladění scoring
a execution modelu výkon výrazně sráží — dostaneš validní běh, jen měříš něco
jiného.

**Ověřeno na účtu 28. 8. 2026** (130 modelů, `GET /v1/models`):

- Completions endpoint bere `gpt-3.5-turbo-instruct`, `gpt-3.5-turbo-instruct-0914`,
  `davinci-002`, `babbage-002`. `gpt-3.5-turbo-completions` z tabulky výše na
  účtu **není**.
- Na resample promptu (`n=2`, temp 0.9, max_tokens 50) obě instruct varianty
  vrací čisté parafráze s `finish_reason: stop`. `davinci-002` a `babbage-002`
  instrukci nenásledují vůbec: odbočí do textu nebo si dogenerují další
  `Input:/Output:` páry, vždy `finish_reason: length`. `babbage-002` vrátil
  „write the **synonym** of the word" — tedy obrácený význam.
- **`echo` + `logprobs` je odmítnuto i na `gpt-3.5-turbo-instruct`:**
  `HTTP 400 — Setting 'echo' and 'logprobs' at the same time is not supported
  for this model.` Likelihood scoring tedy na OpenAI nejde nikde, ani do
  28. 9. Cesta A je doslovná jen pro generační větev, ne pro scoring.

**B. Otevřené váhy přes vLLM (doporučuju).** vLLM podporuje `echo`
i `prompt_logprobs`, takže původní kód jde replikovat doslova, včetně
likelihood větve. Žádný deadline, volitelná velikost modelu i instruct/base
varianta. Zároveň jediná cesta, jak zopakovat škálovací analýzu
(`ada`→`davinci`), protože řada base modelů různých velikostí u OpenAI
neexistuje.

**C. Jen exec-accuracy větev.** Ta funguje na čemkoli včetně chat modelů,
vyžaduje jen úpravu `utility.py` — chat modely na formát `Output:` reagují
ukecaně a rozbíjejí exact match.

### Co nereprodukuješ nikdy

Konkrétní čísla z Fig. 4 (IQM 0.810 vs. 0.749 human). Hlavní model neexistuje.
Replikuješ mechaniku, ne výsledek.

---

## 5. Navazující práce

`google-deepmind/opro` — „Large Language Models as Optimizers", přímý nástupce
APE. Posouvá myšlenku k iterativní optimalizaci s trajektorií předchozích
promptů a jejich skóre v kontextu. Nemá problém s mrtvým endpointem. Pokud jde
o metodu spíš než o repliku článku, jít tudy.

---

## 6. Stav pracovní kopie

Referenční implementace se klonuje takto:

```bash
git clone --depth 1 https://github.com/keirp/automatic_prompt_engineer.git
```

Klíčové soubory: `automatic_prompt_engineer/llm.py` (volání API, scoring),
`configs/default.yaml` a `configs/bandits.yaml`, `experiments/configs/*.yaml`,
`experiments/run_*.py` (skutečné šablony), `evaluation/likelihood.py`,
`experiments/evaluation/instruction_induction/exec_accuracy.py`.

Kód je psaný proti `openai` SDK v0.x (`openai.Completion.create`). Na dnešním
SDK v1.x je potřeba přepsat na `client.completions.create` nebo připnout starou
verzi.
