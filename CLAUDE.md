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

**Resample** (iterative APE, Fig. 3):

```
Generate a variation of the following instruction while keeping the semantic meaning.

Input: [INSTRUCTION]
Output: <COMPLETE>
```

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
  penalties 0.0) je **odvození**, ne fakt z repa — resampling je generování
  kandidátů, takže je to ten správný ze dvou configů, ale nikdo to nenapsal.

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

### Doporučené cesty

**A. Doslovná replika, do 28. 9. 2026.** Jediný rozumný kandidát je
`gpt-3.5-turbo-instruct` ve forward módu — jediný zbylý model InstructGPT linie
na completions endpointu. Ověřit napřed jedním requestem, že `echo` + `logprobs`
projde. Base modely (`davinci-002`, `babbage-002`) splňují mechaniku, ale ne
zero-shot following, a článek sám měří, že nesladění scoring a execution modelu
výkon výrazně sráží — dostaneš validní běh, jen měříš něco jiného.

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
