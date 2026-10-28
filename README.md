# Scalars Over Semantics — Benchmark Data and Reproduction Guide

This repository accompanies **“Doge Does Research #4 — Scalars Over Semantics.”** It compares model outputs when a constrained menu-selection task is presented using descriptive menu text alone versus the same text enriched with structured attributes.

The published dataset contains **540 recorded runs** across three model families:

- Qwen 3.5 2B via Ollama
- Codestral via API
- Gemini Flash Lite via API

The experiment is exploratory. Results describe the recorded runs and should not be interpreted as guarantees of production compliance or as proof of a particular internal model mechanism.

## Repository contents

| File | Purpose |
|---|---|
| `prompt.txt` | Shared persona, task instructions and required JSON response shape |
| `menu.csv` | Baseline menu: dish names and descriptive text |
| `menu_enhanced.csv` | Same menu with additional macro and categorical attributes |
| `foodchoice_ollama.py` | Execution script for the local Ollama experiment |
| `drift_study_ollama.csv` | Recorded Qwen/Ollama results (180 runs) |
| `drift_study_codestral.csv` | Recorded Codestral results (180 runs) |
| `drift_study_gemini.csv` | Recorded Gemini results (180 runs) |

The result CSVs contain the recorded run-level selections, rejected dish, generated explanation fields, shuffled menu order and raw model response. The published CSVs do not contain a separate, adjudicated annotation column for each claimed failure type; examples in the annotation section below are proposed scoring examples, not labels already stored in the files.

## Experimental design

Each run gives the model the same persona and task prompt, a 27-dish menu, and the requirement to select 2–6 dishes and explicitly reject one dish.

Two input conditions are compared:

- **`simple` / Text-Only:** dish names and natural-language descriptions.
- **`enhanced` / Scalar-Augmented:** the same descriptions plus structured attributes, including estimated calories, protein, fat, carbohydrates, sugar, nutrient tags, spice and digestion labels.

The enhanced condition adds both numerical and categorical attributes. This design therefore tests the effect of **additional structured attributes as a package**; it does not isolate the causal effect of numbers alone.

The experiment has 3 models × 2 conditions × 3 temperature settings × 30 runs per configuration = 540 recorded runs.

| Model result file | Temperatures | Runs per condition | Total runs |
|---|---:|---:|---:|
| `drift_study_ollama.csv` | 0.0, 0.3, 0.7 | 90 | 180 |
| `drift_study_codestral.csv` | 0.0, 0.3, 0.7 | 90 | 180 |
| `drift_study_gemini.csv` | 0.0, 0.6, 1.5 | 90 | 180 |

Menu order is shuffled between calls to reduce positional bias. Shuffling reduces a potential confound; it does not prove that all order effects have been eliminated. Temperature settings are model/provider-specific and should not be treated as directly equivalent across models.

## Reproduce the local Ollama experiment

### Requirements

- Python 3.10+ recommended
- Ollama installed and running
- The `qwen3.5:2b` model available in Ollama
- The files `prompt.txt`, `menu.csv`, `menu_enhanced.csv` and `foodchoice_ollama.py` in the same directory

Install the Python dependencies:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install ollama python-dotenv tqdm
ollama pull qwen3.5:2b
```

Run the script from a **working copy** of the repository:

```bash
python foodchoice_ollama.py
```

The script defaults to `qwen3.5:2b`, temperatures `0.0`, `0.3`, and `0.7`, and 30 runs per condition/temperature. It writes `drift_study_ollama.csv` and `drift_study_ollama.json` in the script directory, plus a log file. **Do not run it in the only copy of the published data directory:** the output CSV has the same name as the archived result file and may overwrite it. Keep the released CSVs immutable and run experiments in a separate directory.

The script supports environment variables including:

- `OLLAMA_MODELS` — comma-separated Ollama model names
- `OLLAMA_THINK` — `true`, `false`, or `default`
- `OLLAMA_NUM_CTX` — context window (default `8192`)
- `OLLAMA_NUM_PREDICT` — maximum generated tokens (default `1024`)

For example, in a shell session:

```bash
# macOS/Linux
export OLLAMA_MODELS="qwen3.5:2b"
export OLLAMA_THINK="false"
export OLLAMA_NUM_CTX="8192"
export OLLAMA_NUM_PREDICT="1024"
python foodchoice_ollama.py
```

The exact output can vary with Ollama version, model build, hardware, decoding implementation and other runtime details. Record those details when conducting a new run.

## Codestral and Gemini runs

The recorded Codestral and Gemini CSVs are included in this repository. Their provider-specific execution scripts are **not** included.

The two API runners followed the same broad experiment structure as the Ollama runner—two input conditions, three temperature settings, 30 runs per configuration, shuffled menu order and the same task/output contract—with provider-specific API calls and configuration. The runners are withheld to keep the repository focused on the shared experiment and recorded results.

This explanation helps readers understand the intended consistency of the harness, but it is **not equivalent to publishing executable code**. As a result, the recorded API results can be inspected and analysed, but a reader cannot reproduce the original Codestral or Gemini API calls from this repository alone. New API runs may also differ because provider model aliases, service implementations and defaults can change. If exact execution reproducibility is a goal, publish the runner scripts or a shared provider-adapter implementation and record the exact model identifiers and settings used for the original runs.

## Inspect and summarize the recorded results

The following commands operate on the **existing CSVs** and do not make model/API calls. Install pandas if needed:

```bash
python -m pip install pandas
```

Count runs by model, condition and temperature:

```python
import pandas as pd
from pathlib import Path

files = {
    "Qwen/Ollama": "drift_study_ollama.csv",
    "Codestral": "drift_study_codestral.csv",
    "Gemini": "drift_study_gemini.csv",
}

for label, filename in files.items():
    df = pd.read_csv(filename)
    print(f"\n{label}: {len(df)} rows")
    print(df.groupby(["condition", "temperature"]).size())
```

Inspect a run and its original generated response:

```python
import pandas as pd

df = pd.read_csv("drift_study_codestral.csv")
row = df.loc[df["overall_run_id"].eq(27)].iloc[0]

print("Condition:", row["condition"])
print("Temperature:", row["temperature"])
print("Selected:", row["selected_dish_names"])
print("Raw response:\n", row["raw_response"])
```

These commands summarize the recorded outputs. They do not independently score policy violations, hallucinations or explanation quality.

## How to interpret the recorded outputs

The examples below illustrate useful distinctions when reviewing the raw outputs. They are examples for readers to inspect, **not formal labels already assigned to the CSV rows**. The published CSVs do not contain a fully adjudicated, per-run annotation dataset.

### 1. Selection / constraint violation

A selection violation occurs when a selected dish conflicts with a clearly specified hard constraint in the prompt. Keep hard constraints separate from softer preferences. Because the prompt does not assign a complete numerical threshold to every preference, any aggregate violation rate depends on an explicit scoring rule.

**Example to review — Qwen/Ollama, `overall_run_id = 1`:**

- Condition: `simple`; temperature: `0.0`.
- The run selects `Smoked Dal Makhani`.
- The baseline menu describes the dish as slow-cooked with butter and heavy cream; the enhanced menu also labels its digestion as `Heavy`.

If avoiding heavy-cream gravies is treated as a hard constraint, selecting this dish would count as a selection violation. Do not generalize this rule to every high-fat or high-calorie dish unless the scoring rubric explicitly defines those properties as hard exclusions.

### 2. Unsupported factual claim in an explanation

A claim is unsupported when the supplied prompt and menu do not substantiate it. Unsupported does not automatically mean false; those are different classifications.

**Example to review — Codestral, `overall_run_id = 27`:**

- Condition: `simple`; temperature: `0.0`.
- The run selects `Steamed Sticky Jasmine Rice`.
- Its generated explanation describes the rice as having a “low glycemic index.”

The baseline menu does not provide a glycemic-index value or claim. This is therefore a candidate unsupported claim relative to the supplied evidence. It should only be labelled false if an independent, suitable source establishes that it is false for the item as prepared.

### 3. Unsupported or contradicted attribute claim

Assess attribute claims against the exact menu condition supplied to that run. A claim may be unsupported in the text-only condition but contradicted by explicit metadata in the enhanced condition.

**Example to review — Codestral, `overall_run_id = 8`:**

- Condition: `simple`; temperature: `0.0`.
- The run selects `Mint & Lime Fizz (Virgin Mojito)`.
- Its explanation says the drink provides “a mild caffeine boost.”

The baseline description lists mint, lime, cane-sugar syrup, club soda and ice, but does not state that the drink contains caffeine. This is a candidate unsupported attribute claim. In the enhanced menu, the drink is tagged `Zero Caffeine`; the same caffeine-boost claim in that condition would contradict the supplied record.

### 4. Unsupported explanation versus incorrect selection

Score these separately. A model could choose a dish that is acceptable under the hard-constraint rules but justify it with an unsupported claim. Conversely, it could give a factually accurate explanation and still select a dish that violates a hard constraint. Neither metric should be used as a proxy for the other.

### 5. Schema / task compliance

Check whether the response is parseable JSON, follows the required structure, selects 2–6 menu items, uses names from the supplied menu, and includes an explicitly rejected dish. Where possible, record each failure separately.

For example, the archived Qwen CSV contains rows where `dish_count` is `0` or `12` (inspect `overall_run_id` 61 and 81). These are candidates for task-format/count failures against the prompt's 2–6 selection requirement. Inspect the raw response before deciding whether the failure reflects truncation, JSON parsing, or another cause.

## What the published CSVs do and do not establish

The CSVs let readers inspect the recorded selections and explanations and conduct new analyses. However:

- The CSVs do not contain adjudicated labels for every policy violation, unsupported claim or exact metric citation.
- The prompt contains qualitative preferences and constraints, but not a complete numeric decision function for all dishes.
- The enhanced condition adds both numeric macros and categorical tags, so the results do not isolate the effect of numeric scalars alone.
- A zero observed failure rate in a finite sample is not proof of a zero underlying failure rate.
- The current files are recorded outputs, not a guarantee that a rerun against a changing API or model version will produce identical outputs.

Accordingly, treat aggregate metrics as reproducible only when the corresponding definitions, run-level labels, statistical tests and analysis code are also available.
