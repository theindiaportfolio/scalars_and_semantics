import os
import json
import csv
import time
import random
import re
import logging
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import ollama
from ollama import Client

# --- 1. SETUP LOGGING & ENVIRONMENT ---
logging.basicConfig(
    filename="drift_benchmark.log",
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_path = os.path.join(parent_dir, '.env')

if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

# --- 2. EXPERIMENT CONFIGURATION ---
# Models to benchmark (comma-separated in OLLAMA_MODELS, e.g. "qwen3.5:2b,llama3.2:3b")
models_env = os.getenv("OLLAMA_MODELS", "qwen3.5:2b")
OLLAMA_MODELS = [m.strip() for m in models_env.split(",") if m.strip()]

# Thinking mode (Qwen3.5 is a hybrid reasoning model; small ones can burn thousands
# of tokens, or loop, before answering). OLLAMA_THINK values:
#   false (default) -> think=False      true -> think=True
#   default         -> don't send the parameter (use for models without thinking support)
_think_env = os.getenv("OLLAMA_THINK", "false").strip().lower()
if _think_env in ("1", "true", "yes", "on"):
    THINK = True
elif _think_env in ("default", "auto", "omit"):
    THINK = None
else:
    THINK = False

# Ollama silently truncates prompts that exceed the context window, so set it
# explicitly. The enhanced menu prompt is ~3.5k tokens and needs room for the reply.
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))  # cap on generated tokens

# Files located in the same directory
PROMPT_TEMPLATE_FILE = os.path.join(current_dir, "prompt.txt")
MENU_SIMPLE_FILE = os.path.join(current_dir, "menu.csv")
MENU_ENHANCED_FILE = os.path.join(current_dir, "menu_enhanced.csv")

# Experiment matrix
CONDITIONS = ["simple", "enhanced"]
TEMPERATURES = [0.0, 0.3, 0.7]
RUNS_PER_CONFIG = 30
SLEEP_BETWEEN_CALLS = 0.0  # local inference, no rate limits to respect

# Master Output Files (separate names so they never mix with the Mistral results
# or with the earlier prompt.txt / enhanced_prompt.txt Ollama run)
OUTPUT_CSV = os.path.join(current_dir, "drift_study_ollama.csv")
OUTPUT_JSON = os.path.join(current_dir, "drift_study_ollama.json")

CSV_HEADERS = [
    "overall_run_id",
    "model_name",
    "condition",
    "temperature",
    "run_index",
    "timestamp",
    "status",
    "latency_sec",
    "prompt_tokens",
    "output_tokens",
    "tokens_per_sec",
    "done_reason",
    "dish_count",
    "selected_dish_names",
    "selected_dish_positions",
    "rejected_dish_name",
    "rejected_dish_position",
    "persona_voice_summary",
    "pairing_synergy",
    "rejection_reason",
    "shuffled_menu_order",
    "raw_response"
]

# --- 3. HELPER FUNCTIONS ---
def load_menu_csv(filepath):
    """
    Reads menu CSV supporting utf-8-sig (Excel BOM).
    """
    dishes = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row.get("Category", "").strip()
            name = row.get("DishName", "").strip()
            desc = row.get("DishDescription", "").strip()
            macros = row.get("Macros", "").strip() if "Macros" in row else ""
            if name:
                dishes.append({
                    "category": category,
                    "name": name,
                    "description": desc,
                    "macros": macros
                })
    return dishes

def format_shuffled_menu(dishes, condition):
    """
    Shuffles menu dishes globally and formats each line with pipes.
    Returns: (formatted_menu_string, position_map, order_list)
    """
    shuffled = list(dishes)
    random.shuffle(shuffled)

    lines = []
    position_map = {}
    order_list = []

    for idx, d in enumerate(shuffled, start=1):
        order_list.append(d["name"])
        position_map[d["name"].lower()] = idx

        if condition == "enhanced" and d["macros"]:
            line = f"- [{d['category']}] {d['name']} | {d['description']} | {d['macros']}"
        else:
            line = f"- [{d['category']}] {d['name']} | {d['description']}"
        lines.append(line)

    return "\n".join(lines), position_map, order_list

def inject_menu_into_prompt(template_text, menu_text):
    """
    Injects menu text into {{MENU_SECTION}} placeholder, or replaces
    the existing menu block if placeholder is absent.
    """
    if "{{MENU_SECTION}}" in template_text:
        return template_text.replace("{{MENU_SECTION}}", menu_text)

    # Regex fallback to replace static menu if {{MENU_SECTION}} wasn't manually added
    pattern = r"(### THE 27-DISH MENU:\s*You may ONLY select dishes from this exact list:\s*)(.*?)(---?\s*### YOUR TASK:)"
    if re.search(pattern, template_text, re.DOTALL):
        return re.sub(
            pattern,
            r"\1\n" + menu_text + r"\n\n\3",
            template_text,
            flags=re.DOTALL
        )

    # Ultimate fallback: append if marker missing
    return template_text + "\n\n### THE 27-DISH MENU:\n" + menu_text

def clean_and_parse_json(raw_text: str):
    """
    Strips markdown code fences and extracts valid JSON.
    """
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    json_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(1)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

def find_dish_position(dish_name, position_map):
    """
    Matches dish name against position map using case-insensitive & substring fallback.
    """
    if not dish_name:
        return ""
    name_clean = dish_name.strip().lower()
    if name_clean in position_map:
        return str(position_map[name_clean])

    for canon_name, pos in position_map.items():
        if canon_name in name_clean or name_clean in canon_name:
            return str(pos)
    return "N/A"

def flatten_for_csv(overall_id, model, condition, temp, run_idx, status, latency, stats,
                    parsed_data, raw_content, position_map, order_list):
    """
    Flattens output into a single row for Excel/Pandas analysis.
    """
    row = {
        "overall_run_id": overall_id,
        "model_name": model,
        "condition": condition,
        "temperature": temp,
        "run_index": run_idx,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "latency_sec": round(latency, 2),
        "prompt_tokens": stats.get("prompt_tokens", ""),
        "output_tokens": stats.get("output_tokens", ""),
        "tokens_per_sec": stats.get("tokens_per_sec", ""),
        "done_reason": stats.get("done_reason", ""),
        "dish_count": 0,
        "selected_dish_names": "",
        "selected_dish_positions": "",
        "rejected_dish_name": "",
        "rejected_dish_position": "",
        "persona_voice_summary": "",
        "pairing_synergy": "",
        "rejection_reason": "",
        "shuffled_menu_order": " | ".join(order_list),
        "raw_response": raw_content or ""
    }

    if parsed_data and isinstance(parsed_data, dict):
        selected = parsed_data.get("selected_dishes", [])
        if isinstance(selected, list):
            names = []
            positions = []
            for d in selected:
                if isinstance(d, dict):
                    name = d.get("dish_name", "")
                    names.append(name)
                    positions.append(find_dish_position(name, position_map))

            row["dish_count"] = len(names)
            row["selected_dish_names"] = " | ".join(names)
            row["selected_dish_positions"] = ", ".join(positions)

        row["persona_voice_summary"] = parsed_data.get("persona_voice_summary", "")
        row["pairing_synergy"] = parsed_data.get("pairing_synergy", "")

        rejected = parsed_data.get("explicitly_rejected_dish", {})
        if isinstance(rejected, dict):
            rej_name = rejected.get("dish_name", "")
            row["rejected_dish_name"] = rej_name
            row["rejected_dish_position"] = find_dish_position(rej_name, position_map)
            row["rejection_reason"] = rejected.get("rejection_reason", "")

    return row

def preflight_check(client, models):
    """
    Confirms the Ollama server is reachable and every requested model is downloaded.
    Returns the list of models that are actually available.
    """
    try:
        response = client.list()
    except Exception as e:
        print("Error: could not reach the Ollama server. Start the Ollama app (or run 'ollama serve') and retry.")
        print(f"Details: {e}")
        return []

    entries = response["models"] if isinstance(response, dict) else response.models
    installed = set()
    for m in entries:
        name = getattr(m, "model", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else None)
        if name:
            installed.add(name)
            if name.endswith(":latest"):
                installed.add(name[:-len(":latest")])

    available = []
    for model in models:
        if model in installed:
            available.append(model)
        else:
            print(f"Warning: model '{model}' is not downloaded. Run: ollama pull {model}")
    return available

# --- 4. RESUME / SAFE FILE HANDLING ---
def run_key(model, condition, temp, run_idx):
    return (model, condition, float(temp), int(run_idx))

def load_existing_records(path):
    """
    Loads previous results. Refuses to continue if the file is unreadable, so a
    damaged file is never silently overwritten by a fresh run.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(
            f"Error: could not read existing results file '{path}' ({e}).\n"
            f"Move or repair it, then rerun. Nothing was changed."
        )
    if not isinstance(data, list):
        raise SystemExit(f"Error: '{path}' does not contain a list of run records. Nothing was changed.")
    return data

def save_json_atomic(path, records):
    """
    Writes to a temp file then swaps it in, so an interrupted write can't corrupt the results.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)

def prepare_csv(path):
    """
    Creates the CSV with a header, or verifies an existing file has the same columns.
    Returns the number of data rows already in it.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
            if existing_header != CSV_HEADERS:
                raise SystemExit(
                    f"Error: '{path}' has different columns than this script writes.\n"
                    f"It was probably created by another version. Rename or move it, then rerun."
                )
            return sum(1 for _ in reader)

    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()
    return 0

# --- 5. OLLAMA API CALL WITH RETRIES ---
def _get(obj, name, default=None):
    """Reads a field from an Ollama response whether it is an object or a dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def extract_stats(response):
    """
    Pulls token counts and generation speed from the response metadata.
    """
    eval_count = _get(response, "eval_count")
    eval_duration = _get(response, "eval_duration")  # nanoseconds
    prompt_count = _get(response, "prompt_eval_count")

    tokens_per_sec = ""
    if eval_count and eval_duration:
        tokens_per_sec = round(eval_count / (eval_duration / 1e9), 2)

    return {
        "prompt_tokens": prompt_count if prompt_count is not None else "",
        "output_tokens": eval_count if eval_count is not None else "",
        "tokens_per_sec": tokens_per_sec,
        "done_reason": _get(response, "done_reason", "") or "",
    }

def call_target_model(client, messages, model_name, temperature=0.7, retries=3, backoff_factor=2):
    """
    Calls a local Ollama model, retrying on transient errors (server busy, connection drops).
    Returns (content, error, stats).
    """
    last_error = None

    for attempt in range(retries):
        try:
            kwargs = dict(
                model=model_name,
                messages=messages,
                format="json",
                options={
                    "temperature": temperature,
                    "num_ctx": NUM_CTX,
                    "num_predict": NUM_PREDICT,
                },
            )
            if THINK is not None:
                kwargs["think"] = THINK

            response = client.chat(**kwargs)
            content = _get(_get(response, "message"), "content")
            return content, None, extract_stats(response)

        except KeyboardInterrupt:
            raise
        except ollama.ResponseError as e:
            last_error = f"{e.status_code}: {e.error}"
            logging.warning(f"Ollama error ({model_name}): {last_error}")

            # Retry only on server-side errors; 4xx (e.g. model not found) won't fix itself
            if e.status_code >= 500:
                time.sleep(backoff_factor ** attempt + random.uniform(0.5, 1.5))
                continue
            return None, last_error, {}
        except Exception as e:
            # Typically a connection error (server not running / dropped connection)
            last_error = str(e)
            logging.warning(f"Connection error ({model_name}): {last_error}")
            time.sleep(backoff_factor ** attempt + random.uniform(0.5, 1.5))
            continue

    return None, last_error, {}

# --- 6. MAIN BENCHMARK LOOP ---
def main():
    if not os.path.exists(PROMPT_TEMPLATE_FILE):
        print(f"Error: Prompt template file '{PROMPT_TEMPLATE_FILE}' not found.")
        return
    if not os.path.exists(MENU_SIMPLE_FILE) or not os.path.exists(MENU_ENHANCED_FILE):
        print("Error: menu.csv or menu_enhanced.csv not found in script directory.")
        return

    with open(PROMPT_TEMPLATE_FILE, "r", encoding="utf-8") as f:
        base_prompt_template = f.read()

    menu_simple = load_menu_csv(MENU_SIMPLE_FILE)
    menu_enhanced = load_menu_csv(MENU_ENHANCED_FILE)

    # Connect to Ollama (honors the OLLAMA_HOST env var; defaults to http://localhost:11434)
    client = Client()
    models_to_run = preflight_check(client, OLLAMA_MODELS)
    if not models_to_run:
        print("No runnable models. Exiting.")
        return

    # Load previous results (if any) so an interrupted run can pick up where it stopped
    all_json_records = load_existing_records(OUTPUT_JSON)
    csv_rows = prepare_csv(OUTPUT_CSV)
    if csv_rows != len(all_json_records):
        print(f"Warning: CSV has {csv_rows} rows but JSON has {len(all_json_records)} records. "
              f"They are out of sync (one file was edited or deleted). Resuming from the JSON file.\n")

    # A run counts as done unless it failed on the API side; failed runs are retried.
    # (Their old API_ERROR rows stay in the files for the record, so filter on status.)
    done_keys = set()
    for r in all_json_records:
        if str(r.get("status", "")).startswith("API_ERROR"):
            continue
        try:
            done_keys.add(run_key(r.get("model_name"), r.get("condition"), r.get("temperature"), r.get("run_index")))
        except (TypeError, ValueError):
            continue

    pending = {}
    for model_name in models_to_run:
        pending[model_name] = [
            (condition, temp, run_idx)
            for condition in CONDITIONS
            for temp in TEMPERATURES
            for run_idx in range(1, RUNS_PER_CONFIG + 1)
            if run_key(model_name, condition, temp, run_idx) not in done_keys
        ]

    overall_counter = len(all_json_records)
    total_expected = len(models_to_run) * len(CONDITIONS) * len(TEMPERATURES) * RUNS_PER_CONFIG
    already_done = total_expected - sum(len(jobs) for jobs in pending.values())

    think_label = "default (not sent)" if THINK is None else ("ON" if THINK else "OFF")
    print("=================================================================")
    print("   SHUFFLED FOOD PAIRING DRIFT BENCHMARK MATRIX (OLLAMA)         ")
    print("=================================================================")
    print(f"Models ({len(models_to_run)})       : {', '.join(models_to_run)}")
    print(f"Conditions        : {CONDITIONS} (Dynamic CSV Global Shuffle)")
    print(f"Temperatures      : {TEMPERATURES}")
    print(f"Runs per config   : {RUNS_PER_CONFIG}")
    print(f"Thinking          : {think_label}")
    print(f"Context window    : {NUM_CTX} tokens (max output {NUM_PREDICT})")
    print(f"Total calls       : {total_expected}  (already done: {already_done})")
    print(f"Outputs           : {OUTPUT_CSV}\n                    {OUTPUT_JSON}\n")

    if already_done >= total_expected:
        print("Everything is already complete. Delete the output files to start over.")
        return

    progress_bar = tqdm(total=total_expected, initial=already_done, desc="Overall Progress", unit="call")

    for model_name in models_to_run:
        jobs = pending[model_name]
        if not jobs:
            tqdm.write(f"\n{model_name}: all runs already complete, skipping.")
            continue

        tqdm.write(f"\n=======================================================")
        tqdm.write(f"  STARTING MODEL: {model_name}  ({len(jobs)} runs to do)")
        tqdm.write(f"=======================================================")

        consecutive_failures = 0
        last_group = None

        for job_idx, (condition, temp, run_idx) in enumerate(jobs):
            current_menu = menu_enhanced if condition == "enhanced" else menu_simple

            if (condition, temp) != last_group:
                tqdm.write(f"\n>> Model: {model_name} | Condition: {condition} | Temp: {temp}")
                last_group = (condition, temp)

            overall_counter += 1

            # 1. Shuffle dynamically per run
            menu_text, position_map, order_list = format_shuffled_menu(current_menu, condition)

            # 2. Inject into prompt template
            full_prompt = inject_menu_into_prompt(base_prompt_template, menu_text)

            messages = [{"role": "user", "content": full_prompt}]

            start = time.time()
            raw_response, error, stats = call_target_model(
                client, messages, model_name=model_name, temperature=temp
            )
            latency = time.time() - start

            status = "SUCCESS"
            parsed_json = None

            if error or not raw_response:
                status = f"API_ERROR: {error or 'empty_response'}"
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                parsed_json = clean_and_parse_json(raw_response)
                if not parsed_json:
                    # done_reason == "length" in the CSV means the reply hit the token cap
                    status = "PARSE_ERROR"

            # 3. Flatten and record position metrics
            flat_record = flatten_for_csv(
                overall_counter, model_name, condition, temp, run_idx,
                status, latency, stats, parsed_json, raw_response, position_map, order_list
            )
            with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writerow(flat_record)

            # 4. Master JSON append
            full_record = {
                "overall_run_id": overall_counter,
                "model_name": model_name,
                "condition": condition,
                "temperature": temp,
                "run_index": run_idx,
                "timestamp": flat_record["timestamp"],
                "status": status,
                "latency_sec": flat_record["latency_sec"],
                "prompt_tokens": flat_record["prompt_tokens"],
                "output_tokens": flat_record["output_tokens"],
                "tokens_per_sec": flat_record["tokens_per_sec"],
                "done_reason": flat_record["done_reason"],
                "shuffled_positions": position_map,
                "parsed_data": parsed_json,
                "raw_response": raw_response
            }
            all_json_records.append(full_record)
            save_json_atomic(OUTPUT_JSON, all_json_records)

            progress_bar.update(1)

            if SLEEP_BETWEEN_CALLS > 0:
                time.sleep(SLEEP_BETWEEN_CALLS)

            # Fail-fast safeguard: Skip model if completely broken/inaccessible.
            # The skipped runs are not recorded, so rerunning the script retries them.
            if consecutive_failures >= 3:
                tqdm.write(f"\n[ALERT] 3 consecutive fatal errors on '{model_name}'. Skipping model...")
                tqdm.write("Rerun the script after fixing the problem to retry the remaining runs.")
                remaining = len(jobs) - job_idx - 1
                if remaining > 0:
                    progress_bar.update(remaining)
                break

    progress_bar.close()
    print("\nBenchmark Matrix Completed!")
    print(f"Results saved in:\n  - CSV:  {OUTPUT_CSV}\n  - JSON: {OUTPUT_JSON}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress is saved; rerun the script to resume where it stopped.")