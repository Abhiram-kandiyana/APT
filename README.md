# Active Prompt Tuning

Active Prompt Tuning (APT) is an active-learning workflow for improving visual language model prompts on microscopy classification tasks. APT starts from a small set of expert seed examples, uses a VLM to classify validation and candidate images, selects the next useful examples to add to the prompt, sends those examples through an oracle correction step, and repeats until validation performance reaches a target or the run reaches the round limit.

The current implementation is centered on `main.py`. It supports several selection strategies:

- `dts`: density-threshold sampling over image embeddings.
- `mdl`: minimum-description-length scoring.
- `entropy`: uncertainty-based active selection.
- `random`: random active-set selection.
- `zero_shot`: test-only baseline with no prompt examples.
- `one_shot`: test-only baseline using a fixed corrected prompt set.

## Quick Start

Create an environment for APT.

```bash
conda create -n apt python=3.9
conda activate apt
python -m pip install -r requirements.txt
```

Set your OpenAI API key if you are using the OpenAI VLM backend:

```bash
export OPENAI_API_KEY="..."
```

You can also place the key in a local `.env` file because `main.py` calls `load_dotenv()`.

An example of a run for the current microscopy Lurcher dataset with APT-DTS configuration for one fold:

```bash
python main.py \
  --config config.json \
  --dataset microscopy_lurcher \
  --fold 5 \
  --selection_method dts \
  --model gpt-4o
```

Run several folds in one command:

```bash
python main.py \
  --config config.json \
  --dataset microscopy_lurcher \
  --folds 5,6,8,10 \
  --selection_method dts \
  --model gpt-4o
```

`config.json` stores the project defaults used by the current experiments. Any CLI argument passed explicitly overrides the value loaded from the config file. `--selection_method` should still be passed on the command line because the argument is marked as required by the parser.

## Command-Line Arguments

Core run arguments:

- `--config`: JSON config file loaded before parsing the final command.
- `--dataset`: dataset name under `datasets/`.
- `--fold`: run one fold.
- `--folds`: comma-separated fold list, for example `5,6,10`.
- `--selection_method`: one of `dts`, `mdl`, `entropy`, `random`, `zero_shot`, or `one_shot`.
- `--model`: VLM model name, defaulting to `gpt-4o` unless overridden by config.
- `--vlm_backend`: `auto`, `openai`, `transformers`, or `mlx`.
- `--resume`: resume from the matching checkpoint file.
- `--debug`: use cached/passthrough behavior for development.

Input and output paths:

- `--init_prompts_path`: directory containing seed prompt JSON files.
- `--system_prompt_1`, `--system_prompt_2`: system prompt files.
- `--unlabeled_data_json_path`: explicit candidate-pool JSONL path.
- `--val_json_path`: explicit validation JSONL path.
- `--test_json_path`: explicit test JSONL path.
- `--test_limit_per_class`: cap final test examples per class for smoke tests.
- `--checkpoint_path`: checkpoint output path; a selection suffix is added.
- `--prompt_set_path`: final prompt-set output path; a selection suffix is added.
- `--vlm_log_path`: VLM response log path; a selection suffix is added.
- `--oracle_path`: prompt bank or oracle-cache JSON path; if this path is missing, APT falls back to manual correction.
- `--oracle_script_path`: optional path to APT-v3/oracle.py.
- `--correction_tool_path`: optional path to `apt_correction_tool_v2.py` for manual correction fallback.
- `--logs_dir`: logs and intra-round checkpoints.
- `--prompts_root`: pre/post-oracle prompt files.
- `--val_results_root`: per-round validation predictions.
- `--test_results_root`: final test predictions.
- `--results_root`: consolidated per-fold summaries.

Active-learning controls:

- `--max_rounds`: maximum active-learning rounds.
- `--initial_batch_size`: examples selected each round.
- `--candidate_pool_size`: candidates sampled from the unlabeled pool; `-1` means all candidates.
- `--stopping_accuracy`: stop when validation average class accuracy reaches this percent.
- `--val_batch_size`: batch size for validation VLM calls.
- `--selection_batch_size`: batch size for selection-time VLM calls.
- `--vlm_query_batch_size`: batch size inside oracle prompt generation.
- `--vlm_timeout_s`: timeout per VLM request.
- `--invalid_output_max_retries`: retries for malformed VLM outputs.

MDL and entropy controls:

- `--alpha`: caption-length coefficient.
- `--beta`: caption-redundancy coefficient.
- `--lambda_mdl`: MDL loss tradeoff.
- `--lambda_c`: selection-score tradeoff between uncertainty and expected caption complexity.
- `--K_uncertainty`: stochastic VLM calls for uncertainty estimation.
- `--mdl_tol`: MDL convergence tolerance.
- `--uncertainty_cache_path`: shared JSONL cache for stochastic label counts.

DTS controls:

- `--dts_clip_model_alias`: embedding model alias: `biomedclip`, `clip`, `phikonv2`, or `medsiglip`.
- `--dts_clip_model_name`: explicit embedding model id or legacy alias.
- `--clip_batch_size`: image embedding batch size.
- `--dts_k`: neighborhood graph size.
- `--dts_k_rho`: neighbors used for density proxy.
- `--dts_k_t`: neighbor index used for local threshold radius.
- `--dts_k_b`: neighbors used for boundary score.
- `--dts_mutual_knn`: use hybrid mutual-kNN parent and boundary links.
- `--dts_mcluster_min`: minimum cluster mass for tiny-cluster safeguards.
- `--dts_c_tiny`: maximum selections allowed per tiny cluster.
- `--dts_max_per_basin`: maximum selections per basin.
- `--dts_deg_min_tiny`: tiny-basin fallback minimum mutual degree.
- `--dts_b_min_tiny`: tiny-basin fallback minimum boundary score.
- `--dts_tune_hparams`, `--no_dts_tune_hparams`: enable or disable DTS hyperparameter mutation.

Diagnostics:

- `--diagnostic_mode`: save DTS diagnostic artifacts.
- `--show_interactive`: show diagnostic figures interactively.
- `--diagnostic_every`: save heavy diagnostics every N rounds.
- `--diagnostic_outdir`: diagnostics output directory.
- `--diagnostic_seed`: diagnostics sampling/PCA seed.
- `--max_images_per_panel`: maximum images in diagnostic panels.

Local VLM controls:

- `--local_vlm_device`: `auto`, `cpu`, `mps`, or `cuda` for Transformers backend.
- `--local_vlm_dtype`: `auto`, `float16`, `bfloat16`, or `float32`.
- `--local_vlm_min_pixels`: minimum image pixels for local processor.
- `--local_vlm_max_pixels`: maximum image pixels for local processor.
- `--mlx_model_path`: MLX model directory for `--vlm_backend mlx`.
- `--mlx_resize_shape`: optional MLX image resize shape.
- `--temperature`, `--top_p`, `--max_tokens`: generation parameters.
- `--label_map`: class labels used to map VLM outputs to numeric labels.

Baselines:

- `--selection_method zero_shot`: skips seed prompts, unlabeled data, validation, and oracle correction, then evaluates on the test set.
- `--selection_method one_shot`: loads `--one_shot_prompt_set_path`, skips active selection, and evaluates on the test set.

## Data Layout

APT expects a JSONL file scoped for each fold under:

```text
datasets/<dataset_name>/fold-<fold_number>/
  train.jsonl
  val.jsonl
  val_selected.jsonl
  test.jsonl
```

Each JSONL row should contain an image path and a class label:

```json
{"image_path": "/absolute/path/to/image.jpg", "class": "lurcher"}
```

For the active-learning modes, `train.jsonl` is treated as the unlabeled candidate pool, `val_selected.jsonl` is preferred for validation when present, and `test.jsonl` is used for the final test evaluation. If `val_selected.jsonl` does not exist, the code falls back to `val.jsonl`.

Seed prompts are loaded from:

```text
init_prompts/<dataset_name>_fold=<fold_number>.json
```

System prompts are loaded from the paths passed through `--system_prompt_1` and `--system_prompt_2`; the microscopy defaults are:

```text
system_prompts/microscopy_lurcher_1.md
system_prompts/microscopy_lurcher_2.md
```

For a new dataset, create the fold JSONL files, create a matching seed-prompt JSON file, add or pass dataset-specific system prompts, and pass a label map:

```bash
python main.py \
  --dataset my_dataset \
  --fold 1 \
  --selection_method dts \
  --label_map class_a class_b \
  --system_prompt_1 system_prompts/my_dataset_1.md \
  --system_prompt_2 system_prompts/my_dataset_2.md
```

The current code automatically sets `--label_map wild lurcher` only when `--dataset microscopy_lurcher`.

## Oracle Correction

In every active-learning round, APT asks the VLM to classify the selected active-set images and writes those raw predictions to a pre-correction prompt file under `prompt_sets/`. It then calls the external oracle script to turn those raw predictions into corrected prompt examples for the next round.

By default APT calls:

```text
../APT-v3/oracle.py
```

The paths written for a round look like:

```text
prompt_sets/<dataset_name>/fold-<fold>_<selection>/fold<fold>_round<round>_prompts_<selection>.json
prompt_sets/<dataset_name>/fold-<fold>_<selection>/fold<fold>_round<round>_prompts_<selection>_corrected.json
```

The oracle script has two correction modes:

1. Prompt-bank correction: if a prompt bank is available, the script tries to replace the VLM rationale with a trusted caption from the bank. It first checks for the exact image. If that image is not present, it falls back to an anatomically close image when possible, currently based on parsed microscopy filename metadata such as the same case, slide, and tissue section. If the VLM predicted the wrong class, the oracle output also updates the class to the inferred ground truth.
2. Manual correction: if a wrong prediction cannot be resolved from the prompt bank, the oracle script queues it for the manual correction tool (`apt_correction_tool_v2.py`). Manually corrected entries are marked with `manual_corrected: true`, and APT can merge those corrections back into the oracle cache or prompt bank when `--oracle_path` is supplied.

Use `--oracle_path` to pass a specific prompt bank or oracle-cache JSON file:

```bash
python main.py \
  --dataset microscopy_lurcher \
  --fold 5 \
  --selection_method dts \
  --oracle_path /path/to/LurcherData_prompt_bank.json
```

If `--oracle_path` points to an existing file, APT calls `oracle.py` with that file as `--prompt_bank_path`. If `--oracle_path` is omitted at the class/API level, APT preserves the older behavior and calls `oracle.py` without an explicit prompt-bank path, letting that script try to auto-resolve one from `APT-v3`. In normal CLI runs, the default `--oracle_path` is `oracle.json`; if that file does not exist, or if the user passes any missing/invalid prompt-bank path, APT skips prompt-bank correction and launches the manual correction tool directly. The manual corrections are written to the normal corrected prompt file and then merged into the oracle cache at `--oracle_path`.

Use `--oracle_script_path` if the oracle script lives somewhere else. Use `--correction_tool_path` if the manual correction GUI is not at `apt_correction_tool_v2.py` in this repository. The current `config.json` points at the local Lurcher prompt bank used in the existing experiments; collaborators using another dataset should override `--oracle_path` with their own bank/cache path, or let the missing path trigger manual correction for a new cache.

For a dry run that skips the external oracle and writes passthrough corrected prompts, add:

```bash
--debug
```

## Project Structure

```text
main.py                      Main APT runner and CLI.
config.json                  Current experiment defaults.
utils.py                     JSONL loading, label extraction, split helpers.
dts_sampling.py              DTS embedding and candidate scoring utilities.
dts_diagnostics.py           DTS diagnostics and tuning support.
datasets/                    Fold JSONL files and split-generation helpers.
init_prompts/                Initial prompt examples per dataset/fold.
system_prompts/              Dataset-specific system prompts.
prompt_sets/                 Round prompt sets before and after oracle correction.
logs/                        VLM logs, uncertainty caches, and intra-round logs.
results/                     Consolidated final run summaries.
val_results/                 Validation prediction dumps.
test_results/                Final test prediction dumps.
diagnostics/                 DTS diagnostic artifacts.
models/                      Optional local model files.
```

The repository also contains recovery, migration, verification, and correction helper scripts used during experiments.

## Outputs

APT writes selection-method-specific artifacts so that different runs do not overwrite each other. For example, a DTS run with BiomedCLIP, batch size 10, and candidate pool 100 writes files with a suffix like:

```text
dts_biomedclip_b=10_candidate-size=100
```

Typical outputs include:

- `checkpoint_fold=<fold>_<selection>.json`
- `final_prompt_set_fold=<fold>_<selection>.json`
- `prompt_sets/<dataset>/<fold_selection>/...`
- `val_results/<dataset>/selection_method=<selection>/fold-<fold>.json`
- `test_results/<dataset>/selection_method=<selection>/fold-<fold>.json`
- `results/<dataset>/fold-<fold>/results_selection=<selection>.json`

## Notes For New Datasets

### Choose A Dataset Key

Pick a short dataset key and use it consistently. The current example is `microscopy_lurcher`; if your dataset key is `my_dataset`, APT will look for dataset files, seed prompts, outputs, and prompt sets using that exact token.

The dataset key appears in these places:

```text
--dataset my_dataset
datasets/my_dataset/fold-1/train.jsonl
init_prompts/my_dataset_fold=1.json
prompt_sets/my_dataset/...
results/my_dataset/...
val_results/my_dataset/...
test_results/my_dataset/...
```

Avoid spaces and punctuation in the dataset key. Use lowercase letters, numbers, and underscores.

### Create The Dataset Directory

Create one subdirectory per fold under `datasets/<dataset_key>/`:

```text
datasets/my_dataset/
  fold-1/
    train.jsonl
    val.jsonl
    val_selected.jsonl
    test.jsonl
  fold-2/
    train.jsonl
    val.jsonl
    val_selected.jsonl
    test.jsonl
```

For active-learning runs, `train.jsonl` is the candidate pool. The examples are treated as unlabeled during selection, even though the file still contains class labels for bookkeeping and evaluation helpers. `val_selected.jsonl` is preferred for validation if present; otherwise APT uses `val.jsonl`. `test.jsonl` is used for final evaluation after the active-learning rounds finish.

Each row should be newline-delimited JSON with an image path and a class label:

```json
{"image_path": "/absolute/path/to/image.jpg", "class": "class_a"}
```

Absolute image paths are the least ambiguous option. If you use relative paths, make sure they resolve correctly from the repository root when you run `python main.py`.

### Add Seed Prompts

Create one seed-prompt file per fold:

```text
init_prompts/my_dataset_fold=1.json
init_prompts/my_dataset_fold=2.json
```

The filename must match the `--dataset` value and fold number. These seed prompts are the initial examples APT uses before it starts selecting new examples.

### Add System Prompts

Create dataset-specific system prompts, for example:

```text
system_prompts/my_dataset_1.md
system_prompts/my_dataset_2.md
```

The prompts should define the classification task, the visual/anatomical features the VLM should inspect, the exact output format, and the allowed labels. Pass these paths in the run command:

```bash
python main.py \
  --dataset my_dataset \
  --fold 1 \
  --selection_method dts \
  --label_map class_a class_b \
  --system_prompt_1 system_prompts/my_dataset_1.md \
  --system_prompt_2 system_prompts/my_dataset_2.md
```

Only `microscopy_lurcher` gets an automatic label map (`wild lurcher`). Every other dataset should pass `--label_map` explicitly and keep those labels consistent with the `class` values in the JSONL files and with the labels requested in the system prompts.

### Configure Oracle Correction

For a dataset with an existing prompt bank, pass it explicitly:

```bash
--oracle_path /path/to/MyDataset_prompt_bank.json
```

For a new dataset, you can start without a prompt bank. Pass the path where you want the new oracle cache to live, even if it does not exist yet:

```bash
--oracle_path oracle_my_dataset.json
```

Because the file is missing on the first run, APT will launch manual correction for the selected images and then create/update that JSON file with the corrected examples. Later runs can reuse the same path as a growing oracle cache. If you already have a prompt bank, pass that existing file instead to enable prompt-bank correction before manual fallback.

Use `--debug` for an initial smoke test when you want to verify file paths, fold loading, and output writing without launching the external correction workflow.

### Smoke-Test Before A Full Run

Start with one fold and a small test cap:

```bash
python main.py \
  --dataset my_dataset \
  --fold 1 \
  --selection_method dts \
  --label_map class_a class_b \
  --system_prompt_1 system_prompts/my_dataset_1.md \
  --system_prompt_2 system_prompts/my_dataset_2.md \
  --test_limit_per_class 2 \
  --debug
```

After the smoke test, inspect the generated files under `prompt_sets/`, `val_results/`, `test_results/`, `results/`, and `logs/`. Then remove `--debug`, set the correct oracle paths, and run the full experiment.
