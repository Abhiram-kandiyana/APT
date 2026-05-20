# Active Prompt Tuning

Active Prompt Tuning (APT) is an active-learning workflow for improving visual language model prompts on microscopy classification tasks. APT starts from a small set of expert seed examples, uses a VLM to classify validation and candidate images, selects the next useful examples to add to the prompt, sends those examples through an oracle correction step, and repeats until validation performance reaches a target or the run reaches the round limit.

The current implementation is centered on `code_mdl.py`. It supports several selection strategies:

- `dts`: density-threshold sampling over image embeddings.
- `mdl`: minimum-description-length scoring.
- `entropy`: uncertainty-based active selection.
- `random`: random active-set selection.
- `zero_shot`: test-only baseline with no prompt examples.
- `one_shot`: test-only baseline using a fixed corrected prompt set.

## Quick Start

Create a small environment for APT rather than reusing the full Microscopy conda environment.

```bash
conda create -n apt python=3.9
conda activate apt
python -m pip install -r requirements.txt
```

Set your OpenAI API key if you are using the OpenAI VLM backend:

```bash
export OPENAI_API_KEY="..."
```

You can also place the key in a local `.env` file because `code_mdl.py` calls `load_dotenv()`.

Run the current microscopy Lurcher DTS configuration for one fold:

```bash
python code_mdl.py \
  --config config.json \
  --dataset microscopy_lurcher \
  --fold 5 \
  --selection_method dts \
  --model gpt-4o
```

Run several folds in one command:

```bash
python code_mdl.py \
  --config config.json \
  --dataset microscopy_lurcher \
  --folds 5,6,8,10 \
  --selection_method dts \
  --model gpt-4o
```

`config.json` stores the project defaults used by the current experiments. Any CLI argument passed explicitly overrides the value loaded from the config file. `--selection_method` should still be passed on the command line because the argument is marked as required by the parser.

## Data Layout

APT expects fold-scoped JSONL files under:

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
python code_mdl.py \
  --dataset my_dataset \
  --fold 1 \
  --selection_method dts \
  --label_map class_a class_b \
  --system_prompt_1 system_prompts/my_dataset_1.md \
  --system_prompt_2 system_prompts/my_dataset_2.md
```

The current code automatically sets `--label_map wild lurcher` only when `--dataset microscopy_lurcher`.

## Oracle Correction

During active-learning runs, APT writes the selected examples for each round to `prompt_sets/`, then calls an external oracle correction script. By default the code looks for:

```text
../APT-v3/oracle.py
```

Use `--oracle_script_path` if the oracle script lives elsewhere. Use `--oracle_path` for the prompt-bank or oracle-cache file that should be passed to the oracle and updated after corrections. The current `config.json` points at the local Lurcher prompt bank used in the existing experiments; collaborators on another machine should usually override this path.

For a dry run that skips the external oracle and writes passthrough corrected prompts, add:

```bash
--debug
```

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

## Project Structure

```text
code_mdl.py                  Main APT runner and CLI.
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

1. Put images somewhere stable and use absolute `image_path` values in the JSONL files.
2. Create `train.jsonl`, `val.jsonl` or `val_selected.jsonl`, and `test.jsonl` for each fold.
3. Create `init_prompts/<dataset>_fold=<fold>.json` with seed examples.
4. Write system prompts that describe the task, expected rationale, and exact label vocabulary.
5. Pass `--label_map` for every dataset that is not `microscopy_lurcher`.
6. Override `--oracle_script_path` and `--oracle_path` if the default local APT-v3 oracle paths are not valid.
7. Start with `--debug` or `--test_limit_per_class` for a smoke test before running a full active-learning experiment.
