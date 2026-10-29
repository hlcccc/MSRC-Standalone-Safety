# Reproduction Procedure

## 1. Choose an Authorized Environment

Use a new project/run directory. Do not modify the original experiment or its
frozen artifacts. Use only a GPU that you are authorized to use; do not kill
another user's job or rely on a free-memory reading as permission. The portable
runner does not implement the historical shared-GPU duty-cycle scheduler.

The historical standalone experiment ran on an A100 using Python 3.10.12,
PyTorch 2.7.1 with CUDA 12.6, Transformers 5.13.0, NumPy 2.2.6, bfloat16, SDPA,
batch size one, and greedy decoding capped at 128 non-EOS tokens. It used
Qwen2.5-VL-7B-Instruct at the revision in `generator.spec.json`.

Prefer restoring the separately retained compatible environment. For a new
Linux environment, install the pinned GPU dependencies and then the package:

```bash
python3.10 -m venv local/venv
. local/venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-gpu.txt
python -m pip install --no-deps .
```

`requirements-historical-linux.lock` is the full old Linux environment inventory,
not a portable Windows lock or an assertion that all its packages are needed.
The CPU-only tests need only the core package and NumPy.

Put the checkout and all large/private artifacts on an authorized data volume.
Set caches and temporary storage there **before** downloads or model loading:

```bash
mkdir -p local/tmp local/cache/hf local/cache/torch local/cache/pip local/cache/cuda local/cache/triton
export TMPDIR="$PWD/local/tmp"
export HF_HOME="$PWD/local/cache/hf"
export HF_MODULES_CACHE="$PWD/local/cache/hf/modules"
export TORCH_HOME="$PWD/local/cache/torch"
export PIP_CACHE_DIR="$PWD/local/cache/pip"
export XDG_CACHE_HOME="$PWD/local/cache"
export CUDA_CACHE_PATH="$PWD/local/cache/cuda"
export TRITON_CACHE_DIR="$PWD/local/cache/triton"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
```

Select the authorized physical GPU with `CUDA_VISIBLE_DEVICES` before starting
the process; pass its logical device explicitly, for example `--device cuda:0`.
The default per-process allocation cap is 20 GiB. This is not an isolation
mechanism and does not guarantee that a shared GPU workload will be unaffected.

## 2. Restore Private Inputs, Do Not Invent a New Split

Required private assets are the frozen request manifest, images, generator
files, and audited fit directions. Refitting additionally needs response-prefix
checkpoint labels and matching three-layer hidden-vector features. They are
not supplied by this Git repository.

The historical dataset combines 5,040 MM-SafetyBench variants (1,680 original
questions across SD, SD_TYPO, and TYPO) with 1,680 benign requests. It has 6,720
requests total: 2,688 fit, 672 calibration, 3,360 test. Reuse the audited connected
original-question/exact-request grouping rather than randomly splitting rows.

Input is a UTF-8 JSONL file. Every row must contain:

| Field | Meaning |
| --- | --- |
| `id` | Unique request identifier |
| `group` | Frozen original-question/connected-component group |
| `role` | `fit`, `calibration`, or `test` |
| `label` | Integer request-risk label: 0 benign, 1 risk |
| `question` | Original question text |
| `image` | Relative path beneath the supplied image root |
| `image_sha256` | Lowercase 64-character SHA256 of the image bytes |
| `suite` | Recommended: SD, SD_TYPO, TYPO, or benign |

The manifest must include all three roles. IDs must be unique; a group cannot
cross roles. An identical `(question, image_sha256)` cannot cross roles either.
The package validates those declared assignments; it does not reconstruct or
independently certify the upstream grouping from incomplete provenance.

Obtain the already-prepared snapshot from the private archive. The upstream
preparation/split implementation is retained in the source project's
[`extensions/multilayer/frozen/replay.py`](https://github.com/hlcccc/MSRC/blob/9be6b95808a2a10c81b88d9f7e43a29fca303d60/extensions/multilayer/frozen/replay.py).
Map its audited group/role assignments into the schema above without resplitting;
make image paths relative to the restored image root. This serialization change
changes the manifest's byte hash, not the request identities or role assignments.
No private dataset exporter or raw questions are uploaded here.

Restore generator files beneath `local/models/generator`. Use
`results/historical_20261028/generator.spec.json`, which pins each expected file.
The loader checks those file hashes and image hashes, uses local files only,
and never downloads weights implicitly. Model and data licenses apply separately.

## 3. Import Audited Directions or Fit From Saved Prefix Features

For the actual standalone ablation, reuse the fit-only directions from the
previous multilayer experiment. Their historical SHA256 is
`5dbeafc4e2ee0c3a771680ee32b19ad72372aadebe5a498956e859be9ae270b5`.
Check the source fit provenance and hash against the private archive first.
Then bind the imported projection parameters to this portable manifest/spec:

```bash
python -m msrc_safety import-directions \
  --inputs local/inputs.jsonl \
  --directions local/original-directions.json \
  --generator-spec results/historical_20261028/generator.spec.json \
  --output local/directions.json
```

Import copies only the projection/fit/hold fields, records the old artifact hash,
and does not refit or reuse its legacy temporal gate. Binding is an explicit
operator assertion that the source fit artifact belongs to these inputs/model;
hashing cannot prove that assertion independently.

Alternatively, fit the same projection algorithm from saved fit-role features:

```bash
python -m msrc_safety fit \
  --inputs local/inputs.jsonl --records local/fit-prefixes.jsonl \
  --features local/fit-prefixes.npz \
  --generator-spec results/historical_20261028/generator.spec.json \
  --output local/directions.json
```

The NPZ must contain `layers` with shape `[N, 3, hidden_dimension]`, aligned with
N JSONL checkpoint rows. Each row requires `id`, `group`, `role: "fit"`, positive
one-based `position`, binary **response-prefix** `label`, and
`label_source: "response_prefix_judge"`. A prefix label may differ from the
request label. Duplicate checkpoints, test/calibration roles, and fewer than
32 independent groups per response class are rejected.

The offline teacher/prefix-feature extraction is an upstream data-preparation
stage, not an online guard dependency. Use the previously audited feature/label
archive for the historical fit. This repository consumes those artifacts; it
does not pretend to regenerate teacher annotations from a request label. A new
fit is a new experiment unless the entire input/feature/label identity matches.

## 4. Collect Full Calibration Shadows

```bash
python -m msrc_safety collect \
  --inputs local/inputs.jsonl --directions local/directions.json \
  --generator-spec results/historical_20261028/generator.spec.json \
  --model-dir local/models/generator --image-root local/images \
  --device cuda:0 --memory-limit-gib 20 --max-tokens 128 \
  --run runs/standalone_001
```

Only calibration requests are generated here. Before collection, one sample
per calibration suite is compared token-for-token against native greedy
generation at 32 tokens. A mismatch is an error, not a tolerance to bypass.
Full 128-token-or-EOS shadow scores are retained without any PlugGuard-based
filter or stop. The source-code, manifest, generator-spec, and direction hashes
are bound to this development run. Valid records can be resumed unchanged.

Historical calibration features were reused from a complete frozen shadow
archive. This new collector can regenerate them without that old project, but
is a packaging entry point that still needs a fresh GPU validation run.

## 5. Freeze Before Testing

```bash
python -m msrc_safety freeze \
  --inputs local/inputs.jsonl --directions local/directions.json \
  --generator-spec results/historical_20261028/generator.spec.json \
  --run runs/standalone_001 --inherited-threshold 6.606602553614496
```

All calibration requests and the native-greedy smoke must be complete. Select
the temporal threshold only from full calibration trace maxima, maximizing
request-label accuracy. Strict `>` and the lowest-threshold tie rule are fixed.
The optional inherited threshold is a declared historical comparison, not a
second threshold fitted using test results. Omit it for a single-arm run.

This writes immutable `protocol.json` and `frozen.json`. Hold thresholds come
only from the fit artifact. Test data cannot be used to alter any of these.

## 6. Generate the Frozen Test

```bash
python -m msrc_safety test \
  --inputs local/inputs.jsonl --directions local/directions.json \
  --generator-spec results/historical_20261028/generator.spec.json \
  --model-dir local/models/generator --image-root local/images \
  --device cuda:0 --memory-limit-gib 20 --run runs/standalone_001
```

Each sampled non-EOS token is consumed by a cached forward pass; the three
last-token block vectors are projected and scored **before release decisions**.
A blocked arm stops consuming further scores. If multiple arms exist, the
shared shadow continues until every arm blocks or generation terminates.
There is no independent-arm latency measurement. No answer-safety judge runs.

The command validates existing records before resuming and never overwrites
them. An empty generation is an error, not an assumed safe result. A failed
record/hash check requires investigation; do not edit frozen records to pass.

## 7. CPU Replay and Report

```bash
python -m msrc_safety report \
  --inputs local/inputs.jsonl --directions local/directions.json \
  --generator-spec results/historical_20261028/generator.spec.json \
  --run runs/standalone_001 --output runs/standalone_001/report.json
```

This needs no GPU or model loading. It verifies frozen artifacts, exact request
coverage, per-token event ordering, scoring timestamps inside the generation
interval, terminal states, and every release decision by replaying the same
three-score trace through a fresh controller. Missing test requests yield
`PARTIAL_NOT_FINAL_ACCURACY`, never `COMPLETE`.

Compare confusion matrices, risk recall, benign false-positive rate, and balanced
accuracy as well as request accuracy. Because the historical test is 3:1 risk
to benign, always warning scores 75%. Do not report 75.71% without the 95.71%
benign false-positive rate. Do not interpret causal scoring timestamps as an
annotation of the first harmful token or proof of zero harmful output leakage.

The historical result files are preserved evidence. They are not automatically
overwritten by this procedure. Exact historical reproduction needs the original
private assets and a validated compatible GPU runtime; CPU tests of the new
wrapper do not replace that full-data GPU reproduction.
