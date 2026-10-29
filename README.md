# MSRC Standalone Safety

Experimental risk warning during multimodal generation, using three internal
layer signals, temporal evidence, and buffered output decisions. The online
path does **not** load a PlugGuard head, consume its risk scores, or call its gate.

This is the **standalone ablation**, not the original PlugGuard implementation
and not the PlugGuard-plus-multilayer version. Those projects are unchanged.

## Status and Actual Results

The completed historical test contains 3,360 requests: 2,520 risk-labelled and
840 benign. The primary metric is **request-label warning accuracy**, not
annotated harmful-token early-warning accuracy.

| Standalone variant | Accuracy | Risk recall | Benign false-positive rate |
| --- | ---: | ---: | ---: |
| Inherited threshold, 6.606602553614496 | 57.17% | 46.07% | 9.52% |
| Calibration-only threshold, 0.519962791813474 | 75.71% | 99.52% | 95.71% |

**The recalibrated variant falsely warns on 804 of 840 benign requests.** Always
warning would already obtain 75.00% on this imbalanced test. These results do
not establish practical safety utility, equivalence to PlugGuard, or precise
lead time before the first harmful token. This prototype is not production-ready.

Full aggregate counts, original artifact hashes, calibration statistics, and
limitations are retained in [the historical results](results/historical_20261028).
Historical references to PlugGuard in that report are comparison data, not an
online dependency of this package.

## Implementation

1. Capture the last consumed token at three fixed decoder depths. For the
   28-layer generator these are layers 14, 21, and 28.
2. L2-normalize each vector and apply a fit-only linear risk projection. These
   are three correlated scores, not probabilities or three semantic categories.
3. Average the scores and accumulate positive risk plus positive score rises,
   with exponential decay. One temporal threshold triggers blocking.
4. Use cross-layer disagreement to switch between 16- and 64-token pending
   buffers with hysteresis. Disagreement delays release; it is not itself a
   harmful-content verdict. `HOLD` does not pause model computation.
5. Score after each generated token is consumed, before updating the release
   counter. Stop on an alarm, discard still-pending tokens, and record decisions
   for independent causal replay. Earlier released tokens cannot be recalled.

The mathematical core in `msrc_safety/policy.py` and `msrc_safety/signals.py` is
copied unchanged from the completed experiments. The portable IO, CLI, model
adapter, and workflow are a new packaging layer. See [architecture](docs/ARCHITECTURE.md)
and [provenance](docs/PROVENANCE.md) for the precise boundary.

## CPU Quick Start

Python 3.10 or newer:

```bash
python -m pip install -r requirements-core.txt
python -m pip install --no-deps .
python -m unittest discover -s tests -v
python -m msrc_safety demo
python -m msrc_safety --help
```

The demo and workflow unit-test fixtures are explicitly **synthetic**. Their
scores and accuracy must not be presented as benchmark measurements. Core
imports, demo, fitting, freezing, and replay do not require PyTorch or a GPU.

## Reproduction

[Reproduction instructions](docs/REPRODUCTION.md) cover the input schema,
audited fitted-artifact import or fitting, calibration shadow collection,
threshold freezing, test generation, and CPU replay. Every model/data/output
location and the CUDA device are explicit arguments. No server login, old
experiment directory, or external project import is required by the package.

Dataset contents, images, prefix-labelled feature arrays, fitted directions,
generator/teacher weights, environments, and raw request records are **not**
distributed in Git. Exact numerical reproduction therefore requires separately
retained, identity-matching private artifacts. Cloning alone cannot reproduce
the historical 3,360-request experiment.

The historical run is complete. This repository's repackaged workflow has CPU
tests; a new end-to-end GPU rerun of this packaging has **not** been performed.
The GPU adapter pins Transformers 5.13.0 and requires native-greedy smoke checks.

## Scope and Terms

This repository contains an experimental software module and CLI, not a user
interface or a deployed moderation service. It logs token IDs and release
decisions locally; any live-stream integration must emit only the newly
authorized prefix, never the raw generated-token trace.

No online guard model does **not** mean no external model was used offline:
the historical response-prefix fit labels came from Qwen3Guard. The generator
is Qwen2.5-VL-7B-Instruct and its weights are frozen during fitting/inference.

The existing Apache-2.0 code terms and attribution are retained in [LICENSE](LICENSE)
and [NOTICE](NOTICE). Model and dataset terms remain separate. Repository
organization does not establish algorithmic novelty, exclusive ownership, or
software-copyright registration eligibility.
