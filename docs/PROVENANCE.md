# Provenance, Exclusions, and Verification Scope

The source project is [hlcccc/MSRC](https://github.com/hlcccc/MSRC) at revision
`9be6b95808a2a10c81b88d9f7e43a29fca303d60`. Its original and PlugGuard-plus-multilayer
versions remain separate. The independent ablation was completed on 2026-10-28.

| New path | Original project artifact | SHA256 |
| --- | --- | --- |
| `msrc_safety/policy.py` | `standalone_policy.py` | `88945e1f5c5f57b6deab34514ba22408e0fa9be6f164bfa7b189052782b58adc` |
| `msrc_safety/signals.py` | `safety_lab/multilayer_signals.py` | `dc6fe8fda8473fbc29f338a7295f6cf66b1802cd3ee1c9254a821d73685c36a5` |

The original model adapter was in `run_standalone.py`, whose SHA256 is
`471f971189736adceed7912567e3a450034e64356b125100c6feda8011a9b7e5`.
That server-specific driver is not redistributed: its path/resource/old-project
dependencies have been removed in the portable adapter and workflow. This is
an explicit packaging change, not a claim that the new wrapper itself produced
the historical result. The core algorithm files above are unchanged.

The historical generator is `Qwen/Qwen2.5-VL-7B-Instruct`, revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. Its local file identities appear in
`results/historical_20261028/generator.spec.json`. The offline response-prefix
label teacher is `Qwen/Qwen3Guard-Gen-8B`, revision
`4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb`. The standalone online path does not
load this teacher or a PlugGuard model.

`provenance.json` records original input, fitted-direction, protocol, source,
and frozen-manifest hashes. These identify artifacts; they do not contain the
artifacts or prove their labels independently. Private snapshots are needed
for a byte-identity check and full historical reproduction.

## Publication Boundary

Included: portable source, unchanged mathematical core, tests, dependency
declarations, historical Linux runtime inventory, docs, aggregate metrics,
model identifiers/hashes, and sanitized historical protocol/provenance.

Excluded: credentials, SSH/Jupyter/VNC endpoints, server-specific operational
scripts, account information, raw benchmark requests/images/answers, per-request
events, feature arrays, fitted projection weights, model checkpoints, caches,
environments, and the original project's unrelated code/results.

The raw historical report is safe to retain because it contains aggregate
counts and method-limit flags only. Other historical JSON is projected into
summary files, removing absolute server paths and request identifiers.

## Verification Boundary

Historical evidence: all 3,360 test requests completed; the historical report
records successful causal replay of all 3,360 records. Four calibration-only
native-greedy checks are summarized alongside it. Those are historical findings.

Release checks: independent CPU unit tests cover the frozen math, last-token
capture with tensor doubles, fit/calibration role guards, frozen artifacts,
idempotent resume, causal replay, class imbalance, and tamper/coverage failures.
Workflow fixtures use a deterministic fake decoder, not a GPU model. A clean
CPU installation and CLI checks validate packaging only. No fresh GPU accuracy,
latency, harmful-token lead-time, or complete-data reproduction claim is made.

An independent repository is useful as a source and documentation record. It
does not settle authorship, institutional ownership, permission to reuse data,
software-copyright registration, or patent novelty. Retain human design/change
records and verify rights before preparing legal submissions.
