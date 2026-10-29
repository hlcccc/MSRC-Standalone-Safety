# Architecture and Historical Boundary

## Data Flow

```text
image + question -> frozen Qwen2.5-VL -> sample one token greedily
   -> consume that token in a cached forward pass
   -> decoder blocks 14 / 21 / 28: last-token vectors
   -> L2 normalization -> three linear projections
   -> average risk + positive rise + decaying evidence
   -> temporal alarm / disagreement-driven buffer decision
   -> terminal decision or next token
```

There is no PlugGuard head, base controller, entropy gate, or online safety judge
in this path. A calibration collection can continue a full shadow generation;
test controllers consume only their causal prefix. Multiple threshold arms can
share a shadow generation, so their timings are not independent latency tests.

## Three Layer Signals

`signals.selected_layers(L)` selects `ceil(L/2)`, `ceil(3L/4)`, and `L` (one-based).
`LayerCapture` attaches hooks only to those decoder blocks, not the vision encoder
or final output normalization. Each hook retains `[batch=0, last_token, :]`.

Fit-only response-prefix labels determine the safe/risk vector means and a
shrunk pooled diagonal variance. Each request has total fitting weight one
across its prefix checkpoints; class weighting is then balanced. The direction
is the mean difference divided by the shrunk variance. Projections are
standardized using fit-safe means and standard deviations. Generator weights
are not updated.

The three signals observe different depths of the **same** generator. They
share supervision and can be correlated. They are not separate content-risk
categories, probabilities, or independently thresholded detectors.

## Temporal Evidence and Gate

For the current projected scores `z1, z2, z3`:

```text
s = (z1 + z2 + z3) / 3
e = 0.9 * previous_e + max(0, s - 0.5) + 0.25 * max(0, s - previous_mean)
mean = 0.8 * previous_mean + 0.2 * s
semantic = max(0, s)
disagreement = population_std(z1, z2, z3)
```

The coefficients and buffers are fixed design choices, not learned values.
The standalone policy tests `e > temporal_threshold` using strict inequality.
An alarm has priority over the buffer state and discards all pending tokens.

The hold threshold is the 95th percentile of fit-safe cross-layer disagreement.
Above it, the controller enters `HOLD` and retains up to 64 pending tokens. At
or below half that threshold it returns to `NORMAL`, retaining 16 pending tokens.
Excess tokens become authorized for release. At a normal EOS/token limit it
flushes the pending buffer. `HOLD` withholds output, not GPU computation.

`StandaloneController` implements counters and decisions. The experiment adapter
stores these decisions and raw token IDs for audit; it is not a streaming UI.
A caller must never expose pending/discarded token IDs to users as approved text.

## Fit, Calibration, and Test

The historical manifest has 2,688 fit, 672 calibration, and 3,360 test requests.
Original-question groups and connected exact-request duplicates were isolated
between roles. The directions and hold threshold are reused fit-only artifacts.
The recalibrated temporal threshold maximizes calibration request-label accuracy,
without filtering/truncating traces using PlugGuard alarms; ties choose the
lowest nonnegative threshold. No test request selects a threshold.

The historical test had already been observed in earlier experiments. Reusing
the same split is an ablation, not a newly blinded independent test. Exact
first-harmful-token boundaries and this ablation's answer-safety labels were
not collected. A causal in-generation warning does not by itself prove that
the first harmful token was prevented from reaching a user.

## What Was Repackaged

| Component | Relationship to the completed experiment |
| --- | --- |
| `policy.py` | Unchanged standalone policy and calibration selector |
| `signals.py` | Unchanged three-layer fitting and projection math |
| `generation.py` | Portable version of the historical local generation/capture adapter |
| `assets.py`, `workflow.py`, `cli.py` | New explicit-path IO, workflow, freeze/resume/replay entry points |
| CPU tests | Original behavioral cases plus isolated workflow/capture fixtures |
| Historical result JSON | Real completed-run aggregates, not new packaging test output |

Generic linear probes and buffer/state-machine ideas are established techniques.
The project's engineering contribution is their concrete projection, temporal,
disagreement, gating, and audit implementation. No priority or paper-level
novelty claim is made by this repository.
