# J1 mixed-labeler Terra analysis

Status: **EXPLORATORY_ANALYSIS_MIXED_COMPLETE** — exploratory discovery only.

## ITT metrics

| Arm | micro AP | macro AP | pairwise | Brier | coverage |
|---|---:|---:|---:|---:|---:|
| SAE_CONTEXT | 0.905301 | 0.907844 | 0.888889 | 0.145098 | 0.977778 |
| NLA_ASSISTED | 0.918582 | 0.922474 | 0.906250 | 0.132426 | 0.977778 |
| NLA_CONTRASTIVE | 0.907093 | 0.903056 | 0.886111 | 0.142441 | 0.977778 |
| NLA_MISMATCHED | 0.849382 | 0.839299 | 0.800000 | 0.193581 | 0.977778 |
| NLA_ONLY | 0.890284 | 0.906310 | 0.881944 | 0.161125 | 0.977778 |

## Decision-relevant contrasts

| Contrast | ITT Δmicro AP [95% bootstrap CI] | Luna-Luna Δ | Fable-Fable Δ | sign flag |
|---|---:|---:|---:|---|
| ASSISTED_vs_SAE | 0.013281 [-0.004941, 0.035954] | 0.009202 | -0.017494 | True |
| ASSISTED_vs_MISMATCHED | 0.069199 [0.022793, 0.125741] | 0.103941 | 0.004645 | False |
| CONTRASTIVE_vs_SAE | 0.001793 [-0.026851, 0.028256] | 0.005923 | -0.014761 | True |
| CONTRASTIVE_vs_MISMATCHED | 0.057711 [0.013979, 0.111425] | 0.093552 | 0.007192 | False |

## Decision

**REDESIGN_REPLICATE_BEFORE_CONFIRMATORY**

The complete ITT and Luna-Luna directions are favorable, but the gain over SAE_CONTEXT reverses under Fable-Fable labels. Because Luna labels are evaluated by the same OpenAI/Codex model family, the present signal is labeler-dependent and does not justify an immediate expensive fresh confirmatory run.

This is a design decision, not a confirmatory scientific claim.

## Key limitations

- Discovery-only reused N3 cohort; no confirmatory NLA-assisted-SAE claim is permitted.
- Thirteen batches use Fable and 32 use Luna; labeler is collinear with batch order.
- Terra and Luna are both OpenAI/Codex-family models, so Luna-label performance may include family-specific communication.
- The SAE_CONTEXT baseline is not token/capacity matched; byte-budget equality is not established.
- Only one model, layer, SAE family, and AV-format-eligible feature cohort are evaluated.
- Mismatched NLA is a harmful-content control, not necessarily a strong neutral autointerp baseline.
- Bootstrap intervals are exploratory percentile intervals, not preregistered significance gates.
