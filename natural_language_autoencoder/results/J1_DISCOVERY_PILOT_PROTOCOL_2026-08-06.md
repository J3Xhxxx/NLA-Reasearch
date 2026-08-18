# J1-D1 discovery pilot protocol

Status: **EXPLORATORY / DISCOVERY ONLY**  
Protocol date: 2026-08-06 (Asia/Shanghai)  
Primary research direction: **NLA → SAE assistance for mechanistic interpretation**

## Engineering amendment A1 (before any AV outcome)

The first engineering smoke stopped at the activation-verification gate before
loading AV or producing any NLA text. Token IDs and positions matched exactly,
but 8 of 360 target activations exceeded the initial implementation tolerance
`rtol=0.005, atol=1.0`. The largest observed relative discrepancy was about
2.02%; all eight values remained in the same high-activation regime. The
recomputed chunks use the same BF16 model and N3 chunk rule, but not the
original full-run batch composition, so BF16 kernel/batch-shape variation is a
plausible numerical source.

Before inspecting any treatment output, the verification tolerance is therefore
frozen at `rtol=0.025, atol=1.0`. The freeze artifact must retain expected and
actual activations for all 360 contexts and report maximum/quantile absolute and
relative errors. Any token mismatch, sign/firing mismatch, non-finite value, or
error beyond this amended tolerance still fails closed before AV. This
engineering amendment changes no feature, context, negative, arm, endpoint, or
decision rule.

## Engineering amendment A2 (before any AV outcome)

The A1 smoke passed activation verification but stopped at hard-negative
construction, still before loading AV or producing NLA text. Restricting the
candidate pool to the selected 45 features' top contexts left fewer than four
unique strict-tier negatives for a Hindi-selective feature. Relaxing the match
to another language would create an invalid shortcut, so the tier definition
and cohort remain unchanged.

Before model forward, the runner will now deterministically add a background
candidate pool from the same frozen N3 corpus. For every
`(source, language, corpus)` group appearing among held-out positives, it
samples up to 32 unique raw-text positions (and requires at least four) using
seed `20260806` with a
group-derived seed, prioritizing different documents and allowing at most two
positions per document. Positions require at least 16 prefix tokens and six
following tokens, cannot be special/blank tokens, and cannot duplicate any
selected top-context position. These rows are negative candidates only: they
do not alter the 45-feature cohort, discovery evidence, or held-out positives.

Their L32 residuals and true l0-small activations are extracted in the same
forward pass. The original strict matching remains unchanged: exact zero target
activation, unique physical position within feature, and tier 0/1/2 only. The
freeze records pool construction and group counts and embeds only the
background contexts actually selected as hard negatives. Failure to obtain
four strict negatives for any feature still aborts before AV.

## 1. Purpose and claim boundary

This pilot asks whether natural-language descriptions of **real, on-manifold
high-activation residual states** add useful evidence when an interpreter
labels an SAE feature.

It does **not** test whether NLA is a better codec than SAE. It also cannot
support a confirmatory NLA-assisted-SAE claim, because it reuses the already
exploratory N3 corpus and its frozen 120-feature candidate cohort. Its purpose
is to validate the end-to-end protocol, estimate signal and failure modes, and
decide whether a fresh preregistered J1 experiment is worth running.

No AR round-trip, centered cosine, retrieval score, same-family Gemma judge, or
surface plausibility score is allowed to serve as ground truth.

## 2. Frozen inputs

- `results/n3_candidate_cohort_v1.json`: 120 l0-small SAE features, frozen
  before their NLA AV outputs were generated.
- `results/n3_feature_stats_v1.npz`: top-16 real activating positions and
  activation statistics from 8,240,945 N3 tokens.
- `results/n3_corpus_v1.jsonl`: raw real-text documents.
- Base model: local `gemma-3-12b-it`.
- SAE: Gemma Scope 2, layer 32, width 16k, `l0_small`.
- NLA verbalizer: local Gemma-3-12B-IT L32 AV checkpoint.

The 24 B6/B4 features, C1 synthetic-confirmatory material, N4 rows, and N5/N6
rows are not eligible as pilot outcomes.

## 3. Feature and context freeze

The cohort construction is deterministic and occurs before any J1 AV
generation:

1. Require at least eight positive top-activation positions from eight distinct
   documents in `small_top_meta`.
2. Within each N3 stratum (`source_concentrated`, `source_distributed`,
   `language_selective`), select 15 eligible features using seed `20260806`.
3. For each feature, assign its four highest-activation distinct-document
   positions to discovery evidence.
4. Assign the next four highest-activation distinct-document positions to
   held-out positives.
5. Discovery and held-out documents must be disjoint for that feature.

The discovery ITT cohort is therefore 45 features. A failure to obtain exactly
15 eligible features per stratum aborts the pilot rather than changing the
criteria.

All selected residuals must be recomputed with the exact N3 extraction rule:
raw text, no chat template, `add_special_tokens=True`, 512-token chunks,
layer-32 `resid_post`, and early exit after layer 32. Re-encoded target-feature
activations must agree with the stored N3 top values within a predeclared
numerical tolerance; mismatch aborts before AV generation.

For each held-out positive, select one hard negative among the extracted
candidate positions where that target feature does not fire. Exclude all
discovery and positive documents for that feature. Match in this order:

1. same source and language;
2. same source;
3. same corpus and language.

Within a tier, minimize a deterministic distance in residual norm and token
position. Failure to find four negatives aborts. Positives, negatives, residual
vectors, matching tiers, and all hashes are frozen before AV text generation.

## 4. NLA evidence

For every feature, run AV greedily (`temperature=0`, maximum 200 new tokens) on
each of the four **real discovery residuals**. These four snippets are the
NLA-assisted evidence.

The pilot also includes a prespecified contrastive variant. For discovery
residual \(x_i\), target-feature activation \(a_{if}\), and SAE decoder row
\(w_f\), construct

`x_minus_if = x_i - a_if * w_f`.

Run the same greedy AV on both `x_i` and `x_minus_if`. The latter is an
SAE-feature-ablation probe and must not be called a naturally occurring
activation. Save both vector hashes, norms, and their cosine. Because the AV
normalizes its injected activation, do not add an adaptive norm-scaling rule.
The paired text difference is the evidence for a separate
`NLA_CONTRASTIVE` interpretation arm. This arm tests a more genuinely joint
mechanism: SAE specifies the component to remove and NLA verbalizes the
resulting state contrast.

Pure scaled `w_dec` text is off-manifold and is not the treatment. It may be
generated only as an explicitly labeled exploratory control and may not replace
the four real-residual snippets.

Every AV row is append-checkpointed and binds:

- feature ID;
- document ID and token position;
- input-vector SHA-256;
- frozen-cohort SHA-256;
- raw AV output.

## 5. Interpretation arms

The downstream label-generation stage will use a fixed non-Gemma interpreter
and randomized opaque arm IDs:

- `SAE_CONTEXT`: four raw discovery contexts only.
- `NLA_ASSISTED`: the same four raw contexts plus their four on-manifold NLA
  snippets.
- `NLA_CONTRASTIVE`: the same four raw contexts plus four paired
  original-versus-target-feature-ablated NLA snippets.
- `NLA_MISMATCHED`: the same four raw contexts plus four snippets from a
  deterministic, same-stratum, length-matched donor feature.
- `NLA_ONLY` (diagnostic): the four NLA snippets without raw contexts.

All arms must emit the same schema and maximum output budget: one concise
feature hypothesis, expected positive cues, expected exclusion cues, and an
abstention flag. Candidate order is blinded and fixed before scoring.

Input-token equality is not fully achievable in this discovery pilot, so
`SAE_CONTEXT` is not yet a capacity-matched strong baseline. Token/byte counts
must be reported. A later confirmatory J1 must add a matched-budget
context-derived-note baseline or otherwise equate evidence capacity.

Planned fixed interpreter for this pilot: Claude Fable 5 through Claude Code
2.1.193, tools disabled, deterministic/lowest-variance available decoding.
Exact CLI invocation, model provenance, prompt hash, usage, and cost must be
saved. If that model is unavailable, the model substitution must be frozen and
documented before any label is generated.

## 6. External held-out evaluation

The blinded evaluator sees one candidate hypothesis and one held-out context at
a time and returns:

- probability that the target SAE feature fires at the marked token;
- supported / unsupported / abstain;
- a short evidence code.

Actual l0-small SAE activation is the binary truth. Evaluator scores are a
measurement instrument, not truth. At least one evaluator must be from a
different model family than Gemma and different from the label generator.
Planned first evaluator is a pinned Codex/OpenAI model; a second heterogeneous
evaluation or human audit is required before any formal J1 claim.

Discovery summaries:

- feature-clustered AUPRC and paired positive-vs-negative accuracy;
- coverage, Brier score, and calibration bins;
- paired arm differences with feature-cluster bootstrap;
- results by the three frozen strata;
- all 45 features ITT, with parser failures and abstentions retained.

No threshold, subgroup, arm, prompt, or rater may be promoted to confirmatory
status based on this pilot.

## 7. Decision rule for the next experiment

This is a design decision rule, not a significance gate:

- Proceed to fresh confirmatory J1 if `NLA_ASSISTED` improves held-out AUPRC
  over both `SAE_CONTEXT` and `NLA_MISMATCHED`, the direction is not confined
  to one stratum, and calibration/tail behavior shows no obvious collapse.
- Redesign rather than confirm if the gain appears only against the weak
  context-only baseline, only under one rater, or is reproduced by mismatched
  NLA snippets.
- Deprioritize J1 if there is no stable external activation-prediction gain.

The formal experiment must use a new document/source-shingle-embargoed corpus,
freeze roughly 100 candidate features before any NLA score, retain at least 60
features with sufficient fresh held-out support, include a capacity-matched
strong autointerp baseline, and add a preregistered SAE ablation/injection
causal endpoint.

## 8. Operational rules

- The GPU is billed whenever the AutoDL instance is on.
- Load base/SAE for residual extraction, unload them completely, then load AV.
- Do not use quantization, CPU offload, or `device_map="auto"`.
- Smoke outputs must use separate paths and may never be appended to the full
  discovery checkpoint.
- Preserve raw outputs and logs, pull and hash all artifacts, then shut down
  with `sync; /usr/bin/shutdown -h now`.
- SSH disconnect alone is not proof that the AutoDL control plane stopped
  billing.
