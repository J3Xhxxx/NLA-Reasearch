# N5 selective NLA + SAE-big — preregistration v1

Frozen on 2026-07-30 (Asia/Shanghai), before constructing the N5 position
manifest or generating any N5 activation, AV text, AR reconstruction, SAE
reconstruction, or causal result.

## Purpose and status of prior evidence

N4 was a 200-position exploratory design input. It showed:

- aggregate ratio-of-sums recovery of .94795 for NLA and .96649 for SAE-big;
- a post-hoc, document-separated one-dimensional NLA-cosine gate with a
  nontrivial selection rate and a small positive point-estimate versus
  always-SAE-big;
- rare, real NLA causal failures despite moderately high centered cosine;
- strong post-hoc stable evidence that paragraph 3 was much more causally useful
  than paragraphs 1+2.

N5 is the first independent confirmation attempt for a selective NLA/SAE-big
router and for the paragraph-level causal mechanism. N4 numbers may motivate
this frozen design but may not be pooled into N5 inference.

The gate is a fidelity router, not a compute-saving router: both NLA and SAE-big
reconstructions are available before routing. It does not by itself establish
SAE-feature interpretation or steering.

## Frozen source data and embargo

Base corpus:

- N3 JSONL SHA-256:
  `d40069ab51c294ecbe3e76845d1f2f4dff1bb66a6061c5b6b4c612f7d0ff8816`.
- Pile parquet SHA-256:
  `a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31`.
- XNLI all-languages validation parquet SHA-256:
  `c5e6263b0872a3914c9bc165bfe3883e433aa2066c3fa3b9d142829a9b122518`.
- N4 activation cohort SHA-256:
  `eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66`.

Every N4 Pile document is embargoed. Every XNLI `passage_id` represented in N4
is embargoed across all languages. Pile documents are additionally compared on
the exact prefix consumed by the model: after Unicode normalization, lower
casing, and word extraction, any candidate sharing a contiguous normalized
20-word shingle with an N4 prefix is excluded. Selected N5 Pile documents may
not share such a shingle with each other.

This explicitly blocks parallel-content leakage and repeated boilerplate such
as the Github MIT-license prefix. Full-text SHA uniqueness alone is not treated
as sufficient.

## Frozen cohort

N5 contains **600 independent content groups and 600 positions**, one position
per group:

| Split | Pile | XNLI | Total |
|---|---:|---:|---:|
| discovery | 150 | 50 | 200 |
| held-out | 300 | 100 | 400 |
| total | 450 | 150 | 600 |

Pile uses these 13 sources:

`ArXiv`, `DM Mathematics`, `FreeLaw`, `Github`, `HackerNews`,
`NIH ExPorter`, `OpenWebText2`, `Pile-CC`, `PubMed Abstracts`,
`PubMed Central`, `StackExchange`, `USPTO Backgrounds`, and
`Wikipedia (en)`.

Exact per-source quotas are assigned before eligibility/model outcomes by a
SHA-256 largest-remainder rule: discovery has seven sources with 12 rows and
six with 11; held-out has one source with 24 and twelve with 23. Candidate and
tie ordering is SHA-256 deterministic from the frozen seed, source, content
group, and position.

XNLI uses 150 previously unused parallel passage groups, `passage_id`
124–273 inclusive. Exactly one language version is selected per passage.
Each of the ten languages contributes 5 discovery and 10 held-out groups.
No passage or translation-equivalent content may cross splits.

Each frozen position must satisfy:

- raw text only; no chat template;
- sequence truncated to at most 512 tokens;
- position in `[64, 480]`;
- decoded token is non-special, nonblank, and contains a Unicode word
  character;
- at least 16 following tokens;
- at least 75% of those 16 following tokens are content tokens.

Position selection and full `input_ids` are frozen by a CPU/tokenizer-only stage
before the base model is loaded. Any quota, group, passage, shingle, or
eligibility failure aborts/version-bumps N5 before activation extraction. It
must not be repaired after observing model output.

This is a corpus/source-stratified content-group-heldout evaluation, not a
source-OOD evaluation.

## Frozen representations and causal protocol

- Base/AV/AR: the already-downloaded Gemma-3-12B-IT layer-32 checkpoints.
- SAE-small and SAE-big: the same Gemma Scope 2 width-16k layer-32 checkpoints
  used in N4.
- AV generation: greedy, temperature 0, maximum 200 new tokens.
- Causal intervention: replace `model.layers[32]` output at the exact frozen
  position.
- All nonzero substitutes are norm-matched to the frozen activation.
- Every clean and patched evaluation uses the full frozen sequence, batch size
  one, and an independent forward.
- Clean layer-32 state must be bit-exact with the frozen activation.
- Identity KL must be zero within `1e-5`; a failure aborts the split.
- Primary causal value: `KL(clean || patched)` at the patched position.
- Secondary values: mean KL over the first 16 affected logits and CE on the
  corresponding 16 observed next tokens.
- Mandatory conditions: identity, NLA-orig, SAE-small, SAE-big, p3-only, p12,
  quote-strip-p3, and zero.

Discovery reconstruction and causal results are completed first. The gate and
discovery mean direction are frozen before any held-out AV/AR reconstruction or
held-out causal run.

## H5-A: selective NLA + SAE-big

### Only allowed gate score

The discovery activation mean defines

`m_D = mean(x_discovery) / ||mean(x_discovery)||`.

For any vector `v`, define

`v_c = v - (v · m_D) m_D`.

The only router score is the same scalar family used in the N4 post-hoc pilot:

`q_i = cosine(NLA_reconstruction_i,c, x_i,c)`.

Held-out uses the frozen discovery `m_D`; no held-out centering, normalization
fit, corpus-specific gate, or alternative score is allowed.

### Discovery-only threshold training

Candidate NLA routing fractions are frozen to

`{.20, .25, .30, ..., .80}`.

For each fraction, the threshold is the discovery top-q cutoff. Boundary ties
are resolved by SHA-256 of the frozen row UID. The hybrid uses NLA above the
cutoff and SAE-big otherwise.

A discovery candidate is feasible iff:

1. `sum(KL_big - KL_hybrid) > 0`; and
2. at most 3% of discovery content groups have
   `KL_hybrid - KL_big > 1.0 nat`.

Among feasible candidates, maximize

`sum(KL_big - KL_hybrid) / sum(KL_zero)`.

Differences within `1e-6` use the smaller routing fraction. Freeze `m_D`,
routing fraction, numerical threshold, tie assignments, all input hashes, and
the held-out routing rule before held-out reconstruction.

If no candidate is feasible, freeze always-SAE-big and label
`GATE TRAINING FAILURE`. Held-out may still be run for method and H5-B
reporting, but no selective-success claim is possible. Held-out may never
retune the score, threshold, fraction, or tail constraint.

### Held-out primary estimands

No row-wise division by `KL_zero` is allowed.

Primary aggregate gain:

`G = sum(KL_big - KL_hybrid) / sum(KL_zero)`.

The denominator-free paired effect

`Delta_raw = mean(KL_big - KL_hybrid)`

is mandatory alongside it. With one row per independent content group, all
resampling operates on content groups. Bootstrap uses 50,000 resamples,
stratified by Pile versus XNLI, fixed seed 20260730, and never retrains the
gate.

Catastrophic regret is frozen as:

`KL_hybrid - KL_big > 1.0 nat`.

Its one-sided 95% Clopper–Pearson upper bound is used because an ordinary
bootstrap is invalid for zero rare events. Held-out NLA routing coverage uses a
one-sided 95% stratified-bootstrap lower bound.

### H5-A decision labels

`SELECTIVE IMPROVEMENT` requires all of:

1. a feasible discovery gate;
2. held-out coverage lower bound `> .10`;
3. one-sided 95% bootstrap lower bound for `G` `> 0`;
4. catastrophic-regret rate one-sided 95% exact upper bound `< .03`.

If superiority fails, `SAFE SELECTIVE PARITY` requires conditions 1, 2, and 4,
plus the two-sided 90% bootstrap CI for `G` lying wholly inside
`[-.01, +.01]`.

Otherwise the result is labelled `NO SELECTIVE CLAIM`, with the failed gates
listed. A nonsignificant superiority test may not be called parity.

## H5-B: independent paragraph-channel confirmation

AV text is split on blank lines. If any N5 explanation contains fewer than
three paragraphs, H5-B aborts before paragraph-variant AR scoring; H5-A
continues with the full original text, and no row is removed.

- `p3_only`: final paragraph.
- `p12`: all paragraphs before the final paragraph.
- `quote_strip_p3`: full explanation with quoted spans in p3 replaced by the
  fixed placeholder `"[...]"`.

On the 400 held-out groups define:

`G_p3_p12 = sum(KL_p12 - KL_p3) / sum(KL_zero)`,

`R_s = 1 - sum(KL_s) / sum(KL_zero)`,

and retention `T = R_p3 / R_orig`.

`CHANNEL REPLICATED` requires both:

1. the 95% stratified-bootstrap CI for `G_p3_p12` has lower bound `> 0`;
2. the one-sided 95% stratified-bootstrap lower bound for `T` is `> .90`.

Raw paired KL, KL16, CE16, sign count, quote-strip-p3, centered cosine, and
retrieval remain mandatory diagnostics.

H5-A and H5-B are separately named confirmatory claims. Neither may be used to
rescue failure of the other.

## Numerical rules and mandatory reporting

- All 400 held-out rows are ITT. No outcome-based row deletion, trimming,
  Winsorization, minimum-effect eligibility filter, or denominator clipping.
- Float KL in `[-1e-7, 0)` is recorded and set to zero for aggregate inference.
  Any KL `< -1e-7` aborts QA.
- If held-out `sum(KL_zero) <= 1e-6`, N5 is a data/implementation failure.
- Report raw and aggregate KL, KL16, CE16, medians, sign counts, 95th
  percentile, maxima, corpus/source/language strata, always-NLA,
  always-SAE-big, always-SAE-small, hybrid, and per-row oracle.
- Oracle is an upper bound only and cannot train the gate.
- Report tail(always-NLA) minus tail(hybrid) as a secondary paired result.
- Report all artifact/script/model hashes, exact wall time, forward count, and
  QA failures.

Forbidden substitutions after held-out:

- row-wise recovered, median/trimmed mean as the primary;
- a new KL threshold or catastrophe definition;
- NLA-minus-SAE cosine, multivariate classifiers, or corpus-specific gates;
- a favorable subgroup;
- KL16 or CE16 as a replacement for failed KL-at-position primary.

## Resource expectation and stopping rule

One A800-80GB is sufficient. Six hundred AV generations dominate the expected
1–1.5 GPU-hours; activation extraction and causal patching should take only
minutes. New disk use should be well below 1 GB.

The held-out stage is permitted only after the discovery gate artifact and its
SHA-256 are frozen. N5 ends after the held-out analysis and resource report.
No N6, feature-level C1/Q2 run, shutdown, or power-state change is authorized
without the user's next instruction.

