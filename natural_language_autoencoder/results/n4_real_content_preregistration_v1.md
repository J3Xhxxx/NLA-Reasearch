# N4 real-content replication — preregistration v1

Frozen on 2026-07-30 before any AV explanation, AR reconstruction, SAE
reconstruction, or causal-patch result from the N3 200-position cohort was
generated or inspected.

## Question

The original E1–E7/N1/N2 results used 40 positions from five short chat
prompts, of which 13/40 were chat-template or blank tokens. N4 asks whether the
two important findings survive on natural, non-template content tokens:

1. Does the NLA text channel primarily transmit the final paragraph containing
   likely next-token candidates?
2. Under causal patching, is NLA reconstruction approximately equivalent to
   SAE-big and better than SAE-small?

This experiment does **not** answer F14 Q2 ("what SAE directions are
readable"). The first N3 120-feature candidate cohort is excluded because its
source strata used raw firing shares under highly unequal source exposure.

## Frozen cohort

- Input: `acts_L32_n3_v1.parquet`
- Expected SHA-256:
  `eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66`
- 200 layer-32 `resid_post` activations from 101 natural documents.
- Every sampled token is a decoded content token; template/blank count is zero.
- Realised positions are 64–475. Each position has at least 16 following
  tokens, at least 75% of which are content tokens.
- No row may be removed because of its NLA/SAE score. Provenance failures abort
  the causal run rather than being filtered.

## Frozen methods

- Base/AV/AR: the already-downloaded Gemma-3-12B-IT L32 checkpoints.
- SAE-small and SAE-big: the same Gemma Scope 2 width-16k checkpoints used in
  E1–E13.
- AV generation: greedy, temperature 0, maximum 200 new tokens.
- Centered cosine: project the unit direction of the cohort mean out of both
  prediction and target. A leave-one-document-out mean-direction sensitivity
  analysis is also required.
- Causal patch: replace `model.layers[32]` output at the exact stored position.
  All nonzero substitutes are norm-matched to the stored activation. Identity,
  zero, dataset mean, another activation, and Gaussian are mandatory controls.
- Primary causal endpoint: KL at the patched position,
  `KL(clean next-token distribution || patched next-token distribution)`.
- Secondary causal endpoints: mean KL over positions `p:p+15`, and CE for the
  corresponding 16 observed next tokens.

## Frozen text ablations

The AV explanation must be frozen before any AR score is computed. It is split
on blank lines into three paragraphs. Any row with fewer than three paragraphs
causes an audit stop before AR scoring.

- `orig`: complete AV explanation.
- `p1_only`, `p2_only`, `p3_only`: one paragraph only.
- `p12`: paragraphs 1 and 2, omitting the final candidate paragraph.
- `quote_strip_p2`: quoted spans removed only from paragraph 2.
- `quote_strip_p3`: quoted spans removed only from paragraph 3.
- `quote_strip_all`: every quoted span removed.
- `word_shuffle`: deterministic seeded word permutation.
- Eight fixed generic texts provide the centered generic floor.

No third-party paraphrase or entity-swap claim is made in N4.

## Confirmatory endpoints

The independent unit is the document, not the position. With at most two rows
per document, row metrics are first averaged within document. Confidence
intervals resample documents, stratified by `pile` versus `xnli`.

### H1 — channel localization replicates

Define

`share(v) = (mean centered_cos(v) - generic_floor) /
            (mean centered_cos(orig) - generic_floor)`.

H1 is confirmed only if:

1. `share(p3_only) >= 0.80`;
2. `share(p12) <= 0.50`; and
3. the 95% document-bootstrap CI for
   `centered_cos(p3_only) - centered_cos(p12)` is entirely above zero.

`quote_strip_p3`, `p1_only`, `p2_only`, and `word_shuffle` are mechanism
diagnostics, not additional confirmation gates.

### H2 — causal reconstruction ranking replicates

For substitute `s`, define per-row

`KL_recovered(s) = 1 - KL_s / max(KL_zero, 1e-6)`.

The headline reports equal-document-weighted means and document-bootstrap CIs.

- NLA versus SAE-big is an equivalence test with margin ±0.05 in mean
  KL-recovered. Equivalence requires the 90% document-bootstrap CI for
  `NLA - SAE-big` to lie wholly inside `[−0.05, +0.05]`.
- NLA superiority over SAE-small requires the 95% document-bootstrap CI for
  `NLA - SAE-small` to be entirely above zero.

Raw paired KL differences are reported alongside the recovered fraction.

### H3 — the text-localized channel is causally useful

Patch reconstructions from `p3_only`, `p12`, and `quote_strip_p3`.

H3 is confirmed if:

1. the 95% document-bootstrap CI for
   `KL_recovered(p3_only) - KL_recovered(p12)` is above zero; and
2. mean `KL_recovered(p3_only)` is at least 75% of mean
   `KL_recovered(orig)`.

This endpoint was not run in N1/N2 and is therefore a new confirmatory
mechanism test, not a replication.

## Mandatory reporting

- Identity provenance minimum cosine and identity KL.
- All 200 ITT rows and all 101 document clusters.
- Source/corpus/language strata, without selecting a favourable stratum.
- Mean, median, document-bootstrap CI, and sign count for paired contrasts.
- Centered cosine, raw cosine, retrieval, KL, and fixed-window CE must remain
  separate; no single metric may be relabelled as "interpretability".
- Script/input/output SHA-256 manifest and wall-clock/GPU-forward counts.

## Decision rule

- If H1 survives but H2 does not, retain the "readable next-token predictor"
  mechanism and drop method-level causal parity.
- If H2 survives but H1 does not, retain reconstruction parity but treat the
  paragraph mechanism as an artefact of the original five prompts.
- If both survive, proceed to the corrected, held-out 100+ feature C1/Q2
  benchmark.
- If neither survives, F11/F12 remain restricted to the contaminated 40-row
  pilot and should not anchor a paper claim.
