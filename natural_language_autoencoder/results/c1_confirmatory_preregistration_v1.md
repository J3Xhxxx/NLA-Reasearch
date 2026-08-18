# C1 confirmatory synthetic cohort — preregistration v1

Status: frozen before generation of the new corpus and before any activation,
SAE, AV, or AR result from this cohort is inspected.

Date: 2026-07-30 (Asia/Shanghai)

## Question and scope

The computational question is whether natural-language descriptions that name a
feature's independently assigned concept reconstruct closer to that feature's
centered SAE decoder direction than equally formatted, within-superdomain hard
negatives. This is an internal confirmatory test of the C1 round-trip signal. It
does not by itself establish that NLA discovers human-interpretable SAE
features, improves SAE training, or enables causal steering.

The cohort is deliberately new: 24 English concept clusters in six
superdomains, with four discovery documents and two held-out documents per
concept. The concept list, three description templates, hard-negative mapping,
and prior-feature denylist are frozen in
`server/c1_confirmatory_concepts_v1.json` and
`server/c1_confirmatory_denylist_v1.json`.

## Corpus generation and admissibility

One fixed local Gemma-3-12B-IT checkpoint generates six user requests per
concept in a single batch: four discovery and two held-out requests. Sampling
uses master seed 20260730, temperature 0.7, top-p 0.95, top-k 64, repetition
penalty 1.0, and at most 1,800 new tokens. The prompt targets 95–125 words per
request while the wider mechanical acceptance interval remains 80–170 words.
Retries are allowed only for deterministic format/admissibility failures and
use the predeclared seed `master + 100 * concept_index + attempt`, with attempts
0, 1, and 2. The first admissible batch is retained.

Mechanical admissibility is automatic and label-blind beyond the supplied
concept:

- exactly four train and two test strings;
- 80–170 whitespace-delimited words per string;
- no empty or exact-duplicate requests;
- normalized word-5-gram Jaccard below 0.15 for every train/test pair;
- the exact concept title occurs in at most one discovery request and in no
  held-out request;
- no URL, four-digit year, code/list/table/formula marker;
- no literal concept ID or superdomain ID, and no meta-language identifying a
  training/test split, dataset, held-out prompt, SAE feature, or activation
  vector (ordinary uses of words such as “test” remain allowed);
- valid UTF-8 and no non-finite or missing fields.

After mechanical acceptance and before activation extraction, a fixed manual
audit records pass/fail reason codes for topic adherence, named entities,
product names, duplicated scenarios, and contamination by the paired hard
negative. The auditor has no access to SAE, AV, AR, or endpoint values.
Mechanical failure after three attempts, or any manual failure, stops v1; text
is not edited and documents are not selectively regenerated. The complete
manifest and physically separate discovery/held-out manifests and their SHA256
values are then frozen.

## Feature selection

Layer-32 residual activations are extracted from all eligible prompt tokens at
positions 50 and later. The Gemma Scope 16k `l0_small` SAE is applied to these
residuals. A document's score for a feature is the mean of its top three token
activations; selection receives only the discovery manifest and its 96
documents. The held-out manifest is generated, serialized, and hashed at corpus
freeze, but it is not supplied to the extractor/selector and its 48 documents
are not read, encoded, or scored until feature IDs and reference candidates are
frozen.

For each of the 24 concepts, candidate features are ranked by

`max(AUROC - 0.5, 0) * max(raw mean difference, 0) * (0.5 + 0.5 * support precision)`.

Each feature is eligible only for the concept on which its composite score is
largest (ties follow frozen concept order). The deterministic round-robin
selector admits at most four features per concept. It first uses the strict
tier:

- train AUROC at least 0.85 against all other discovery concepts;
- firing support in at least three of four positive documents;
- positive raw mean difference;
- positive-document dominance at most 0.70;
- centered decoder projected-norm ratio at least 0.20.

If fewer than four features qualify, the selector may fill remaining slots from
the predeclared relaxed tier:

- train AUROC at least 0.75;
- firing support in at least three of four positive documents;
- positive raw mean difference;
- positive-document dominance at most 0.85;
- centered decoder projected-norm ratio at least 0.20.

Across all concepts, a candidate is skipped when its absolute centered decoder
cosine with an already selected direction exceeds 0.80. Stable feature-ID order
breaks exact ranking ties. The conservative denylist excludes all 1,282 unique
feature IDs that appeared either in the prior selection asset's selected rows
or its human-readable top-200-per-label tables, plus the legacy exclusion list.

The experiment proceeds to AV/AR only if at least 60 features are selected, at
least 18 concepts have one or more selected features, and at least nine frozen
reciprocal hard-negative pairs have selected features on both sides. Otherwise
the selection failure is reported and no post-hoc threshold relaxation is
allowed.
If the gate passes, every deterministically selected feature is retained in the
intention-to-test analysis regardless of held-out performance. Held-out AUROC,
effect, and support are descriptive moderators only.

## Frozen descriptions and score

For feature \(f\) assigned to concept \(c\), the three correct descriptions are
the three frozen templates populated with \(c\)'s title and summary. The three
hard negatives use the same templates populated with the concept specified by
\(c\)'s frozen within-superdomain `hard_negative_id`. Thus wording and template
count are paired.

NLA AR maps each unique description to a residual vector. Let \(m\) be the unit
mean residual direction estimated with equal document weight from discovery
documents only. Both reconstructed vectors and SAE decoder rows are projected
orthogonally to \(m\) and unit-normalized. The score \(q(f,d)\) is their cosine.

The feature-level paired effect is

`delta_f = mean_t q(f, correct_template_t) - mean_t q(f, hard_negative_template_t)`.

The concept-cluster effect is the mean of `delta_f` over all selected features
assigned to that concept. Because each reciprocal pair reuses the same two
candidate concepts in opposite roles, the pair effect is the equal mean of its
two concept-cluster effects. Complete reciprocal pairs, not features, candidate
strings, or individual concepts, are the primary independent units. A pair is
complete only when both concepts have at least one selected feature; incomplete
concepts remain in descriptive reports but not the primary estimand.

## Primary analysis

The sole confirmatory hypothesis is that the equally weighted mean of the
complete reciprocal-pair effects is greater than zero.

- Primary p-value: exact one-sided random-sign test over all \(2^n\) joint
  reciprocal-pair sign assignments, where \(9 \le n \le 12\). Both concepts in
  a pair always flip together.
- Decision threshold: alpha 0.05.
- Effect report: equal-pair mean and median, positive-pair fraction, and a 95%
  percentile interval from 20,000 pair bootstrap resamples using NumPy PCG64
  seed 20260731.
- Robustness report: exact one-sided binomial sign test over non-zero pair
  effects, plus the 24 concept-cluster effects. Neither is a second route to
  declaring the primary positive.

There is one primary hypothesis, so no multiplicity correction is applied.
Missing AR reconstruction or non-finite score invalidates the run rather than
being imputed or dropped. Candidate text is deduplicated only for computation;
each prespecified feature-template pairing remains in the analysis.

## Prespecified secondary analyses

The following are descriptive/exploratory and cannot rescue a failed primary:

- feature-level effect distribution and results stratified by strict/relaxed
  selection tier;
- correct descriptions versus every other concept in the same superdomain;
- concept and feature retrieval ranks among the selected directions;
- association of `delta_f` with held-out AUROC, held-out effect, and held-out
  support;
- comparison with greedy NLA AV explanations and a base-model autointerpretation
  generated only from discovery contexts;
- NLA explanation specificity under blinded human evaluation;
- sensitivity to leave-one-superdomain-out and leave-one-concept-out analyses.

The previous Gemma-family automatic judge is excluded as a ground-truth or
confirmatory endpoint because it was uninformative in the pilot. Any automatic
text metric will be clearly marked exploratory.

## Blinded human endpoint for a paper-level claim

Before candidate sources are revealed, the pipeline will emit a randomized
rating packet containing discovery and held-out activation contexts plus
source-hidden candidates. At least three independent raters should score
correctness, specificity, and unsupported assertions under a separately frozen
rubric. Inter-rater agreement and cluster-aware uncertainty must be reported.
Until those ratings, a real-corpus replication, and preferably a second
model/layer are complete, the result is evidence for a computational
round-trip distinction rather than a complete top-conference claim.

## Operational constraints

The AutoDL server remains powered on after all jobs. No script in this
experiment may contain shutdown, poweroff, instance-stop, or auto-release
logic. Wall time, peak GPU memory, and available disk space are recorded.
