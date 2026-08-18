# N6+ matched candidate-content causal specificity — preregistration v2

Status: **BINDING when and only when this file has a matching
`n6_plus_preregistration_v1.md.sha256` sidecar.**

The cohort, parser, donor assignment, conditions, endpoints, bootstrap, and
decision labels below are frozen before any N6 AR reconstruction,
candidate-probability scoring, or causal-patch outcome is generated. N4 and N5
are design data only and are embargoed from the N6 cohort.

## 1. Scientific question

N5 confirmed on a fresh 400-group held-out cohort that the final paragraph
(`p3`) of an NLA activation-vector explanation is a causally dominant,
near-sufficient reconstruction channel. N5 did not distinguish:

1. recipient-specific continuation candidates;
2. the target-token and context anchors in the first two quoted spans; and
3. candidate-list format, generic lexical content, or familiar wording.

N6+ tests whether candidate spans 3 onward carry incremental,
recipient-specific predictive content beyond the anchors and list format.

## 2. Frozen systems, layer, and provenance

- Base model: `google/gemma-3-12b-it`.
- Intervention site: layer 32 `resid_post`, at the frozen content-token
  position, using the exact N5 hook and full 512-token sequence.
- AV: `kitft/nla-gemma3-12b-L32-av`.
- AR: `kitft/nla-gemma3-12b-L32-ar`.
- SAE comparator: the exact N5 Gemma Scope 2 layer-32 resid-post w16k
  `SAE-big`.
- Dtype for base-model activation extraction and causal scoring: `bfloat16`.
- Batch size for provenance-critical model stages: 1.
- Every local model file must match the frozen N5 combined model manifest.

Frozen input identities:

- N3 corpus JSONL:
  `d40069ab51c294ecbe3e76845d1f2f4dff1bb66a6061c5b6b4c612f7d0ff8816`
- N3 corpus manifest:
  `500d5b88b78c8bc06ff7965c0dffcc25cb5b0e9f50bfa8ec1ae009f9312d6046`
- Pile parquet:
  `a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31`
- N4 activation parquet:
  `eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66`
- N5 cohort plan v2:
  `6e7394476c4769bcb3334bbc82ca078fc778e4f006d9d600dac3882983cafb4c`
- N5 combined model manifest:
  `4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735`
- Frozen N5 centered-cosine gate, used only for descriptive geometry:
  `036477f21fb550b317978a880df0a708dcf42a5201d301ca0757fade3baea059`
- N6 code manifest:
  `4f7e7f612e80a73766d95f2dfd34dcdc37d1b0a6c43a82255d0f3ea87970bc0d`

Every N6 artifact must bind the preregistration, code manifest, model
manifest, and all upstream artifacts it consumes by SHA-256.

## 3. Fresh provisional cohort

### 3.1 Population and sources

N6 uses Pile only. The 13 source labels are:

`ArXiv`, `DM Mathematics`, `FreeLaw`, `Github`, `HackerNews`,
`NIH ExPorter`, `OpenWebText2`, `Pile-CC`, `PubMed Abstracts`,
`PubMed Central`, `StackExchange`, `USPTO Backgrounds`, and
`Wikipedia (en)`.

The selection seed is `20260803`.

### 3.2 Position and independence rules

The N5 v2 content-token rules are retained:

- sequence length 512;
- position in `[64, 480]`;
- at least 16 continuation tokens;
- at least 75% of the continuation passes the frozen content-token predicate;
- one position per independent content group;
- deterministic SHA-256 ordering;
- no chat-template or structural-token fallback.

Every N4 or N5 Pile document/content-group identity is excluded. Decoded
prefixes must have no case-normalized 20-word shingle overlap with the frozen
N4/N5 prefix union. The N6 provisional cohort also enforces the same
20-word-shingle exclusion globally between accepted N6 rows.

Rows are considered in one global deterministic SHA-256 order. Each source is
capped at 40 rows, but a source is not required to fill its cap. The
provisional cohort must contain at least 480 distinct rows in total; otherwise
N6 aborts before any N6 model output. The expected size is approximately 500
and the hard maximum is 520.

## 4. AV-format eligibility and analysis cohort

AV explanations are generated for every provisional row. Eligibility is then
determined using only the explanation text, frozen tokenizer, and upstream
target token—never AR, SAE, logits, causal outcomes, or semantic judgment.

The byte parser:

- encodes the explanation as UTF-8;
- splits blank-line-separated paragraphs with
  `(?:\r?\n)[ \t\f\v]*(?:\r?\n)`;
- strips each paragraph and discards empty pieces;
- defines `p3` as the last paragraph and `p12` as all prior paragraphs joined
  by exactly two LF bytes;
- pairs consecutive ASCII `0x22` quote bytes and rejects an odd quote count.

A row is eligible exactly when:

- it has at least three non-empty paragraphs;
- `p3` contains 6–8 paired ASCII quoted spans;
- quoted span 1 is the target-token anchor;
- quoted span 2 is the context anchor;
- spans 3 onward are 4–6 candidate spans;
- every candidate is non-empty after Unicode NFKC followed by Python
  `str.split()` whitespace collapse with one ASCII space;
- every candidate has a non-empty canonical first-token encoding; and
- after Unicode NFKC, casefolding, and removal of all Unicode whitespace, the
  normalized upstream target token is a substring of normalized span 1.

The context anchor is recorded but is not an eligibility gate. Candidate
normalization for donor exclusion is deliberately case-sensitive after NFKC
and whitespace collapse; it does not casefold.

The confirmatory population is the **AV-format-eligible subset** of the fresh
provisional cohort, not an intention-to-treat population over all provisional
rows. Eligibility counts, reasons for rejection, and source composition are
mandatory reports.

Eligible rows are partitioned into hard cells `(Pile source, candidate count)`.
Cells with fewer than two eligible rows are discarded. Within each retained
cell, the first two rows under the frozen cell hash order are seeded into the
analysis cohort. Remaining rows are filled to exactly 400 by deterministic,
source-balanced round-robin ordering. If the mandatory seeds exceed 400 or the
retained capacity is below 400, N6 aborts before AR, candidate-mass, or causal
scoring.

## 5. Frozen donor assignment

Donors are selected only from the frozen 400-row analysis cohort.

Hard constraints:

- donor and recipient have the same Pile source and candidate count;
- row, content-group, and document identities differ;
- no candidate string is exactly shared after the frozen case-sensitive NFKC
  and whitespace-collapse normalization; and
- the assignment is a one-to-one derangement within every hard cell.

Within each hard cell, a deterministic pure-Python Hungarian assignment
lexicographically minimizes the sums of:

1. absolute difference in the number of unique canonical first-token IDs;
2. absolute difference in total candidate tokenizer length;
3. L1 distance between sorted per-candidate tokenizer lengths;
4. absolute difference in total `p3` tokenizer length; and
5. SHA-256 tie rank.

The implementation converts the lexicographic objective to exact integer
weights, each larger than the maximum possible aggregate contribution of all
lower-priority components. Forbidden edges receive a cost larger than every
feasible assignment. Failure to find a full derangement aborts before any
outcome stage. The donor map, component costs, weights, and transformed-text
hashes are frozen and reported.

## 6. Frozen text variants and reconstruction conditions

The parser labels:

- quoted span 1: `target_anchor`;
- quoted span 2: `context_anchor`;
- quoted spans 3 onward: `candidates`.

The fixed stripping placeholder is `[...]`. Transformations operate on `p3`
quote interiors only. Every byte outside quote interiors is recipient-exact;
for `p3_cross_matched`, both anchors and all non-candidate bytes are
recipient-exact.

The ten causal conditions are:

1. `identity`: exact clean activation.
2. `orig`: AR reconstruction from the full recipient AV explanation.
3. `p3_true`: AR reconstruction from recipient `p3` unchanged.
4. `p3_cross_matched`: recipient `p3` with only candidate interiors replaced
   one-for-one by its frozen donor’s candidate interiors.
5. `p3_candidate_strip`: recipient `p3` with only candidate interiors replaced
   by `[...]`.
6. `p3_anchor_strip`: recipient `p3` with only the first two quote interiors
   replaced by `[...]`.
7. `p3_all_quote_strip`: recipient `p3` with every quote interior replaced by
   `[...]`.
8. `p12`: all paragraphs before `p3`.
9. `sae_big`: frozen SAE-big reconstruction.
10. `zero`: the exact zero vector.

No candidate may be regenerated, paraphrased, edited by an LLM, or selected
using an outcome. All nonzero substitutes are row-wise norm-matched to the
clean activation with the exact N5 rule; `zero` remains exactly zero.

## 7. Causal and predictive-alignment outcomes

### 7.1 Causal fidelity

For every row and condition, patch the substitute at the exact frozen
layer-32 position.

Primary per-row outcome:

- `KL(clean || patched)` at the patched position.

Mandatory diagnostics:

- mean `KL(clean || patched)` over the first 16 continuation positions;
- clean-target cross-entropy over those 16 positions;
- exact clean-activation provenance;
- all per-condition recoveries and tail counts.

All primary aggregate effects are ratios of sums. Row-wise division by
`KL_zero` is forbidden.

Definitions:

`G_specific = sum(KL_cross - KL_true) / sum(KL_zero)`

`G_content = sum(KL_candidate_strip - KL_true) / sum(KL_zero)`

`M_majority = [sum(KL_cross - KL_true)
               - 0.5 * sum(KL_candidate_strip - KL_true)]
              / sum(KL_zero)`

`G_candidate_anchor =
    sum(KL_candidate_strip - KL_anchor_strip) / sum(KL_zero)`

`R_s = 1 - sum(KL_s) / sum(KL_zero)`

`T_p3 = R_p3_true / R_orig`

The raw paired mean `mean(KL_cross - KL_true)` is mandatory beside
`G_specific`.

### 7.2 Clean predictive alignment

For both true and cross candidate sets:

1. prepend one ASCII space unless the raw candidate already begins with
   Unicode whitespace or a Unicode `P*` punctuation character;
2. tokenize without special tokens;
3. take the first token ID;
4. deduplicate first-token IDs within the set; and
5. sum their probabilities under the clean target-model next-token
   distribution.

Let `P_true`, `P_cross`, `n_true`, and `n_cross` denote the two set masses and
their numbers of unique canonical first-token IDs.

The confirmatory paired score is:

`A_meanmass = mean[log(P_true / n_true + 1e-12)
                   - log(P_cross / n_cross + 1e-12)]`.

The raw set-mass score is mandatory but secondary:

`A_setmass = mean[log(P_true + 1e-12)
                  - log(P_cross + 1e-12)]`.

Raw and canonical tokenizations, set masses, unique-ID counts, hit@1/5/10/50,
and membership of the observed next token in each set are mandatory
diagnostics.

## 8. Bootstrap and intervals

- Analysis unit: one frozen content group / row.
- Analysis size: exactly 400.
- Resamples: exactly 50,000 ordinary row bootstraps.
- RNG: `numpy.random.Generator(numpy.random.PCG64(20260803))`.
- The same row-index matrix is shared across every endpoint.
- Quantiles: NumPy `method="linear"`.
- Point estimates use all 400 rows.
- Two-sided 95% intervals use the 2.5% and 97.5% quantiles.
- One-sided 95% lower bounds use the 5% quantile.

The draw-index SHA-256 is reported. An independent audit re-reads raw frozen
artifacts, regenerates the draw matrix, recomputes every endpoint and label,
and must agree with the main analysis to `1e-12`.

## 9. Confirmatory decisions

### H6-A: sample-specific causal candidate channel

Label `SAMPLE-SPECIFIC CHANNEL CONFIRMED` only if all three gates pass:

1. the two-sided 95% CI lower bound for `G_specific` is greater than zero;
2. the two-sided 95% CI lower bound for `G_content` is greater than zero; and
3. the one-sided 95% lower bound for `T_p3` is greater than `0.90`.

Otherwise label `NO SAMPLE-SPECIFIC CHANNEL CLAIM` and list failed gates.

### H6-B: first-token predictive alignment

Label `PREDICTIVE ALIGNMENT CONFIRMED` only if the two-sided 95% CI lower bound
for `A_meanmass` is greater than zero. Otherwise label
`NO PREDICTIVE ALIGNMENT CLAIM`.

The headline phrase
`sample-specific natural-language predictive-state code` is allowed only if
both H6-A and H6-B pass.

`M_majority` is a named secondary endpoint. Label
`MAJORITY-OF-CANDIDATE-BENEFIT SUPPORTED` only if its two-sided 95% CI lower
bound is greater than zero. It is not an H6-A gate.

`G_candidate_anchor` is a named secondary endpoint. Label
`CANDIDATE DOMINANCE SUPPORTED` only if its two-sided 95% CI lower bound is
greater than zero. It is not an H6-A gate.

Failure interpretations are frozen:

- `G_specific` fails: no evidence for incremental candidate identity beyond
  anchors/format; this does not prove that `p3` is generic.
- `G_content` passes but `G_specific` fails: generic candidate content or list
  structure may help, but recipient-specific content is unestablished.
- H6-A passes but H6-B fails: candidate identity matters causally, but
  first-token predictive alignment is unestablished.
- H6-B passes but H6-A fails: AV candidates correlate with the clean
  next-token distribution, but they are not established as the causal `p3`
  mechanism.

## 10. Mandatory descriptive reporting

The following are descriptive and cannot redefine the frozen claims:

- eligibility and rejection counts;
- source and candidate-count composition;
- per-source and leave-one-source-out point estimates;
- candidate, `p3`, and total explanation token lengths;
- target-anchor gate and context-anchor diagnostics;
- true/cross lexical overlap;
- centered cosine for every reconstruction condition using the frozen N5 gate
  direction;
- counts where `KL_cross - KL_true > 1` nat;
- counts where `KL_true - KL_sae_big > 1` nat;
- raw KL, first-16 KL, and first-16 CE summaries;
- all recoveries and `T_p3`;
- candidate hit and observed-token membership diagnostics.

No subgroup, tail, or geometry diagnostic may be promoted post hoc to a
confirmatory claim.

## 11. Fail-closed checks

N6 aborts without a scientific label on any failure of:

- preregistration, code, model, or upstream artifact hashes;
- N4/N5 identity or shingle embargo;
- provisional minimum size;
- exact activation provenance;
- AV-format parser and target-anchor contract;
- exactly 400 unique analysis rows/content groups/documents;
- complete one-to-one donor derangement in every hard cell;
- byte identity outside intended quote interiors;
- exact row/condition mapping;
- finite model and analysis outputs;
- positive aggregate `sum(KL_zero)`;
- identity absolute KL at the patched position or mean first-16 KL exceeding
  `1e-5`; or
- disagreement between the main and independent analyses.

KL values in `[-1e-7, 0)` are numerical roundoff and are clamped to zero and
listed. Any KL value below `-1e-7` aborts.

## 12. Scope and stopping rule

N6 can establish or reject a sample-specific candidate mechanism only within
the fresh Pile, AV-format-eligible subset of Gemma-3-12B-IT layer 32. It does
not establish proposition-level human faithfulness, global superiority over
SAE-big, safe routing, steering, cross-layer generality, or cross-model
generality.

The authorized run ends after artifact pull, independent audit, resource
report, and automatic server shutdown. The shutdown path is:

`sync; /usr/bin/shutdown -h now`

It must run on both success and failure and must not rely on `systemctl`. No
N7, router retry, C1, steering, cross-layer, or cross-model experiment is
authorized by this preregistration.
