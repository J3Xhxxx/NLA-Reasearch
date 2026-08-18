# N7: Gemma-3-27B-IT L41 matched candidate-content replication

## Preregistration v1 — DRAFT

Status: **NOT BINDING. NO N7 CONFIRMATORY RUN IS AUTHORIZED BY THIS FILE.**

This draft must not receive a `.sha256` sidecar. It becomes binding only after:

1. engineering smoke rows have been permanently embargoed;
2. every unresolved item in Section 16 has been filled;
3. the N7 code, model, environment, corpus, and embargo manifests have been frozen;
4. the final text has been copied to the exact basename
   `n7_27b_l41_n6_replication_preregistration_v1.md`;
5. that exact file has a matching
   `n7_27b_l41_n6_replication_preregistration_v1.md.sha256` sidecar; and
6. all binding hashes are verified before any analysis-cohort AV explanation,
   AR reconstruction, candidate-mass score, or causal-patch outcome is generated.

The final binding file and sidecar must have matching basenames. No stale v0,
draft, or N6 sidecar may satisfy the binding condition.

Date drafted: 2026-08-06.

Proposed experiment ID:

`N7_27B_L41_MATCHED_CANDIDATE_CONTENT_REPLICATION_V1`

---

## 1. Purpose

The broader research program is not a ranking contest between NLA and SAE.
Its original goal is bidirectional mechanistic-interpretability assistance:

1. use NLA to generate, refine, and triage hypotheses about SAE features,
   decoder directions, interactions, and residual structure; and
2. use SAE features and interventions to ground, audit, and improve NLA
   explanations.

N7 does not test either assistance claim directly. It tests a prerequisite:
whether the causal, sample-specific natural-language signal that would justify
using NLA as an interpretability assistant replicates in a second released
model-scale system. A successful N7 therefore licenses the next joint
experiment; it does not complete the NLA↔SAE research program.

N6 confirmed, on a fresh 400-group Pile cohort for Gemma-3-12B-IT layer 32,
that:

1. the final AV paragraph (`p3`) carries a causally useful channel;
2. continuation-candidate spans carry incremental recipient-specific causal
   information beyond recipient anchors and list format; and
3. the candidate first-token set is aligned with the clean target model's
   next-token distribution.

N7 tests whether the same pre-specified mechanism externally replicates in the
released Gemma-3-27B-IT NLA system at its corresponding layer 41.

This is a **second-model/cross-scale replication**. It is not a causal test of
parameter count, not an emergence-threshold experiment, and not a direct
12B-versus-27B superiority test.

The scientific construct, parser, donor intervention, primary endpoints,
decision thresholds, analysis size, and fail-closed rules are retained from
N6. Changes forced by the target system—model identity, layer, hidden width,
model-specific hashes, and model-specific descriptive centering—are declared
explicitly below.

---

## 2. Relation to N6

Binding N6 reference:

- preregistration:
  `results/n6_pull_staging/n6_pull_20260803T061302Z/n6_plus_preregistration_v2.md`
- preregistration SHA-256:
  `cbc8a5395844b5a61a3a52a543f978c89273d6e9c786db0262d4a4c936faf6a8`
- final main analysis:
  `results/n6_pull_staging/n6_pull_20260803T061302Z/n6_analysis_v1.json`
- independent audit:
  `results/n6_pull_staging/n6_pull_20260803T061302Z/n6_independent_audit_v1.json`

N6 is design data only. N6 effect sizes may justify the value of replication,
but no N6 row, document, content group, prefix shingle, donor, explanation, or
causal outcome may enter the N7 confirmatory population.

N7 preserves an analysis size of exactly 400. A later reduction to 200 would
be a different, explicitly resource-limited protocol and would require a new
preregistration version before any N7 analysis-cohort model output.

---

## 3. Frozen target systems

The final binding version must replace every `[TO FREEZE]` field below with an
exact local path, Hugging Face revision, complete file manifest, and SHA-256.

### 3.1 Base target model

- repository: `google/gemma-3-27b-it`
- exact revision: `[TO FREEZE]`
- local root: `[TO FREEZE]`
- architecture: 62 transformer blocks
- hidden width: `5376`
- intervention block: zero-based block `41`
- hook: `model.layers[41].output`
- activation semantics: output of block 41 / `resid_post`
- Transformers hidden-state equivalent: `hidden_states[42]`
- extraction and causal-scoring dtype: `torch.bfloat16`
- device map: explicit single-device `cuda:0`
- batch size: `1`
- sequence length: `512`
- `use_cache=False` for provenance-critical causal forwards

The code must use `model.layers[41]`; `hidden_states[42]` is an indexing
cross-check, not authorization to patch block 42.

### 3.2 Activation verbalizer

- repository: `kitft/nla-gemma3-27b-L41-av`
- exact revision: `[TO FREEZE]`
- local root: `[TO FREEZE]`
- sidecar role: `av`
- sidecar extraction layer: `41`
- sidecar hidden width: `5376`
- sidecar injection scale: must be recorded from frozen `nla_meta.yaml`
- runtime dtype: `torch.bfloat16`
- generation:
  - greedy
  - `temperature=0.0`
  - `do_sample=False`
  - `max_new_tokens=200`

The released AV repository is stored in float32. Loading it explicitly as
bfloat16 is a pre-specified engineering adaptation required for a single-GPU
run; it must be verified in smoke and frozen before confirmatory outputs.
Quantization is forbidden.

### 3.3 Activation reconstructor

- repository: `kitft/nla-gemma3-27b-L41-ar`
- exact revision: `[TO FREEZE]`
- local root: `[TO FREEZE]`
- sidecar role: `ar`
- sidecar extraction layer: `41`
- sidecar hidden width: `5376`
- critic backbone layers: exactly `42` (`K+1` for `K=41`)
- `value_head.safetensors`: required and hashed
- runtime dtype: `torch.bfloat16`

AR must return one finite vector of shape `(5376,)` for every reconstruction
text. No projection, padding, truncation, or width adapter is allowed.

### 3.4 SAE comparator

- repository: `google/gemma-scope-2-27b-it`
- exact revision: `[TO FREEZE]`
- exact directory:
  `resid_post_all/layer_41_width_16k_l0_big`
- local root: `[TO FREEZE]`
- required files:
  - `config.json`
  - `params.safetensors`
- expected hook:
  `model.layers.41.output`
- expected width: `16384`
- expected hidden width: `5376`
- expected L0 family: `big` / approximately `120`
- SAE reconstruction compute dtype: `torch.float32`
- patch conversion: the resulting finite float32 vector is norm-matched in
  float64/float32 analysis code and cast to the target hidden dtype only at
  assignment.

`resid_post`, a different layer, a 262k SAE, a 12B SAE, or a PT-model SAE is
not an admissible substitute.

The L41 16k small SAE may be downloaded for a separate descriptive inventory,
but it is not an N7 condition, cannot affect selection or decisions, and must
not be silently substituted for SAE-big.

### 3.5 Sequential loading contract

On a single GPU:

1. base extraction is a standalone stage;
2. AV generation is a standalone stage;
3. AV is fully released before AR is loaded;
4. AR is fully released before SAE reconstruction if memory requires;
5. base causal scoring is a standalone stage.

AV and AR must not be resident on the same 80 GB GPU. `device_map=auto`,
unregistered CPU offload, meta-tensor execution, 8-bit loading, 4-bit loading,
or any other memory-driven semantic change is forbidden.

---

## 4. Environment and provenance

The final binding run must freeze:

- OS/container image identity;
- Python version;
- NumPy version;
- PyTorch version;
- CUDA runtime and driver;
- Transformers version;
- safetensors version;
- pyarrow version;
- scipy version;
- SAE loading code version;
- exact `nla_inference.py`;
- tokenizer and model config files;
- every N7 Python and shell script;
- every consumed upstream artifact.

Code hash equality is not sufficient if tokenizer or library behavior differs.
The environment manifest is therefore binding.

Required binding artifacts:

- `n7_code_manifest_v1.txt`
- `n7_model_weights_v1.sha256`
- `n7_environment_v1.json`
- `n7_upstream_inputs_v1.json`
- `n7_embargo_v1.json`
- final preregistration and matching sidecar

Every output must embed the SHA-256 of the preregistration, code manifest,
model manifest, environment manifest, upstream input manifest, and every
direct input it consumes.

No fixed number of model-manifest entries is assumed. The N6 hard-coded
25-entry contract must not be reused.

---

## 5. Engineering smoke before binding

### 5.1 Scope

Before the final binding preregistration, run an engineering-only smoke on
20–40 independent Pile content groups.

Smoke rows, documents, content-group identities, original source indices, and
20-word prefix shingles are permanently embargoed from the N7 provisional and
analysis cohorts.

Smoke may inspect:

- asset completeness;
- tokenizer identity and deterministic tokenization;
- block-41 hook provenance;
- vector shapes and finite values;
- AV/AR sequential loading;
- parser success under the already-fixed N6 parser;
- identity KL;
- peak VRAM, disk, throughput, and ETA;
- checkpoint/resume and shutdown behavior.

Smoke must not calculate or inspect the N7 confirmatory endpoint set:

- `G_specific`
- `G_content`
- `T_p3`
- `A_meanmass`
- `M_majority`
- `G_candidate_anchor`

Smoke is not a miniature scientific analysis.

### 5.2 Smoke gates

The final preregistration may be frozen only if all of the following pass:

1. every required shard, index, config, tokenizer file, sidecar, and value head
   is present and hashed;
2. free disk before download/run satisfies the frozen operational budget;
3. no incomplete or duplicate HF-cache copy threatens the budget;
4. base BF16, batch-1 execution remains entirely on `cuda:0`;
5. `model.layers[41].output` yields finite vectors of width 5376;
6. `hidden_states[42]` and the hook agree on smoke examples within a
   pre-specified numerical tolerance `[TO FREEZE]`;
7. SAE config and parameter shapes exactly match L41/5376/16384;
8. AV BF16 generates at least one explanation accepted by the unchanged byte
   parser;
9. AR has a finite value-head weight of shape `(5376,5376)` and returns finite
   `(5376,)` vectors;
10. identity patched-position KL and mean first-16 KL are each at most `1e-5`;
11. no OOM, CPU offload, meta tensor, NaN, or wrong-layer provenance occurs;
12. smoke logs, exit status, staged pull, and real shutdown path are verified.

If the exact parser fails in smoke, N7 aborts as an engineering/protocol
failure. The parser may not be relaxed to improve 27B eligibility while
retaining the same replication label.

Any code or environment change after smoke requires rerunning the affected
smoke gates before binding.

---

## 6. Input corpus and fresh provisional cohort

### 6.1 Frozen input corpus

The old N3/N6 `pile-10k` artifact cannot be the sole N7 sampling frame.
The frozen N6 availability audit accounts for every HackerNews document after
N4/N5/N6 use, and leaves at most 28 unused DM Mathematics documents before
target-position eligibility. Even an optimistic cap calculation is therefore
`0 + 28 + 11*40 = 468 < 480`. The old frame cannot reach the frozen
provisional minimum.

N7 therefore requires a new Pile extension artifact that:

- contains the same 13 source labels in Section 6.2;
- contains stable raw document text and source/document identities;
- is frozen before any N7 analysis-cohort model output;
- has enough raw-text/tokenizer capacity to satisfy the unchanged source cap
  and provisional minimum without adaptive source substitution;
- is document- and shingle-disjoint from N4, N5, N6, and smoke after the
  Section 6.3 embargo;
- has an exact dataset repository, configuration, split, revision, extraction
  script, row range, local path, schema, manifest, and SHA-256.

Binding N7 corpus fields:

- dataset repository/configuration/split: `[TO FREEZE]`
- immutable dataset revision: `[TO FREEZE]`
- extraction script and SHA-256: `[TO FREEZE]`
- local corpus artifact and SHA-256: `[TO FREEZE]`
- corpus manifest and SHA-256: `[TO FREEZE]`
- per-source raw document counts: `[TO FREEZE]`

Corpus selection may inspect only raw text, source labels, stable identities,
the frozen tokenizer, and embargo membership. It may not inspect AV, AR, SAE,
target logits, candidate text, or causal outcomes.

The following old artifacts remain mandatory upstream embargo/provenance
inputs, not the N7 sampling frame:

- N3 corpus JSONL:
  `d40069ab51c294ecbe3e76845d1f2f4dff1bb66a6061c5b6b4c612f7d0ff8816`
- N3 corpus manifest:
  `500d5b88b78c8bc06ff7965c0dffcc25cb5b0e9f50bfa8ec1ae009f9312d6046`
- old Pile parquet:
  `a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31`

The final binding version must verify the old and new corpus artifacts locally
and bind every actual path and hash.

### 6.2 Population and sources

N7 is Pile-only and uses English real-text documents. The 13 frozen source
labels are:

`ArXiv`, `DM Mathematics`, `FreeLaw`, `Github`, `HackerNews`,
`NIH ExPorter`, `OpenWebText2`, `Pile-CC`, `PubMed Abstracts`,
`PubMed Central`, `StackExchange`, `USPTO Backgrounds`, and
`Wikipedia (en)`.

The selection seed is `20260803`, retained from N6 to preserve the deterministic
protocol. Freshness is enforced by hard embargo, not by changing the seed.

All tokenization, sequence construction, target token IDs, and positions are
regenerated with the frozen 27B tokenizer. No 12B input-ID array is accepted
as N7 provenance merely because the two model families are expected to share
a tokenizer.

### 6.3 Embargo

Before selection, construct and freeze a model-independent embargo manifest
from the original source documents and raw Pile identities.

Exclude:

- every N4 document and content group;
- every N5 discovery or held-out document and content group;
- every N6 provisional or analysis document and content group;
- every engineering-smoke document and content group;
- all matching source/original-index identities;
- all decoded prefixes with any normalized 20-word shingle overlap with the
  combined N4/N5/N6/smoke prefix union.

Shingle normalization:

- Unicode NFKC;
- lowercase;
- Python Unicode `\w+` word extraction;
- consecutive 20-word shingles.

The N7 provisional cohort must also have zero global 20-word prefix-shingle
overlap among accepted rows.

Mandatory reports:

- identity counts by embargo source;
- shingle counts and set hashes;
- overlap counts, all exactly zero after filtering;
- input document, content-group, and original-index hashes;
- smoke overlap exactly zero.

### 6.4 Position rules

The N6 rules are retained:

- sequence length `512`;
- frozen target position in `[64,480]`;
- at least `16` continuation tokens;
- at least `75%` of the continuation passes the frozen content-token
  predicate;
- one position per independent content group;
- deterministic SHA-256 position and candidate ordering;
- no chat-template, structural-token, or short-prompt fallback.

Define the exact provisional-cohort hash:

`H(*parts) =
    SHA256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()`

The ordering keys are:

`document_order =
    H(selection_seed, "pile-document", source,
      stable_document_id, text_sha256)`

`position_order =
    H(selection_seed, "position", source, content_group_id, position)`

`candidate_order =
    H(selection_seed, "candidate", source, content_group_id, position)`

`row_uid =
    H(selection_seed, "n7-provisional-row",
      source, content_group_id, position)`

Rows are considered in one global deterministic SHA-256 order. Each source is
capped at 40 rows after the global candidate ordering, but a source is not
required to fill its cap.

The provisional cohort must contain at least 480 distinct rows. The expected
size is approximately 500 and the hard maximum is 520. Failure to reach 480
aborts N7 before any provisional-row AV output.

---

## 7. Activation extraction and descriptive centering

Extract the exact output of zero-based block 41 for the full frozen 512-token
sequence at the exact target position.

Required provenance:

- exact input IDs and their SHA-256;
- token ID and decoded token;
- source, document, content group, and original index;
- position;
- hook module name;
- hook call count;
- sequence length;
- hidden width 5376;
- model/config/tokenizer hashes;
- dtype bfloat16;
- batch size 1;
- no template fallback;
- one activation per provisional row in identical order.

The clean activation array is stored as finite float32 values after capture.

The old N5 L32 centered-cosine gate/direction
`036477f21fb550b317978a880df0a708dcf42a5201d301ca0757fade3baea059`
is invalid for N7 and must never be loaded.

For descriptive geometry only, define:

`m_hat_27b_l41 = mean(clean provisional activations)`

and report:

`centered_cosine(x_hat,x) =
    cosine(x_hat-m_hat_27b_l41, x-m_hat_27b_l41)`

This vector is frozen immediately after activation extraction and before any
provisional-row AV explanation is generated. Its data, code, and hash are
reported. It must not affect eligibility, donor assignment, reconstruction,
causal outcomes, H7 decisions, or stopping.

---

## 8. AV-format eligibility and analysis cohort

AV explanations are generated for every provisional row only after the final
preregistration and all manifests are binding.

Eligibility uses only:

- the AV explanation text;
- the frozen tokenizer;
- the frozen upstream target token.

It must never use:

- AR output;
- SAE activation or reconstruction;
- target-model logits;
- causal outcomes;
- centered cosine;
- N6 effect size;
- semantic or human judgment.

### 8.1 Frozen byte parser

The parser:

1. encodes the explanation as UTF-8;
2. splits blank-line-separated paragraphs with
   `(?:\r?\n)[ \t\f\v]*(?:\r?\n)`;
3. strips each paragraph and discards empty pieces;
4. defines `p3` as the last paragraph;
5. defines `p12` as all prior paragraphs joined by exactly two LF bytes;
6. pairs consecutive ASCII `0x22` quote bytes;
7. rejects an odd ASCII quote count.

Curly quotes are not converted. No LLM cleanup, repair, paraphrase, or format
normalization is allowed.

### 8.2 Eligibility gates

A row is eligible exactly when:

- it has at least three non-empty paragraphs;
- `p3` contains 6–8 paired ASCII quoted spans;
- quoted span 1 is the target-token anchor;
- quoted span 2 is the context anchor;
- spans 3 onward are 4–6 candidate spans;
- every candidate is non-empty after Unicode NFKC and Python `str.split()`
  whitespace collapse with one ASCII space;
- every candidate has a non-empty canonical first-token encoding; and
- after NFKC, casefolding, and removal of all Unicode whitespace, the
  normalized upstream target token is a substring of normalized span 1.

The context anchor is recorded but is not an eligibility gate.

Candidate normalization for donor exclusion remains case-sensitive after NFKC
and whitespace collapse; it does not casefold.

The confirmatory population is the 27B **AV-format-eligible subset** of the
fresh provisional cohort. Eligibility counts, rejection reasons, source
composition, and candidate counts are mandatory.

### 8.3 Exact 400-row analysis cohort

Eligible rows are partitioned into hard cells:

`(Pile source, candidate_count)`

Define the exact analysis/donor hash:

`F(seed,domain,*parts) = SHA256(concat(fields)).hexdigest()`

where `fields` are, in order,

`("n7-freeze-v1", str(seed), domain, *map(str,parts))`

and every field is encoded as:

`len(field_utf8).to_bytes(8,"big") || field_utf8`

with no separator or terminator.

Rules:

- discard cells with fewer than two eligible rows;
- within every retained cell, seed the first two rows under
  `F(seed,"cell-row",source,candidate_count,row_uid)`;
- fill the remaining slots to exactly 400 with deterministic,
  source-balanced round-robin ordering;
- require 400 unique rows, content groups, and documents.

Retained cells are ordered by `F(seed,"cell-order",source,candidate_count)`.
Sources are ordered by `F(seed,"source-round-robin",source)`. Remaining rows
within a source are ordered by
`F(seed,"source-fill",source,candidate_count,row_uid)`.

If mandatory seeds exceed 400 or retained capacity is below 400, N7 aborts
before AR reconstruction, candidate-mass scoring, or causal scoring.

No smaller adaptive cohort is allowed.

---

## 9. Frozen donor assignment

Donors are selected only from the frozen 400-row N7 analysis cohort.

Hard constraints:

- recipient and donor share the same Pile source;
- recipient and donor have the same candidate count;
- row, document, content-group, and original-index identities differ;
- no candidate string is exactly shared after case-sensitive NFKC and
  whitespace-collapse normalization;
- assignment is a one-to-one derangement in every hard cell.

Within each cell, the deterministic pure-Python Hungarian assignment
lexicographically minimizes aggregate:

1. absolute difference in unique canonical first-token ID count;
2. absolute difference in total candidate tokenizer length;
3. L1 distance between sorted per-candidate tokenizer lengths;
4. absolute difference in total `p3` tokenizer length;
5. SHA-256 tie rank.

Tie hashes are
`F(seed,"donor-tie",source,candidate_count,recipient_uid,donor_uid)`.
Donor blocks are ordered by
`F(seed,"donor-block",source,candidate_count)`.

Exact integer weights are used, each larger than the maximum aggregate
contribution of all lower-priority components. Forbidden edges receive a cost
larger than every feasible assignment.

Failure to find a complete derangement aborts before outcomes.

The donor map, hard-cell membership, component costs, integer weights,
recipient/donor identities, candidate-normalization hashes, and every
transformed-text hash are frozen.

---

## 10. Frozen text variants and conditions

The parser labels:

- span 1: `target_anchor`
- span 2: `context_anchor`
- spans 3 onward: `candidates`

The stripping placeholder is exactly:

`[...]`

All transformations operate only on quote interiors in `p3`. Every byte
outside the intended quote interiors is recipient-exact.

For `p3_cross_matched`:

- both recipient anchors remain exact;
- all recipient non-candidate bytes remain exact;
- only candidate interiors are replaced one-for-one by the frozen donor's
  candidate interiors.

The ten conditions are:

1. `identity`: exact clean activation.
2. `orig`: AR reconstruction from the full recipient explanation.
3. `p3_true`: AR reconstruction from recipient `p3`.
4. `p3_cross_matched`: recipient `p3` with only candidate interiors replaced.
5. `p3_candidate_strip`: only candidate interiors replaced by `[...]`.
6. `p3_anchor_strip`: only the first two quote interiors replaced by `[...]`.
7. `p3_all_quote_strip`: all quote interiors replaced by `[...]`.
8. `p12`: all paragraphs before `p3`.
9. `sae_big`: frozen L41 16k SAE-big reconstruction.
10. `zero`: exact zero vector.

No candidate may be regenerated, edited, paraphrased, translated, selected,
or repaired by an LLM.

Every nonzero substitute is row-wise norm-matched to the clean activation with
the N6 rule. `zero` remains exactly zero.

Norm matching is part of the binding code and must have an independent
self-test for zero vectors, finite values, and exact per-row target norms.

---

## 11. Causal and predictive-alignment outcomes

### 11.1 Causal fidelity

For every row and condition, patch the substitute at the exact frozen block-41
position in the full 512-token target-model forward.

Primary per-row outcome:

`KL(clean || patched)` at the patched position.

Mandatory diagnostics:

- mean `KL(clean || patched)` over the first 16 continuation positions;
- clean-target cross-entropy over those 16 positions;
- exact clean-activation provenance;
- per-condition sums, means, medians, quantiles, recoveries, and tail counts.

All primary aggregate effects use ratios of sums. Row-wise division by
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

The raw paired mean:

`mean(KL_cross - KL_true)`

is mandatory beside `G_specific`.

### 11.2 Clean predictive alignment

For true and cross candidate sets:

1. prepend one ASCII space unless the raw candidate already begins with
   Unicode whitespace or Unicode `P*` punctuation;
2. tokenize without special tokens;
3. take the first token ID;
4. deduplicate first-token IDs within the set;
5. sum their probabilities under the clean target-model next-token
   distribution.

Let `P_true`, `P_cross`, `n_true`, and `n_cross` be the set masses and unique
canonical first-token counts.

Confirmatory score:

`A_meanmass = mean[log(P_true / n_true + 1e-12)
                   - log(P_cross / n_cross + 1e-12)]`

Secondary raw set-mass score:

`A_setmass = mean[log(P_true + 1e-12)
                  - log(P_cross + 1e-12)]`

Mandatory diagnostics:

- raw candidates;
- canonical token IDs;
- unique-ID counts;
- true and cross masses;
- hit@1/5/10/50;
- observed-next-token membership;
- tokenizer hashes.

These endpoints concern candidate first tokens, not complete candidate
sequences or proposition truth.

---

## 12. Bootstrap and intervals

- analysis unit: one frozen content group / row
- analysis size: exactly `400`
- resamples: exactly `50,000`
- bootstrap type: ordinary row bootstrap
- RNG:
  `numpy.random.Generator(numpy.random.PCG64(20260803))`
- one shared row-index matrix for every endpoint
- draw-index shape: `(50000,400)`
- draw-index dtype: NumPy `int32`
- byte serialization: C order
- expected draw-index SHA-256:
  `478a7789dfcac82b7b2c3663a60da82147e6014942fd29291e4c0a8a3688297e`
- quantiles: NumPy `method="linear"`
- point estimates: all 400 rows
- two-sided 95% interval: 2.5% and 97.5% quantiles
- one-sided 95% lower bound: 5% quantile

The draw-index array dtype, byte order, shape, and SHA-256 are frozen above.
The independent audit regenerates the draw matrix from the binding environment
and must match its bytes and endpoint values. A different hash aborts before
scientific labels.

No source-stratified, cluster, Bayesian, studentized, BCa, adaptive, or
outlier-trimmed interval may replace the primary ordinary bootstrap.

---

## 13. Confirmatory decisions

### H7-A: 27B sample-specific causal candidate channel

Label:

`27B SAMPLE-SPECIFIC CHANNEL REPLICATED`

only if all three gates pass:

1. the two-sided 95% CI lower bound for `G_specific` is greater than zero;
2. the two-sided 95% CI lower bound for `G_content` is greater than zero;
3. the one-sided 95% lower bound for `T_p3` is greater than `0.90`.

Otherwise label:

`NO 27B SAMPLE-SPECIFIC CHANNEL REPLICATION CLAIM`

and list every failed gate.

### H7-B: 27B first-token predictive alignment

Label:

`27B PREDICTIVE ALIGNMENT REPLICATED`

only if the two-sided 95% CI lower bound for `A_meanmass` is greater than zero.

Otherwise label:

`NO 27B PREDICTIVE ALIGNMENT REPLICATION CLAIM`

### Cross-model headline

The phrase:

`SAMPLE-SPECIFIC NATURAL-LANGUAGE PREDICTIVE-STATE CODE REPLICATED ACROSS 12B AND 27B SYSTEMS`

is allowed only if:

1. H7-A passes;
2. H7-B passes;
3. the independent audit passes;
4. every fail-closed check passes.

This headline does not claim every model, layer, position, or NLA checkpoint.

### Named secondary endpoints

`M_majority`:

- label `27B MAJORITY-OF-CANDIDATE-BENEFIT SUPPORTED` only if its two-sided
  95% CI lower bound is greater than zero;
- otherwise `NO 27B MAJORITY CLAIM`.

`G_candidate_anchor`:

- label `27B CANDIDATE DOMINANCE SUPPORTED` only if its two-sided 95% CI lower
  bound is greater than zero;
- otherwise `NO 27B CANDIDATE DOMINANCE CLAIM`.

Neither is an H7-A gate.

No formal 12B-versus-27B difference, equivalence, non-inferiority, superiority,
or emergence claim is preregistered. Comparing N6 and N7 point estimates is
descriptive only.

---

## 14. Failure interpretations

- `G_specific` fails:
  no replication evidence for incremental candidate identity beyond
  anchors/format in this 27B eligible population; this does not prove `p3` is
  generic.
- `G_content` passes but `G_specific` fails:
  generic candidate content or list structure may help, but
  recipient-specific content is unestablished.
- H7-A passes but H7-B fails:
  candidate identity matters causally, but the first-token predictive-alignment
  mechanism did not replicate.
- H7-B passes but H7-A fails:
  AV candidates correlate with the clean distribution, but they are not
  established as the causal `p3` mechanism.
- `T_p3` fails:
  p3 is not established as near-sufficient at the frozen 0.90 retention gate,
  even if candidate contrasts are positive.
- capacity, parser, hash, provenance, identity-KL, or finite-value failure:
  protocol abort with no scientific H7 label.
- OOM or unavailable assets:
  infrastructure abort, not scientific falsification.

A failed N7 gate does not erase N6's 12B/L32 result. It establishes a boundary
for the frozen 27B/L41 protocol.

---

## 15. Mandatory descriptive reporting

The following are mandatory but cannot redefine the frozen decisions:

- provisional and eligible counts;
- all eligibility rejection reasons;
- source and candidate-count composition;
- per-source point estimates;
- leave-one-source-out point estimates;
- candidate, `p3`, and explanation token lengths;
- target-anchor and context-anchor diagnostics;
- true/cross lexical overlap;
- centered cosine for all reconstruction conditions using only the new frozen
  `m_hat_27b_l41`;
- raw KL, first-16 KL, and first-16 CE;
- aggregate recoveries and `T_p3`;
- counts where `KL_cross-KL_true > 1 nat`;
- counts where `KL_true-KL_sae_big > 1 nat`;
- hit@1/5/10/50 and observed-token membership;
- donor matching costs and hard-cell feasibility;
- peak GPU memory, stage throughput, wall time, disk use, and failure/retry
  counts.

No subgroup, source, tail, geometry, or cross-model contrast may be promoted
post hoc to a confirmatory claim.

---

## 16. Unresolved items required before binding

The final v1 cannot be frozen until all rows below are resolved without using
N7 confirmatory outcomes.

| Item | Required resolution | Current draft state |
|---|---|---|
| Base revision and files | exact revision, shard/index/config/tokenizer hashes | unresolved |
| AV revision and files | exact revision, 22-shard/index/sidecar/tokenizer hashes | unresolved |
| AR revision and files | exact revision, 8-shard/index/value-head/sidecar hashes | unresolved |
| SAE-big revision and files | exact L41 `resid_post_all` config/params hashes | unresolved |
| Runtime environment | complete package/container/driver manifest | unresolved |
| Smoke cohort | frozen identities and 20-word shingle embargo | unresolved |
| Smoke tolerances | hook-vs-hidden-state comparison tolerance | unresolved |
| New Pile extension | exact dataset revision, extraction range, artifact and manifest hashes | unresolved |
| Old embargo inputs | local verified paths for N3/Pile/N4/N5/N6 artifacts | unresolved |
| N4/N5/N6 embargo | one combined model-independent manifest and zero-overlap QA | unresolved |
| N7 scripts | new versioned stage code and wrappers | unresolved |
| Code manifest | exact hash list, no stale N5/N6 path references | unresolved |
| Centering artifact | provisional-activation mean file and contract | unresolved |
| Output schemas | exact JSON/NPZ/parquet field contracts | unresolved |
| Bootstrap byte contract | implement and self-test the frozen int32/C-order/hash contract | specified; implementation unresolved |
| Monitor/pull | bounded retry, digit-safe key parser, staged verify | unresolved |
| Shutdown | non-systemd success/failure shutdown and logging | unresolved |
| Resource budget | verified free disk, GPU type/VRAM, expected wall time | unresolved |

Resolving these engineering items may produce a new draft revision. Once the
binding v1 exists, none may be changed in place.

---

## 17. Fail-closed checks

N7 aborts without a scientific label on any failure of:

- preregistration sidecar or basename;
- code, model, environment, corpus, upstream, or embargo hash;
- missing model shard/index/config/tokenizer/sidecar/value-head file;
- wrong base family, wrong layer, wrong hook, wrong hidden width, or wrong SAE;
- N4/N5/N6/smoke identity overlap;
- N4/N5/N6/smoke or internal N7 20-word shingle overlap;
- provisional minimum size;
- exact activation provenance;
- AV parser and target-anchor contract;
- exactly 400 unique analysis rows/content groups/documents;
- complete one-to-one donor derangement in every retained cell;
- byte identity outside intended quote interiors;
- exact row/condition mapping;
- vector shape not equal to 5376;
- any NaN or infinity;
- aggregate `sum(KL_zero) <= 0`;
- identity absolute patched-position KL or mean first-16 KL greater than
  `1e-5`;
- any KL below `-1e-7`;
- main/independent endpoint disagreement above `1e-12`;
- formal-label disagreement;
- unregistered quantization, offload, parser repair, row removal, or fallback.

KL values in `[-1e-7,0)` are listed and clamped to zero. Values below
`-1e-7` abort.

No outcome row may be silently dropped.

---

## 18. Planned implementation stages

Proposed new files; exact names and hashes are unresolved until binding:

| Proposed stage | Responsibility |
|---|---|
| 57 | freeze 27B fresh cohort and N4/N5/N6/smoke embargo |
| 58 | extract L41 activations and freeze descriptive centering vector |
| 59 | generate AV explanations |
| 60 | parse eligibility, freeze 400 rows, donors, and text variants |
| 61 | sequential AR and SAE-big reconstruction |
| 62 | base-model causal patch and candidate mass |
| 63 | primary analysis and frozen labels |
| 64 | independent raw-artifact audit |

N6 stage 49–56 files remain immutable historical code. The N7 implementation
must not overwrite any N6 artifact.

Every stage:

- validates all direct input hashes before work;
- refuses to overwrite an existing final output;
- writes resumable checkpoints only under a binding contract hash;
- atomically publishes final outputs;
- records peak memory, wall time, exit status, and hashes;
- fails closed.

---

## 19. Resource and operational stopping rule

Expected minimum model footprint before HF cache and experiment intermediates:

- base: approximately 54.904 GB;
- AV: approximately 108.077 GB on disk;
- AR: approximately 37.600 GB;
- required SAE-big: approximately 0.705 GB;
- optional SAE-small: approximately 0.705 GB.

Prepare at least 250–350 GB free disk. The N6-era 150 GB volume with
approximately 81 GB free is insufficient.

The initial smoke target is one A800/H800-class GPU:

- 80 GB may work only with explicit BF16, batch 1, and sequential loading;
- 96 GB is safer;
- no claim is made that smoke will pass until measured.

GPU billing begins while the instance is on, regardless of utilization.

The final authorized run ends after:

1. raw artifacts are complete or a fail-closed error is recorded;
2. independent audit finishes or its failure is recorded;
3. resource report is written;
4. staged artifacts are pulled and verified when connectivity permits;
5. `sync; /usr/bin/shutdown -h now` is issued on success or failure.

Do not use `systemctl poweroff`. SSH disconnect is not independent proof that
the AutoDL control plane is off; the user should confirm in the console without
restarting the instance.

This draft alone does not authorize SSH, model download, GPU startup, or
shutdown.

---

## 20. Scope

If successful, N7 may establish that the N6 sample-specific predictive-state
mechanism externally replicates in the fresh Pile AV-format-eligible subset of
the released Gemma-3-27B-IT L41 paired AV/AR system.

It cannot establish:

- proposition-level human faithfulness;
- correctness of full continuation candidates;
- global superiority or equivalence to SAE-big;
- safe selective routing;
- steering;
- all-layer or all-position generality;
- all-model generality;
- a parameter-count causal effect;
- an emergence threshold.

Those require separate preregistrations and data.
