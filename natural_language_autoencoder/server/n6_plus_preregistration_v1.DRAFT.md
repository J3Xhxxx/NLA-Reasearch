# N6+ matched candidate-content causal specificity — preregistration v1 DRAFT

Status: **DRAFT — not frozen**. This file becomes binding only after:

1. the parser and donor matcher pass N5-only CPU tests;
2. all implementation self-tests pass;
3. the final file is renamed without `.DRAFT`;
4. a SHA-256 sidecar is written; and
5. that hash is embedded in every N6 artifact before any N6 AR reconstruction,
   base-model candidate-mass score, or causal-patch outcome is generated.

No N6 outcome may be inspected before the binding version is frozen. N4 and N5
may be used as design data and are embargoed from the N6 cohort.

## 1. Scientific question

N5 established that the final paragraph (`p3`) of an AV explanation is a
causally dominant and near-sufficient reconstruction channel. It did not
separate three possible mechanisms:

1. recipient-specific continuation content;
2. the target-token/context anchors in the first two quoted spans; and
3. candidate-list format, lexical density, or generic familiar wording.

N6+ tests whether the quoted continuation candidates after the two anchors
carry recipient-specific predictive content.

## 2. Frozen model and activation site

- Target model: `google/gemma-3-12b-it`.
- Activation site: layer 32 `resid_post`, using the exact N5 hook and provenance
  contract.
- AV: `kitft/nla-gemma3-12b-L32-av`.
- AR: `kitft/nla-gemma3-12b-L32-ar`.
- SAE comparator: the exact N5 Gemma Scope 2 layer-32 resid-post w16k SAE-big.
- Model-file identities must match the N5 combined model manifest unless a
  pre-outcome manifest audit documents a byte-identical replacement.

## 3. Cohort

### 3.1 Source corpus

N6 uses Pile only. A fresh XNLI cohort is not used because N5 already consumed
150 of the 186 independently grouped XNLI premise groups available under the
corrected v2 grouping; reusing them would invalidate independence.

Pile sources are the same 13 frozen N5 sources:

`ArXiv`, `DM Mathematics`, `FreeLaw`, `Github`, `HackerNews`,
`NIH ExPorter`, `OpenWebText2`, `Pile-CC`, `PubMed Abstracts`,
`PubMed Central`, `StackExchange`, `USPTO Backgrounds`, and
`Wikipedia (en)`.

### 3.2 Position rules

The N5 v2 content-token rules are retained:

- sequence length 512;
- position in `[64,480]`;
- at least 16 continuation tokens;
- at least 75% of the continuation passes the frozen content-token predicate;
- one position per independent content group;
- deterministic SHA-256 ordering;
- no chat-template or structural-token fallback.

Every N4 and N5 document/content-group identity is embargoed. Candidate prefixes
must also have no case-normalized 20-word shingle overlap with N4 or N5 frozen
prefixes.

### 3.3 Provisional and analysis cohorts

Before any N6 model output, freeze 455 provisional Pile groups: 35 from each of
the 13 sources.

After AV generation but before any AR, candidate-mass, SAE, or causal outcome,
apply the frozen text-only eligibility parser:

- at least three blank-line-separated paragraphs;
- p3 is the last paragraph;
- at least six and at most eight ASCII double-quoted spans in p3;
- the first quoted span is the target-token anchor;
- the second quoted span is the local-context anchor;
- spans 3 onward are candidate spans, giving 4–6 candidates;
- every candidate span is non-empty after Unicode NFKC and whitespace
  normalization;
- every candidate has a non-empty canonical first-token encoding.

The target-token anchor comparison is an audit diagnostic, not an additional
eligibility condition unless N5-only parser validation shows a deterministic,
tokenizer-safe normalization rule with no semantic judgment.

Select exactly 400 eligible groups in frozen within-source SHA-256 order.
Source quotas are 30 each plus one extra for the first ten sources under
`SHA256(seed, "analysis-extra-source", source)` ordering. If a source cannot
fill its frozen quota from its 35 provisional rows, N6 aborts before AR or
causal scoring. Eligibility rate and every rejected row remain mandatory
reports.

## 4. Frozen p3 parser and variants

The parser splits p3 with the ASCII non-greedy quote regex used in N5. Let:

- `anchor_token` be quoted span 1;
- `anchor_context` be quoted span 2;
- `candidates` be quoted spans 3 onward.

All transformations operate on p3 only. Unless a quoted span is explicitly
replaced, every byte outside the replaced span must be identical to recipient
`p3_true`.

Conditions:

1. `orig`: full AV explanation.
2. `p3_true`: recipient p3 unchanged.
3. `p3_cross_matched`: recipient p3 with candidate spans replaced by a donor's
   candidate spans; both anchors and all non-candidate bytes remain recipient
   bytes.
4. `p3_candidate_strip`: only candidate-span contents are replaced by the
   fixed placeholder `[...]`.
5. `p3_anchor_strip`: only the first two quoted-span contents are replaced by
   `[...]`; all candidate spans remain unchanged.
6. `p3_all_quote_strip`: all quoted-span contents in p3 are replaced by
   `[...]`.
7. `p12`: every paragraph before p3.
8. `sae_big`: frozen SAE-big reconstruction.
9. `identity`: exact clean activation.
10. `zero`: all-zero activation.

No candidate span may be regenerated, paraphrased, edited by an LLM, or chosen
using AR, SAE, target-model logits, or causal outcomes.

## 5. Donor assignment

Donors are selected only among the frozen 400 analysis rows.

Hard constraints:

- donor and recipient row/content/document identities differ;
- candidate count is identical;
- no normalized candidate span is exactly shared between donor and recipient;
- assignment is a one-to-one derangement within candidate-count blocks.

The deterministic assignment minimizes, in order:

1. absolute difference in total candidate tokenizer length;
2. sum of absolute differences in sorted per-candidate tokenizer lengths;
3. absolute difference in total p3 tokenizer length;
4. a penalty for different Pile source;
5. SHA-256 tie rank.

The exact integer cost scaling and matching algorithm must be frozen in the
binding preregistration. Failure to find a full derangement aborts before AR
scoring. The donor map, component costs, and all transformed-text hashes are
frozen and reported.

## 6. Outcomes

### 6.1 Causal fidelity

For each condition, patch the reconstruction at the exact layer-32 position and
measure:

- primary: `KL(clean || patched)` at the patched position;
- mandatory diagnostics: mean KL over the first 16 continuation positions and
  clean-target CE over the same 16 positions.

Identity KL must be at most `1e-5`; otherwise the run aborts without a
scientific decision. Negative numerical KL values within the N5 tolerance are
clamped to zero and listed; larger violations abort.

All primary aggregate quantities use ratios of sums. Row-wise division by
`KL_zero` is forbidden.

Define:

`G_specific = sum(KL_cross - KL_true) / sum(KL_zero)`.

`G_content = sum(KL_candidate_strip - KL_true) / sum(KL_zero)`.

`M_majority = [sum(KL_cross - KL_true)
               - 0.5 * sum(KL_candidate_strip - KL_true)]
              / sum(KL_zero)`.

`G_candidate_anchor =
    sum(KL_candidate_strip - KL_anchor_strip) / sum(KL_zero)`.

`R_s = 1 - sum(KL_s) / sum(KL_zero)`.

`T_p3 = R_p3_true / R_orig`.

The raw paired mean `mean(KL_cross-KL_true)` is mandatory beside
`G_specific`.

### 6.2 Clean predictive alignment

From the clean target-model next-token distribution at each recipient position:

1. canonicalize each candidate by prepending one ASCII space unless it already
   begins with whitespace or punctuation;
2. tokenize without special tokens;
3. take the first token ID;
4. deduplicate IDs within a candidate set; and
5. sum their clean next-token probabilities.

Let `P_true` and `P_cross` be this mass for the true and matched-cross candidate
sets. The paired alignment score is:

`A_mass = mean[log(P_true + 1e-12) - log(P_cross + 1e-12)]`.

Raw masses, unique first-token counts, top-k hits, and both raw/no-leading-space
tokenizations are mandatory diagnostics. Only the frozen canonical
leading-space rule enters the confirmatory endpoint.

## 7. Bootstrap

- Analysis unit: one frozen content group / row.
- Resamples: 50,000 ordinary group bootstraps because N6 is Pile-only.
- Shared resample indices across every endpoint.
- Seed: `20260803`.
- Quantiles: NumPy linear quantiles.
- Point estimates always use all 400 rows.

The binding implementation must include a deterministic synthetic end-to-end
self-test and a second independent audit script that does not import the main
analysis module.

## 8. Confirmatory decisions

### H6-A: sample-specific candidate channel

Label `SAMPLE-SPECIFIC CHANNEL CONFIRMED` only if all hold:

1. the two-sided 95% bootstrap CI lower bound for `G_specific` is greater than
   zero;
2. the two-sided 95% bootstrap CI lower bound for `G_content` is greater than
   zero;
3. the one-sided 95% bootstrap lower bound for `M_majority` is greater than
   zero; and
4. the one-sided 95% lower bound for `T_p3` is greater than `.90`.

Otherwise label `NO SAMPLE-SPECIFIC CHANNEL CLAIM` and list failed gates.

### H6-B: predictive alignment

Label `PREDICTIVE ALIGNMENT CONFIRMED` only if the two-sided 95% bootstrap CI
lower bound for `A_mass` is greater than zero.

Otherwise label `NO PREDICTIVE ALIGNMENT CLAIM`.

H6-A and H6-B are separate. Neither may rescue failure of the other. The phrase
“sample-specific natural-language predictive-state code” is allowed only when
both H6-A and H6-B pass.

`G_candidate_anchor` is a named secondary mechanistic endpoint. It may support
candidate dominance only when its two-sided 95% CI lower bound is above zero;
it is not part of the two primary labels.

## 9. Mandatory subgroup and tail reporting

Descriptive only:

- Pile source;
- candidate count 4/5/6;
- p3 and candidate token lengths;
- anchor-token match status;
- true/cross lexical overlap;
- centered cosine by reconstruction condition;
- counts where cross is worse than true by more than 1 nat;
- counts where true is worse than SAE-big by more than 1 nat.

No subgroup may be promoted to a confirmatory claim or used to redefine the
cohort, parser, donor rule, endpoint, or decision labels.

## 10. Stopping and resource rules

Abort before outcome analysis on any failure of:

- N4/N5 embargo or source quotas;
- model/hash binding;
- activation provenance;
- parser/eligibility/source-quota contract;
- complete donor derangement;
- byte-identity outside intended spans;
- exact row/condition mapping;
- identity KL;
- finite numerical outputs.

N6+ ends after artifact pull, independent audit, resource report, and automatic
server shutdown. The shutdown path must be:

`sync; /usr/bin/shutdown -h now`

and must not rely on `systemctl`. The remote supervisor must shut down on both
success and failure, after allowing a bounded result-pull acknowledgement
window. No N7, router retry, C1, steering, or cross-model experiment is
authorized by this preregistration.

