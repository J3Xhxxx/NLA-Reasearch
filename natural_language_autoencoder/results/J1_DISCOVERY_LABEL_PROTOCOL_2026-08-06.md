# J1-D1 downstream label/evaluation addendum

Status: **EXPLORATORY / DISCOVERY ONLY**
Freeze date: 2026-08-06 (Asia/Shanghai), before any J1 label-generation outcome
Parent protocol: `J1_DISCOVERY_PILOT_PROTOCOL_2026-08-06.md`
Parent protocol SHA-256: `638cb1454c4e248cd56e1148f88bc91b0840797ff7ad787f1ef095bd920777cf`

## 1. Scope

This addendum governs only the downstream interpretation and blinded-evaluation
stages of J1-D1. It does not change the 45-feature cohort, discovery contexts,
held-out positives, hard negatives, residual vectors, NLA AV generation,
SAE-feature ablations, external activation truth, endpoints, strata, or
discovery-only claim boundary frozen by the parent protocol.

The addendum was written before invoking Claude Fable 5 on any J1 prompt and
before invoking the Terra evaluator on any J1 hypothesis or held-out context.
Generic CLI/schema smoke tests contained no J1 data and are not outcomes.

## 2. Label evidence correction

The label generator must never receive held-out contexts, hard negatives,
held-out SAE truth, or numeric target-feature activation values. The initial
implementation had included each discovery context's activation magnitude as
prompt metadata. That field is removed before any label call because the
parent protocol defines `SAE_CONTEXT` as raw discovery contexts only and the
other context-bearing arms as those same contexts plus their specified text.

All context-bearing arms use the identical four marked discovery contexts:

- `SAE_CONTEXT`: marked raw contexts only;
- `NLA_ASSISTED`: the contexts plus four on-manifold NLA snippets;
- `NLA_CONTRASTIVE`: the contexts plus four raw-versus-feature-ablated NLA pairs;
- `NLA_MISMATCHED`: the contexts plus four donor-feature NLA snippets;
- `NLA_ONLY`: the four on-manifold NLA snippets without contexts.

Opaque identifiers hide feature and arm names. Evidence modality cannot be
fully blinded because the treatments necessarily contain different evidence
structures. This is an explicit discovery-pilot limitation and prohibits
interpreting the label generator as unaware of treatment modality.

## 3. Cross-feature call isolation

Putting all five arms for one feature in the same model call would allow direct
cross-arm information transfer: an NLA-bearing case could reveal the concept
while the model writes the nominal context-only hypothesis. Therefore the
frozen call schedule contains 45 batches of five cases with:

- exactly one case from each arm per batch;
- five different target features per batch;
- each feature's five arms assigned to five different batches;
- exactly 225 feature-arm cases overall;
- deterministic assignment from seed `20260806`;
- no feature ID, arm name, condition map, held-out value, or truth in a public
  prompt.

The private case-to-feature/arm map is saved only for deblinding after all
hypotheses have been parsed. This preserves 45 label-model calls while blocking
same-feature cross-arm transfer within a call.

## 4. Mismatched control

Within each of the three frozen strata, donor features are assigned by a
deterministic minimum-cost bijective derangement. A target cannot donate to
itself. Pair cost is the sum, across the four matched discovery indices, of the
absolute difference in NLA-snippet UTF-8 byte length. Exact bitmask dynamic
programming with deterministic tie-breaking fixes the assignment.

The artifact records target/donor per-index lengths, total absolute cost, and
maximum per-snippet length difference. This is a length-balancing control, not
proof of semantic exchangeability; residual semantic mismatch remains a
limitation.

## 5. Interpreter and evaluator

The planned label generator is Claude Fable 5 through Claude Code, requested
as model alias `fable`, effort `low`, tools disabled, structured JSON output.
The runner saves the exact CLI version, requested and resolved model
provenance, immutable prompts/schema, usage, cost, raw output, parser result,
and append-only attempts. A model substitution is forbidden unless explicitly
enabled and frozen before any J1 label call.

The planned evaluator is `gpt-5.6-terra` through Codex CLI `0.146.1`, with
read-only sandboxing, an ephemeral empty working directory, no repository
access, and a frozen JSON schema. Each feature is evaluated in one blinded
batch containing five hypotheses and eight held-out contexts. SAE activation
from the GPU freeze remains binary truth; Terra is only a measurement
instrument.

## 6. Integrity and completion

All input sidecars and upstream bindings must verify before job freeze.
Checkpoint rows bind the exact prompt, input artifacts, protocol addendum,
script, model request, and CLI version. Successful resumed rows must reparse
from retained raw output and reproduce the stored structured cases and stdout
hash. Failed append-only rows remain in history but do not count as completed
and may be followed by one valid success on resume.

No label result is emitted unless all 225 feature-arm cases parse successfully.
No evaluation result is complete unless all 1,800 hypothesis-context scores
parse successfully. Failures remain ITT and withhold complete analysis rather
than deleting rows.

## 7. Claim boundary

J1-D1 remains a reused-cohort discovery pilot. Its output may decide whether a
fresh, source/shingle-embargoed, capacity-matched, causally evaluated
confirmatory J1 is worth preregistering. It cannot itself establish a formal
NLA-assisted-SAE gain, regardless of effect size or interval.
