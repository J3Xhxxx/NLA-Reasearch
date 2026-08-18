# J1-D1 Luna completion and mixed-labeler addendum

Status: **EXPLORATORY / DISCOVERY ONLY**
Freeze date: 2026-08-06 (Asia/Shanghai), before any Luna J1 label outcome
User directive: retain completed batches `0..12`; use fresh Luna Max
subagents for batches `13..44`; do not repeat the first 13 batches

## 1. Bound parent artifacts

- GPU freeze SHA-256:
  `8f7690f8b12842b32ce5cb32af7ee941b2ce2f71fcc0270768cf0f84edcb50d3`
- AV result SHA-256:
  `d93d99e3c84b07a6f76b3b4549bb16fbe520f9c7aa59579f98e57bb2a85749a4`
- Original J1 protocol SHA-256:
  `638cb1454c4e248cd56e1148f88bc91b0840797ff7ad787f1ef095bd920777cf`
- Downstream label addendum SHA-256:
  `75fe0caf64a2e598cd7509ca13ab83c8fc09bcfe43f28c5107e6d5b9097e0da3`
- Frozen label jobs SHA-256:
  `411c67acd230018c60d50194d51c70f3cde847c7f85b38713456450b609f4aad`
- Original label runner SHA-256:
  `132697447b3169dabab373fec2d0647f396424e611d080515f6ae85118ce37f9`
- Append-only checkpoint snapshot before Luna completion SHA-256:
  `439f8af0c438e8a79a13af5bb6dd463932ca1d56e225912da3dba9edd8cd12e4`

The checkpoint contains valid Fable successes for exactly batches `0..12`.
Historical `403 insufficient balance` error rows for later batches are retained
as execution history but are not labels.

## 2. Explicit protocol deviation

The first 13 successful batches were produced by `claude-fable-5`. The user
subsequently required Luna Max and explicitly instructed that these batches
must not be repeated. Batches `13..44` will therefore be produced by fresh
Codex Luna workers.

The final discovery artifact is intentionally labeled **mixed-labeler**:

- batches `0..12`: Claude Fable 5;
- batches `13..44`: `gpt-5.6-luna`, Luna worker role, reasoning effort `max`;
- no claim of a single fixed interpreter;
- no substitution is hidden or described as a clean replication;
- no Fable request is issued during completion.

This deviation is acceptable only for exploratory protocol triage. A future
confirmatory J1 must use one frozen interpreter or independently replicate the
complete label set with each interpreter.

## 3. Why arm counts remain balanced

The already frozen cross-feature schedule contains exactly one case from each
of the five arms in every batch. Consequently every arm has:

- 13 Fable-generated hypotheses;
- 32 Luna-generated hypotheses.

This prevents a global arm-count imbalance. It does not remove feature-level
confounding, because the feature assigned to a given arm differs by batch.
Labeler identity must therefore be retained for every feature-arm hypothesis
and used in robustness analyses.

## 4. Luna execution isolation

Only the frozen public prompt for one batch may be shown to a Luna worker.
Public inputs contain case IDs and evidence only. They must not contain:

- feature IDs;
- arm or condition names;
- private condition maps;
- held-out contexts, hard negatives, activations, or truth;
- outputs from any other Fable or Luna batch.

Each batch `13..44` is assigned to a newly spawned `luna_worker` agent with
`fork_turns="none"`. One agent handles exactly one batch and is not reused.
This prevents same-feature or cross-batch label transfer through conversational
memory.

The agent returns exactly five cases:

```json
{
  "cases": [
    {
      "case_id": "opaque id",
      "hypothesis": "at most 32 words",
      "positive_cues": ["..."],
      "exclusion_cues": ["..."],
      "abstain": false,
      "confidence": 0.0
    }
  ]
}
```

Every output artifact records the canonical subagent task name, Luna worker
role, model family, reasoning effort, public-prompt SHA-256, and output
SHA-256. Parser failure, missing IDs, duplicate IDs, extra IDs, or malformed
fields fails closed. No partial scientific analysis is allowed.

## 5. Mixed artifact

A new versioned mixed-labeler result is created; the original Fable jobs and
checkpoint are never overwritten or relabeled.

The merger must:

1. independently validate the 13 retained Fable success rows against their
   frozen prompts and raw outputs;
2. validate 32 Luna output artifacts against batches `13..44`;
3. require exactly 45 batches and 225 unique cases;
4. preserve the original private case-to-feature/arm map only in the merged
   evaluator input artifact, never in a label-model prompt;
5. attach `labeler`, `batch_id`, and provenance to every hypothesis;
6. emit no interpretation-quality metric.

## 6. Terra evaluation

After and only after all 225 labels validate, freeze the same held-out
evaluation:

- 45 features;
- five hypotheses per feature;
- four frozen held-out positives and four exact-zero hard negatives;
- 40 scores per feature, 1,800 scores total;
- evaluator `gpt-5.6-terra`;
- Codex CLI `0.146.1`;
- read-only sandbox, ephemeral empty work directory, no tools/repository data;
- SAE activation from the GPU freeze is truth; Terra is a measurement
  instrument.

No complete result is emitted unless all 1,800 pairs parse.

## 7. Required analyses

Report the original exploratory ITT summaries:

- micro pooled AP and macro mean per-feature AP;
- mean within-feature positive-versus-negative pairwise accuracy;
- Brier score, non-abstain coverage, and calibration bins;
- results by the three frozen strata;
- feature-cluster bootstrap with 20,000 shared resamples;
- direct `NLA_ASSISTED - NLA_MISMATCHED` and
  `NLA_CONTRASTIVE - NLA_MISMATCHED` contrasts.

Add mixed-labeler robustness:

1. counts and metrics by labeler and arm;
2. for every main pairwise arm contrast, a paired analysis restricted to
   features for which both hypotheses were generated by Luna;
3. the analogous Fable-Fable subset when its size is reportable, without
   treating it as powered;
4. mixed-pair features shown separately;
5. labeler identity included in every raw score row;
6. an explicit flag if the all-labeler effect and Luna-Luna paired effect
   differ in sign;
7. no post hoc threshold or subgroup promotion.

## 8. Decision rule

Proceed to design a fresh confirmatory J1 only if:

- `NLA_ASSISTED` improves over both `SAE_CONTEXT` and `NLA_MISMATCHED` in the
  complete ITT analysis;
- the corresponding Luna-Luna common-feature contrasts have the same
  favorable direction;
- the apparent gain is not confined to one stratum;
- calibration, Brier score, coverage, and negative-tail behavior show no
  obvious collapse;
- mismatched NLA text does not reproduce the gain.

`NLA_CONTRASTIVE` is separately promising if it beats both `SAE_CONTEXT` and
`NLA_MISMATCHED` under the same robustness requirements.

If only the mixed full analysis is positive, or the Luna-Luna direction
reverses, the correct decision is **redesign/replicate**, not confirm.
Regardless of outcome, J1-D1 cannot itself establish an NLA-assisted-SAE
performance gain.

## 9. Infrastructure

The GPU stage is complete and all remote artifacts are local. No GPU is needed
for Luna labels, Terra evaluation, or statistics. The AutoDL host was sent
`sync; /usr/bin/shutdown -h now` before this addendum was frozen; a subsequent
SSH attempt timed out. Do not restart it for this pipeline.
