# J1-D1 mixed-labeler analysis plan

Status: **FROZEN EXPLORATORY ANALYSIS PLAN**
Freeze date: 2026-08-06 (Asia/Shanghai), before any Luna J1 label outcome or
Terra J1 evaluation outcome

This plan operationalizes the mixed-labeler robustness requirements in
`J1_DISCOVERY_LUNA_COMPLETION_PROTOCOL_2026-08-06.md` without changing the
discovery-only claim boundary.

## 1. Bound inputs

- Luna completion protocol SHA-256:
  `b8653f78accc76b8a3b61a460c812df73cf76ab0111bc202da6bdc1d030081e3`
- Frozen label jobs SHA-256:
  `411c67acd230018c60d50194d51c70f3cde847c7f85b38713456450b609f4aad`
- Pre-completion checkpoint snapshot SHA-256:
  `439f8af0c438e8a79a13af5bb6dd463932ca1d56e225912da3dba9edd8cd12e4`

The frozen assignment is batches `0..12 = Fable` and
`13..44 = Luna`. Every arm therefore has exactly 13 Fable and 32 Luna
hypotheses.

## 2. Pre-outcome design audit

All 45 batches contain one case from each arm and five distinct features. Each
feature occurs once in every arm and in five distinct batches.

Across complete five-arm feature profiles:

- 9 features are Fable-only;
- 28 features are Luna-only;
- 8 features are mixed-labeler:
  `2096, 2700, 3176, 3441, 15742, 15793, 16016, 16059`.

For the four decision-relevant paired contrasts, the homogeneous-labeler and
mixed-pair counts are:

| Contrast | Luna-Luna | Fable-Fable | Mixed |
|---|---:|---:|---:|
| `NLA_ASSISTED - SAE_CONTEXT` | 31 | 12 | 2 |
| `NLA_ASSISTED - NLA_MISMATCHED` | 30 | 11 | 4 |
| `NLA_CONTRASTIVE - SAE_CONTEXT` | 30 | 11 | 4 |
| `NLA_CONTRASTIVE - NLA_MISMATCHED` | 31 | 12 | 2 |

Labeler is exactly collinear with batch/call order, so a regression adjustment
cannot be interpreted as causal deconfounding.

## 3. Complete ITT analysis

All 45 features, five arms, and 1,800 Terra scores remain in the primary
exploratory ITT analysis. For each arm report:

- pooled/micro average precision (the protocol's AUPRC endpoint);
- unweighted mean per-feature average precision;
- mean within-feature 4-by-4 positive-versus-negative pairwise accuracy;
- Brier score;
- non-abstain coverage;
- five fixed calibration bins;
- results in each of the three frozen feature strata.

Use 20,000 paired feature-cluster bootstrap resamples with seed `20260806`.
Report raw deltas and percentile 95% intervals. No p-value, threshold, arm, or
subgroup is promoted to confirmatory status.

## 4. Labeler robustness

Repeat every decision-relevant paired contrast on:

1. the Luna-Luna common-feature subset shown above;
2. the Fable-Fable common-feature subset shown above, marked low-powered;
3. mixed-pair features, reported separately rather than adjusted away;
4. the 37-feature all-arms homogeneous subset;
5. the 28 Luna-only and 9 Fable-only complete five-arm subsets.

Within every subset, resample only eligible feature clusters and use the same
20,000-resample seed contract. Record labeler identity on every hypothesis and
every raw Terra score. Flag any sign difference between complete ITT,
Luna-Luna, and Fable-Fable estimates. The Fable estimate is a robustness check,
not a powered gate.

## 5. Calibration and negative-tail diagnostics

For each arm and required subset also report:

- mean and 95th percentile probability on exact-zero hard negatives;
- hard-negative rates with probability at least `0.5` and at least `0.8`;
- mean and 5th percentile probability on held-out positives;
- held-out-positive rates with probability at most `0.5` and at most `0.2`;
- abstention rates separately for positives and negatives.

An "obvious collapse" flag is raised descriptively if, relative to either
required comparator, an assisted arm has any of:

- Brier degradation greater than `0.05`;
- non-abstain coverage loss greater than `0.10`;
- hard-negative mean probability increase greater than `0.10`;
- hard-negative `p >= 0.8` rate increase greater than `0.10`.

These thresholds are design-risk flags, not significance tests.

## 6. Decision rule

`NLA_ASSISTED` is worth taking to a fresh confirmatory J1 only if:

1. complete-ITT micro AP is higher than both `SAE_CONTEXT` and
   `NLA_MISMATCHED`;
2. both corresponding Luna-Luna paired deltas have the same favorable sign;
3. the Fable-Fable estimates do not reverse both required contrasts;
4. favorable effects occur in at least two of the three frozen strata rather
   than only one;
5. the mismatched control does not match or exceed assisted performance;
6. no obvious calibration, coverage, or hard-negative-tail collapse is
   flagged.

`NLA_CONTRASTIVE` is judged separately under the same rule.

If complete ITT is positive but Luna-Luna reverses, if benefit appears only
under one labeler, or if mismatched text reproduces it, the decision is
**REDESIGN / REPLICATE**. If neither assisted arm shows a stable external
activation-prediction gain, the decision is **DEPRIORITIZE J1**.

No outcome from this mixed-labeler discovery pilot may be described as a
confirmed NLA-assisted-SAE gain.
