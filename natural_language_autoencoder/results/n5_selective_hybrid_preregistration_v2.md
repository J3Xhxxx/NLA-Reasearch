# N5 selective NLA + SAE-big - preregistration v2 amendment

Frozen on 2026-07-30 (Asia/Shanghai), before any N5 base-model activation,
AV explanation, AR reconstruction, SAE reconstruction, causal result, or gate
score was generated.

This document supersedes v1 for N5. The frozen v1 file has SHA-256
`110c952805e5c8d469815c9f60a6fcf4520537452c60c6305e1ea699bd3b82b0`.
Every v1 clause remains in force except where explicitly replaced below.

## Reason for the amendment and audit trail

The first tokenizer-only v1 cohort construction aborted before writing a cohort
plan. The raw XNLI validation table has 2,490 rows, but the same parallel
premise is normally repeated on three consecutive rows for different
hypotheses. The old N3 construction grouped every eight raw rows. Because eight
is not divisible by three, an identical premise can cross adjacent old
`passage_id` values. For example, the v1 audit found that the Vietnamese text
in old passage 151 shared exact 20-word sequences with old passages 150 and
152. Thus old passage IDs 124-273 were not 150 independent content groups.

The v1 implementation correctly stopped at its text audit. No N5 cohort plan or
model-derived N5 output existed when this amendment was written. The correction
below changes only the definition of an independent XNLI content unit. It does
not inspect or optimize any activation, explanation, reconstruction, causal
metric, gate score, or held-out outcome.

The failed v1 logs and scripts remain audit artifacts and must not be deleted or
presented as results.

## Replacement for the XNLI cohort definition

The frozen XNLI source remains the `facebook/xnli` all-languages validation
parquet with SHA-256
`c5e6263b0872a3914c9bc165bfe3883e433aa2066c3fa3b9d142829a9b122518`.
The ten languages remain `en`, `es`, `zh`, `de`, `fr`, `ru`, `ar`, `hi`,
`tr`, and `vi`.

The 2,490 raw rows are converted to independent parallel-premise units before
any passage is formed:

1. For each raw row, take the exact ordered tuple of its ten language premise
   strings.
2. Canonically UTF-8 serialize that tuple and compute SHA-256.
3. Rows with the same ten-language tuple are one parallel-premise unit,
   regardless of whether the duplicates are consecutive. The unit's order key
   is the first raw row index at which it appears.
4. The implementation must report the raw-row multiplicity distribution, the
   number of unique units, and the raw-row-to-unit mapping hash.

The N4 embargo is then lifted from old passage IDs to actual content:

1. Read every N4 XNLI old `passage_id`.
2. Map all eight raw validation rows used by each such old passage to the
   parallel-premise identities above.
3. Embargo every resulting identity across all languages.

Candidate XNLI content groups are formed before tokenizer eligibility:

1. Remove all embargoed units.
2. Sort remaining units by first raw row index.
3. Partition that sequence into consecutive, nonoverlapping blocks of four
   units; discard only a final block shorter than four.
4. A candidate group's language text is the four premise strings in original
   unit order joined by one space.
5. Its content identity is the ordered tuple of its four unit SHA-256 values.
   No unit may occur in more than one candidate group.

Candidate groups and positions use the frozen seed `20260730`. For each
candidate-language pair, apply the unchanged v1 tokenizer-only content-token
eligibility rule and deterministic SHA position selection. Candidate groups
are processed in SHA-256 order. A deterministic augmenting-path matching assigns
eligible groups to the 150 frozen split/language slots: for every language,
5 discovery and 10 held-out slots. Candidate-to-slot edge order is SHA-256
deterministic. Before all slots are filled, a candidate that cannot augment the
current matching is skipped and recorded; processing stops immediately once
all 150 slots are filled. If all slots cannot be filled, v2 aborts before any
model forward.

Exactly one language is selected for each chosen group. The 150 selected groups
therefore contain 600 distinct unique parallel-premise units. The implementation
must recompute that:

- all 600 unit identities are unique;
- none is in the N4 content embargo;
- no candidate group or unit crosses discovery and held-out;
- each language contributes exactly 5 discovery and 10 held-out groups.

XNLI does not use the Pile 20-word shingle rule as its grouping primitive.
Exact parallel-premise identity and nonoverlapping four-unit blocks are its
independence rule. The v1 Pile document embargo and Pile-only normalized
20-word shingle embargo remain unchanged.

## Cohort counts and unchanged protocol

N5 still contains exactly 600 independent groups and one frozen position per
group:

| Split | Pile | XNLI | Total |
|---|---:|---:|---:|
| discovery | 150 | 50 | 200 |
| held-out | 300 | 100 | 400 |
| total | 450 | 150 | 600 |

All Pile sources, per-source quotas, token eligibility, sequence length,
position range, continuation rule, deterministic ordering, and abort-before-
model-output discipline are unchanged from v1.

Discovery reconstruction and causal results must still finish before the gate,
discovery mean direction, and their hashes are frozen. No held-out AV/AR
reconstruction or held-out causal run may begin before that gate artifact
exists.

The only allowed H5-A score, fraction grid, discovery feasibility conditions,
objective, held-out estimands, 50,000 stratified bootstrap, exact catastrophic
regret bound, and decision labels remain exactly those in v1. The H5-B
paragraph definitions, global fewer-than-three-paragraph abort, causal
conditions, estimands, and decision labels also remain exactly those in v1.
All v1 ITT, numerical, mandatory-reporting, forbidden-substitution, resource,
and stopping rules remain in force.

## Frozen model identity

The combined base/AV/AR/SAE model-file manifest has SHA-256
`4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735`.
It lists 25 configuration/index/weight files. Before cohort construction was
resumed, `sha256sum -c` checked all 25 entries and returned exit status 0.
Every N5 model stage must bind this manifest hash; reconstruction and causal
stages must additionally record the actual model files they consume.

## Stopping rule

Any failure of the corrected unique-premise grouping, N4 content embargo,
nonoverlap checks, exact quotas, tokenizer eligibility, hash chain, channel
audit, identity KL, or clean-activation provenance aborts the relevant stage.
It may not be repaired after observing held-out outcomes.

N5 ends after the held-out analysis and resource report. No N6, feature-level
C1/Q2 experiment, shutdown, or power-state change is authorized without the
user's next instruction.
