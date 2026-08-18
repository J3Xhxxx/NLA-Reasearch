# N5 v1 tokenizer-only cohort abort

Date: 2026-07-30 (Asia/Shanghai)

Status: **ABORTED BEFORE ANY N5 MODEL FORWARD**

The v1 cohort constructor stopped before writing
`n5_cohort_plan_v1.json`. Its final audit error was:

> deterministic XNLI assignment creates an N5 internal 20-word shingle
> conflict: passage_id=151 lang=vi

The stopped constructor is preserved as
`server/42_n5_freeze_cohort_v1_aborted.py` with SHA-256
`f019bf24dd043541b4e3cbe3a1c08ab5a12be5bf06ebda3be33b88aa661547d3`.
The preserved log has SHA-256
`babec6f9517843249c832ae672e16f928c34e601358a39e57545c8bdd090368a`.

## Root cause

XNLI validation contains 2,490 rows but only 830 unique exact ten-language
premise tuples; every unique tuple occurs exactly three times for different
hypotheses. N3 grouped every eight raw rows. Because 8 is not divisible by 3,
an identical premise can cross old passage boundaries.

The stopped target old passage 151/vi shared:

- 12 normalized 20-word shingles with old passage 150/vi; and
- 31 normalized 20-word shingles with old passage 152/vi.

Across the raw table:

- 2,490 raw rows collapse to 830 unique parallel-premise units;
- the 27 N4 XNLI old passage IDs touch 83 such units;
- 747 nonembargoed units remain;
- those form 186 complete nonoverlapping four-unit candidates, with three
  units left over.

The raw-row parallel-unit identity sequence audit hash was
`219beb639c27289bea460156d3a09c3117caaba5674499259e16120319045393`.

## Resolution

No row was substituted and no favorable translation was selected. N5 v1 was
abandoned. The correction was frozen before model output in
`n5_selective_hybrid_preregistration_v2.md`: exact parallel-premise identity is
now the XNLI independence unit, N4 is embargoed at that identity level, and
candidate groups use nonoverlapping blocks of four unique units.
