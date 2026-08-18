# J2 preflight hash correction — 2026-08-07

## Scope

This note records a metadata correction made during the preflight of the
formal J2 SAE-projection language-loop run. It is a pre-outcome bookkeeping
event; it does not alter the frozen cohort, estimands, condition definitions,
or case-study selection rules.

## Chronology

1. The first remote `66 --dry-run` failed before producing any AV output. The
   failure was caused by the frozen SHA string for `n4_explanations`: the
   string in the initial upload contained one extra `b` (65 characters),
   whereas artifact SHA-256 digests are 64 characters.
2. The correct SHA-256 for the existing `n4_explanations` artifact is:

   `b656ded845c8fd122e4dcb1391ba5d81e1a903f80a69c30575bf26910e200942`

3. At the time of that failed preflight, the remote results directory
   contained no `j2_sae_projection*` outputs. Thus the failed preflight did
   not create partial J2 data or contaminate the run.
4. The protocol, scripts `66`–`68`, runner, and manifest were corrected and
   re-hashed. The remote sidecar checks, Python compilation, and shell syntax
   checks then passed.
5. The second dry-run returned:

   - `contract = 8ee91239f024019bb174f6b5e774e45cb8f7bb183d53e8d62a5f391ec9e12625`
   - `complete = 0/400`
   - `missing = 400`
   - `dry_run = True`

   Only after this clean preflight did formal J2 generation begin.

## Interpretation

The event is a **pre-outcome metadata correction**, not an experimental retry
or an outcome-dependent change. No cohort row, model condition, endpoint,
estimand, or post-hoc case rule was changed. The initial failed dry-run
generated no J2 AV/AR output, so there is no discarded or mixed partial result
to include in analysis.

