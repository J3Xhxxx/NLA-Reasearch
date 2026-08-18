# C1 v3r2 scenario-anchor audit failure

Date: 2026-07-30

## Decision

The final permitted pre-text scenario-anchor iteration, v3r2, failed its
independent conservative semantic audit. The frozen stop rule therefore
prohibits C1 v3 text generation in this run.

No v3 request text was generated. No v3 activation was extracted, no SAE
feature was selected, no AV or AR output was produced, and no downstream
endpoint or held-out metric was inspected.

## Frozen evidence

- v3r2 anchors:
  `server/c1_confirmatory_scenario_anchors_v3r2.json`
  - SHA-256:
    `3b38876a663ea3a3a9a1623017242a06e0f51b667109cf60bb8de549cb21600a`
- independent audit:
  `results/c1_confirmatory_scenario_anchor_audit_v3r2.json`
  - SHA-256:
    `23908a7784e3e49f96daf0437186cc9bc9d6c1f453e6d8c2c1c9405e1786b1ef`
  - status: `FAIL`
- v3r2 audit addendum:
  `server/c1_confirmatory_manual_audit_addendum_v3r2.json`
  - SHA-256:
    `d98776815bde9c16b3e1a757f238542c127c7eeca6aff86e4fa54b137722394c`
- preregistration amendment:
  `results/c1_confirmatory_preregistration_v3_amendment.md`
  - SHA-256:
    `3ef67bfd29549afa966fa003bc37a2ef58e0e97b32f9a99df85b63b7f8787314`

The failed v3 draft and its audit remain unchanged and are bound by v3r2:

- failed v3 anchors SHA-256:
  `061903133b748ccffe2f85f697c5e6a7d53fd631e63fb1c0302ae42fbd59e6d5`
- failed v3 anchor audit SHA-256:
  `5dcc267ad9f6bf9765bbb7c6ab3965498c275c7df3165780f5f1d79301c2a396`

## Coverage and result

- Structure: PASS
  - 24 concepts
  - 144 anchors
  - 96 discovery and 48 held-out anchors
  - 12 reciprocal hard-negative pairs
  - unique IDs and scenario texts
- Direct pair-level mapping, hard-negative safety, semantic difficulty, and
  style/framing balance: 12/12 PASS
- Concept batches: 17 PASS, 7 FAIL
- Overall pairs after constituent-concept conjunction: 6 PASS, 6 FAIL
- The earlier `fault_rupture_mechanics` versus `slope_failure_mechanics`
  framing imbalance was repaired and passed.

The seven failed concept batches were:

1. `error_detecting_codes`: discovery and held-out both test the
   check-length/overhead/undetected-error tradeoff.
2. `protein_quality_control`: discovery and both held-out anchors reuse
   defective nascent-protein folding, chaperone load, disposal, and
   aggregation.
3. `microbial_quorum_sensing`: discovery and held-out both center on a sharp
   density-dependent extracellular-signal threshold and abrupt response.
4. `microbial_cross_feeding`: held-out is a direct counterfactual continuation
   of the same vitamin-dependent auxotrophic partnership used in discovery.
5. `groundwater_contaminant_transport`: discovery and both held-out anchors
   reuse in-situ dissolved-plume degradation.
6. `quarantine_regimes`: multiple discovery and held-out anchors reuse
   detention duration, release criteria, and residual release risk.
7. `phonological_assimilation`: discovery and held-out both manipulate speech
   rate and categorical-versus-gradient neighboring-sound influence.

Exact anchor IDs and full reasons are preserved in the audit JSON.

## Operational state

The AutoDL instance was deliberately left running. At the stopping decision,
the A800 had no allocated GPU memory and no v3 output existed in
`/root/autodl-tmp/results` or `/root/autodl-tmp/activations`. No shutdown or
automatic release action was issued.

## What requires a new decision

Continuing from this point is not a retry of the frozen run. It requires an
explicitly disclosed new design, for example:

1. a v4 corpus design with a new preregistration and a fresh whole-corpus
   generation attempt; or
2. a changed generalization estimand that defines held-out separation at the
   application, perturbation, evidence, or document-source level instead of
   requiring a different substantive mechanism inside a narrow concept.

The second option is scientifically cleaner if the present failure reflects a
mismatch between the intended concept-generalization question and an overly
strict “different mechanism” rubric. Neither option should reuse or edit a
subset of generated text, and neither has been started.
