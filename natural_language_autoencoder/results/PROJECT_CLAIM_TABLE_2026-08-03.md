# NLA × SAE project claim table — 2026-08-03

This table is the paper-facing status after N6+. It separates confirmatory
claims from exploratory findings, failed claims, protocol aborts, and questions
that remain untested. Numeric conclusions must be read with the cohort and
endpoint in the same row.

Current primary sources:

- `results/J2_FINAL_ANALYSIS_2026-08-07.md`
- `results/J2_INDEPENDENT_AUDIT_2026-08-07.md`
- `results/j2_sae_projection_analysis_v1.json`
- `results/J1_MIXED_DISCOVERY_FINAL_2026-08-06.md`
- `results/j1_blinded_eval_analysis_mixed_v2.json`
- `results/j1_blinded_eval_result_mixed_v2.json`
- `results/N6_FINAL_ANALYSIS_2026-08-03.md`
- `results/n6_pull_staging/n6_pull_20260803T061302Z/n6_analysis_v1.json`
- `results/n6_pull_staging/n6_pull_20260803T061302Z/n6_independent_audit_v1.json`
- `RECOVERY_2026-08-03.md`
- `RECOVERY_2026-08-01.md`
- `results/N5_ANALYSIS_V2.md`
- `results/n5_independent_audit_v2.json`
- `results/N4_REAL_CONTENT_CAUSAL_AUDIT_2026-07-30.md`
- `results/REVIEW_OPUS_2026-07-30.md`

Older root documents remain useful history but are not current sources for
server state, experimental priority, or N5/N6 decisions.

## A. Confirmatory claims

| ID | Claim | Evidence | Allowed wording | Scope / limitation |
|---|---|---|---|---|
| C-N6-A | Candidate interiors in p3 carry incremental, recipient-specific causal information beyond the fixed recipient anchors and list format. | Fresh 400-group Pile N6+: `G_specific=.117954`, 95% CI `[.102860,.133995]`; raw `KL_cross-KL_true=1.827883` nat/row, CI `[1.591758,2.074946]`; `G_content=.154382`, CI `[.136602,.173507]`; `T_p3=.995175`, one-sided lower `.993194`. All three frozen gates passed. | `H6-A: SAMPLE-SPECIFIC CHANNEL CONFIRMED`. Within the AV-format-eligible Gemma-3-12B-IT L32 population, recipient-specific candidate content has incremental causal utility. | Single model/layer and fresh Pile eligible subset. The effect is through the paired AV/AR system and does not establish human-faithful explanation. |
| C-N6-B | AV candidate first-token sets are aligned with the clean target model's next-token distribution. | `A_meanmass=9.008470`, 95% CI `[8.444855,9.572597]`; hit@1 true/cross `66.5%/3.75%`; observed-token membership `49.0%/2.75%`. | `H6-B: PREDICTIVE ALIGNMENT CONFIRMED`. Together with H6-A, the frozen headline `sample-specific natural-language predictive-state code` is allowed. | The confirmatory endpoint concerns deduplicated first-token set mass, not full-sequence probability or proposition truth. |
| C-N6-anchor-neg | Candidates were not established as dominant over target/context anchors. | Frozen secondary `G_candidate_anchor=-.045042`, 95% CI `[-.078669,-.012779]`; candidate-strip recovery `.820385`, anchor-strip `.775343`, all-quote-strip `.367074`. | `NO CANDIDATE DOMINANCE CLAIM`. Anchors and sample-specific candidates are complementary components of p3. | Ablation effects are nonlinear and must not be converted into additive percentages of information. |
| C-N5-B | The final, candidate-bearing paragraph is the dominant and near-sufficient text channel for downstream reconstruction fidelity. | Fresh 400-group N5 held-out: `G_p3_p12=+0.179490`, 95% CI `[+0.146504,+0.215996]`; `R_orig=.964506`, `R_p3=.952928`; `T=.987996`, one-sided 95% lower `.983015`; 326/400 paired signs favor p3 over p12. | `H5-B: CHANNEL REPLICATED`. For Gemma-3-12B-IT L32, p3 is causally dominant and near-sufficient relative to the full AV text. | Single model and layer. p12 still has `R=.773438`, so p3 is not unique. Existing quote stripping removes all quoted spans, not candidate spans alone. |
| C-N5-A-neg | The frozen one-variable centered-cosine router does not establish selective improvement or safe parity over SAE-big. | Fresh 400-group held-out: coverage `.655` (one-sided lower `.615`); `G=+.002419`, one-sided lower `-.001326`; raw gain `+.032168` nat, 95% CI `[-.026114,+.091261]`; catastrophe 9/400, exact one-sided upper `3.893%`. | `H5-A: NO SELECTIVE CLAIM`. The tested router failed its superiority and tail-safety gates. | This rejects the frozen score/rule, not every possible router. It must not be reported as improvement or parity. |
| C-N3-coverage | Zero activation on the old small synthetic corpus was a coverage failure, not feature death. | All eight formerly zero-activation features fired on the 8.24M-token real corpus; 24/24 legacy features were alive. | A small synthetic corpus cannot be used as a feature-death gate. | Activation on real text does not validate the old feature label or human readability. |

## B. Strong exploratory or replicated diagnostics

| ID | Finding | Evidence | Safe interpretation | Why not confirmatory |
|---|---|---|---|---|
| E-geometry | Centering removes a large common residual-stream direction and leaves NLA with better geometric reconstruction than the two frozen SAEs. | Centered cosine: NLA `.85927`, SAE-small `.65839`, SAE-big `.72456`; optimally scaled centered FVE `.707/.459/.543`. A nearest-other-real-activation control already reaches cosine `.541` and FVE `.336`. | NLA preserves more sample-specific activation geometry than these two SAE checkpoints at this operating point. | Original 40-position cohort contains 13 chat/template positions; no capacity/rate matching; geometric advantage did not become causal superiority. |
| E-B2 | NLA has a larger continuous retrieval separation but not better top-k accuracy. | 40-way centered retrieval top-1: NLA `92.5%`, SAE-small/big `95%/95%`; normalized d-prime about `6.49/5.12/5.51`. | NLA correct matches are more separated from mismatches on average, while discrete top-1 does not beat SAE. | Small, partly flawed 40-position cohort. |
| E-causal-codec | NLA and SAE-big occupy a similar aggregate causal-fidelity regime, with heavier NLA failure tails. | N2: KL recovery NLA/SAE-big `.757/.771` on the old 40 positions. N4 real-content: aggregate recovery NLA `.94795`, SAE-big `.96649`; unresolved paired mean, rare NLA catastrophes. N5 router tail reproduced genuine risk. | Centered geometry cannot substitute for sample-level downstream KL; SAE-big remains the safer fidelity baseline. | N2 cohort flawed; N4 endpoints partly post hoc after a denominator failure; N5 did not directly preregister codec equivalence. |
| E-text-ablation | Lexical/quoted content in p3 matters strongly. | N5 `R_p3=.952928` versus existing full-text `quote_strip_p3 R=.402280`. Earlier N1 showed paraphrase retention `.973`, entity-swap retention `.897`, and p3-only retention `.961`. | The channel is not explained by paragraph presence alone; semantic/lexical content carries substantial reconstruction signal. | Existing quote stripping removes the target-token quote, context-fragment quote, and candidate quotes together. It does not isolate candidate identity. |
| E-N6-majority | Recipient-specific identity accounts for more than half of the frozen candidate-content benefit on the aggregate causal scale. | Pre-registered secondary `M_majority=.040763`, 95% CI `[.028680,.053252]`. | `MAJORITY-OF-CANDIDATE-BENEFIT SUPPORTED`. | This was not an H6-A gate and does not imply that candidates dominate anchors. |
| E-feature-readable-subset | SAE decoder directions have a heterogeneous readable subset rather than uniformly weak readability. | q-plus is bimodal: 10/24 at or above `.362`, 13/24 below `.15`; 9/10 high group retrieve their own feature at rank 1. | Ask which feature directions are readable, not whether the median feature is weakly readable. | Feature labels have incomplete external validation and were selected in earlier exploratory work. |
| E-label-risk | Internal AV/AR round-trip strength does not certify a faithful feature label. | Surface audit includes 9/24 clear mismatches and extensive unsupported specificity; old pooled correlation was untestable because of synthetic coverage and ceiling artifacts. | NLA/AR may help generate or triage hypotheses but cannot validate SAE labels by itself. | Existing external-label datasets and judges were underpowered or miscalibrated. |
| E-J1-D1 | Correct NLA content can help hypothesis generation relative to mismatched NLA, but NLA-assisted improvement over the stronger SAE-context baseline is not yet stable across labelers. | Mixed-label discovery, 45 features and 1,800 Terra scores: Assisted − SAE micro AP `+.013281`, 95% feature-bootstrap CI `[-.004941,+.035954]`; Assisted − Mismatched `+.069199`, CI `[+.022793,+.125741]`. Assisted − SAE is `+.009202` in Luna–Luna and `−.017494` in Fable–Fable subsets. | `REDESIGN_REPLICATE_BEFORE_CONFIRMATORY`. The result motivates a fixed heterogeneous-labeler replication and capacity-matched baseline; it does not establish NLA-assisted SAE. | Exploratory N3-derived cohort; labeler is confounded with batch/feature, Luna and Terra share a model family, input budgets differ, and no new causal intervention endpoint was run. |
| E-J2-P0 | A naive SAE→AV→AR serial loop improves centered geometry over native SAE but loses original SAE sparse-code identity and worsens downstream causal fidelity. | Reused N4 cohort, 200 positions/101 documents. Loop−native centered cosine `+.109487` small and `+.095760` big. Loop−direct-NLA code cosine `−.029295/−.031677`. Loop−direct-NLA mean KL `+.697680 [.467955,.988165]` small and `+.620861 [.320137,.980321]` big; loop−native KL `+.652083 [.318883,1.043851]` and `+.844832 [.460211,1.295714]`. Independent raw-data recomputation max error `7.86e−08`. | `DO NOT CONFIRM THE SAME SERIAL DESIGN`. AV/AR can act like an activation-manifold prior in geometry while compounding functionally destructive codec error. Continue SAE→NLA only through a redesigned structured/conditional interface with a fresh external causal endpoint. | Exploratory reused cohort; local rescue cases are post-hoc and heterogeneous; no pre-output selector was frozen; this rejects naive serial composition, not all SAE-grounded NLA methods. |

## C. Failed, aborted, or not-testable lines

| ID | Status | Correct interpretation |
|---|---|---|
| F-H5-A | Confirmatory negative | The tested centered-cosine router failed. Do not retune on N5 and relabel it successful. |
| F-N4-H2 | Not confirmatory | Row-wise recovered ratios had near-zero-denominator pathologies; stable sensitivity summaries cannot rescue the preregistered endpoint. |
| F-C1-v3r2 | Protocol abort before model outcomes | The synthetic corpus audit failed its construction gate. This is neither evidence for nor against the scientific C1 hypothesis. |
| F-old-AUC | Not testable | Eight of 24 features did not fire on the tiny test corpus and language AUCs were ceiling-saturated; the pooled rho near zero was measurement failure, not a demonstrated zero relationship. |
| F-steering | Not run | AV/AR carrier ablation, insertion, or amplification without patching through downstream target-model computation is not steering. |
| F-J1-D1-old-interruption | Superseded infrastructure history | The earlier Fable run stopped after 13/45 batches, but those 13 were preserved and the remaining 32 were completed with Luna Max under frozen mixed-label protocols. The old “no result” state is obsolete; use `E-J1-D1` and the 2026-08-06 final report. |

## D. Claims that are currently prohibited

- NLA is globally better than SAE.
- NLA is a better causal codec than SAE-big.
- The N5 hybrid is selectively better or safely equivalent to SAE-big.
- p3 is the only text channel.
- Candidate content is the only or dominant component of p3.
- First-token predictive alignment proves full-continuation or proposition-level truth.
- Reconstruction score proves proposition-level truth or human-faithful explanation.
- C1 failed scientifically.
- Carrier readout is steering.
- Synthetic non-activation proves a dead or invalid feature.
- J1-D1 establishes a confirmatory NLA-assisted-SAE gain, or justifies an
  immediate fresh confirmatory launch.
- J2-P0 establishes SAE-grounded NLA improvement, a safe ordering rule, or a
  validated rescue router.
- A centered-geometry improvement in J2 implies sparse-code grounding or
  downstream causal improvement.

## E. Paper-ready center of gravity

The defensible current paper story is:

> Activation reconstruction through language must be evaluated separately at
> the levels of centered geometry, text-channel content, downstream causal
> fidelity, and tail safety. In Gemma-3-12B-IT L32, the released NLA carries a
> sample-specific, next-token-aligned predictive-state code: recipient-specific
> candidate content has incremental causal utility within a replicated,
> near-sufficient final-paragraph channel. This does not establish
> proposition-level human faithfulness, superiority over SAE-big, or a safe
> selective router.

The paper should pair the N6 confirmatory mechanism result with the N5
confirmatory router failure. This turns a possible contradiction into the main
evaluation lesson: a reconstruction can contain genuine, interpretable
predictive content without providing a calibrated sample-level safety score.

This is the strongest paper story supported by completed confirmatory data, not
the project's final research objective. The original objective remains
bidirectional NLA↔SAE assistance. J1-D1 is now the first complete direct
NLA→SAE discovery pilot. It shows a large penalty for mismatched NLA content
but only a small, labeler-dependent gain over SAE_CONTEXT, so it does not yet
justify preregistering a fresh confirmatory J1. A heterogeneous-labeler
replication plus a capacity-matched strong baseline is required first. J2-P0 is
the first complete SAE→NLA serial-composition audit. Its geometry/code/causal
dissociation is a useful negative mechanism result: naive
`SAE reconstruction→AV→AR` is not the desired mutual-assistance interface.
Future J2 work must expose structured sparse grounding or counterfactual SAE
effects to NLA and be judged on a fresh external causal endpoint.
