# B6+B4 Factorial — Stratified Result

## Protocol and completion

- Directions: 45; jobs: 590 (450 direction + 140 carrier).
- AV generation: 73.8 minutes; AR reconstruction: 25.9 seconds.
- Explanation tag success: 99.2%.
- Greedy generation is the primary result; four temperature-0.7 draws estimate within-feature generation stability.

## Selection transfer

| Stratum | Selected | Heldout-valid | Yield |
|---|---:|---:|---:|
| all | 24 | 14 | 58.3% |
| domain | 15 | 6 | 40.0% |
| language | 9 | 8 | 88.9% |

Primary frozen-cohort intervals:

- Semantic ITT q+ median 0.114 (bootstrap 95% [0.009, 0.453]); polarity median 0.071 ([0.020, 0.245]).
- Heldout-valid q+ median 0.181 ([0.018, 0.446]); polarity median 0.102 ([0.024, 0.260]).
- Fixed generic texts: mean absolute centered cosine 0.034 across all directions and 0.037 for semantic_new.
- Heldout AUC vs greedy q+ Spearman: -0.015.

## Stochastic stability

| Cohort | n | median mean q+ | median q+ SD | median mean polarity | mean sign consistency |
|---|---:|---:|---:|---:|---:|
| semantic_new_itt | 24 | 0.074 | 0.014 | 0.050 | 66.7% |
| semantic_new_heldout_valid | 14 | 0.188 | 0.017 | 0.109 | 78.6% |
| domain_itt | 15 | 0.066 | 0.018 | 0.044 | 68.3% |
| domain_heldout_valid | 6 | 0.408 | 0.031 | 0.221 | 87.5% |
| language_itt | 9 | 0.098 | 0.011 | 0.060 | 63.9% |
| language_heldout_valid | 8 | 0.059 | 0.008 | 0.048 | 71.9% |
| active_nonselective | 8 | 0.014 | 0.020 | 0.020 | 43.8% |
| gaussian | 8 | 0.007 | 0.004 | 0.003 | 31.2% |

- Greedy-vs-stochastic-mean q+ Spearman: ITT 0.943; heldout-valid 0.987; domain-valid 0.943; language-valid 0.976.

## Frozen active-control pairs

- Analyzable pairs: 8/8; median matching distance 6.833.
- Median semantic-minus-control q+ difference: 0.045 (75.0% positive); polarity difference: 0.026.

## Carrier-conditioned AV/AR readout

| Cohort | n | no-op | amplify median cos / norm | ablate median cos / norm | insert median cos / norm |
|---|---:|---:|---:|---:|---:|
| semantic_new_itt | 24 | 9 | 0.000 / 1876 | 0.069 / 2218 | 0.006 / 2222 |
| semantic_new_nonzero | 15 | 0 | 0.042 / 2676 | 0.210 / 3278 | 0.097 / 2317 |
| semantic_new_heldout_valid | 14 | 0 | 0.019 / 2698 | 0.206 / 3342 | 0.113 / 2440 |
| semantic_legacy | 3 | 0 | 0.113 / 3258 | 0.163 / 2839 | 0.056 / 2610 |

## Greedy isolated-direction results

| Cohort | n | q+ median | r− median | polarity median | sign acc. | signed Top-1 | feature Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| semantic_new_itt | 24 | 0.114 | 0.031 | 0.071 | 75.0% | 33.3% | 33.3% |
| semantic_new_heldout_valid | 14 | 0.181 | 0.042 | 0.102 | 85.7% | 39.3% | 39.3% |
| domain_itt | 15 | 0.118 | 0.028 | 0.073 | 80.0% | 30.0% | 30.0% |
| domain_heldout_valid | 6 | 0.407 | 0.039 | 0.224 | 100.0% | 41.7% | 41.7% |
| language_itt | 9 | 0.109 | 0.042 | 0.069 | 66.7% | 38.9% | 38.9% |
| language_heldout_valid | 8 | 0.066 | 0.046 | 0.052 | 75.0% | 37.5% | 37.5% |
| active_nonselective | 8 | -0.004 | 0.025 | 0.015 | 25.0% | 18.8% | 18.8% |
| gaussian | 8 | 0.005 | 0.001 | 0.004 | 37.5% | 0.0% | 0.0% |

## Interpretation limits

- domain positives within a split are translations of one topic, not independent topics
- language strata include script, tokenization, and length effects
- heldout-valid means label-selective, not human-validated monosemantic
- structural control has n=1 and active controls cover only eight approximate pairs
- isolated -w_dec is an OOD signed-axis test, not a semantic antifeature
- carrier tests measure verbalization/reconstruction sensitivity, not downstream model behavior
- same-family AV-to-AR round-trip is internal communication, not external label fidelity
