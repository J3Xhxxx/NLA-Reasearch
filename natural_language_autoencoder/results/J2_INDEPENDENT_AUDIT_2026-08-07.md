# J2-P0 independent numeric and case audit — 2026-08-07

Status: **PASS**

Scope: read-only post-run audit of the exploratory J2-P0 experiment. This
document verifies numerical reproducibility and frozen case selection; it does
not upgrade J2-P0 to confirmatory evidence.

## 1. Inputs and audit separation

Two independent Luna Max audits were kept separate:

- the numeric auditor did not read case text and recomputed vector geometry
  from `n4_recon_vectors_v1.npz` plus
  `j2_sae_projection_vectors_v1.npz`, fixed-point/code summaries from the 200
  row-level records in `j2_sae_projection_recon_v1.json`, and causal summaries
  from the row-level records in `j2_sae_projection_causal_v1.json`;
- the case auditor did not alter any file and checked the already frozen
  metric-only shortlist against the subsequently rendered case bundle.

The automatic analysis is numerically reproducible. Across rebuilt vector
metric bundles, paired geometry contrasts, causal summaries, and paired causal
contrasts, the largest absolute difference from
`j2_sae_projection_analysis_v1.json` was `7.86e-08`, attributable to float32
storage and aggregation. Identity, zero, and clean-control rechecks differed
by exactly zero.

## 2. Fixed-point recomputation

For `E(x) → D(E(x)) → E(D(E(x)))`:

| SAE | support Jaccard | code cosine | L0 first → second | births / deaths | birth-mass ratio |
|---|---:|---:|---:|---:|---:|
| small | 0.830918 | 0.960296 | 20.005 → 17.005 | 0.200 / 3.200 | 0.005200 |
| big | 0.736028 | 0.942678 | 127.680 → 114.160 | 12.325 / 25.845 | 0.041713 |

The reconstructed activation vectors remain close to their first SAE
reconstruction (`raw cosine = .999267/.997378`), but the sparse codes are not
idempotent. SAE-big has substantially more support churn.

## 3. Sparse-code grounding recomputation

The relevant grounding comparison is whether encoding
`NLA(SAE(x))` returns closer to the original `E(x)` than encoding direct
`NLA(x)`.

| SAE | path | support Jaccard to `E(x)` | code cosine to `E(x)` |
|---|---|---:|---:|
| small | `E(NLA(x))` | 0.547780 | 0.866536 |
| small | `E(NLA(SAE(x)))` | 0.511973 | 0.837241 |
| big | `E(NLA(x))` | 0.455666 | 0.830055 |
| big | `E(NLA(SAE(x)))` | 0.393049 | 0.798378 |

Paired loop-minus-direct deltas were:

- small: support Jaccard `−0.035807`, code cosine `−0.029295`;
- big: support Jaccard `−0.062617`, code cosine `−0.031677`.

Thus the serial SAE-first loop did not ground the NLA reconstruction in the
original SAE code. It moved farther away, especially for SAE-big.

## 4. Geometry and causal recomputation

Centered cosine to the original activation:

| condition | small | big |
|---|---:|---:|
| native SAE | 0.586929 | 0.640781 |
| direct NLA | 0.835669 | 0.835669 |
| SAE → AV → AR loop | 0.696417 | 0.736541 |

The language loop therefore improved centered geometry over native SAE by
`+0.109487` (small) and `+0.095760` (big), but remained below direct NLA by
`−0.139253` and `−0.099128`.

The downstream causal endpoint gives the opposite practical verdict. Lower
KL/CE is better:

| condition | mean KL at patch | KL recovery | KL16 recovery | CE16 recovery |
|---|---:|---:|---:|---:|
| direct NLA | 0.628851 | 0.947946 | 0.945482 | 0.944603 |
| native SAE-small | 0.674448 | 0.944171 | 0.938502 | 0.924367 |
| small loop | 1.326531 | 0.890194 | 0.885883 | 0.881034 |
| SAE-small after direct NLA | 1.314217 | 0.891213 | 0.886090 | 0.869140 |
| native SAE-big | 0.404879 | 0.966485 | 0.962306 | 0.954888 |
| big loop | 1.249711 | 0.896553 | 0.893815 | 0.897795 |
| SAE-big after direct NLA | 1.082186 | 0.910420 | 0.902681 | 0.888709 |

Document-cluster bootstrap 95% intervals for loop-minus-baseline KL at the
patched position were wholly harmful:

- small loop minus direct NLA: `+0.697680 [.467955, .988165]`;
- small loop minus native SAE-small: `+0.652083 [.318883, 1.043851]`;
- big loop minus direct NLA: `+0.620861 [.320137, .980321]`;
- big loop minus native SAE-big: `+0.844832 [.460211, 1.295714]`.

Loop versus reverse-order `SAE(NLA(x))` was unresolved:

- small: `+0.012314 [−.329422, .305184]`;
- big: `+0.167525 [−.099750, .437530]`.

Both serial orders are therefore similarly damaging on average; the data do
not establish a causal ordering advantage.

## 5. Frozen case audit

The shortlist contains 18 predeclared SAE-by-category cells with top 3 rows
each (`54` memberships) and `35` unique positions. The case bundle contains
exactly those 35 positions. Category, rank, selection metric, input hash, and
sidecar checks had zero mismatches.

Recommended post-hoc mechanism cases are:

- `idx75`: both small and big loops beat direct NLA, native SAE, and reverse
  order; the cleanest local synergy candidate;
- `idx168`: small has the larger sparse-code rescue but worse causal fidelity,
  while big causally rescues; a strong operating-point/non-commutativity case;
- `idx185/186`: adjacent positions in the same Apache-license document flip
  the successful SAE width/order, showing extreme position sensitivity;
- `idx34`: plausible Linux-themed text and improved code similarity coexist
  with severe causal catastrophe;
- `idx122`: a fluent date/report-format explanation accompanies a large
  numerical/date identity failure;
- `idx51`: high raw cosine and language/register preservation coexist with
  genre drift and causal damage;
- `idx135`: high native geometry coexists with high SAE-big code churn and a
  worse language loop.

These are frozen-shortlist case studies, not independent replications or
population-level effects.

## 6. Audit verdict

The generated analysis is trustworthy. J2-P0 is a clear exploratory negative
for **naive serial SAE→NLA grounding**:

1. SAE→AV→AR moves native SAE reconstructions toward the original activation
   in centered geometry;
2. it nevertheless loses more of the original SAE sparse code than direct NLA;
3. it substantially worsens downstream causal fidelity relative to both
   direct NLA and native SAE;
4. occasional strong local rescues exist, but are heterogeneous across SAE
   width, document position, and composition order.

Do not launch a fresh confirmatory experiment claiming that this same serial
design improves grounding. Any continued SAE→NLA work should redesign the
interface around conditional or structured sparse grounding, freeze a new
held-out cohort, and retain an external causal endpoint.
