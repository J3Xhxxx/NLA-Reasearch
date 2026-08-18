# J2-P0 final analysis — naive serial SAE→NLA grounding

Date: 2026-08-07  
Status: **EXPLORATORY MECHANISM AUDIT COMPLETE**  
Decision: **DO NOT CONFIRM THE SAME SERIAL DESIGN**

## Executive verdict

J2-P0 is the first completed project experiment to run the full path

`real activation x → SAE reconstruction → AV → AR`.

It does not establish that SAE projection improves NLA. The result is more
specific and scientifically useful:

> Passing an SAE reconstruction through the NLA language bottleneck behaves
> like an activation-manifold prior: it moves the vector closer to the
> original activation in centered geometry. But this apparent denoising loses
> original SAE sparse-code identity and compounds downstream causal error.

The average result is negative for naive serial composition, with sharply
heterogeneous local rescue and catastrophe cases. Geometry, sparse-code
similarity, fluent AV text, and downstream causal fidelity are distinct
measurements; none can substitute for the causal endpoint.

J2-P0 reuses the N4 200-position/101-document cohort, so every conclusion in
this report is exploratory. It can motivate a new design but cannot be called
confirmatory.

## 1. Frozen design and provenance

Before any new J2 AV output, the protocol froze four paths:

1. direct NLA: `NLA(x) = AR(AV(x))`;
2. native SAE: `SAE(x) = D(E(x))`;
3. SAE-first language loop: `NLA(SAE(x))`;
4. reverse order: `SAE(NLA(x))`.

Both the frozen SAE-small and SAE-big checkpoints were evaluated. Endpoints
covered centered activation geometry, SAE fixed-point and sparse-code
retention, AV text change, KL at the patched position, mean KL over the first
16 continuations, and first-16-token cross entropy.

The formal run completed:

- AV: `400/400`;
- causal cohort: `200/200` positions from `101/101` documents;
- causal forwards: `1301`;
- frozen shortlist: `18` SAE-by-category cells, `35` unique positions;
- identity/zero/clean control difference: exactly `0`;
- all 18 pulled artifact sidecars matched.

A pre-outcome dry run caught one transcription error in a hard-coded upstream
SHA string. No J2 output existed at that point. The string was corrected,
protocol and manifest were re-frozen, and the correction is recorded in
`J2_PREFLIGHT_HASH_CORRECTION_2026-08-07.md`.

## 2. The central dissociation

### 2.1 The language loop improves geometry over native SAE

Centered cosine to `x`:

| condition | SAE-small | SAE-big |
|---|---:|---:|
| native SAE | 0.586929 | 0.640781 |
| SAE→AV→AR loop | 0.696417 | 0.736541 |
| direct NLA | 0.835669 | 0.835669 |

Document-bootstrap paired gains of the loop over native SAE were:

- small: `+0.109487`, 95% CI `[+.094272,+.124555]`;
- big: `+0.095760`, 95% CI `[+.080511,+.110902]`.

Yet the loop remained worse than direct NLA:

- small: `−0.139253 [−.153628,−.125607]`;
- big: `−0.099128 [−.111160,−.087318]`.

Raw cosines were all about `.989–.995`, reinforcing the established warning
that the common residual-stream direction masks sample-specific differences.

### 2.2 The language loop does not preserve the original SAE code

Relative to direct `NLA(x)` re-encoded through the same SAE, the SAE-first loop
was farther from the original `E(x)`:

| SAE | loop − direct support Jaccard | loop − direct code cosine | loop − direct top-20 overlap |
|---|---:|---:|---:|
| small | −0.035807 | −0.029295 | −0.720 |
| big | −0.062617 | −0.031677 | −0.735 |

All corresponding document-bootstrap intervals excluded zero. SAE-small
reduced new-feature mass slightly, but did so while losing more original
support; SAE-big worsened both retention and new-feature mass.

Native SAE is itself not a perfect sparse-code fixed point:

- small: support Jaccard `.830918`, code cosine `.960296`,
  L0 `20.005→17.005`;
- big: support Jaccard `.736028`, code cosine `.942678`,
  L0 `127.680→114.160`.

The vectors remain nearly identical in raw cosine, while the active sparse
support changes. This is a relevant warning for any method that treats a
single SAE encode/decode pass as an idempotent semantic projection.

### 2.3 The geometry gain is causally harmful on average

| condition | mean KL at patch | aggregate KL recovery |
|---|---:|---:|
| direct NLA | 0.628851 | 0.947946 |
| native SAE-small | 0.674448 | 0.944171 |
| small loop | 1.326531 | 0.890194 |
| SAE-small after direct NLA | 1.314217 | 0.891213 |
| native SAE-big | 0.404879 | 0.966485 |
| big loop | 1.249711 | 0.896553 |
| SAE-big after direct NLA | 1.082186 | 0.910420 |

Loop-minus-baseline mean KL, with document-bootstrap 95% intervals:

- small versus direct NLA: `+0.697680 [.467955,.988165]`;
- small versus native SAE: `+0.652083 [.318883,1.043851]`;
- big versus direct NLA: `+0.620861 [.320137,.980321]`;
- big versus native SAE: `+0.844832 [.460211,1.295714]`.

KL16 and CE16 give the same harmful direction with intervals above zero.
Neither loop has a resolved advantage over reverse-order `SAE(NLA(x))`.

The dissociation occurs within rows, not only in aggregate:

- geometry improves over native SAE in `168/200` small and `169/200` big rows;
- geometry improves while causal KL worsens in `95/200` small and `109/200`
  big rows;
- Spearman correlation between geometry gain and causal change is
  `−.311` small and `−.353` big.

### 2.4 AV text is highly unstable under small activation changes

AV text overlap between `AV(SAE(x))` and `AV(x)` was low despite raw vector
cosines near `.99`:

| SAE | token Jaccard | sequence similarity | quoted-token Jaccard |
|---|---:|---:|---:|
| small | 0.3001 | 0.3319 | 0.1938 |
| big | 0.3330 | 0.3525 | 0.2390 |

Lengths and quote counts were similar, so this is not merely a response-length
artifact. SAE projection can preserve broad format or register while changing
sample-specific propositions and continuation identity.

## 3. Case-study findings

All cases below come from the metric-only shortlist frozen before human text
reading. They remain post-hoc mechanism hypotheses.

- **idx75, OpenSubtitles — local synergy.** Both SAE widths lower KL relative
  to direct NLA, native SAE, and reverse order. The text paths retain a
  “Do not assume” warning despite broader discourse drift. This is the best
  candidate for studying when sparse projection stabilizes a local NLA state.
- **idx168, Turkish XNLI — operating-point reversal.** SAE-small produces the
  larger code-similarity rescue but worsens causal fidelity; SAE-big improves
  causal fidelity. Sparse-code rescue alone is not a calibrated causal score.
- **idx185/186, adjacent Apache-license positions — position brittleness.**
  Within the same document, the successful SAE width flips: big helps at
  `idx185` while small nearly perfectly rescues `idx186`; the other width is
  catastrophic. These positions are a paired illustration, not independent
  replications.
- **idx34, Ubuntu IRC — plausible but causally wrong.** Linux/dbus-themed AV
  text and code similarity coexist with KL around `21–24` for the loops.
- **idx122, Enron — numeric identity corruption.** A fluent schedule/report
  explanation changes date/number identity, while the big loop alone becomes
  catastrophic (`KL=7.832`).
- **idx130, HackerNews — causal fidelity without proposition fidelity.**
  The big loop lowers KL to `.059`, although its AV explanation is
  propositionally dubious. A good patch need not be a human-faithful account.

The local cases show that a more selective interface may be possible, but they
do not identify a pre-output router and cannot support a population claim.

## 4. Scientific decision

### What J2-P0 supports

- The complete `x→SAE→AV→AR` path had not been tested previously; it now has.
- AV/AR can pull an SAE reconstruction toward the empirical activation
  manifold in centered geometry.
- Naive composition is not additive: serial codec errors destroy sparse-code
  identity and causal fidelity.
- SAE width, composition order, and token position produce real mechanism
  heterogeneity.
- The project must evaluate NLA×SAE assistance with an external causal or
  behavioral endpoint, not geometry, code similarity, or fluent text alone.

### What it does not support

- SAE-grounded NLA is not established.
- The serial loop is not better than direct NLA or native SAE.
- Neither serial order is established as safer.
- The selected rescue cases do not justify a post-hoc router claim.
- J2-P0 is not confirmatory because it reuses the N4 cohort.

### Go/no-go

**No-go** for a fresh confirmation of the same
`SAE reconstruction → free-form AV → AR` design.

The next SAE→NLA experiment should first be redesigned on CPU. The most
promising interface is not another serial reconstruction, but conditional
structured grounding: expose sparse feature identities/effects or
counterfactual SAE interventions to NLA, require NLA to make a falsifiable
prediction, and score it on a fresh held-out causal intervention. Before
spending GPU, use J2 only as discovery data to ask whether local rescue can be
predicted from pre-output variables such as SAE fixed-point stability,
activation norm, code churn, source/language, candidate identity, and
NLA–SAE disagreement. Any rule must then be frozen and tested on new data.

## 5. Authoritative artifacts

- frozen protocol:
  `J2_SAE_PROJECTION_LANGUAGE_LOOP_PROTOCOL_2026-08-06.md`;
- frozen run manifest:
  `J2_SAE_PROJECTION_LANGUAGE_LOOP_RUN_MANIFEST_2026-08-06.json`;
- raw vectors/reconstructions:
  `j2_sae_projection_vectors_v1.npz`,
  `j2_sae_projection_recon_v1.json`;
- raw causal rows:
  `j2_sae_projection_causal_v1.json`;
- generated analysis:
  `j2_sae_projection_analysis_v1.json` and
  `J2_SAE_PROJECTION_ANALYSIS_V1.md`;
- frozen case selection and rendered bundle:
  `j2_sae_projection_case_shortlist_v1.json`,
  `j2_sae_projection_case_bundle_v1.json`, and
  `J2_SAE_PROJECTION_CASE_BUNDLE_V1.md`;
- independent audit:
  `J2_INDEPENDENT_AUDIT_2026-08-07.md`.

All listed run artifacts have matching local SHA-256 sidecars.
