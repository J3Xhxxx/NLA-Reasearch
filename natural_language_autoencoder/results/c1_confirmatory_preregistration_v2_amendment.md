# C1-confirmatory corpus generation v2 — preregistration amendment

Date: 2026-07-30 (Asia/Shanghai)

This amendment was written after the v1 corpus generator stopped on mechanical
word-count failures and before any new-cohort activation, SAE, AV, AR, held-out
metric, or endpoint was computed. The observed v1 failures are frozen in
`c1_confirmatory_generation_v1_failure.md`.

All scientific choices in `c1_confirmatory_preregistration_v1.md` remain fixed:
the 24 concepts, 12 reciprocal hard-negative pairs, English-only scope, three
reference templates, conservative 1,282-feature denylist resolution, discovery
only selection, strict/relaxed feature gates, stopping gates, centered score,
reciprocal-pair primary estimand, exact pair sign-flip test, secondary analyses,
and held-out embargo.

Only these corpus-generation feasibility parameters change:

- master seed: 20260731;
- attempt seed: `20260731 + 100 * concept_index + attempt`;
- maximum attempts per six-document concept batch: four;
- prompt target: 105–135 English words per request;
- mechanical accepted interval: 70–170 words, inclusive;
- maximum new tokens remains 1,800;
- temperature 0.7, top-p 0.95, top-k 64, repetition penalty 1.0 remain fixed.

The lower mechanical bound remains safely above the layer-32 extractor's
minimum eligible position after chat templating. Document word count and token
count will be reported by concept. A v2 concept still fails after four attempts,
or any post-generation manual audit failure, stops v2; no text is edited and no
document is selectively regenerated.

Because this amendment was motivated by generator compliance, any final paper
must disclose it. The downstream feature/AR result may be described as
confirmatory conditional on the frozen v2 corpus, but not as a wholly untouched
end-to-end preregistration from corpus generation onward.
