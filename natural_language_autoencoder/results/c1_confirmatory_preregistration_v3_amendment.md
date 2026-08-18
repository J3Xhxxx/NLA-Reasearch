# C1 confirmatory corpus v3 amendment

Frozen before any C1 activation extraction, SAE feature selection, AV
generation, AR reconstruction, held-out scoring, or endpoint analysis.

## Why v2 stopped

C1 corpus v2 completed mechanical generation, then underwent the frozen
two-reviewer semantic audit before activation extraction. Both independent
reviewers returned `FAIL`. They agreed on the same ten document-level failures,
including reciprocal hard-negative contamination, one prompt whose example did
not instantiate its assigned syntactic concept, and use of `Earth` under the
frozen no-specific-place rule. They also found repeated discovery/held-out
scenarios and concept-specific template shortcuts in several six-document
batches.

The aggregate v2 audit therefore stops v2. No v2 document is edited, no subset
is regenerated, and no v2 prompt is used for activation extraction.

## Corpus-only v3 revision

V3 is a complete new 144-document corpus. It preserves the 24 concepts, six
superdomains, 12 reciprocal within-superdomain hard-negative pairs, four
discovery plus two held-out documents per concept, reference templates,
denylist, feature-selection gates, estimands, tests, model identities, and
held-out embargo.

Only corpus construction changes:

1. A frozen file supplies six concept-specific scenario anchors per concept:
   four discovery and two held-out. All 144 anchors are fixed before v3 text
   generation.
2. The six slots share a common discourse-role schedule across all concepts,
   while their applications are concept-specific. This addresses style
   imbalance without making the target label recoverable from a unique prompt
   template.
3. Each generation prompt includes the reciprocal hard-negative title and
   scope as an explicit exclusion. A request must centrally require the target
   and must not require the excluded concept as a co-equal goal.
4. Every request must be a single English prose question. Named people,
   organizations, products, specific places (explicitly including Earth),
   URLs, four-digit years, code, formulas, tables, and lists are forbidden.
5. The exact target and hard-negative titles are forbidden in every generated
   request. The anchor is an intent constraint and may not be copied as
   metadata or discussed as an instruction.
6. All 24 concept batches use a new master seed, `20260801`, with attempt seed
   `master + 100 * concept_index + attempt`. Attempts 0–3 are evaluated in
   ascending seed order using only the frozen mechanical validator. The first
   complete six-document batch that passes is retained immediately; later
   attempts are not generated or compared. Semantic or manual ranking among
   mechanically admissible batches is prohibited. If no attempt passes, v3
   stops.
7. Sampling remains `temperature=0.7`, `top_p=0.95`, `top_k=64`,
   `repetition_penalty=1.0`, and `max_new_tokens=1800`. Accepted prompts contain
   80–150 words; the generation target is 95–120 words. Train–test word
   5-gram Jaccard must remain below 0.15.

### Pre-text anchor-design iteration

The first complete v3 anchor draft was frozen with SHA-256
`061903133b748ccffe2f85f697c5e6a7d53fd631e63fb1c0302ae42fbd59e6d5`
and independently audited before any generated v3 text existed. That audit
returned `FAIL` with SHA-256
`5dcc267ad9f6bf9765bbb7c6ab3965498c275c7df3165780f5f1d79301c2a396`,
principally because several held-out anchors recombined discovery mechanisms.
The failed anchor file is preserved unchanged and cannot be supplied to the
generator.

One final full anchor asset, identified as v3r2, may be authored using only the
failed anchor audit, concept specification, and already disclosed v2 corpus
development information. It must retain all 24 concepts and the common six-slot
schedule, bind the failed draft and audit hashes, and undergo a fresh
independent semantic audit before stage-0 freeze. Stage 0 and the generator must
bind only the exact v3r2 hash. If v3r2 fails, anchor iteration and v3 stop in
this run; no generated text is produced.

The human corpus-design process had access to the complete v2 generated corpus,
both detailed v2 reviews, the aggregate failure, the frozen rubric, and the
concept specification. This pre-activation information informed the v3
safeguards and anchors. The generator itself receives only the frozen v3
assets: concept specification, scenario anchors, rubric addendum,
preregistration amendment, and stage-0 freeze. No v2 sentence is retained, and
no v2 document is selectively reused. The anchors were explicitly designed to
avoid the scenarios identified as duplicated or contaminated; generic
scientific settings that are unavoidable for a concept are not claimed to be
statistically independent of v2. V3 is therefore disclosed as adaptive corpus
redevelopment, not as a completely independent corpus-generation attempt.

The six split assignments and discourse roles are fixed identically for every
concept before generation. Four named slots are always discovery and two named
slots are always held out; there is no difficulty-based reassignment after
anchor or text inspection. The held-out embargo concerns downstream decisions:
corpus designers and semantic reviewers necessarily inspect held-out text, but
after the two semantic reviews are independently hash-locked and aggregated,
held-out text cannot be supplied to the extractor, selector, benchmark builder,
feature/reference designer, or any downstream decision process until feature
selection and the candidate benchmark have been frozen.

## V3 audit and stopping rule

After mechanical generation, the base semantic rubric plus a v3 addendum frozen
before generation are applied by two independent reviewers before any
activation extraction. The addendum defines anchor adherence, prohibition on
anchor copying or meta-discussion, exact-title exclusion, the single English
prose-question rule, discourse-schedule adherence, and pair-level
hard-negative/style checks. Each reviewer examines all 144 documents, all 24
six-document batches, and both sides of every reciprocal pair. Each reviewer
must write and hash-lock a complete decision artifact without access to the
other review. A deterministic conservative-conjunction aggregator is run only
after both files are locked; reviewers do not coordinate or revise judgments
toward `PASS`. Both reviews must pass every check. Any failure or disagreement
stops v3; generated text is not edited and no document or concept is
selectively regenerated.

If v3 passes, the downstream C1 analysis remains confirmatory conditional on
this disclosed corpus-development amendment. If v3 fails, no further corpus
revision is made in this run without a new explicit design decision.
