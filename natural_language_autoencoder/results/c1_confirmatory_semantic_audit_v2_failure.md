# C1 confirmatory corpus v2 semantic-audit failure

Date: 2026-07-30 (Asia/Shanghai)

Corpus v2 completed mechanical generation with 144 documents, then stopped at
the pre-activation semantic gate. Two independent Codex reviewers inspected all
24 concepts, 144 documents, and 12 reciprocal pairs under the rubric frozen
before text inspection. Both returned `FAIL`.

Frozen assets:

- Combined manifest:
  `a3d3ffdd18c8b26891031c08dba0cce5b5eea5ec8182ceef0656d8b248106c51`
- Discovery manifest:
  `7b1ce526e8bc897b7f5c22e8ec7d2b0b5e04d70a30c3ca74411f40f8bec4fd2a`
- Held-out manifest:
  `2d83053ac2bfd62722bbaa92e06e1c1408a83e5e032e0cd17ba8878db7f56b8a`
- Frozen rubric:
  `821eb57d3b8c2109acd623137b7f6ba35cb67e74f459bded791ad2eb2d7ec48a`
- Reviewer A:
  `06ca5fa5eec08e6ed10f73e5905bbb3bb1e23dfb441a2c6e1b2ce5c19c84a8d3`
- Reviewer B:
  `78a3708341a5efdc5d620ec1bdc2f9207ebaed3488e7d2f1b6f7b95b5818619f`
- Conservative aggregate audit:
  `255100f62db6e7df1b1a05c8d4451b0213cd17742b2e8e70ab7a4f233355fa35`

The reviewers independently agreed on ten document-level failures. These
included reciprocal hard-negative contamination, one example that did not
instantiate its assigned syntactic concept, and uses of `Earth` that violated
the frozen no-specific-place rule. Both also found repeated discovery/held-out
scenarios and concept-specific template shortcuts in multiple six-document
batches.

No v2 activation Parquet was created. No SAE feature, AV output, AR output,
held-out metric, or endpoint was inspected. The v2 documents are preserved
unchanged; none may be edited, reused, or selectively regenerated. The next
attempt, if run, must be a complete new corpus under the separately frozen v3
amendment.
