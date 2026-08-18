# B6+B4 `+w_dec` Surface-Semantics Audit

> Date: 2026-07-26  
> Scope: 24 frozen `semantic_new` features, greedy `+w_dec` explanation only.

This is a post-hoc, non-blind, single-reviewer diagnostic. The reviewer could
see the frozen label, top activating contexts, explanation, and direction
scores. It is not an independent human evaluation and has no inter-rater
reliability estimate.

## Rubric and counts

- **Strict match:** the explanation's central topic agrees with both the
  frozen label and the main activating contexts.
- **Coarse/language-only:** only a broad domain, language, or format signal is
  retained; the concrete subject drifts or is unsupported.
- **Obvious mismatch:** the main explanation is in a different domain, follows
  the wrong word sense, or fails to identify a clear non-English direction.
- **Indeterminate:** used for English-language features because AV explains all
  directions in English by default.

| Stratum | Strict | Coarse/language-only | Mismatch | Indeterminate |
|---|---:|---:|---:|---:|
| Domain (`n=15`) | 5 | 5 | 5 | 0 |
| Language (`n=9`) | 0 | 2 | 4 | 3 |
| Total (`n=24`) | **5** | **7** | **9** | **3** |

Strict feature-context agreement was therefore 5/24 (20.8%). Under a relaxed
criterion that also accepts broad-domain or correct-language signals, at most
12/24 (50.0%) retained some expected axis information. Nine of 24 (37.5%)
were obvious surface mismatches.

## Relationship to round-trip score

- Strict-match features had median `q+ ≈ 0.458`.
- Obvious mismatches had median `q+ ≈ 0.008`.
- Among the ten features with `q+ > 0.3`: four were strict, two coarse, one an
  obvious mismatch, and three English-language features were indeterminate.

Thus `q+` contains useful triage information but is not a correctness
certificate.

| Feature | Frozen evidence | `q+` | + feature rank | Surface result |
|---|---|---:|---:|---|
| f14470 | history; printing press / 印刷 | **0.507** | **1** | Minecraft, HP LaserJet, graphics card |
| f15207 | biology; virus / 病毒 | 0.070 | **1** | EternalBlue/WannaCrypt malware |
| f2725 | biology; vaccine / 疫苗 | **0.673** | **1** | Correct vaccine topic, unsupported CDC/mRNA/Pfizer detail |
| f10000 | software; message queue/broker | **0.458** | **1** | Correct broker core, unsupported SFS/AMQP/RabbitMQ detail |

Exact direction retrieval can therefore be correct while the natural-language
surface interpretation is wrong.

## Unsupported specificity

All inputs were isolated decoder directions and contained no current sentence.
Nevertheless, 24/24 explanations asserted a concrete phrase, code fragment, or
scenario. Under a conservative count, at least 22/24 included entities or
technical details not supported by the frozen prompts/top contexts, including
Minecraft, The Last of Us, HP LaserJet, NumPy, `std::sort`, EternalBlue,
WannaCrypt, CDC, and Pfizer.

This should be described as **specificity unsupported by the available feature
evidence**, not necessarily as a false world-knowledge statement.

## Interpretation

AV→AR round-trip and surface quality are coarsely associated, but high
round-trip fidelity does not prove label fidelity. B6+B4 supports using `q+`
as a triage signal inside a C2 labeling pipeline; it does not support
automatically accepting the generated label. C1 correct/wrong/human-label
comparisons, C7 third-party paraphrase tests, and blind multi-rater semantic
evaluation remain necessary.
