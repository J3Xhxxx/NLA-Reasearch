# Structure-mask AR probe protocol (idx 75, v1)

Status before outcome: **FROZEN EXPLORATORY SINGLE-CASE PROTOCOL**

## Fixed sample

- Frozen N4/J2 row: `idx=75`
- Source: `OpenSubtitles`
- `doc_id=5849`, `position=276`, token=` assume`
- Selection reason: readable real context and a clearly structured three-paragraph
  AV explanation. The sample is selected without observing the new masked-AR result.

## Intervention

Use the exact frozen `direct_n4` AV explanation. Scan Unicode characters from left
to right, toggling quote state only at the ASCII double quote (`"`). Every
alphabetic Unicode character (`str.isalpha()`) **outside** double-quoted spans is
replaced by one ASCII `X`. Characters inside quotes, whitespace, punctuation,
digits, paragraph boundaries, and total character length are preserved exactly.

This operationalizes “structural language” as the unquoted AV scaffolding while
retaining quoted lexical evidence/candidates. It preserves character count, not
token count; AR tokenizer counts must therefore be reported for both variants.

## Endpoints

Both original and masked explanations are reconstructed by the same frozen
Gemma-3-12B-IT L32 AR checkpoint and scored against the same frozen real
activation `x[75]`:

1. official direction cosine;
2. official direction MSE, `2 * (1 - cosine)`;
3. centered cosine using the frozen N4 mean direction;
4. original-vs-masked AR reconstruction cosine;
5. explanation and full AR-prompt token counts.

## Scope

This is a deliberately chosen single mechanism case. It can demonstrate a local
effect and motivate a cohort experiment, but cannot estimate a population effect.
Masking with repeated `X` also changes tokenization and creates OOD text, so any
effect combines removal of scaffold semantics with that perturbation.
