# C1-confirmatory corpus generation v1 — frozen failure record

Status: stopped exactly as preregistered; no activation, SAE, AV, AR, or endpoint
was computed.

The first concept, `automatic_memory_reclamation`, exhausted its three fixed
generation attempts:

- attempt 0: one request had 77 words;
- attempt 1: two requests had 74 and 72 words;
- attempt 2: one request had 78 words.

All failures were solely below the v1 mechanical minimum of 80 words. The raw
generated text was retained in the append-only checkpoint, but no semantic
result was used to alter the concept list, hard-negative mapping, denylist,
feature selector, candidate templates, or statistical analysis.

The failed job ran for about 122 seconds after monitoring began, peaked at
24,251 MiB GPU memory and 171 W, and consumed approximately 4.93 Wh by
trapezoidal integration of the two-second GPU samples.

Frozen evidence:

- `c1_confirmatory_corpus_checkpoint_v1.failed.jsonl`: 28,852 bytes,
  SHA256 `94737dccd2b03014dc38af9117cc2929c28eeafa81aecadecae8e162af218f9b`
- `c1_confirmatory_corpus_v1.failed.log`: 5,144 bytes,
  SHA256 `574d1357950cd5aa24399f94d3491581c8a37a766ce586dcd92c1d3c9d2e3c72`
- `c1_confirmatory_corpus_gpu_v1.failed.csv`: 2,965 bytes,
  SHA256 `850f7008df3aab0bfb0299b9ca240aa910d56aadc53d605c373d368d683718d9`

The v2 amendment changes only the corpus generator's master seed, retry count,
target wording, and mechanically accepted word interval. It is explicitly an
adaptation to a generation-feasibility failure, not an untouched preregistration.
The confirmatory analysis begins only after a v2 corpus passes mechanical and
manual audits and a new pre-generation freeze is recorded.
