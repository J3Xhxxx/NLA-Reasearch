# J1 discovery GPU artifact audit summary

Recorded 2026-08-06 (Asia/Shanghai). Scope is the J1-D1 exploratory GPU
discovery stage and its CPU-only structural/numeric audit. This record makes
no semantic judgment and no confirmatory claim; it embeds metadata and audit
facts only, not AV explanations or generated semantic text.

## Authoritative result

`results/j1_independent_audit_v3.json` is the authoritative audit: **PASS,
26/26 checks, 0 errors**. The audit validates the frozen inputs, vector
schema/alignment and finite values, activation verification, SAE ablation
recomputation, AV plan/checkpoint/result bindings, model bindings, and
sidecars.

The initial unnumbered audit attempt failed serialization before an artifact
was emitted. It is therefore recorded as a pre-artifact event and has no hash
or size.

## Audit history

| Attempt | Status | Sole failure or outcome |
|---|---|---|
| unnumbered | failed before artifact | serialization failed before output |
| v1 | FAIL, 25/26 | audit wrongly required every negative to be a background row |
| v2 | FAIL, 25/26 | audit wrongly rejected tokenizer-decoded whitespace tokens |
| v3 | PASS, 26/26 | no errors; authoritative result |

The v1/v2 failures are audit-logic failures only. They do not change the
frozen cohort or GPU outputs. V3 accepts the prescribed selected-context
negative pool and valid whitespace-decoded tokens.

## Frozen design and counts

- Protocol: `J1_DISCOVERY_PILOT_PROTOCOL_2026-08-06.md`, exploratory/discovery
  only; AV is greedy (`temperature=0`, maximum 200 new tokens), with no AR or
  judge stage in this artifact.
- Freeze: 45 features, 15 per each of 3 strata; 4 discovery and 4 held-out
  positives per feature; discovery and held-out contexts total 180 each.
- Hard negatives: 180; frozen background metadata pool: 690 rows.
- AV plans, append-only checkpoint rows, and result rows: 360 each.
- Vector archive rows: 1,050; SAE decoder matrix shape: `[16384, 3840]`.

## Numeric metrics

- Activation verification: 360 contexts, `atol=1.0`, `rtol=0.025`, maximum
  absolute error `90.22119140625`, maximum relative error
  `0.02016730393599307`, and 0 firing/sign mismatches.
- SAE-ablation recomputation: 360 pairs; maximum ablated absolute and
  relative errors, maximum cosine error, and maximum norm error are all `0.0`.

## Local artifact manifest

The companion JSON contains the local byte size and SHA-256 for the formal
protocol, freeze, vectors, checkpoint, result, full log, audit v1/v2/v3 JSON
and reports, `server/57_j1_discovery_pilot_gpu.py`, and
`server/60_j1_independent_audit.py`.

| Role | Local path | Bytes | SHA-256 (prefix) |
|---|---|---:|---|
| protocol | `results/J1_DISCOVERY_PILOT_PROTOCOL_2026-08-06.md` | 11,197 | `638cb1454c4e…` |
| freeze | `results/j1_discovery_freeze_v1.json` | 626,281 | `8f7690f8b128…` |
| vectors | `results/j1_discovery_vectors_v1.npz` | 90,484,418 | `fa9cc1350863…` |
| checkpoint | `results/j1_discovery_av_checkpoint_v1.jsonl` | 484,949 | `29d184e81b5a…` |
| result | `results/j1_discovery_result_v1.json` | 508,588 | `d93d99e3c84b…` |
| log | `results/j1_discovery_full_v1.log` | 27,801 | `e82d23e1401a…` |
| audit v1/v2/v3 | `results/j1_independent_audit_v{1,2,3}.json` | 1,813,935 / 1,813,881 / 1,813,922 | see JSON manifest |
| script57 | `server/57_j1_discovery_pilot_gpu.py` | 70,067 | `a551287053802…` |
| script60 | `server/60_j1_independent_audit.py` | 59,226 | `75aeaff4a63e…` |

All paths, complete hashes, and the three Markdown audit report hashes are in
`j1_discovery_gpu_manifest_v1.json`. The manifest and this summary are both
local metadata artifacts and intentionally contain no AV text.
