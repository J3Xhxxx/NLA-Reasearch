# J1 Luna nested-execution engineering amendment

Status: **FROZEN BEFORE BATCHES 27..44**
Freeze date: 2026-08-06 (Asia/Shanghai)

This amendment changes only the canonical task paths used to launch the
remaining fresh Luna workers. It does not change any public prompt, case,
label schema, batch assignment, labeler/model, reasoning effort, arm,
feature, evaluator input, endpoint, or decision rule.

## Cause

After batches `13..26` were complete, the collaboration runtime continued to
retain three completed root-level child threads and refused new root-level
children with `agent thread limit reached`. Reusing one of those completed
workers to label another batch would violate the frozen one-batch-per-worker
isolation rule.

## Isolation-preserving remedy

Completed root-level workers may act only as orchestration parents. An
orchestration parent:

- must not open the new batch prompt, sidecar, output, or labels;
- spawns a new `luna_worker` with `fork_turns="none"`;
- gives that child exactly one public batch and one output path;
- does not label, edit, summarize, or inspect the child's cases;
- records only completion and artifact hashes.

The new child remains a fresh Luna Max context and handles exactly one batch.

## Frozen canonical task map

The public index's root-level `expected_agent_task` is superseded only for the
following batches:

| Batch | Canonical fresh worker task |
|---:|---|
| 27 | `/root/luna_batch_18/luna_batch_27` |
| 28 | `/root/luna_batch_18/luna_batch_28` |
| 29 | `/root/luna_batch_24/luna_batch_29` |
| 30 | `/root/luna_batch_24/luna_batch_30` |
| 31 | `/root/luna_batch_26/luna_batch_31` |
| 32 | `/root/luna_batch_26/luna_batch_32` |
| 33 | `/root/luna_batch_18/luna_batch_33` |
| 34 | `/root/luna_batch_18/luna_batch_34` |
| 35 | `/root/luna_batch_24/luna_batch_35` |
| 36 | `/root/luna_batch_24/luna_batch_36` |
| 37 | `/root/luna_batch_26/luna_batch_37` |
| 38 | `/root/luna_batch_26/luna_batch_38` |
| 39 | `/root/luna_batch_18/luna_batch_39` |
| 40 | `/root/luna_batch_18/luna_batch_40` |
| 41 | `/root/luna_batch_24/luna_batch_41` |
| 42 | `/root/luna_batch_24/luna_batch_42` |
| 43 | `/root/luna_batch_26/luna_batch_43` |
| 44 | `/root/luna_batch_26/luna_batch_44` |

Batches `13..26` retain their already frozen root-level task paths.

## Merger requirements

The mixed-label merger must verify the exact canonical path above for every
batch `27..44`, bind this amendment and its SHA-256, and retain the actual
agent path per hypothesis. A suffix-only or arbitrary-agent match is not
allowed.

This is an engineering/provenance amendment, not permission to inspect partial
outcomes or alter scientific labels.
