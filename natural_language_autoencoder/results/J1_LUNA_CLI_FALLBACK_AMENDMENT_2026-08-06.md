# J1 Luna CLI fallback engineering amendment

Status: **FROZEN BEFORE BATCHES 27..44**
Freeze date: 2026-08-06 (Asia/Shanghai)

This amendment supersedes the unused nested-task mapping in
`J1_LUNA_NESTED_EXECUTION_AMENDMENT_2026-08-06.md`. The nested spawn attempt
failed with `agent thread limit reached` before any batch `27..44` prompt was
read or any corresponding label was produced.

The scientific protocol, public prompts, case assignments, model family,
reasoning effort, output schema, and downstream endpoints are unchanged.

## 1. Worker transport

Batches `27..44` are completed by one fresh local Codex CLI process per batch:

- model: `gpt-5.6-luna`;
- reasoning effort: `max`;
- Codex CLI: `0.146.1`;
- `--ephemeral` with no resume/session persistence;
- `--ignore-user-config`;
- empty temporary working directory;
- read-only sandbox;
- no web, repository, connector, or external tool use;
- one public batch serialized into stdin;
- a frozen strict JSON output schema;
- raw CLI event/output artifacts retained and hashed.

The CLI process is an isolated Luna Max execution worker. It receives no
conversation history and no other batch. Any observed tool call, malformed
schema, wrong/missing case ID, extra case, empty or over-32-word hypothesis,
or incomplete process result fails closed.

## 2. Public-data boundary

The runner may read the authoritative public index and exactly one public
batch artifact. The model input contains only:

- opaque batch and case IDs;
- the already frozen public evidence;
- the already frozen public labeling instruction.

It must not contain the private condition map, feature ID, arm, truth,
activation, held-out context, hard negative, or any prior label output.

## 3. Frozen worker identifiers

The exact recorded `agent_task` for batch `XX` is:

`codex-cli://j1/luna_batch_XX`

for every integer `XX` from `27` through `44`, using two decimal digits.
The merger must check this exact mapping, not a suffix or arbitrary string.

Batches `13..26` retain their completed collaboration-worker canonical task
paths. Batches `0..12` remain the unrepeated Fable successes.

## 4. Integrity

For each batch the runner records:

- public prompt artifact SHA-256;
- embedded prompt-text SHA-256;
- model, reasoning effort, CLI version, sandbox, and ephemeral flag;
- exact invocation-contract SHA-256;
- raw structured-output SHA-256;
- exact final output artifact SHA-256 and sidecar;
- process exit status and absence of tool calls.

The mixed-label merger must bind this amendment and its SHA-256. This
transport amendment is exploratory provenance only and cannot create a
confirmatory claim.
