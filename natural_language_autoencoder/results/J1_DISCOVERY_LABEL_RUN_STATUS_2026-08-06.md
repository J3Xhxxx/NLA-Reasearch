# J1-D1 label-run interruption record

> **SUPERSEDED HISTORICAL RECORD.** 本文件只记录最初 13 个 batches 后的中断，
> 不再代表当前状态。本轮没有调用 Claude/Fable：原有 `0..12` 被保留，Luna Max
> 已完成 `13..44`，Terra 1,800-score 盲评与统计分析也已完成。当前权威裁决见
> `J1_MIXED_DISCOVERY_FINAL_2026-08-06.md`：
> `REDESIGN_REPLICATE_BEFORE_CONFIRMATORY`。

Status: **PAUSED — EXTERNAL CLAUDE BALANCE BLOCKER**  
Recorded: 2026-08-06 (Asia/Shanghai)  
Scientific scope: **exploratory / discovery only**

## Frozen inputs

- GPU freeze SHA-256:
  `8f7690f8b12842b32ce5cb32af7ee941b2ce2f71fcc0270768cf0f84edcb50d3`
- AV result SHA-256:
  `d93d99e3c84b07a6f76b3b4549bb16fbe520f9c7aa59579f98e57bb2a85749a4`
- Parent protocol SHA-256:
  `638cb1454c4e248cd56e1148f88bc91b0840797ff7ad787f1ef095bd920777cf`
- Downstream addendum SHA-256:
  `75fe0caf64a2e598cd7509ca13ab83c8fc09bcfe43f28c5107e6d5b9097e0da3`
- Label runner SHA-256:
  `132697447b3169dabab373fec2d0647f396424e611d080515f6ae85118ce37f9`
- Frozen jobs SHA-256:
  `411c67acd230018c60d50194d51c70f3cde847c7f85b38713456450b609f4aad`

The frozen schedule contains 45 cross-feature batches and 225 opaque cases.
Every batch contains five different features and exactly one case from each
arm; each feature's five arms occur in five different Fable calls.

## Interruption facts

The run produced 24 append-only checkpoint rows:

- 13 successful batches: `0..12`;
- 11 retained failed attempts: `13..21`, `23`, and `24`;
- batch `22` and batches `25..44` had no completed attempt when the process was
  interrupted;
- no final label-result artifact was emitted;
- no scientific arm comparison or partial analysis was performed.

Starting with batch 13, Claude returned HTTP `403` with the explicit message
`insufficient balance`. This is an external account-balance failure, not a
parser, label-quality, model-substitution, or experimental endpoint failure.
The process was deliberately interrupted to avoid issuing the remaining
requests against an exhausted balance.

Checkpoint snapshot at interruption:

- path: `results/j1_discovery_labels_checkpoint_v1.jsonl`;
- bytes: `280796`;
- SHA-256:
  `439f8af0c438e8a79a13af5bb6dd463932ca1d56e225912da3dba9edd8cd12e4`.

The 13 successful rows report `$3.5432775`. One failed 403 envelope additionally
reports `$0.07027`; the retained raw envelopes therefore report approximately
`$3.6135475` total attempted cost. The 32 unfinished batches are estimated at
`$8.72` from the successful-call mean; adding 25% headroom gives an advised
minimum balance of about `$11`.

## Safe resume

After the Claude balance is replenished, run exactly:

```powershell
python D:\Projects\natural_language_autoencoder\server\58_j1_discovery_labels.py
```

Do not delete or edit the jobs or checkpoint. The runner reparses and
hash-verifies all successful rows, retains failed history, and calls only
batches without a valid success. A final result is withheld until all 45
batches and all 225 cases succeed.

After the complete label artifact exists, freeze the Terra evaluation job with:

```powershell
python D:\Projects\natural_language_autoencoder\server\59_j1_discovery_evaluate.py --dry-run
```

Audit that job before removing `--dry-run`. Incomplete 13/45 labels must never
be passed to the evaluator or used for scientific analysis.

## Infrastructure state

The A800 GPU stage completed, passed the authoritative v3 independent audit,
and all formal artifacts were pulled locally. The remote host received:

```text
sync; /usr/bin/shutdown -h now
```

SSH closed immediately and a subsequent five-second connection attempt timed
out. This strongly indicates the instance stopped, but only the AutoDL control
plane can prove that billing has ended; do not restart the instance merely to
check.
