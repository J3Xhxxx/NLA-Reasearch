# J1-D1 mixed-label discovery：最终结果与启动裁决

> 日期：2026-08-06（Asia/Shanghai）  
> 状态：`EXPLORATORY_DISCOVERY_COMPLETE`  
> 科学范围：discovery only；不得据此宣称 NLA-assisted SAE 已被确认。

## 1. 本轮实际完成了什么

本轮保留了原有 Fable batches `0..12`，没有重跑、删除或改写它们；随后使用
Luna Max 完成 batches `13..44`：

- batches `13..26`：由 fresh `luna_worker` 子代理逐批完成；
- batches `27..44`：协作运行时因已完成线程仍占用槽位，按事前冻结的 fallback
  amendment，改为每批启动一个 fresh、ephemeral 的
  `gpt-5.6-luna` / `max` Codex CLI 进程；
- Luna 共完成 32 batches、160 hypotheses，18 个 CLI batches 全部首轮成功，
  无工具调用、无 parser retry；
- batch 20 仅修正了 `public_prompt_sha256` 的字符抄写错误，五条标签内容未变；
- 合并后的最终标签集包含 45 batches、225 hypotheses：每个 arm 恰好
  13 个 Fable 标签和 32 个 Luna 标签。

随后使用冻结的 `gpt-5.6-terra` 盲评：

- 45 个 feature；
- 每个 feature 5 个互盲 hypothesis arms；
- 每个 hypothesis 评估 4 个 held-out positives 与 4 个 exact-zero hard
  negatives；
- 共 `45 × 5 × 8 = 1,800` 个 score；
- 45/45 evaluator calls 成功，1,800/1,800 scores 完整，0 failures。

公开 evaluator prompt 不含 arm、labeler、truth 或私有 feature 元数据。结构审计确认
225 个 feature–arm group 均为 8 行，且均严格包含 4 positive / 4 negative。

## 2. 设计审计

45 个 cross-feature batches 均含五个不同 feature，且五种 arm 各出现一次。  
标签器分配在 arm 层面精确平衡，但不是随机化变量：

- 9 个 feature 的五臂全部由 Fable 标注；
- 28 个 feature 的五臂全部由 Luna 标注；
- 8 个 feature 在五臂间混合标签器：
  `2096, 2700, 3176, 3441, 15742, 15793, 16016, 16059`。

因此可以报告完整 ITT 和严格 same-labeler paired subsets，但不能把
Fable–Luna 差异解释为纯模型效应：labeler 与 batch/order/feature 仍有混杂。

## 3. 完整 ITT 结果

| Arm | micro AP | macro AP | pairwise accuracy | Brier（低为好） | coverage |
|---|---:|---:|---:|---:|---:|
| `SAE_CONTEXT` | 0.905301 | 0.907844 | 0.888889 | 0.145098 | 0.977778 |
| `NLA_ASSISTED` | **0.918582** | **0.922474** | **0.906250** | **0.132426** | 0.977778 |
| `NLA_CONTRASTIVE` | 0.907093 | 0.903056 | 0.886111 | 0.142441 | 0.977778 |
| `NLA_MISMATCHED` | 0.849382 | 0.839299 | 0.800000 | 0.193581 | 0.977778 |
| `NLA_ONLY` | 0.890284 | 0.906310 | 0.881944 | 0.161125 | 0.977778 |

`NLA_ASSISTED` 是完整 ITT 中表现最好的 arm，但与真正关键的
`SAE_CONTEXT` 差距很小。`NLA_MISMATCHED` 明显更差，说明错误 NLA 内容会伤害
feature hypothesis；它是 harmful-content control，不是强 baseline。

## 4. 预先指定的四个 contrast

以下区间为 20,000 次 feature-cluster percentile bootstrap；属于探索性区间，
不是预注册显著性 gate。

| Contrast | ITT Δmicro AP [95% CI] | Luna–Luna Δ | Fable–Fable Δ |
|---|---:|---:|---:|
| Assisted − SAE | +0.013281 `[-0.004941, +0.035954]` | +0.009202 | **−0.017494** |
| Assisted − Mismatched | +0.069199 `[+0.022793, +0.125741]` | +0.103941 | +0.004645 |
| Contrastive − SAE | +0.001793 `[-0.026851, +0.028256]` | +0.005923 | **−0.014761** |
| Contrastive − Mismatched | +0.057711 `[+0.013979, +0.111425]` | +0.093552 | +0.007192 |

解释：

1. 相对 `SAE_CONTEXT`，Assisted 的 ITT 增益只有 `+0.0133`，区间跨零；
   Contrastive 几乎为零。
2. 相对 `NLA_MISMATCHED` 的大幅正差说明“内容正确性重要”，但不证明 NLA
   辅助优于 SAE-only。
3. 两个候选联合臂相对 SAE 的方向均在 Luna–Luna 与 Fable–Fable 子集中翻转。
4. Luna 生成标签、Terra 评分都属于 OpenAI/Codex 模型家族；因此当前的
   Luna-only 正方向不能排除 family-specific communication。

按 strata 看，Assisted 相对 SAE 与 Mismatched 在
`source_concentrated`、`source_distributed`、`language_selective`
三个 strata 的点估计都为正；Contrastive 只在其中两个 strata 同时有利。
这支持继续研究 Assisted，但不能消除标签器依赖。

## 5. 额外风险

- `SAE_CONTEXT` 并非 capacity/token matched：平均输入约 684 bytes，而
  Assisted 约 3,441 bytes、Contrastive 约 6,188 bytes。
- discovery 复用了 N3 cohort；不是 fresh confirmatory cohort。
- 只有 Gemma-3-12B-IT、L32、一个 SAE family 和 AV-format-eligible feature
  子总体。
- Terra 是异于 target model 的 evaluator，但不是异于 Luna 的模型家族。
- 当前端点是 held-out activation discrimination；尚无新的外部 causal
  intervention endpoint。

## 6. 正式裁决

> **`REDESIGN_REPLICATE_BEFORE_CONFIRMATORY`**

当前版本**不值得立即启动昂贵的 fresh confirmatory J1**。这不是说 J1 方向失败：

- 正确 NLA 内容相对 mismatched 内容的优势大而稳定；
- Assisted 在完整 ITT 和三个 strata 中方向均有利；
- 没有明显 coverage、calibration 或 tail collapse。

但真正需要建立的 `NLA_ASSISTED > SAE_CONTEXT` 目前效应小、区间跨零，并且
在 Fable–Fable subset 反向。立即在 fresh cohort 上放大样本，只会把尚未解决的
测量设计不确定性一起放大。

## 7. 启动 fresh confirmatory J1 前的必做步骤

1. **跨家族重标同一 discovery benchmark。** 用一个固定的非 OpenAI
   interpreter（或人类标签）对五个 arms 全量重标，不能只补某些 feature/arm；
   Terra 仍保持全盲评分。
2. **加入 capacity-matched 强 baseline。** 至少让 SAE-only baseline 获得与
   Assisted 相同的输入预算、同数量的 activation contexts/hard negatives，并加入
   一个强 autointerp baseline，避免把上下文长度优势误当成 NLA 优势。
3. **预先冻结 go/no-go gate。** 至少要求 Assisted − SAE 在异构标签器下仍为正、
   不发生主要 strata 反向，且 calibration/tail 不恶化；这些 gate 必须在看到新标签
   outcome 前冻结。
4. 只有通过上述 discovery replication，才设计 fresh、source/shingle-embargoed
   confirmatory cohort，并加入独立的 causal intervention endpoint。

## 8. 权威 artifacts

- mixed labels：
  `j1_discovery_labels_mixed_result_v3.json`  
  SHA-256 `2ca779f8ffb89d93531fef31beb12a5d81b0185d18d7d02e6450c296ce562b8b`
- frozen Terra job：
  `j1_blinded_eval_job_mixed_v2.json`  
  SHA-256 `9fd8628a46155a98e0670a79a947cadf69014dddb4aecce02ad0cf281eb599cb`
- raw Terra result：
  `j1_blinded_eval_result_mixed_v2.json`  
  SHA-256 `893e59583d69f25979fc4c2324e47b55ffe829c891a0c9d555cc21f74c51b9b8`
- full statistical analysis：
  `j1_blinded_eval_analysis_mixed_v2.json`  
  SHA-256 `455cfabf06a611ed183af17c9fd84eb0c504c071e93bcea2c357bbeffae891cb`
- paper-facing summary：
  `J1_BLINDED_EVAL_ANALYSIS_MIXED_v2.md`  
  SHA-256 `03463dc5c78f42bf640446b080478f9291673a139e855a1b46ff7bd339600740`
- analysis script：
  `server/65_j1_analyze_mixed_eval.py`  
  SHA-256 `fa90ca50de104225635731ff4f695b683572c4725e12fd4cbe835939b15cc185`

分析脚本以相同 seed 重跑后，权威 analysis JSON 的 SHA-256 仍为
`455cfabf...891cb`，证明统计输出具有确定性。

Luna Max 另行只读重算了全部 arm 指标、四个预设 contrast 与结构计数，最大绝对
差仅 `1.1102230246251565e-16`（浮点舍入），裁决完全一致。见
`J1_MIXED_INDEPENDENT_AUDIT_2026-08-06.md`，SHA-256
`f98dddf53a9dfb486f86566bc0243d85d6e5bd7f30178f9399120ae990a31da2`。

## 9. 服务器状态

本轮标签、Terra 评分与统计分析均为本地/API 工作，不需要 GPU。此前 J1 AV 的 GPU
阶段已完成；已向 AutoDL 发送 `sync; /usr/bin/shutdown -h now`，随后 SSH 超时。
不要为复查而重启实例；是否停止计费仍应以 AutoDL 控制台状态为准。
