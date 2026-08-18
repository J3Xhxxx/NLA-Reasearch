# J1-D1 mixed-label Terra analysis：独立只读审计

> 日期：2026-08-06  
> 执行者：Luna Max 子代理  
> 输入：仅
> `j1_blinded_eval_result_mixed_v2.json` 与
> `j1_blinded_eval_analysis_mixed_v2.json`  
> 约束：未修改 artifact，未调用 evaluator，未复用分析脚本的指标实现。

## 结论

**PASS。** 独立重算的 arm 指标、四个预设 contrast、样本计数和最终裁决均与
权威 analysis JSON 一致。最大绝对差为
`1.1102230246251565e-16`，仅为浮点舍入误差。

## 结构

- 45 features × 5 arms × 8 contexts = 1,800 scores；
- 225 个 feature–arm groups，每组 4 positives + 4 exact-zero negatives；
- 每个 arm 360 scores；
- 每个 stratum 15 features、每个 arm–stratum 120 scores；
- Fable 520 scores、Luna 1,280 scores；
- 40 个 abstentions 全部来自 feature `10075`；
- 全臂 coverage 均为 `352/360 = 0.9777777778`。

## 独立重算的 arm 指标

| Arm | micro AP | macro AP | pairwise | Brier | coverage |
|---|---:|---:|---:|---:|---:|
| `NLA_ASSISTED` | 0.9185816061 | 0.9224735450 | 0.9062500000 | 0.1324255556 | 0.9777777778 |
| `NLA_CONTRASTIVE` | 0.9070932174 | 0.9030555556 | 0.8861111111 | 0.1424413889 | 0.9777777778 |
| `NLA_MISMATCHED` | 0.8493824450 | 0.8392989418 | 0.8000000000 | 0.1935811111 | 0.9777777778 |
| `NLA_ONLY` | 0.8902836083 | 0.9063095238 | 0.8819444444 | 0.1611252778 | 0.9777777778 |
| `SAE_CONTEXT` | 0.9053005147 | 0.9078439153 | 0.8888888889 | 0.1450983333 | 0.9777777778 |

## 裁决核对

完整 ITT 与 Luna–Luna 中，两个候选臂相对 SAE 和 Mismatched 的 micro-AP
方向均为正；但 Fable–Fable 中：

- `NLA_ASSISTED − SAE_CONTEXT = −0.0174938071`
- `NLA_CONTRASTIVE − SAE_CONTEXT = −0.0147605595`

因此 analysis 中的 labeler-dependence flag、
`immediate_fresh_confirmatory_launch=false` 和
`REDESIGN_REPLICATE_BEFORE_CONFIRMATORY` 均由原始评分支持。

独立审计结论同样是：**不支持立即启动 fresh confirmatory J1**。先做固定的异构
non-OpenAI interpreter / 人类全量重标，并加入 capacity-matched 强 baseline。
