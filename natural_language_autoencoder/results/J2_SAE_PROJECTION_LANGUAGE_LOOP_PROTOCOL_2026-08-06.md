# J2-P0 protocol：SAE projection → NLA language loop

> Freeze date: 2026-08-06（在任何新 `AV(SAE(x))` 输出之前）  
> Status: **FROZEN EXPLORATORY PROTOCOL**  
> Claim scope: discovery / mechanism audit only

## 1. 原始问题

项目的双向辅助主线包含 `SAE→NLA`：SAE 的稀疏投影能否为 NLA 提供 grounding，
使自然语言通道更稳定、更可审计？

本实验首次执行完整串联：

```text
真实 residual x
  → SAE encode/decode: s = D(E(x))
  → AV(s): SAE-projected natural-language explanation
  → AR(AV(s)): c
  → activation geometry、SAE-code fixed point、下游 causal KL、case study
```

历史上已做但不等价的实验：

- `x` 分别进入 NLA 与 SAE，再比较两种 reconstruction；
- 单个 SAE decoder direction `w_dec[j]` 进入 AV→AR；
- SAE residual `x − SAE(x)` 进入 AV→AR；
- feature-ablated real activation 进入 AV，作为 J1 contrastive hypothesis。

它们都没有测试 `AV(D(E(x)))`，也没有测
`E(x)` 与 `E(D(E(x)))` 的 SAE fixed-point 差异。

## 2. 冻结 cohort 与已有输入

复用 N4 的 200 个真实 content-token positions、101 篇文档。复用是为了让新串联与
已经冻结的 direct-NLA、SAE-native、causal-patch 逐行配对；因此本实验不能成为
fresh confirmatory evidence。

绑定输入：

- `results/acts_L32_n3_v1.parquet`  
  SHA-256 `eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66`
- `results/n4_recon_vectors_v1.npz`  
  SHA-256 `e9d89713dc64381a52f05224d6522abb0ec547777a8c6a7f08b841a72a339967`
- `results/n4_explanations_v1.json`  
  SHA-256 `b656ded845c8fd122e4dcb1391ba5d81e1a903f80a69c30575bf26910e200942`
- `results/n4_causal_patch_v1.json`  
  SHA-256 `8dd532f65d8c9c153f04ba433cc6f160798598fbbcbee388c15fb4a75a366233`
- `results/n4_analysis_v1.json`  
  SHA-256 `3c8a4d87d7289ac6c41b58e2bbdd6955585db46eaaa5306822d9d802259943cc`
- `results/n5_model_weights_v1.sha256`（同一 AV/AR、SAE 与 base model 的
  full-file manifest）  
  SHA-256 `4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735`
- `server/pilot_common.py`  
  SHA-256 `69fb1b40d60d075c615acdaa23acf4f85c17b5b4cf02e2cc18113c4e14ecf63a`

模型与层保持 N4 不变：

- Gemma-3-12B-IT，L32 `resid_post`；
- NLA AV/AR：`nla-gemma3-12b-L32-av/ar`；
- Gemma Scope 2 12B-IT L32 width-16k：
  `l0_small` 与 `l0_big`。

## 3. 条件

已有：

1. `x`：真实 activation；
2. `nla_direct = AR(AV(x))`；
3. `sae_small = D_small(E_small(x))`；
4. `sae_big = D_big(E_big(x))`。

新增：

5. `small_loop = AR(AV(sae_small))`；
6. `big_loop = AR(AV(sae_big))`；
7. `sae_small_2 = D_small(E_small(sae_small))`；
8. `sae_big_2 = D_big(E_big(sae_big))`。
9. `E_small(small_loop)` 与 `E_big(big_loop)`：测量语言 round-trip 是否保留
   原 SAE sparse code；
10. `direct_small = D_small(E_small(nla_direct))` 与
    `direct_big = D_big(E_big(nla_direct))`：把 direct NLA reconstruction
    重新投影到同一 SAE，形成顺序相反的 `SAE(NLA(x))` comparator。

small/big 两个 operating points 必须同时运行，避免看到某一 SAE 后再挑选稀疏度。
所有 AV generation 使用 greedy decoding、`temperature=0`、
`max_new_tokens=200`，与 N4 direct AV 一致。

## 4. 问题与 estimands

### Q1：SAE manifold 是否更容易经语言 round-trip？

对每个 SAE operating point 报告：

- `cos_c(loop, sae)` 与 leave-one-document-out centering；
- `cos_c(loop, x)`：完整串联对真实 activation 的保真度；
- `cos_c(sae, x)`：native SAE 上界；
- `cos_c(nla_direct, x)`：已有 direct-language comparator；
- fixed-scale centered FVE 与 norm ratio；
- 以下逐行 paired deltas：
  - `cos_c(loop, x) − cos_c(nla_direct, x)`；
  - `cos_c(loop, x) − cos_c(sae, x)`；
  - `cos_c(loop, sae) − cos_c(nla_direct, x)`。

最后一个 delta 的 target 不同，只能回答“SAE manifold 是否更容易被同一 NLA
language bottleneck 编码”，不能写成端到端优越性。

### Q2：SAE 是否接近 encode/decode fixed point？

分别比较 `a=E(x)`、`a'=E(D(E(x)))`、
`a_direct=E(AR(AV(x)))` 与
`a_loop=E(AR(AV(D(E(x)))))`：

- active-support Jaccard、precision、recall；
- weighted code cosine；
- L0 change、activation-mass ratio；
- births、deaths；
- `cos(sae_2, sae)` 与 `cos(sae_2, x)`；
- 每行 top-20 contribution features 在第一次与第二次编码中的保留率。
- 另报 `a_loop` 相对 `a` 和 `a'` 的 support Jaccard、weighted cosine、
  L0 与 top-20 保留率；这是 SAE-grounding 的直接 code endpoint。
- 以 `a_direct` 为顺序对照，报告
  `sim(a_loop,a) − sim(a_direct,a)`，并直接比较
  `a_loop` 与 `a_direct`。这检验先做 SAE grounding 是否比
  direct NLA 后再 SAE 投影更保留原始 sparse code；它仍是探索性 estimand。

纯 `w_dec` 方向属于 off-manifold probe，不混入本节总体结论。

### Q3：串联在下游因果端点上保留什么？

在 N4 同一 token/layer 做 patch，新增 `small_loop`、`big_loop`、
`direct_small` 与 `direct_big`，逐行配对已有：

- `nla_direct`
- `sae_small`
- `sae_big`
- identity / zero controls

报告 raw `KL(clean || patched)`、KL16、CE16，以及稳定的 aggregate
ratio-of-sums。CE16 的 loss recovery 定义为
`1 − Σ(CE_condition−CE_clean) / Σ(CE_zero−CE_clean)`；KL 两项相应以
zero-control KL 的总和为分母。不得使用病态 row-wise recovered ratio
作为主结论。新旧运行的 identity、zero 与 clean-CE 另作逐行复现门禁：
所有对应 raw metric 的总体最大绝对差必须 `≤ 1e-6`。

### Q4：AV 文本发生了什么变化？

逐行比较 `AV(x)` 与 `AV(sae_small/big)`：

- normalized token-set Jaccard；
- sequence similarity；
- quoted-span count 与 quoted-token overlap；
- explanation length；
- `AR` 空间中的 direct/loop 相似度。

这些是文本变化诊断，不是人类命题忠实度评分。

## 5. 预先冻结的 case-study 选择

在查看新文本前冻结以下类别，每类按定义取前三名，允许重叠：

1. **high native fidelity / high code churn**：`cos_c(sae,x)` 位于上 50%，
   support Jaccard 最低；
2. **language-loop rescue**：`KL_loop − KL_nla_direct` 最低；
3. **language-loop catastrophe**：`KL_loop − KL_sae_native` 最高；
4. **geometrically tiny / textually large change**：
   `cos_raw(sae,x)` 位于上 50%，文本 Jaccard 最低；
5. **worst SAE-manifold round-trip**：`cos_c(loop,sae)` 最低；
6. **fixed-point leakage**：`E(D(E(x)))` 相对 `E(x)` 的 birth activation
   mass ratio 最高；
7. **language-code leakage**：`E(loop)` 相对 `E(x)` 的 birth activation
   mass ratio 最高；
8. **SAE-grounding code rescue**：
   `cos_code(E(loop),E(x)) − cos_code(E(nla_direct),E(x))` 最高；
9. **SAE-grounding code catastrophe**：上述差值最低。

冻结 shortlist JSON 后才允许人工阅读并撰写 case study。Case study 是机制假设生成，
不得代替总体统计。

对 shortlist 中涉及的 top active features，可在第二阶段额外生成至多 24 个
`AV(b_dec + a_j w_dec[j])` 单 feature/contribution probes，明确标为 post hoc；
这些 off-manifold probes不能回写总体效应。

## 6. 不允许的表述

- SAE projection 已改善 NLA；
- 串联 codec 优于 direct NLA 或 native SAE；
- `E(D(E(x)))` 的 feature births 一定是真实新概念；
- AV 文本变化证明 feature splitting、superposition 或人类忠实解释；
- 从本 reused cohort 宣称 confirmatory SAE-grounded NLA。

允许的表述是：本实验测量串联组合的几何、code fixed-point、文本和下游因果行为，
并生成可供 fresh J2 设计检验的机制假设。

## 7. 执行门禁与成本

执行顺序：

1. 冻结本 protocol、脚本与 run manifest 的 SHA-256；
2. 验证所有 N4 输入 hash；
3. 400 次 AV（200 small + 200 big）；
4. 冻结新 explanations；
5. 400 次 AR；
6. 计算两个 SAE 的 fixed-point；
7. 对两个 loop 与两个 `SAE(nla_direct)` 顺序对照做 causal patch；
8. CPU-only 完整分析并冻结 case shortlist；
9. 才可查看 shortlist 文本。

SAE 权重与输入必须通过冻结 hash。重新计算的一次 SAE reconstruction 与 N4
冻结向量还必须满足 `max_abs ≤ 1e-5` 且逐行最大 relative-L2 `≤ 1e-6`；
第二次 encode 的输入始终使用 N4 冻结 reconstruction，而不是本次重算副本。

预计 A800-80GB：

- AV/AR + SAE fixed-point：约 20–35 分钟；
- 四个新增 causal conditions：约 8–15 分钟；
- 含模型加载与传输的总 GPU 窗口：约 40–65 分钟；
- 新增结果落盘预计低于 100 MiB。

任何阶段结束后都不得让 GPU 空转。完成结果同步并校验后发送
`sync; /usr/bin/shutdown -h now`；AutoDL 控制台仍是计费状态的唯一独立确认。
