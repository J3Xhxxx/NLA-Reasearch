# N6+ 最终审计与科学结论 — 2026-08-03

> 状态：**正式实验完成；独立审计通过；产物已回传并校验；已发送真实关机命令。**
>
> 本文是 N6+ 的 paper-facing source of truth。数值来源于冻结的原始 JSON/NPZ，
> 而不是从日志或旧结论文件转录。

## 1. 一句话结论

N6+ 在全新 Pile cohort 上确认：NLA 第三段中的候选内容不是仅靠列表格式、
词汇密度或通用候选措辞起作用；**属于当前样本的候选内容同时具有增量因果效用，
并与 clean target model 的 next-token 分布显著对齐。**

冻结的正式标签为：

- `H6-A: SAMPLE-SPECIFIC CHANNEL CONFIRMED`
- `H6-B: PREDICTIVE ALIGNMENT CONFIRMED`
- headline：
  `SAMPLE-SPECIFIC NATURAL-LANGUAGE PREDICTIVE-STATE CODE CONFIRMED`

但候选内容没有压倒 target/context anchors。冻结的 secondary endpoint 反而给出
`NO CANDIDATE DOMINANCE CLAIM`。因此最准确的机制表述是：

> p3 是一个由 anchors、样本特定 continuation candidates 与固定文本结构共同组成的
> predictive-state code；候选内容有不可替代的增量贡献，但不是已证明的唯一或主导成分。

## 2. 冻结设计与 provenance

### 2.1 版本与哈希

- binding preregistration：
  `n6_plus_preregistration_v2.md`
- preregistration SHA-256：
  `cbc8a5395844b5a61a3a52a543f978c89273d6e9c786db0262d4a4c936faf6a8`
- frozen code manifest SHA-256：
  `4f7e7f612e80a73766d95f2dfd34dcdc37d1b0a6c43a82255d0f3ea87970bc0d`
- frozen model manifest SHA-256：
  `4a5c6212dcf5851b9bea2313b4898f977fef0ee36aecc922d10f0375cbc94735`
- main analysis SHA-256：
  `c31d114f9909c2f5371b49730ab60d156a59d238457ada5e48b1a9fa1999789b`
- independent audit SHA-256：
  `d15badffd6b33e986364a5d3d0925cb95e68461dc27b172e44b447046fea6fe0`

v1 启动检查发现本地 `nla_inference.py` 比服务器实际推理代码少两个
Transformers 兼容补丁。该问题在任何 N6 stage 或 outcome 产生前修正；正式 v2
从 2026-08-03 12:16:54 CST 开始，绑定的是下载后逐字节匹配服务器的推理代码。
因此这不是观察结果后的分析修改。

### 2.2 Cohort

- population：全新 Pile content-token positions，仅英文真实语料；
- sources：13 个 Pile sources；
- provisional：501 个互不重叠的 content groups；
- AV-format eligible：434/501，`86.63%`；
- frozen analysis cohort：400 rows / 400 content groups / 400 documents；
- candidate count：4 个候选 105 行、5 个候选 192 行、6 个候选 103 行；
- 每个 source 22–32 行；
- N4/N5 document、content-group 与 20-word shingles 全部 embargo；
- donor 在 `(source, candidate_count)` hard cell 内做 one-to-one derangement，
  且 recipient/donor 的 row、document、content group 不同，无相同规范化候选字符串。

67 个 text-ineligible rows 的冻结拒绝原因：

| 原因 | 行数 |
|---|---:|
| quoted-span 数量不是 6–8 | 44 |
| target-anchor substring gate 失败 | 12 |
| ASCII 双引号不平衡 | 10 |
| 空规范化候选 | 1 |

所以正式外推对象是 **fresh Pile 中能被该固定 AV 格式解析的子总体**，不是所有激活位置。

### 2.3 干预

关键对照保持 recipient 的两个 anchors 和所有引号外 bytes 不变，只替换候选内部：

- `p3_true`：原样本 p3；
- `p3_cross_matched`：换成同 source、同候选数、长度代价匹配的 donor candidates；
- `p3_candidate_strip`：只移除 candidates；
- `p3_anchor_strip`：只移除两个 anchors；
- `p3_all_quote_strip`：移除全部 quoted interiors；
- 另有 `orig`、`p12`、`sae_big`、`identity` 和 `zero`。

所有非零重建逐行 norm-match 到 clean activation。主因果端点是 patched position 的
`KL(clean || patched)`，聚合时只使用 ratio of sums，禁止逐行除以 `KL_zero`。

## 3. 预注册主结果

| 端点 | Point | 95% interval / gate | 裁决 |
|---|---:|---:|---|
| `G_specific` | **+0.117954** | `[+0.102860,+0.133995]` | pass |
| raw `KL_cross-KL_true` | **+1.827883 nat/row** | `[+1.591758,+2.074946]` | descriptive support |
| `G_content` | **+0.154382** | `[+0.136602,+0.173507]` | pass |
| `T_p3=R_p3/R_orig` | **0.995175** | one-sided 95% lower **0.993194** | pass |
| `A_meanmass` | **+9.008470 log units** | `[+8.444855,+9.572597]` | pass |

因此 H6-A 的三个 gate 和 H6-B 的一个 gate 全部通过。

`A_meanmass` 的指数约为 `8,172×`，表示 true candidate set 相对 matched-cross set
的“每个唯一 first-token candidate 的平均 clean probability”的几何平均比值。
它不是整条候选字符串的概率比，也不是 proposition-level truth 分数。

## 4. Secondary endpoints 与 recovery 分解

### 4.1 候选的样本特异性占了显著多数

`M_majority=+0.040763`，95% CI
`[+0.028680,+0.053252]`，所以冻结标签是
`MAJORITY-OF-CANDIDATE-BENEFIT SUPPORTED`。

这表示 `G_specific` 显著大于 `0.5 * G_content`。它支持“候选收益的大部分与
recipient-specific identity 有关”，但它不是 H6-A 的必要 gate。

### 4.2 候选不支配 anchors

`G_candidate_anchor=-0.045042`，95% CI
`[-0.078669,-0.012779]`，所以不能写 candidate dominance。

对应 recovery：

| Reconstruction | Aggregate recovery |
|---|---:|
| `identity` | 1.000000 |
| `orig` | 0.979493 |
| `sae_big` | 0.978140 |
| `p3_true` | 0.974768 |
| `p3_cross_matched` | 0.856814 |
| `p3_candidate_strip` | 0.820385 |
| `p12` | 0.795530 |
| `p3_anchor_strip` | 0.775343 |
| `p3_all_quote_strip` | 0.367074 |
| `zero` | 0.000000 |

只去 anchors 比只去 candidates 的损伤更大；同时全去 quoted content 的损伤远大于
两个单独 ablation。最合理的解释是 anchors 与候选之间存在互补或非线性交互，
而不是把 recovery 机械分摊成可加的“信息百分比”。

`orig`、`p3_true` 与 `sae_big` 的点估计很接近，但 N6 没有预注册三者的 superiority
或 equivalence gate。它们不能被写成 NLA 优于、等价于或安全替代 SAE-big。

## 5. 稳健性与诊断

### 5.1 独立复算

独立 stage 56：

- 重新读取 frozen plan、variants、reconstruction JSON/NPZ 与 causal JSON；
- 重新生成 `PCG64(20260803)` 的 50,000 个 shared bootstrap draws；
- bootstrap index bytes SHA-256：
  `478a7789dfcac82b7b2c3663a60da82147e6014942fd29291e4c0a8a3688297e`；
- 56 个 endpoint numeric leaves 全部在 `1e-12` 内；
- 唯一非零差异是 `M_majority` CI 下界的 `3.47e-18`（1 ULP）；
- 五项正式 label 逐字一致；
- `all_checks_pass=true`、`formal_decisions_exact=true`。

Numerical QA：

- identity patched-position KL 与 KL16 最大绝对值均为 0；
- `sum(KL_zero)=6198.638175 > 0`；
- 一项 `p3_true KL=-2.20e-9` 属冻结容差内浮点 roundoff，已记录并钳为 0；
- 无被丢弃的 outcome cell。

### 5.2 Source 与 candidate-count 稳定性

这些是描述性检查，不替代主 bootstrap：

- 13/13 source 的 `G_specific`、`G_content`、`A_meanmass` 点估计均为正；
- leave-one-source-out：
  - `G_specific` 位于 `0.1123–0.1250`；
  - `G_content` 位于 `0.1453–0.1649`；
  - `A_meanmass` 位于 `8.797–9.295`；
- candidate count 4、5、6 三个子组的三个效应也都为正。

因此结果不是由单一 source 或候选数 cell 驱动。

### 5.3 可直接理解的 predictive diagnostics

| 指标 | True candidates | Cross-matched candidates |
|---|---:|---:|
| hit@1 | 66.50% | 3.75% |
| hit@5 | 84.00% | 11.75% |
| hit@10 | 87.00% | 16.50% |
| hit@50 | 95.75% | 35.75% |
| observed next token 在 set 中 | 49.00% | 2.75% |

True/cross 候选 Unicode-word Jaccard overlap 的均值仅 `0.0546`，说明 donor
替换不是同义候选的小扰动。

Tail diagnostics：

- `KL_cross-KL_true > 1 nat`：210/400；
- `KL_true-KL_sae_big > 1 nat`：18/400。

前者说明 sample-specific effect 不只是少数异常值；后者提醒 `p3_true` 仍存在
相对 SAE-big 的失败尾部，不能把 N6 变成安全 routing claim。

### 5.4 Geometry 仍不是 causal safety score

Centered cosine 均值：

| Reconstruction | Centered cosine |
|---|---:|
| `orig` | 0.846 |
| `p3_true` | 0.784 |
| `sae_big` | 0.664 |
| `p3_candidate_strip` | 0.450 |
| `p3_anchor_strip` | 0.406 |
| `p3_cross_matched` | 0.392 |
| `p3_all_quote_strip` | 0.044 |
| `p12` | 0.630 |

方向与机制结果大体一致，但 N5 已经正式否定单变量 centered-cosine router。
这些几何数值只能作为机制诊断，不能重新包装成逐样本安全证书。

## 6. N1–N6 后的统一结论

| 层面 | 当前最稳结论 | 仍不能声称 |
|---|---|---|
| 几何 | 去均值后 NLA 比两只冻结 SAE 保存更多 sample-specific activation geometry | 几何优势等于因果优势 |
| 因果 codec | NLA reconstruction 有很高 aggregate causal recovery；p3 near-sufficient | NLA 全面优于或等价于 SAE-big |
| 文本机制 | p3 中 recipient-specific candidate content 有独立因果效用 | p3 只有 candidates；candidates 支配 anchors |
| predictive alignment | true candidate first-token sets 与 clean next-token distribution 强对齐 | 完整候选序列正确；命题级忠实 |
| safety/router | SAE-big 是重要的 tail comparator；冻结 centered-cosine router 失败 | 已有可部署的 selective hybrid |
| SAE 辅助解释 | NLA/AR 可生成和分诊 SAE feature hypotheses | round-trip 分数证明 feature label 正确 |
| steering | 尚未执行真正的 intervention-to-behavior steering | carrier readout 或 causal patch 等于 steering |

项目中心问题已经从“NLA 是否全面胜过 SAE”收窄为：

> 自然语言 reconstruction 的几何可解码性、文本内容、下游因果保真度和尾部安全性
> 必须分开评估。对 Gemma-3-12B-IT L32，NLA 的关键文本通道是一个
> sample-specific、next-token-aligned predictive-state code；它具有强因果效用，
> 但尚不是经验证的人类忠实解释、全局更优 codec 或安全 router。

## 7. 论文判断

N6 使项目比 N5 后**明显更乐观**。此前最强正结果只是“candidate-bearing p3
通道存在”；现在 matched substitution 排除了“只偏好候选列表格式/通用词汇”的主要
替代解释，并由独立 predictive-alignment 端点交叉确认。

一篇机制/评测论文现在有三个互相咬合的贡献：

1. 将 activation reconstruction 拆成 geometry、text content、causal fidelity 与
   tail safety 四层评估；
2. 用预注册 matched-content intervention 识别自然语言 code 的样本特定因果内容；
3. 同时报告冻结 router 的 confirmatory negative，证明高 reconstruction geometry
   不能自动转化为安全选择器。

这已经达到“认真写顶会稿”的证据门槛，但不是稳收。最大审稿风险依次是：

1. single model / single layer；
2. population 限于 AV-format-eligible fresh Pile；
3. paired AV/AR 仍可能利用其训练形成的语言协议；
4. 没有人类 proposition-level fidelity 验证；
5. 没有 rate/capacity-matched NLA–SAE 比较。

## 8. 下一步顺序

### 8.1 立即执行，CPU-only

1. 冻结 N1–N6 claim table，开始写 Methods、N4–N6 Results 与 Limitations；
2. 从现有 JSON 生成论文表格和四张主图：
   recovery decomposition、true-vs-cross paired effect、source robustness、
   geometry-vs-causal mismatch；
3. 做 N4/N5/N6 failure-tail forensic audit，但只作为 discovery；
4. 整理 artifact/hash appendix。

### 8.2 下一项最值得的新 GPU 实验

若存在兼容的第二层 NLA AV/AR checkpoint，优先做一份全新预注册的
**cross-layer N6 replication**。鉴于当前 effect size 很大，可先做约 200 个独立
content groups 的 power calculation 与 protocol audit，再决定最终样本量。

若没有第二层 checkpoint，则优先级改为：

1. 第二模型/第二 NLA checkpoint replication；
2. candidate-semantic robustness：在不改变候选命题的前提下测试表面改写，
   区分公开自然语言语义与 paired AV/AR 私有协议；
3. capacity-matched post-training rate–distortion operating curve。

不建议下一步立刻重跑 centered-cosine router、C1 synthetic corpus、steering 或
另一份同层同模型 400-row N6。它们对当前论文最主要的外推缺口信息增益较低。

## 9. 产物、资源与关机

正式 v2：

- supervisor wall time：`1:54:51`；
- stage 49–56：`1:49:48`；
- AV generation：`3505.95 s`；
- causal stage：`4400 forwards`，`393.25 s`；
- pipeline exit：0；
- 结束快照：A800-SXM4-80GB，`0 MiB / 0%`；
- 磁盘：70/150 GB；内存：45 GiB / 1 TiB。

本地 verified pull：

- 目录：`results/n6_pull_staging/n6_pull_20260803T061302Z/`
- 51 files，45,583,606 bytes；
- 15 个 SHA-256 sidecars 全部通过；
- stage 49–56 与 supervisor exit markers 全为 0。

Stage-50 activation parquet 与其 metadata 位于远端 `activations/`，不在冻结
`n6_*/N6_*` pull allowlist 中；它们是可重用上游 cache，不是 stage-55/56
独立审计的必需输入。数据盘持久化，今后应在不启用 GPU 的模式下补拉，且不需要重跑。

本地 launcher 在 12:59 遇到一次 SSH reset 后退出；root 在 pull-ready 后接管，
完成本地哈希校验再写 exact ACK。supervisor 随后执行：

```bash
sync; /usr/bin/shutdown -h now
```

两次后续 SSH 探测均返回 exit 255 / connection closed，符合实例已关机。
由于 `198.18.x` 是 Clash TUN fake-IP，最终控制面状态仍以 AutoDL 控制台为准；
不要为了确认而重新开机。

## 10. 禁止的表述

- 不说“NLA 生成了忠实的人类语义解释”；
- 不说“p3 或 candidates 是唯一通道”；
- 不说“candidates 比 anchors 更重要”；
- 不说“NLA 优于、等价于或可安全替代 SAE-big”；
- 不说“centered cosine 可以安全 routing”；
- 不说“candidate first-token alignment 等于完整 continuation 正确”；
- 不外推到 Gemma-3-12B-IT L32、fresh Pile AV-format-eligible subset 之外；
- 不把 reconstruction、carrier ablation 或 causal patch 称为 steering。
