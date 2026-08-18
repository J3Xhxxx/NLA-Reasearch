# NLA × SAE 项目交接与恢复指南

> 更新时间：2026-08-07（J2-P0 完成、独立审计通过、AutoDL 已执行关机）  
> 当前科学状态：N5/N6 确认性结论不变；首个直接测试 `NLA→SAE` 辅助的
> J1-D1 discovery 已完成 225 个 hypotheses、Terra 1,800/1,800 盲评分和完整统计。
> 正式裁决为 `REDESIGN_REPLICATE_BEFORE_CONFIRMATORY`：不得立即启动或声称
> fresh confirmatory J1。先读
> `results/J1_MIXED_DISCOVERY_FINAL_2026-08-06.md` 与
> `results/j1_blinded_eval_analysis_mixed_v2.json`。  
> 当前执行状态：首个完整
> `x → SAE(x) → AV → AR` 四路径探索性审计 J2-P0 已完成。400/400 AV、
> 200/200 causal positions、101/101 documents 均完成；18 个拉回 artifact
> sidecar 全部匹配。两名 Luna Max 分别从 raw rows/NPZ 独立重算数值和审查预冻
> case，自动分析最大绝对误差仅 `7.86e−08`，case membership 零 mismatch。
> 权威裁决是 `DO NOT CONFIRM THE SAME SERIAL DESIGN`：language loop 相对
> native SAE 改善 centered geometry，却损失原始 sparse code，并显著恶化
> downstream KL。详见 `results/J2_FINAL_ANALYSIS_2026-08-07.md` 与
> `results/J2_INDEPENDENT_AUDIT_2026-08-07.md`。运行结束后 supervisor 已执行
> `sync; /usr/bin/shutdown -h now`，SSH 随即关闭；控制台是计费停止的最终证明，
> 不要为复查而重启。

## 最新执行覆盖：J2-P0（exploratory complete）

J2-P0 是项目首次直接测试 `SAE→NLA grounding`，但复用 N4 的 200 行、
101 文档真实 cohort，因此永远只能是 mechanism discovery，不能写成
confirmatory evidence。四路径为：

1. direct NLA：`NLA(x)=AR(AV(x))`；
2. native SAE：`SAE(x)=D(E(x))`；
3. SAE-first loop：`NLA(SAE(x))`；
4. reverse order：`SAE(NLA(x))`。

### 最终 J2 裁决

- loop 相对 native SAE 的 centered cosine 提高
  `+.109487`（small）和 `+.095760`（big），但仍比 direct NLA 低
  `−.139253/−.099128`；
- loop 相对 direct NLA re-encoding 的原始 SAE code cosine 下降
  `−.029295/−.031677`，support Jaccard 下降 `−.035807/−.062617`；
- causal KL 显著恶化：
  - small loop − direct NLA `+.697680 [.467955,.988165]`；
  - small loop − native SAE `+.652083 [.318883,1.043851]`；
  - big loop − direct NLA `+.620861 [.320137,.980321]`；
  - big loop − native SAE `+.844832 [.460211,1.295714]`；
- loop 与 reverse-order `SAE(NLA(x))` 的 causal 差异均跨零，因此没有建立
  顺序优势；
- AV 文本 token Jaccard 仅 `.300/.333`，尽管对应 activation raw cosine
  约 `.99`；几何、sparse code、流畅文本和 causal fidelity 必须分开评估；
- 预冻 shortlist 中存在真实局部 rescue（如 `idx75`）以及同一文档相邻位置、
  SAE width 和顺序发生方向翻转（`idx168`、`idx185/186`），但不能据此宣称
  population improvement 或事后 router。

结论不是“SAE 与 NLA 不能互助”，而是**朴素串联不是可靠的互助接口**。
`SAE reconstruction → free-form AV → AR` 在几何上像 activation-manifold prior，
在功能上却累积错误。不得对同一串联设计直接启动 fresh confirmatory J2。
若继续 SAE→NLA，应先 CPU-only 设计 structured/conditional grounding：让 SAE
feature identity、counterfactual intervention 或约束候选成为可检验输入，并在全新
held-out cohort 上以外部 causal endpoint 裁决。

当前 authoritative J2 chain：

1. `results/J2_SAE_PROJECTION_LANGUAGE_LOOP_PROTOCOL_2026-08-06.md`
2. `results/J2_SAE_PROJECTION_LANGUAGE_LOOP_RUN_MANIFEST_2026-08-06.json`
3. `results/j2_sae_projection_vectors_v1.npz`
4. `results/j2_sae_projection_recon_v1.json`
5. `results/j2_sae_projection_causal_v1.json`
6. `results/j2_sae_projection_analysis_v1.json`
7. `results/j2_sae_projection_case_shortlist_v1.json`
8. `results/j2_sae_projection_case_bundle_v1.json`
9. `results/J2_FINAL_ANALYSIS_2026-08-07.md`
10. `results/J2_INDEPENDENT_AUDIT_2026-08-07.md`

## 最新状态覆盖：J1-D1（mixed-label discovery complete）

遇到与后文 N7 优先级或旧状态冲突时，以本节及其冻结文件为准。

### 最终 discovery 裁决

原有 Fable batches `0..12` 被原样保留；Luna Max 完成 batches `13..44`，没有
重复前 13 个 batches。最终每个 arm 都有 13 个 Fable 与 32 个 Luna hypotheses。
Terra 随后完成 45 features × 5 arms × 8 contexts = 1,800 个盲评分，0 failures。

完整 ITT 的 micro AP 为：

- `SAE_CONTEXT 0.905301`
- `NLA_ASSISTED 0.918582`
- `NLA_CONTRASTIVE 0.907093`
- `NLA_MISMATCHED 0.849382`
- `NLA_ONLY 0.890284`

`NLA_ASSISTED − SAE_CONTEXT = +0.013281`，feature-cluster bootstrap 95% CI
`[-0.004941, +0.035954]`；该方向在 Luna–Luna 为 `+0.009202`，在
Fable–Fable 为 `−0.017494`。Contrastive 相对 SAE 也出现同类 sign flip。
因此正确 NLA 内容相对 mismatched 内容确实有价值，但尚未建立对强
SAE_CONTEXT baseline 的跨标签器稳定增益。Luna 与 Terra 同属 OpenAI/Codex
家族，且 SAE baseline 未做 capacity matching；本轮只支持重新设计与异构标签器
复制，不支持立即 fresh confirmatory。

新的 authoritative J1 discovery chain：

1. `results/J1_DISCOVERY_LUNA_COMPLETION_PROTOCOL_2026-08-06.md`
2. `results/J1_DISCOVERY_MIXED_ANALYSIS_PLAN_2026-08-06.md`
3. `results/J1_LUNA_CLI_FALLBACK_AMENDMENT_2026-08-06.md`
4. `results/j1_discovery_labels_mixed_result_v3.json`
5. `results/j1_blinded_eval_job_mixed_v2.json`
6. `results/j1_blinded_eval_result_mixed_v2.json`
7. `results/j1_blinded_eval_analysis_mixed_v2.json`
8. `results/J1_BLINDED_EVAL_ANALYSIS_MIXED_v2.md`
9. `results/J1_MIXED_DISCOVERY_FINAL_2026-08-06.md`
10. `results/J1_MIXED_INDEPENDENT_AUDIT_2026-08-06.md`

下一步不是新增 GPU batch，而是用固定的非 OpenAI interpreter（或人类）全量重标
同一五臂 benchmark，并加入 capacity-matched SAE-only 与强 autointerp baseline。
只有跨家族下 `NLA_ASSISTED > SAE_CONTEXT` 仍稳定，才冻结 fresh、
source/shingle-embargoed、带 causal intervention endpoint 的确认性 J1。

### 以下为本轮完成前的历史执行记录

本轮选择 J1 而非先跑 N7，因为项目原始中心问题是 NLA 与 SAE
能否相互辅助。J1-D1 直接测试：真实高激活 residual 的 NLA 文本，是否能在新文档
held-out SAE activation discrimination 上改善 SAE feature hypothesis。

它严格是 **exploratory / discovery only**：

- 复用 N3 的冻结 120-feature 候选集，从三个 strata 各固定抽取 15 个，共 45；
- 每个 feature 使用 4 个 discovery contexts、4 个不同文档的 held-out positives 和
  4 个 exact-zero hard negatives；
- GPU 生成 180 个 on-manifold raw NLA snippets 和 180 个
  SAE-feature-ablated contrastive snippets；
- 不使用 AR round-trip、centered cosine 或同源 Gemma judge 作为 truth；
- SAE 实际 activation 是二元 truth，异构 evaluator 只作为测量仪器；
- 即使结果很强，也只能决定是否值得另起 fresh、capacity-matched、带 causal endpoint
  的确认性 J1，不能直接宣称 NLA-assisted SAE 已建立。

当前 authoritative J1 GPU chain：

1. `results/J1_DISCOVERY_PILOT_PROTOCOL_2026-08-06.md`
2. `results/j1_discovery_freeze_v1.json`
3. `results/j1_discovery_vectors_v1.npz`
4. `results/j1_discovery_av_checkpoint_v1.jsonl`
5. `results/j1_discovery_result_v1.json`
6. `results/j1_discovery_gpu_manifest_v1.json`
7. `results/j1_independent_audit_v3.json`
8. `results/J1_DISCOVERY_LABEL_PROTOCOL_2026-08-06.md`
9. `results/j1_discovery_labels_jobs_v1.json`
10. `results/J1_DISCOVERY_LABEL_RUN_STATUS_2026-08-06.md`

GPU 结果状态为 `EXPLORATORY_DISCOVERY_AV_COMPLETE`，360/360 AV rows；
authoritative audit v3 为 `PASS`、26/26 checks、0 errors。v1/v2 是被完整保留的
audit-script 假设错误，不是数据失败：

- v1 错误要求所有 hard negatives 都必须来自 A2 background pool；
- v2 错误拒绝 tokenizer 解码出的合法 whitespace token；
- v3 允许 80 个 selected-context negatives 与 100 个 background negatives，
  并确认 180/180 均为 tier 0、target activation exact zero。

标签阶段在任何 outcome 前增加了单独 addendum：

- Fable prompt 中删除 numeric SAE activation；
- 同一 feature 的五个 arms 不得同 call，改成 45 个 cross-feature batches；
- 每批五种 arms 各一、来自五个不同 features；
- `NLA_MISMATCHED` 使用 same-stratum、最小 UTF-8 长度差的精确双射 derangement；
- 所有 prompt、raw envelope、cost/usage、model、checkpoint 与协议 hash 均 fail closed。

当前标签 checkpoint 有 13 个成功 batch（`0..12`）和 11 个保留的 403 失败尝试；
batch 22 与 25–44 尚未完成。没有最终 label result，也没有运行 Terra 或查看 partial
arm outcome。充值约 `$11` 后，从项目根目录原样运行：

```powershell
python server\58_j1_discovery_labels.py
```

该命令会验证并保留历史，只调用没有有效 success 的 batch。禁止：

- 删除或编辑 `j1_discovery_labels_checkpoint_v1.jsonl`；
- 用 13/45 做部分分析；
- 因余额问题临时改用另一个 label model；
- 在 label 完整前运行正式 Terra evaluation；
- 把 J1-D1 写成 confirmatory result。

标签 45/45 完成后，先运行：

```powershell
python server\59_j1_discovery_evaluate.py --dry-run
```

审计冻结的 45-feature × 5-hypothesis × 8-context evaluator job 后，才移除
`--dry-run`。Terra 固定通过 Codex CLI `0.146.1`、read-only sandbox、
ephemeral empty directory 运行；完整结果需要 1,800/1,800 scores。

本文面向接手本项目的下一位模型。目标不是重述每一段历史，而是规定：

1. 如何在不被旧文档误导的情况下建立全局认识；
2. 哪些数据和裁决是当前 source of truth；
3. 当前论文可以说什么、不能说什么；
4. N7 第二模型复现已经设计到哪里；
5. 下一步应按什么顺序执行，何时必须停止；
6. 如何避免 GPU 空转、错误重跑和不可恢复的文件覆盖。

---

## 0. 十分钟内建立正确全局观

### 0.1 三个目录

主研究目录：

`D:\Projects\natural_language_autoencoder`

- 存放从 E1/B2 到 N6 的实验脚本、结果、审计、时间线和运维脚本；
- 当前 N5/N6 的本地 frozen artifacts 在这里；
- 该目录没有可靠 Git 历史，不要把文件修改时间等同于实验冻结时间。

上游 NLA 代码目录：

`D:\Projects\nla-from-autodl\natural_language_autoencoders`

- 存放 NLA 的训练、数据生成、推理代码和 27B 配置；
- 当前工作树是 dirty，`main` 相对远端 ahead 5；
- 不要 reset、checkout 或覆盖已有修改；
- 上游 `remote_results` 中的早期 Qwen7B 烟测不是本项目 Gemma-3-12B 正式结果。

Claude 历史配置：

`D:\Projects\.claude`

- `settings.local.json` 主要是历史命令权限白名单；
- 它不是科学事实来源，也不能证明当前 SSH 主机、端口、服务器状态或本轮授权；
- 其中包含旧 SSH、SCP、`systemctl poweroff` 和 `shutdown -h now` 权限，读取时应把它当风险记录，而不是执行清单。

### 0.2 先读什么

建议的人类/模型阅读顺序：

1. `RESEARCH_TIMELINE_2026-08-06.md`  
   完整研究时间线、数据总账、脚本地图、claim 边界和未来路线。
2. `D:\Projects\comprehension.md`  
   较新的代码调用链理解笔记；用于定向，不是数值 source of truth。
3. `RECOVERY_2026-08-03.md`  
   N6 完成后的短交接。
4. `results\N6_FINAL_ANALYSIS_2026-08-03.md`  
   N6 的 paper-facing 解释。
5. `results\n6_pull_staging\LATEST_VERIFIED_PULL.txt` 指向的 staged 目录，然后读：
   - `n6_analysis_v1.json`
   - `n6_independent_audit_v1.json`
   - `n6_plus_preregistration_v2.md`
   - `n6_code_manifest_v2.txt`
6. `results\PROJECT_CLAIM_TABLE_2026-08-03.md`
7. N5 frozen chain：
   - `n5_selective_hybrid_preregistration_v2.md`
   - `n5_cohort_plan_v2.json`
   - `n5_analysis_v2.json`
   - `n5_independent_audit_v2.json`
   - `N5_ANALYSIS_V2.md`
8. `results\N4_REAL_CONTENT_CAUSAL_AUDIT_2026-07-30.md`
9. `results\REVIEW_OPUS_2026-07-30.md`
10. 只有需要追历史时，才读旧 `README.md`、`Conclude.md`、`Prompt.md`、`THOUGHT.md`、`Analysis.md`、`continue.md` 和 `POSSBILITY.md`。

遇到数字或裁决冲突时，严格采用以下证据优先级：

1. frozen raw JSON/NPZ；
2. independent audit JSON；
3. frozen preregistration、code/model manifest 和 hash sidecar；
4. paper-facing final analysis；
5. recovery、timeline 和 claim table；
6. 旧总结、日志和对话摘录。

特别注意：`results\n6_live_analysis_v1.json`、`n6_launcher*.log` 只是运行历史。N6 的正式事实只能来自 `LATEST_VERIFIED_PULL.txt` 指向的 staged pull。

---

## 1. 项目究竟在研究什么

### 1.1 原始研究目标：双向辅助，而不是给两个 codec 排名

本项目最初且持续的中心问题是：

> **能否让 NLA 与 SAE 相互辅助，从而改进现有 SAE/Mechanistic
> Interpretability 工作，并产生比单独使用任一工具更可靠、更可检验的机制解释？**

它包含三个相连方向：

1. **NLA → SAE / 现有 Mech Interp**
   - 用自然语言为 SAE feature、decoder direction、feature interaction 和 residual
     提出可读的机制假设；
   - 根据真实 max-activating contexts 生成候选 label、hard negatives、
     counterfactuals 和后续实验；
   - 对大量 SAE features 做 hypothesis generation、triage 和实验优先级排序；
   - 但必须由新语料激活、held-out discrimination、causal/logit intervention 或
     人类/异构模型证据验证，NLA/AR 不能自我验真。
2. **SAE → NLA / 自然语言 Mech Interp**
   - 用稀疏、可定位的 SAE features 为 NLA explanation 提供 activation-level
     grounding；
   - 检查 NLA 文本中的 proposition/candidate 是否对应真实激活 feature；
   - 用 SAE reconstruction、feature ablation 和 feature intervention 诊断 NLA 的
     causal failures、hallucination 和 tail risk；
   - 测试 SAE 是否能让 NLA explanation 更可校准、更可干预。
3. **NLA ↔ SAE 联合闭环**
   - NLA 提出人类可读假设；
   - SAE 定位稀疏机制和可操作方向；
   - causal patch、feature intervention、logit/behavior endpoint 负责裁决；
   - 研究互补信息、共同盲点、residual coverage 和联合表示，而不是只比较谁的
     cosine 更高。

因此，项目最终要回答的是“**交叉辅助是否改善机制解释**”，而不是“NLA 是否全面
胜过 SAE”。

### 1.2 为什么前面做了大量 NLA–SAE 重建比较

同层重建、centered geometry、retrieval、KL recovery 和 router 比较是**校准与前置
验证**：

- 建立两种工具在同一 activation、同一 layer、同一下游 causal endpoint 上的共同
  评测坐标；
- 检查 NLA 文本是否真的携带 sample-specific、causally relevant information；
- 确认 NLA 是否有资格被用于生成/筛选 SAE hypotheses；
- 用 SAE 作为稀疏、可定位且 causal fidelity 较强的参照，暴露 NLA 的失败尾部；
- 防止在 NLA 只是模板、private code 或无因果效用文本时，过早把它用于 autointerp。

这些比较从来不是项目终点。N6 建立的是一个重要前提：paired NLA 文本确实包含
样本特定、next-token-aligned 的因果信息。N5 router negative 又说明，简单
centered-cosine 规则不足以把该信息安全地接入 SAE。**目前尚未正式建立
NLA-assisted SAE、SAE-grounded NLA 或联合 Mech Interp 的性能增益。**

### 1.3 当前实验对象

目标激活为目标模型某层某位置的 residual-stream activation `x`。

NLA 路径：

`x → AV（activation verbalizer）→ 自然语言文本 → AR（activation reconstructor）→ x_hat_NLA`

SAE 路径：

`x → sparse features → SAE decoder → x_hat_SAE`

然后把不同重建向量 patch 回目标模型的同一层、同一位置，观察下游分布变化。

本项目已经证明必须把四类问题分开：

1. **几何可解码性**：重建向量在 centered activation space 中是否接近 `x`；
2. **文本通道内容**：AV 文本的哪一段、哪些 spans 携带信号；
3. **下游因果保真度**：patch 后目标模型分布是否接近 clean run；
4. **尾部安全性**：平均效果好时，是否仍存在少数灾难性样本。

这四层不能互相替代。高 centered cosine 不是逐样本 causal safety certificate；AR 能 round-trip 某段文字也不是 human-faithful explanation 的证明。

---

## 2. 当前最重要的正式结论

### 2.1 N6：样本特定 predictive-state code 已确认

系统与范围：

- base：Gemma-3-12B-IT；
- intervention：L32 `resid_post`；
- population：fresh Pile、通过固定 AV byte parser 的 eligible subset；
- `n=400` 独立 content groups/documents；
- paired AV/AR protocol；
- comparator：同层 Gemma Scope 2 16k SAE-big。

正式结果：

| Endpoint | Point estimate | Interval / bound | Decision |
|---|---:|---:|---|
| `G_specific` | `+0.117954` | 95% CI `[+0.102860,+0.133995]` | pass |
| raw `KL_cross-KL_true` | `+1.827883 nat/row` | 95% CI `[1.591758,2.074946]` | support |
| `G_content` | `+0.154382` | 95% CI `[+0.136602,+0.173507]` | pass |
| `T_p3` | `0.995175` | one-sided 95% lower `0.993194` | pass |
| `A_meanmass` | `+9.008470` | 95% CI `[+8.444855,+9.572597]` | pass |

冻结标签：

- `H6-A: SAMPLE-SPECIFIC CHANNEL CONFIRMED`
- `H6-B: PREDICTIVE ALIGNMENT CONFIRMED`
- 允许的 headline：
  `SAMPLE-SPECIFIC NATURAL-LANGUAGE PREDICTIVE-STATE CODE CONFIRMED`

正确解释：

> p3 中属于 recipient 自身的 continuation-candidate 内容，在保留 recipient anchors、列表格式和所有引号外 bytes 时，仍有独立的下游因果效用，并与 clean target model 的 next-token 分布对齐。

重要限定：

- `A_meanmass` 是候选 canonical first-token set 的平均概率质量对齐，不是完整候选序列概率；
- 它不是 proposition-level truth 或人类忠实度；
- 结论只适用于该 AV-format-eligible 子总体；
- 结论尚未跨层、跨模型。

### 2.2 candidates 有贡献，但没有压倒 anchors

冻结 secondary：

- `M_majority=+0.040763`，95% CI `[+0.028680,+0.053252]`；
- `G_candidate_anchor=-0.045042`，95% CI `[-0.078669,-0.012779]`。

因此：

- 可以说 recipient-specific identity 占 candidate-content benefit 的显著多数；
- 不能说 candidates 比 target/context anchors 更重要；
- 正确标签是 `NO CANDIDATE DOMINANCE CLAIM`；
- anchors、candidates 和固定文本结构最可能是互补、非线性协作。

N6 aggregate recovery：

| Condition | Recovery |
|---|---:|
| identity | `1.000000` |
| orig | `0.979493` |
| SAE-big | `0.978140` |
| p3 true | `0.974768` |
| p3 cross matched | `0.856814` |
| candidate strip | `0.820385` |
| p12 | `0.795530` |
| anchor strip | `0.775343` |
| all quoted content strip | `0.367074` |
| zero | `0.000000` |

`orig`、`p3_true` 和 `SAE-big` 点估计接近，但 N6 没有 superiority 或 equivalence gate。不得据此写 NLA 优于、等价于或可安全替代 SAE-big。

### 2.3 N5：p3 channel 复制成功，但简单 router 失败

H5-B：

- `G_p3_p12=+0.179490`；
- 95% CI `[+0.146504,+0.215996]`；
- `R_orig=0.964506`；
- `R_p3=0.952928`；
- `T=0.987996`，one-sided lower `0.983015`；
- frozen label：`CHANNEL REPLICATED`。

可说 p3 causally dominant、near-sufficient，相对于完整 AV explanation 保留大部分因果效用；不能说 p3 是唯一通道。

H5-A：

- frozen router coverage `0.655`；
- `G=+0.002419`；
- one-sided lower `-0.001326`；
- catastrophic regret `9/400`；
- exact one-sided 95% upper `3.893%`。

两个 gate 都没有通过，因此唯一正确标签是：

`H5-A: NO SELECTIVE CLAIM`

这否定的是 frozen 单变量 centered-cosine router，不是否定所有可能 router。

### 2.4 其他边界

- N3 证明旧 synthetic corpus 上的 zero activation 是 coverage failure，不是 feature death；
- centered geometry 优势是探索性结果，不等于 causal superiority；
- C1-confirmatory 在语料 construction gate 处停止，没有模型 outcome，不能说科学假设失败；
- C2 没有执行产物；
- 真正的 intervention-to-behavior steering 尚未运行。

---

## 3. 现有证据能支持的论文主线与禁止表述

这里的“论文主线”只表示 **N1–N6 现有数据已经足以支持的论文**，不等于项目最初
或最终的研究目标。原始的 NLA↔SAE 双向辅助主张仍需要新的联合实验。

最稳的论文中心：

> Activation reconstruction through language must be evaluated separately at the levels of centered geometry, text-channel content, downstream causal fidelity, and tail safety. In Gemma-3-12B-IT L32, the paired NLA system carries a sample-specific, next-token-aligned predictive-state code, but this does not establish proposition-level human faithfulness, superiority over SAE-big, or a safe selective router.

应把 N6 的 confirmatory mechanism positive 与 N5 的 confirmatory router negative 并列。二者共同说明：

> 一个 reconstruction 可以携带真实、可解释、因果有效的 predictive content，同时仍不能提供校准良好的逐样本安全分数。

当前禁止：

- “NLA 全面优于 SAE”；
- “NLA 是更好的 causal codec”；
- “NLA 与 SAE-big 已等价或可安全替代”；
- “N5 hybrid/selective router 成功”；
- “p3 是唯一通道”；
- “candidates 是唯一或主导成分”；
- “first-token alignment 证明完整 continuation 正确”；
- “AV/AR round-trip 证明 proposition 或 SAE label 对”；
- “C1 科学假设失败”；
- “synthetic non-activation 证明 feature dead”；
- “carrier readout 或 causal patch 就是 steering”；
- “结论已经跨层、跨模型、跨所有位置”。

---

## 4. 关键 frozen 数据与复现状态

### 4.1 N6

本地 verified staged pull：

`results\n6_pull_staging\n6_pull_20260803T061302Z`

指针：

`results\n6_pull_staging\LATEST_VERIFIED_PULL.txt`

核心文件：

- `n6_plus_preregistration_v2.md`
- `n6_code_manifest_v2.txt`
- `n6_provisional_cohort_v1.json`
- `n6_av_explanations_v1.json`
- `n6_variants_donor_v1.json`
- `n6_recon_v1.json`
- `n6_recon_vectors_v1.npz`
- `n6_causal_candidate_mass_v1.json`
- `n6_analysis_v1.json`
- `n6_independent_audit_v1.json`
- `N6_ANALYSIS_V1.md`
- `n6_resource_report_v1.txt`

核验事实：

- staged pull 共 51 个文件；
- 15/15 advertised `.sha256` sidecars 通过；
- stage 49–56 和 supervisor `.exit` 全为 `0`；
- independent audit：
  - `status=complete`
  - `all_checks_pass=true`
  - `formal_decisions_exact=true`
  - 56 个 numeric leaves 与主分析在 `1e-12` 内一致。

N6 Stage-50 activation parquet 没有拉回，但正式 reconstruction、causal analysis 和 audit 不依赖本地重跑该 parquet。不要为了补它重跑 N6。

N6 prereg v2 的开头误写了匹配 `v1.md.sha256`；实际代码校验自身 sidecar，实际 v2 hash 和审计链通过。N7 必须修复这个文字问题，binding 文件和 sidecar basename 必须逐字一致。

### 4.2 N5

N5 frozen chain 的 17 个核心文件及 sidecars 已重新只读核验通过。`server\48_n5_independent_audit.py --self-test` 通过。

### 4.3 已知复现缺口

- 主项目没有 Git 历史；
- N4/N5 实际权重本体仍主要由远端 manifest 记录，本地不能重新 hash 本体；
- B2/N3 部分旧资产缺 sidecar；
- B6 当前脚本与运行时 hash 有漂移；
- N6 activation parquet 未拉回；
- C1-confirmatory 无模型 outcome；
- C2 和真正 steering 未执行；
- 旧手工 surface audit 缺完整独立标注脚本。

这些缺口要诚实写入 artifact appendix，不能用事后创建的 hash 冒充运行前冻结。

---

## 5. 脚本地图与不可误跑项

N6 正式链：

| Stage | Script | 作用 |
|---|---|---|
| 49 | `server\49_n6_freeze_cohort.py` | 冻结 fresh provisional cohort |
| 50 | `server\50_n6_extract_activations.py` | 12B L32 activation extraction |
| 51 | `server\51_n6_generate_av.py` | AV explanation generation |
| 52 | `server\52_n6_freeze_variants.py` | parser、eligible cohort、donor、text variants |
| 53 | `server\53_n6_reconstruct.py` | AR 与 SAE-big reconstruction |
| 54 | `server\54_n6_causal_patch.py` | causal KL 与 candidate probability mass |
| 55 | `server\55_n6_analyze.py` | 50k bootstrap 与正式 labels |
| 56 | `server\56_n6_independent_audit.py` | 原始产物独立复算 |

这些文件是 frozen N6 历史。N7 应 clone/version 或参数化为新文件，不能原地编辑后仍称 N6 frozen code。

危险项：

- `server\n6_supervisor_template.sh`
  - EXIT trap 在成功、失败、ACK timeout 时都会执行真实 shutdown；
  - N6 已完成，禁止重跑。
- `server\n5_resume_heldout_and_shutdown.sh`
  - 使用 `systemctl poweroff`；
  - 历史上容器没有 systemd，曾造成日志声称 POWER_OFF、实例实际仍空转；
  - 禁止重跑。
- `server\n6_launch_monitor_pull.ps1`
  - `-Execute` 会真的启动 supervisor；
  - `-EmergencyShutdownOnFailure` 可关机；
  - pull-ready key regex 不接受 key 中数字，需在 N7 新版本修复；
  - 不能修改 frozen N6 版本。
- `sync.ps1`
  - wildcard 上传/下载范围过宽，可能覆盖远端或本地 results；
  - N7 应使用 staging 目录、manifest 和逐文件 hash。
- `remote.py`
  - 可执行任意远端 shell，不应当作无害探测。
- 上游训练/发布脚本
  - `train_actor.py` 可能清理旧 `iter_*` checkpoint；
  - `fp8_cast_bf16.py` 可覆盖目标目录；
  - SGLang patch 脚本会原地修改外部 checkout；
  - 当前 N7 replication 不需要运行这些脚本。

---

## 6. N7：为什么选第二模型 27B/L41

选择：

`N7_27B_L41_MATCHED_CANDIDATE_CONTENT_REPLICATION_V1`

而不是 12B 第二层，理由：

1. 当前没有一套可形成 AV → AR → patch → SAE comparator 闭环的 12B 第二层公开 NLA checkpoint；
2. 27B 有完整的 L41 AV、AR、base 和同层 Gemma Scope 2 SAE；
3. 复制 N6 最强、最简洁的 mechanism，比把整个项目迁移到 27B 更有信息增益；
4. 成功可以直接缓解 single-model/scale 限制；
5. 失败也是预注册的 scale boundary，不抹除 12B/N6 结论。

这是一项**第二模型/跨规模外部复现**，不是“参数量导致涌现”的因果实验。12B 与 27B 两点不能单独建立参数规模因果或 emergence threshold。

N7 预注册草案：

`results\n7_27b_l41_n6_replication_preregistration_v1.DRAFT.md`

当前状态：

- DRAFT，非 binding；
- 没有 `.sha256` sidecar；
- 没有 N7 code manifest；
- 没有 N7 model manifest；
- 没有 N7 cohort、AV、AR、SAE、logit 或 causal outcome；
- 本地没有 27B 权重；
- 没有授权旧 N6 supervisor 或旧 remote wrapper 代跑 N7。

---

## 7. N7 必须保持不变的科学协议

为形成严格 N6 replication，N7 正式分析仍采用：

- Pile-only 13 sources；
- sequence length 512；
- position `[64,480]`；
- 至少 16 continuation tokens；
- continuation 至少 75% 通过 frozen content predicate；
- one position per content group；
- source cap 40；
- provisional minimum 480、预期约 500、hard maximum 520；
- frozen byte parser；
- p3 为最后一段，p12 为此前段落；
- ASCII quotes，6–8 spans，其中 4–6 candidate spans；
- AV-text/parser-only eligibility；
- `(source,candidate_count)` hard cells；
- 每 cell 先 seed 2 行，再 source-balanced 填到 exactly 400；
- deterministic one-to-one Hungarian donor derangement；
- recipient anchors 和所有非 candidate bytes 保持不变；
- 十个 conditions：
  - identity
  - orig
  - p3_true
  - p3_cross_matched
  - p3_candidate_strip
  - p3_anchor_strip
  - p3_all_quote_strip
  - p12
  - sae_big
  - zero
- 所有非零向量逐行 norm-match，zero 保持 exact zero；
- batch 1、full 512、BF16 base causal scoring、horizon 16；
- `KL(clean || patched)`；
- ratio-of-sums，禁止 rowwise `/ KL_zero`；
- `G_specific`、`G_content`、`T_p3`、`A_meanmass` 的原公式和原 gate；
- `M_majority`、`G_candidate_anchor` 作为 named secondary；
- 50,000 次 shared ordinary row bootstrap；
- PCG64、linear quantiles；
- identity KL、negative KL、finite、hash、provenance 等 fail-closed checks；
- raw artifact independent audit 到 `1e-12`；
- 完成后 staged pull、逐文件 hash verify、resource report 和关机。

新增 freshness 要求：

- 排除 N4、N5 和 N6 的 document/content-group/original-index identity；
- 对 N4/N5/N6 prefix union 做 20-word normalized shingle embargo；
- N7 内部也维持全局 20-word shingle non-overlap；
- smoke rows 及其文档/shingles 永久排除；
- 不能复用 N6 400 行作为 N7 confirmatory population。

旧 N3/N6 的 `pile-10k` 不能作为 N7 唯一 sampling frame。N6 已几乎耗尽
HackerNews 和 DM Mathematics 等低容量 source 的未用 eligible 文档；继续保持
source cap 40 和 provisional minimum 480 时，严格 embargo 后容量不足。N7 必须先
构建并冻结一份覆盖同 13 source 的新 Pile extension artifact。旧 N3/Pile 数据只
作为 embargo/provenance 输入，新语料的 dataset revision、抽取范围、脚本、schema、
per-source counts 和 SHA-256 必须在正式模型输出前冻结。

如果以后希望在 N6 原 400 行上跑 27B paired comparison，它只能是单独的 secondary/descriptive cross-scale experiment，不能替代 N7 fresh confirmatory replication。

---

## 8. N7 模型、层与资产契约

必须冻结 exact HF revision 和本地每个文件 SHA-256：

### Base

- `google/gemma-3-27b-it`
- 62 transformer blocks；
- `d_model=5376`；
- hook：zero-based block 41 的 `model.layers[41].output`；
- 等价 hidden-state 索引通常是 `hidden_states[42]`，不能误写成 layer 42。

### NLA AV

- `kitft/nla-gemma3-27b-L41-av`
- sidecar 必须确认：
  - role `av`
  - `d_model=5376`
  - extraction layer 41
- 仓库名义约 108.077 GB / 100.654 GiB；
- 权重发布为 F32，单卡运行必须在预注册中显式冻结 BF16 load/cast；
- 不允许把量化作为临时 OOM 修复。

### NLA AR

- `kitft/nla-gemma3-27b-L41-ar`
- role `ar`；
- critic extraction layer 41；
- config 42 layers，即 block 41 后的 critic prefix；
- 名义约 37.600 GB / 35.018 GiB。

### SAE comparator

仓库：

`google/gemma-scope-2-27b-it`

严格目录：

`resid_post_all\layer_41_width_16k_l0_big`

必须确认：

- `hook_in`/`hook_out = model.layers.41.output`
- width 16384
- model family 为 Gemma-3-27B-IT
- `params.safetensors` 与 `config.json` 都有 hash

可一并下载 small 进入 model manifest 作为 descriptive comparator，但 N7 十条件正式协议只使用 SAE-big。

禁止：

- `resid_post` 与 `resid_post_all` 混用；
- L42 或 PT 模型混用；
- 下载整个 Gemma Scope 2 27B 仓库；
- 下载 SAE `examples.safetensors`；
- 用 12B SAE 或 262k SAE 替代冻结的 16k big。

---

## 9. 27B 资源现实

必要资产名义落盘：

| Asset | Approx. size |
|---|---:|
| 27B base | 54.904 GB |
| 27B AV | 108.077 GB |
| 27B AR | 37.600 GB |
| two selected 16k SAEs | 1.410 GB |
| Total before cache/intermediates | 201.991 GB / 188.118 GiB |

因此：

- 准备至少 250–350 GB **可用**磁盘；
- N6 结束时旧盘只有约 81 GB free，绝对不够；
- 注意 HF cache 和 `local_dir` 可能产生 blob 复本与 `.incomplete` 峰值；
- 下载前先做完整文件计划和 `df`；
- 只下载明确 allow-list 的文件。

单卡：

- 80 GB A800 可能完成 BF16、batch 1、顺序加载的 smoke；
- 96 GB 更稳；
- AV 与 AR 不能像旧 `server\03_run_nla.py` 那样同时常驻单张 80 GB 卡；
- 必须 AV generate → unload/free → AR reconstruct；
- base causal scoring 也应单独阶段加载；
- `device_map=auto` 可能静默 CPU offload 或产生 meta-tensor 问题，不得作为默认；
- 若 BF16 sequential smoke 仍 OOM，立即停并报告，不得现场改成 8-bit/4-bit 后继续称严格 replication。

GPU 实例“开着就计费”，不是有 utilization 才计费。任何 GPU 阶段都必须在启动前准备好：

1. 完整脚本；
2. frozen inputs；
3. model/code manifest；
4. staged pull 路径；
5. 成功和失败两条 shutdown 路径；
6. 用户当前明确授权。

---

## 10. N7 执行顺序

### Phase A：CPU-only，当前应继续做

1. 不改 frozen N6 文件，clone/version stage 49–56 为 N7；
2. 清除所有 12B、L32、N5 gate、旧 model manifest 和 `n6_*` 输出 hardcode；
3. 新建 exact 27B asset allow-list；
4. 构建并冻结新的 13-source Pile extension，不以旧 `pile-10k` 作为唯一 sampling frame；
5. 固定容器、Python、NumPy、PyTorch、Transformers 和实际 SAE loader
   （若改用 SAE Lens，则同时冻结其版本）；
6. 生成模型、tokenizer/config、code 和 upstream input manifest；
7. 实现 N4/N5/N6 + smoke embargo；
8. 新建 L41/27B descriptive centering reference；它不能进入 H7 gate；
9. 修复：
   - pull-ready key parser；
   - bounded SSH retry；
   - staged pull；
   - shutdown 真实状态记录；
10. 给所有 N7 脚本写 CPU self-tests；
11. 生成 code manifest 后，才准备 engineering smoke。

### Phase B：20–40 groups engineering smoke

smoke 必须使用永不进入正式分析的独立文档。只允许回答工程问题：

1. 所有 shards、index、sidecar 和 tokenizer 文件是否完整；
2. 27B tokenizer 在 cohort/parser 两个加载路径中是否一致；
3. block 41 hook 是否得到 `[seq,5376]`；
4. base BF16、batch 1 是否没有 CPU/meta offload；
5. SAE shape、hook 和 finite 输出是否正确；
6. AV BF16 单独加载是否能产生 parser-compatible explanation；
7. 卸载 AV 后，AR 是否能重建 `[5376]`；
8. identity patched-position KL 和 first-16 KL 是否小于等于 `1e-5`；
9. peak VRAM、磁盘、吞吐和 ETA；
10. failure shutdown 与 staged logs 是否可靠。

smoke 不允许：

- 查看或估计正式 `G_specific`、`G_content`、`T_p3`、`A_meanmass`；
- 用 smoke 结果调整端点、阈值、donor cost 或候选定义；
- 为提高 eligible rate 放宽 ASCII quote/parser；
- 把 smoke rows 放回正式 cohort。

### Phase C：冻结

只有 smoke 通过，才创建无 `.DRAFT` 的 binding preregistration，并同时冻结：

- exact prereg basename 与 matching `.sha256`；
- code manifest；
- model/tokenizer/config manifest；
- environment manifest；
- corpus、embargo 和 upstream artifact hashes；
- selection/bootstrap seed；
- output schema；
- independent-audit expected contracts。

冻结必须发生在任何 N7 analysis-cohort AV output、AR reconstruction、candidate mass 或 causal outcome 之前。

### Phase D：正式 confirmatory run

按 stage 顺序执行 exactly 400 rows。任何 fail-closed gate 失败：

- 不产生科学 label；
- 不修数据后续跑；
- 保存失败原因、资源报告和 hashes；
- 拉回可诊断 artifacts；
- 关机。

### Phase E：拉回、审计、关机

完成后：

1. 独立 audit 重新读取 raw artifacts；
2. staged pull；
3. 本地逐文件 hash verify；
4. 检查所有 exit codes；
5. 检查正式 labels 与 audit；
6. 保存 resource report；
7. 发送 `sync; /usr/bin/shutdown -h now`；
8. SSH 断开只表明连接消失，不等于 AutoDL 控制面已关机；
9. 用户在控制台确认，不要为了确认而重启。

不要依赖 `systemctl poweroff`。

---

## 11. N7 成功与失败如何解释

如果 27B/L41 的 H7-A 与 H7-B 全部通过：

- 可以说 N6 的 sample-specific predictive-state mechanism 在第二模型规模/对应 NLA 层上外部复制；
- 可以显著缓解 single-model/scale limitation；
- 仍不能说所有模型、所有层或所有位置都成立；
- 仍不能说人类命题忠实、NLA 优于 SAE-big 或 router 安全。

如果部分或全部 gate 不通过：

- 正式报告 `NO ... CLAIM` 及失败 gate；
- 这是预注册的 cross-scale/model boundary；
- 不抹除 12B N6 的 cohort-specific confirmatory result；
- 不允许事后缩小 subgroup、改 parser、改 donor 或改 threshold 后称 replication 成功。

如果工程 smoke 失败：

- 这不是科学 falsification；
- 正确状态是 infrastructure/protocol abort；
- 记录 OOM、shape、layer、parser 或 asset contract 的具体失败原因；
- 在修改协议后必须重新出新的 draft/frozen version，不能覆盖旧 frozen bytes。

---

## 12. 最小只读恢复检查

以下 PowerShell 不启 GPU、不连接远端、不写新的研究结果：

```powershell
$root = 'D:\Projects\natural_language_autoencoder'
$stage = (Get-Content -Raw -Encoding UTF8 "$root\results\n6_pull_staging\LATEST_VERIFIED_PULL.txt").Trim()
if (!(Test-Path -LiteralPath $stage -PathType Container)) {
    throw "No verified N6 staging directory"
}

$timeline = "$root\RESEARCH_TIMELINE_2026-08-06.md"
$declared = ((Get-Content -Raw "$timeline.sha256").Trim() -split '\s+')[0].ToLower()
$actual = (Get-FileHash -Algorithm SHA256 $timeline).Hash.ToLower()
if ($declared -ne $actual) {
    throw "Timeline hash mismatch"
}

$sidecars = Get-ChildItem -LiteralPath $stage -Filter '*.sha256' -File
if ($sidecars.Count -ne 15) {
    throw "Expected 15 N6 sidecars, found $($sidecars.Count)"
}
$sidecars | ForEach-Object {
    $target = $_.FullName.Substring(0, $_.FullName.Length - 7)
    $declared = ((Get-Content -Raw $_.FullName).Trim() -split '\s+')[0].ToLower()
    $actual = (Get-FileHash -Algorithm SHA256 $target).Hash.ToLower()
    if ($declared -ne $actual) {
        throw "Hash mismatch: $target"
    }
}

$exits = Get-ChildItem -LiteralPath $stage -Filter '*.exit' -File
if ($exits.Count -ne 9) {
    throw "Expected 9 N6 exit files, found $($exits.Count)"
}
$exits | ForEach-Object {
    if ((Get-Content -Raw $_.FullName).Trim() -ne '0') {
        throw "Nonzero exit: $($_.Name)"
    }
}

$analysis = Get-Content -Raw "$stage\n6_analysis_v1.json" | ConvertFrom-Json
$audit = Get-Content -Raw "$stage\n6_independent_audit_v1.json" | ConvertFrom-Json
"$($analysis.status) / $($audit.status) / $($audit.stage55_comparison.all_checks_pass) / $($audit.stage55_comparison.formal_decisions_exact)"

python "$root\server\48_n5_independent_audit.py" --self-test

$upstream = 'D:\Projects\nla-from-autodl\natural_language_autoencoders'
git -C $upstream status --short --branch
git -C $upstream log -1 --format='%H %ad %s' --date=iso-strict
```

预期：

- N6 15 个 sidecar 全通过；
- stage 49–56 和 supervisor exits 全为 0；
- `complete / complete / True / True`；
- N5 audit self-test PASS；
- upstream 显示 dirty/ahead 状态，必须保留。

---

## 13. 接手后的默认优先级

如果用户没有给出新的覆盖指令：

1. 保持 AutoDL 关闭；先把 J2 final/audit、N5/N6、J1-D1 整理进 claim table、
   Methods、主图和 artifact appendix；
2. CPU-only 复核 J2 local rescue/catastrophe 的 pre-output predictor，但只能用于
   discovery；不得在同一 cohort 上宣布 router 成功；
3. 重设计真正的 `J2 / SAE→NLA` 接口：使用 structured sparse grounding 或
   SAE counterfactual intervention，而不是重复 serial reconstruction；
4. `J1 / NLA→SAE` 先做固定非 OpenAI/人类全量重标、capacity-matched SAE-only
   与强 autointerp baseline，跨家族稳定后才冻结 fresh confirmatory；
5. 对任何新 J1/J2 协议做独立代码与统计审计，并向用户报告磁盘、GPU、耗时与
   stopping rule；
6. 只有在协议、cohort、manifest 全部冻结且用户明确说服务器已开时才启动 GPU；
7. pull、独立审计、hash verify、`shutdown -h now`；
8. N7 27B/L41 replication 作为单模型风险对冲，可在联合实验设计成熟后并行，
   但不再自动压过原始双向辅助主线；
9. 同步推进 Methods、N5/N6/J1/J2 Results、Limitations 和 artifact appendix。

不要优先重启：

- 旧的 C1 synthetic-corpus 协议（C1 科学问题本身仍是中心方向）；
- router 事后调参；
- steering；
- SAE residual dark matter；
- 27B “思维涌现”宽题。

不得再把 C2“NLA 辅助 feature labeling”整体列为低优先级。应当淘汰的是没有外部
真值、用小 synthetic corpus 和同源 round-trip 自我验真的旧协议；真实语料、
held-out activation 与客观 causal endpoint 下的双向辅助实验仍是项目主线。

---

## 14. 一句话交接

本项目的原始主线始终是 NLA 与 SAE 的双向辅助：让 NLA 为 SAE/Mech Interp
提出可读假设，让 SAE 为 NLA 提供稀疏 grounding，再由外部 causal/behavioral
端点裁决。E1–N6 的比较和机制实验是在为这个联合闭环建立可信前提，而不是给两个
codec 排名。当前已经确认 Gemma-3-12B-IT L32 的 paired NLA 文本含有样本特定、
next-token-aligned 且下游因果有效的信息，但 J1-D1 尚未证明跨标签器稳定的
NLA-assisted SAE 增益；J2-P0 进一步证明朴素的
`SAE reconstruction→AV→AR` 串联虽然改善 centered geometry，却损失 sparse-code
identity 并恶化 causal fidelity。下一步不是再确认同一串联，而是把真实语料、
held-out activation 与 causal endpoint 下的 structured NLA→SAE / SAE→NLA
联合接口重新设计并冻结。Gemma-3-27B-IT L41 的 N7 是有价值的外部复制，但只是
风险对冲，不能替代原始双向辅助主线。
