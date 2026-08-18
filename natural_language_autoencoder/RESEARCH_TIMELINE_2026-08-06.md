# NLA × SAE 研究总账与时间线

> 整理日期：2026-08-06（Asia/Shanghai）  
> 范围：`D:\Projects\.claude`、`D:\Projects\nla-from-autodl\natural_language_autoencoders`、`D:\Projects\natural_language_autoencoder`  
> 目的：把 Fable 5 与 Jason 开始以来的研究问题、实验、数据、脚本、失败、正式裁决和未来路线编织成一条可追溯的时间线。

## 2026-08-06 追加：J1-D1 首个直接 NLA→SAE discovery 完成

这是本时间线目前最新的事件，覆盖下文“J1 尚未完成”的旧状态，但不改变 N5/N6
确认性结论。

1. 保留 Fable batches `0..12`，Luna Max 完成 `13..44`；最终 45 batches、
   225 hypotheses，每个 arm 恰好 13 Fable + 32 Luna。
2. Terra 对 45 features × 5 arms × 8 contexts 完成 1,800/1,800 个盲评分，
   0 failures；每个 feature–arm 均为 4 held-out positives +
   4 exact-zero hard negatives。
3. 完整 ITT micro AP：SAE `0.905301`、Assisted `0.918582`、Contrastive
   `0.907093`、Mismatched `0.849382`、NLA-only `0.890284`。
4. Assisted − SAE 为 `+0.013281`，95% feature-bootstrap CI
   `[-0.004941,+0.035954]`；Luna–Luna 为正而 Fable–Fable 为负。
5. 裁决：`REDESIGN_REPLICATE_BEFORE_CONFIRMATORY`。这是原始双向辅助主线的
   第一个完整 discovery 结果，但不是 confirmatory 增益。先做固定异构
   non-OpenAI/人类重标与 capacity-matched 强 baseline，再决定 fresh、
   source/shingle-embargoed 且带 causal endpoint 的确认性 J1。

权威文件：

- `results/J1_MIXED_DISCOVERY_FINAL_2026-08-06.md`
- `results/j1_discovery_labels_mixed_result_v3.json`
- `results/j1_blinded_eval_result_mixed_v2.json`
- `results/j1_blinded_eval_analysis_mixed_v2.json`

---

## 0. 当前结论先行

这项工作的原始中心从来不是给 NLA 与 SAE 排名，而是：

> **能否用 NLA 辅助 SAE 与现有 Mechanistic Interpretability，或反过来用 SAE
> 为 NLA 提供稀疏 grounding，从而形成更可靠、更可检验的联合机制解释。**

早期 NLA–SAE reconstruction、geometry、retrieval 和 causal-fidelity 比较，是为
这个双向辅助目标建立共同坐标、可信度和失败边界的实验分支。它们不是项目终点。

截至当前数据，已经被收窄并加固的是其中一个关键前提：

> 激活的自然语言重建必须分别评估 **去均值几何、文本通道内容、下游因果保真度和尾部安全性**。  
> 对 Gemma-3-12B-IT 第 32 层 `resid_post`，现有 NLA 的第三段承载一个与下一 token 分布对齐、具有样本特异性和显著下游因果效用的自然语言 predictive-state code；但这不等于人类忠实解释、NLA 全面优于 SAE-big、安全的逐样本 router，或已经完成 steering。

这使 NLA 成为值得继续用于 SAE hypothesis generation/triage 的候选辅助信号，但
**尚未证明 NLA-assisted SAE 优于 SAE-only，也尚未证明 SAE-grounded NLA 优于
NLA-only**。原始的双向协同问题仍待正式实验。

截至本文件日期，最稳的证据组合是：

1. **N6 confirmatory positive**：recipient 自己的 candidate content 相比 matched-cross candidates 有独立因果收益；预测对齐端点也通过。
2. **N5 confirmatory positive**：包含 candidates 的第三段 `p3` 在全新 held-out cohort 上是 causally dominant、near-sufficient channel。
3. **N5 confirmatory negative**：冻结的单变量 centered-cosine router 没有建立 selective improvement，也没有通过 tail-safety gate。
4. **N3 diagnostic correction**：小型 synthetic corpus 上的零激活是 coverage failure，不能证明 feature death。
5. **方法论结论**：高 centered cosine、内部 AV/AR round-trip 和身份检索都不能代替 sample-level causal KL 或外部语义忠实度。

现在已经有一篇 scope 严格的 mechanism/evaluation paper 所需的核心证据，但尚未完成论文正文、主图、artifact appendix，也尚未补齐第二模型/规模复制和人类 proposition-level fidelity 验证。

---

## 1. 三个目录分别是什么

### 1.1 `D:\Projects\.claude`

这里只有 `settings.local.json`，内容是 Claude Code 的本地工具权限白名单和历史 SSH/SCP/关机命令授权。

- 它不是研究设计、数据或实验结论来源。
- 里面的旧端口、SSH 命令和运维授权可能过时，不应当作当前 AutoDL 配置。
- 它允许过 `systemctl poweroff` 和 `shutdown -h now`，因此属于运维风险面，而不是科学资产。

### 1.2 `D:\Projects\nla-from-autodl\natural_language_autoencoders`

这是上游开源 NLA 完整训练/推理仓库，提供：

- AV：activation vector → natural-language text；
- AR：text → reconstructed activation vector；
- activation 数据生成；
- AV/AR SFT；
- GRPO/RL；
- checkpoint 转换、发布和推理演示。

它定义了本项目复用的核心机制和工程约束，例如 raw activation 不在 datagen 阶段归一化、document-level split、sidecar contract、注入 token 检查以及 `cp_size == 1`。

这里的 `remote_results/*.json` 是 2026-06-16 的 Qwen2.5-7B L20 推理烟测，不属于后来的 Gemma-12B NLA×SAE正式数据。

### 1.3 `D:\Projects\natural_language_autoencoder`

这是我们真正的 NLA×SAE 研究项目：

- `server/`：远端 AutoDL 实验和本地分析脚本；
- `results/`：JSON、JSONL、NPZ、Parquet、日志、哈希、预注册与审计；
- `activations/`：少量冻结语料 manifest；
- `README.md`、`Conclude.md`、`THOUGHT.md`、`Prompt.md`：历史设计与交接；
- `RECOVERY_2026-08-03.md`、`results/N6_FINAL_ANALYSIS_2026-08-03.md`：当前最新结论。

当前 `results/` 共有约 235 个文件、252.54 MiB。项目不是 Git 仓库，因此历史版本只能依赖文件时间、哈希 sidecar、manifest、日志和交接文档恢复。

---

## 2. 证据优先级与编号说明

### 2.1 当前 source-of-truth 顺序

发生数字或表述冲突时，按以下顺序采信：

1. `results/n6_pull_staging/n6_pull_20260803T061302Z/n6_analysis_v1.json`
2. 同目录 `n6_independent_audit_v1.json`
3. `results/N6_FINAL_ANALYSIS_2026-08-03.md`
4. `results/n5_analysis_v2.json`、`results/n5_independent_audit_v2.json`
5. `results/N5_ANALYSIS_V2.md`
6. `results/N4_REAL_CONTENT_CAUSAL_AUDIT_2026-07-30.md`
7. `results/REVIEW_OPUS_2026-07-30.md`
8. `results/PROJECT_CLAIM_TABLE_2026-08-03.md`

`README.md`、`Conclude.md`、`Prompt.md`、`THOUGHT.md`、`POSSBILITY.md` 和 `continue.md` 仍有重要历史价值，但关于“下一实验”、服务器状态及 N5/N6 判决的内容可能过时。

### 2.2 为什么编号看起来会重名

| 编号体系 | 含义 | 例子 |
|---|---|---|
| A/B/C | Fable 阶段提出的研究想法与方向 | B2 检索、B3 实体替换、C1 evaluator、C7 私有编码检查 |
| E1–E11 | 对早期已执行工作的回顾性流水编号 | E1 baseline、E8 B6+B4、E10 causal patch、E11 text ablation |
| F1–F14 | 从实验中总结出的发现/勘误 | F1 centered geometry、F11 p3、F12 causal parity、F14 统计修正 |
| N1–N6 | 2026-07-30 后按执行顺序推进的新实验 | N1 文本变体、N2 causal patch、N3 real corpus、N4–N6 |
| H5/H6 | 预注册的正式假设与裁决 | H5-A router negative、H5-B p3 positive、H6-A/B positive |

特别注意：

- N1/N2 分别对应回顾编号 E11/E10；
- “C1-confirmatory”是 feature-label/evaluator 路线，不是 N1–N6 中的某个编号；
- C2 只是设计提案，从未产生 `c2_*` 脚本或结果；
- carrier ablation/insertion 只经过 AV/AR，不是 steering。

---

## 3. 总时间线

| 时间 | 阶段 | 最终意义 |
|---|---|---|
| 2026-06-16 | 上游 NLA 推理烟测 | 验证 AV/AR、answer probe 和远端环境能工作 |
| 2026-06-27–07-04 | Gemma-12B L32 基础设施与 E1 baseline | 建立同激活、同层的 NLA vs SAE 比较管线 |
| 2026-07-04 | E2/E3：SAE方向与残差 pilot | 暴露“字典方向看似可读、残差不可读”的初始现象 |
| 2026-07-10 | E4 injection + E5 centered correction | 发现均值方向混淆；去均值成为全项目纪律 |
| 2026-07-11 | Fable 交接文档 | 固化早期结果、路线和操作经验 |
| 2026-07-26 | E7/B2 + E8/B6+B4 | 检索不胜 SAE；方向可读性异质且不能自动验真 |
| 2026-07-30 上午 | E9/C1 pilot | relative AR delta 有弱信号，absolute q 和同源 judge 失败 |
| 2026-07-30 下午 | C1-confirmatory corpus v1–v3r2 | 连续门禁失败，停止于语料设计，零科学 outcome |
| 2026-07-30 下午 | Opus 独立复核/F14 | 修正 q+ 双峰、伪 AUC 与错误 gate 顺序 |
| 2026-07-30 晚 | N1/N2 | p3 通道定位；第一次下游 causal patch；暴露旧 cohort 污染 |
| 2026-07-30 晚 | N3 | 8.24M real tokens；证明 synthetic non-activation 是 coverage failure |
| 2026-07-30 晚 | N4 | 200 个真实 content tokens 的 causal replication；p3 因果价值出现 |
| 2026-07-30–08-01 | N5 | router 正式失败；p3 channel 在 400 held-out groups 上正式复制 |
| 2026-08-03 | N6+ | matched candidate-content substitution 确认 sample-specific predictive code |
| 2026-08-05 | 27B 可行性审计 | L41 NLA/SAE 完整可用；适合作为 N7 复制，不宜整体迁移 |
| 2026-08-06 | 本研究总账 | 当前文件；统一时间线、数据、脚本和未来路线 |

---

## 4. 逐阶段研究叙事

### 4.0 2026-06-16：先证明 NLA 推理链能运行

最初不是直接比较 SAE，而是先在上游仓库验证 NLA 推理：

- Qwen2.5-7B-Instruct，第 20 层；
- base model 生成、activation 抓取；
- AV-only 与 AV→AR round trip；
- question/answer/full-context token probe。

数据：

- `nla-from-autodl/natural_language_autoencoders/remote_results/answer_probe_av_only_test.json`
- `.../answer_probe_with_ar_test.json`
- `.../question_probe_test.json`
- `.../full_context_probe_test.json`

这些文件只证明基础设施与接口工作，不构成后续 Gemma-12B 科学结论。

### 4.1 2026-06-27–07-04：E1，同层同激活的 NLA vs SAE baseline

#### 问题

在同一批 Gemma-3-12B-IT L32 `resid_post` 激活上，NLA 与 Gemma Scope 2 SAE 的方向重建谁更好？

#### 设计

- 5 条英文 instruct prompts；
- 每条 8 个位置，共 40 个 activation vectors；
- `d_model=3840`；
- NLA：AV → explanation → AR；
- SAE：16k JumpReLU，small 与 big 两档 L0；
- 最初比较未去均值 cosine、`mse_nrm=2(1-cos)`，SAE 另报 FVE/L0。

#### 初始数字

- NLA raw cosine：`0.996`
- SAE-small：`0.9925`
- SAE-big：`0.9936`
- SAE FVE：small `0.6076`，big `0.6747`
- SAE实测 mean L0：small `15.2`，big `75.5`

这些数字后来被 E5/F13 限定：共享均值方向严重抬高 raw cosine，而且 13/40 个位置是 chat template/blank token。

#### 数据

- `results/nla_results.json`：40 行 AV explanation、位置和 NLA round-trip 分数；
- `results/sae_results.json`：SAE-small；
- `results/sae_results_big.json`：SAE-big；
- `results/comparison.json`、`comparison.md`：按 `(doc_id, position)` 合并；
- `results/pipeline.log`：旧流水日志；
- 原始 `acts_L32.parquet` 当前未保存在本地，但后来的 `results/recon_vectors.npz['x']` 保留了 40×3840 activation。

#### 脚本

`02_extract_activations.py → 03_run_nla.py / 04_run_sae.py → 05_compare.py`

#### 当前等级

探索性 baseline。raw cosine 只能作为历史，不得直接用于论文主张。

### 4.2 2026-07-04：E2，NLA 能不能读 SAE decoder directions

#### 问题

把 SAE 的 `w_dec[j]` 当作一个 activation direction 输入 AV/AR，是否能得到可靠自然语言标签？

#### 设计

top 12、random 6、Gaussian 4，共 22 个方向。

#### 结果

初看 top feature 的 raw median `|cos|≈0.80`，似乎非常可读；E5 去均值后：

- top mean `|cos_c|=0.189`
- random `0.105`
- Gaussian `0.011`

大量“正负极性”其实是 feature direction 与数据集均值方向的对齐符号，而不是 AV 真读出了反义语义。

#### 数据与脚本

- `results/wdec_pilot.json`
- `results/centered_rescore.json → wdec_centered`
- `results/recon_vectors.npz`
- `server/06_pilot_wdec.py`

#### 当前等级

探索性。支持“少数方向有弱内部读出信号”，不支持用 AR round-trip 自动签发 feature label。

### 4.3 2026-07-04：E3，SAE residual / “dark matter”是否可读

#### 问题

SAE 没解释掉的 residual 是否包含 NLA 能读出的自然语言信息？

#### 结果

- raw residual round-trip cosine：`0.0356`
- centered residual cosine：`-0.0287`
- mean residual variance fraction：约 `0.124`

当前 AV 对 SAE residual 基本不可读。

#### 数据与脚本

- `results/resid_pilot.json`
- `results/centered_rescore.json → resid_centered`
- `server/07_pilot_residual.py`

#### 当前等级

负向探索结果。只否定当前 checkpoint 对离流形 residual 的能力，不证明所有 NLA 都不可能读 SAE residual。

### 4.4 2026-07-10：E4，向 residual 注入已知方向

#### 问题

如果把已知 SAE feature direction 以不同强度注入 residual，NLA 能否把它检测出来？

#### 设计

4 features × 4 residuals × `α∈{0,.25,.5,1,2}`，80 行。

#### 结果

raw 指标有 `≈0.919` 的巨大平坦地板；去均值后的 mean `|cos|`：

- α=0：`0.072`
- α=.25：`0.0435`
- α=.5：`0.0436`
- α=1：`0.0556`
- α=2：`0.0915`

即便信号强到约 2× residual norm，检测曲线仍近乎平坦。AV 对 off-manifold 输入常 collapse 成相似模板。

#### 数据与脚本

- `results/injection_pilot.json`
- `results/injection.log`
- `results/centered_rescore.json → injection_centered_curve`
- `server/08_pilot_injection.py`

#### 当前等级

负向探索结果；不是 steering，因为向量没有重新写回 base model 后续计算。

### 4.5 2026-07-10：E5，均值方向修正成为全项目转折点

#### 发现

AR 对许多泛型文本的重建都靠近 activation mean direction，未去均值 cosine 因而“白拿”约 0.975。

对双方都投影掉 `m_hat` 后：

| 方法 | centered cosine | 最优单尺度 centered FVE |
|---|---:|---:|
| NLA | **0.85927** | **0.707** |
| SAE-small | 0.65839 | 0.459 |
| SAE-big | 0.72456 | 0.543 |
| nearest-other real activation oracle | 0.541 | 0.336 |
| fixed generic text（N1后精测） | 约 -0.005 | — |

早期 `pred_resid` 的 `0.041` 不是固定 generic floor，而是逐样本 residual-text weak-signal control。

#### 意义

NLA 确实比两只冻结 SAE 保留更多 sample-specific activation geometry，但 N2/N4/N5 后来证明，这种几何优势不能自动转化为下游因果优势或逐样本安全分数。

#### 数据与脚本

- `results/centered_rescore.json`
- `results/recon_vectors.npz`
- `results/rescore.log`
- `server/09_rescore_centered.py`
- 更严格复算：`server/25_local_recheck.py → results/local_recheck_opus.json`

#### 当前等级

强探索性几何结果和方法论卫生规范，不是 NLA causal superiority。

### 4.6 2026-07-10前后：E6，解释模板化与编造审计

40 条 NLA explanation 中发现至少 3 条具体实体错配，例如将源上下文替换成其他景点/对象；错配行与其他行的 raw cosine 均值同为约 `0.9962`。residual explanation 的 4-gram 重叠约为 full-activation explanation 的 3.1 倍，说明越离流形越容易模板 collapse。

证据来自：

- `results/nla_results.json`
- `results/resid_pilot.json`
- 汇总于 `results/POSSBILITY.md` F6/F7 和 `Conclude.md`

这一阶段没有保存独立统计脚本或完整标注表，因此可复现性弱于后续冻结实验。它仍提供了重要警告：**高 round-trip score 不等于表面文本忠实。**

### 4.7 2026-07-26：E7/B2，centered 40-way retrieval

#### 问题

如果重建向量真的保留样本身份，它能否从 40 个真实 activation 中找回原样本？

#### 结果

| 方法 | Top-1 | Top-5 | MRR | mean margin |
|---|---:|---:|---:|---:|
| NLA | 92.5% | 100% | .9583 | **.3195** |
| SAE-small | 95.0% | 100% | .9708 | .2085 |
| SAE-big | 95.0% | 100% | .9708 | .2531 |
| residual-text control | 17.5% | 40.0% | .2883 | -.0507 |

尺度归一化后的 z-margin/d′：

- NLA：`6.49`
- SAE-small：`5.12`
- SAE-big：`5.51`

所以预设的“NLA Top-k 胜过 SAE”没有成立；NLA 的连续配对间隔更大，但离散准确率略低。

#### 数据与脚本

- `results/retrieval_eval.json`
- 输入 `results/recon_vectors.npz`
- `results/b2_retrieval.log`
- `server/10_retrieval_eval.py`

#### 完整性限制

`retrieval_eval.json` 和 `recon_vectors.npz` 没有本地 `.sha256` sidecar，只能依赖 JSON 内嵌 hash 和既有日志。

### 4.8 2026-07-26：E8/B6+B4，factorial feature readout

#### 设计

- 24 个文档、1,365 个 L32 activations；
- 4 domains × 3 languages × train/test；
- train-only selection；
- 24 semantic-new + 4 legacy + 1 structural + 8 active-nonselective + 8 Gaussian，共 45 directions；
- 590 个 AV jobs：方向 `±v`、随机重复和自然 carrier ablate/insert/amplify。

#### 原始结果

- semantic-new ITT `q+` median `0.1136`
- `r−` median `0.0306`
- polarity `0.0708`
- sign accuracy `75%`
- signed retrieval Top-1 `33.3%`
- heldout-valid `14/24`，其中 domain `6/15`、language `8/9`
- nonzero carrier：ablate `≈0.210`，insert `≈0.097`，amplify `≈0.042`
- surface audit：strict `5/24`，coarse `7/24`，mismatch `9/24`，indeterminate `3/24`

#### F14 后的正确解释

`q+` 不是均匀微弱，而是双峰：

- 10 个 `q+≥0.362`；
- 13 个 `<0.15`；
- 高组 9/10 的方向检索 rank=1；
- median `0.114` 落在两峰断层里。

旧的 pooled `Spearman(q+, synthetic test AUC)≈0` 不能证明“不相关”：

- 8/24 feature 在 synthetic test 完全不激活，AUC=0.5 只是 tie convention；
- language 层 8/9 AUC=1.0，发生天花板饱和；
- domain 层相关反而约 `+0.40`。

因此该 synthetic gate 测到的是语料覆盖，不是 feature quality；N3 随后直接验证了这一点。

#### 数据

- `results/acts_L32_factorial_v1.parquet`
- `results/b6b4_factorial_selection.json`
- `results/b6b4_factorial_feature_stats.npz`
- `results/b6b4_factorial_vectors.npz`：`x=(1365,3840)`，`directions=(45,3840)`
- `results/b6b4_factorial_av_rows.jsonl`
- `results/b6b4_factorial_result.json`
- `results/b6b4_factorial_recon_vectors.npz`：590×3840
- `results/b6b4_factorial_analysis.json/.md`
- `results/b6b4_surface_audit.md`
- `results/b6b4_checksums.sha256`

`b6b4_factorial_av_rows.partial.jsonl` 和 `debug_b6b4_*` 是 stale/debug，不是 canonical。

#### 完整性限制

`b6b4_checksums.sha256` 的 17 个条目中 16 个通过；当前 `server/11_extract_factorial_activations.py` 与运行时冻结 hash 不一致，说明代码在运行后发生漂移。正式引用应依赖冻结数据，不应声称当前 11 号脚本与当时 bit-identical。

#### 当前等级

探索性、异质的内部 readout 信号。NLA/AR 可以生成或分诊 feature hypotheses，但不能自动验真；carrier readout 不是 steering。

### 4.9 2026-07-30 上午：E9/C1 external-validity protocol pilot

#### 问题

relative AR score 是否能区分正确粗轴与受控错误轴，并预测 heldout selectivity 或独立 evaluator？

#### 设计

- 24 个已看过的 E8 features；
- 432 个实际 scored candidates；
- 主对比：正确 coarse axis reference − reciprocal hard negative；
- train references、autointerp、NLA original、paraphrase、generic 和 judge 均为探索性。

#### 结果

- mean delta：`0.01550`，95% feature-bootstrap CI `[0.00510,0.02712]`
- median delta：`0.00897`
- 16/24 为正
- exact one-sided sign test：`p=.0758`
- joint sign-flip mean test：`p=.00393`
- feature→7-axis Top-1：`50%`
- train reference − hard negative：`0.195`
- axis delta 事后预测 heldout-valid：AUC `0.764`
- NLA original 24-way text→feature Top-1：`62.5%`
- blind Gemma context judge 对 NLA 仅 `2/24` 通过
- 同源 base judge 校准失败：pooled q→judge AUC `0.410`

一次 NLA paraphrase 大幅降分，但 median 只保留原文 60.1% 字符，混入长度/压缩混淆，不能证明 private code。

#### 数据

- `results/c1_pilot_benchmark.json`
- `results/c1_pilot_checkpoint_v2.jsonl`
- `results/c1_pilot_result.json`
- `results/c1_pilot_recon_vectors.npz`：432×3840，24 个 semantic directions
- `results/c1_pilot_analysis.json/.md`
- `results/c1_pilot_secondary_analysis.json/.md`
- `results/c1_pilot_validation.json`
- `results/c1_pilot_checksums.sha256`：11/11 通过

`c1_pilot_attempt1.*` 和 smoke checkpoint 是失败/调试历史。

#### 当前等级

协议 pilot，而非确认性结果。relative delta 值得在全新 feature cohort 上预注册复现；absolute q、同源 judge 和身份检索都不能作为标签真值。

### 4.10 2026-07-30 下午：C1-confirmatory 语料门禁连续失败

这一阶段最大的产物是“为什么没有科学结果”。

#### V1

生成前 stage0 freeze 完成，但生成文本在字数/格式门禁失败。

#### V2

机械生成 144 文档：

- 24 concepts；
- 96 discovery；
- 48 heldout；
- 12 reciprocal pairs。

两名独立 reviewer 都判 FAIL，并一致发现 10 个问题文档；没有提取 activation。

#### V3/V3r2

- V3 anchor audit：20/24 concept 的 discovery/heldout 机制重叠，FAIL；
- V3r2 改写 33/144 anchors；
- 最终 concept `17 PASS / 7 FAIL`；
- pair overall `6/12 PASS`；
- 按冻结 stopping rule 停止。

#### 数据

- `results/c1_confirmatory_preregistration_v1.md`
- `..._v2_amendment.md`
- `..._v3_amendment.md`
- `results/c1_confirmatory_stage0_freeze_v1.json/.sha256`
- `..._v2.json/.sha256`
- `results/c1_confirmatory_corpus_checkpoint_v1.failed.jsonl`
- `results/c1_confirmatory_corpus_checkpoint_v2.jsonl`
- `results/c1_confirmatory_corpus_report_v2.json`
- `activations/c1_confirmatory_all_v2.jsonl`、`discovery_v2.jsonl`、`heldout_v2.jsonl`
- `results/c1_confirmatory_manual_review_A_v1.json`
- `..._B_v1.json`
- `results/c1_confirmatory_manual_audit_v1.json`：FAIL
- `results/c1_confirmatory_scenario_anchor_audit_v3r2.json`：FAIL
- `results/c1_confirmatory_scenario_anchor_v3r2_failure.md`

#### 当前等级

**protocol abort before model outcomes**。没有 C1-confirmatory activation、feature selection、AV/AR score 或统计 endpoint；因此不能说“C1 科学假设失败”。这条经历促使项目转向真实语料，而不是继续修第五版 synthetic rubric。

### 4.11 2026-07-30 下午：Opus 独立复核与 F14

Opus 直接从 NPZ/JSON 重算，不采信旧 conclusion 文件，产出：

- `server/25_local_recheck.py → results/local_recheck_opus.json`
- `server/26_local_recheck_b6b4.py → results/local_recheck_b6b4_opus.json`
- `server/27_local_recheck_stratified.py → results/local_recheck_stratified_opus.json`
- `results/REVIEW_OPUS_2026-07-30.md`

复核加固了 centered geometry 和 B2 margin，但修正了三件事：

1. q+ 是双峰，不是“弱而系统”；
2. pooled `ρ≈0` 是 measurement failure，不是已证明零关系；
3. synthetic heldout activation gate 会错误剔除最可读 feature。

它还指出全项目当时缺乏因果端点，并推动 N1/N2、N3、N4 当晚执行。

### 4.12 2026-07-30 晚：N1，文本通道拆解与 C7/B3

#### 设计

在旧 40 行 explanation 上冻结 11 类文本变体：

- 原文；
- p1/p2/p3/p12；
- quote stripping；
- word shuffle；
- 第三方长度匹配 paraphrase；
- entity swap；
- fixed generic controls。

#### 结果

- original centered cosine：`0.8593`
- p3-only：`0.8259`，保留 `96.1%`
- p12：`0.6087`
- p1：`0.2403`
- p2：`0.5900`
- quote-strip-p3：约 `0.176`，保留 `20.8%`
- entity swap：`0.7711`，保留 `89.7%`
- third-party paraphrase：`0.8511`，保留 `97.3%`
- fixed generic floor：约 `-0.005`

这说明 narrow private-string-code 不是主要解释：第三方等义改写几乎保留全部几何信号。但它也不证明人类命题忠实；实体身份只贡献很少，真正承载大量信号的是 p3 中的 candidate-bearing quoted content。

#### 数据与脚本

- `results/c7b3_variants_v1.json/.sha256`
- `results/c7b3_scores_v1.json`
- `results/c7b3_recon_v1.npz`
- `server/28_build_text_variants.py`
- `server/29_score_text_variants.py`
- 汇总见 `results/n1n2_analysis.json`

#### 当前等级

强探索性通道诊断，但 cohort 有 F13 污染，后来由 N4–N6 在真实 content tokens 上加固。

### 4.13 2026-07-30 晚：N2，第一次下游 causal patch

#### 设计

将 NLA、SAE-small、SAE-big、zero、Gaussian、mean 等重建向量 patch 回 Gemma-3-12B-IT L32，测 patched position 的 `KL(clean || patched)` 和 continuation diagnostics。

#### 对照

- identity KL：`0`
- provenance cosine：`1.000000`
- Gaussian KL：约 `31.07`
- zero KL：约 `16.34`

#### 结果

- aggregate KL recovered：NLA `0.7569`
- SAE-small：`0.7128`
- SAE-big：`0.7712`
- NLA 与 SAE-big 没有建立优势；NLA 的 centered-cosine 优势没有转化为 causal superiority。
- cosine 与 KL 跨方法能排序，但方法内不稳定，NLA 内甚至方向相反。
- CE-based loss-recovered 作废：评测窗口被 chat template tokens 支配，clean CE 约 21 nats。
- 旧 40 行中 13/40 是 template/whitespace；这成为 F13。

#### 数据与脚本

- `results/causal_patch_v1.json`
- `results/n1n2_analysis.json`
- `results/n1n2_checksums.sha256`：UTF-16LE，9/9 hash 通过
- `server/30_causal_patch.py`
- `server/31_analyze_n1n2.py`
- `server/32_diag_ce_window.py`

#### 当前等级

探索性因果诊断；旧 cohort 污染限制外推，但它决定性地否定了“几何优势自动等于因果优势”。

### 4.14 2026-07-30 晚：N3，真实语料底座

#### 设计

- Pile 截到 8M tokens；
- XNLI 多语言 parallel corpus 约 241k tokens；
- 32 个来源；
- 总计 `8,240,945` tokens；
- 对两档 16k SAE 计算全 feature 激活频率、来源分布和 top contexts；
- 重抽无 template 的 200-row content-token cohort。

#### 结果

| 指标 | SAE-small | SAE-big |
|---|---:|---:|
| alive features | 16,339 / 16,384 | 16,202 / 16,384 |
| alive fraction | .9973 | .9889 |
| mean L0 | 17.96 | 122.59 |

旧 synthetic test 上完全不激活的 8 个 features，在真实语料上全部激活；24/24 legacy features 无一死亡。

N3 对 q+ readability predictors 的 n=24 分析均不确定：

- `Spearman(q+, log real frequency)=.202`
- `Spearman(q+, source count)=.257`
- `Spearman(q+, synthetic AUROC)=-.278`
- 置信区间均很宽。

冻结了 120 个候选 feature，但来源 exposure 极度不等，不能直接升级成昂贵 confirmatory benchmark。

#### 数据

- `results/n3_corpus_v1.jsonl`：约 65.4 MB
- `results/n3_corpus_v1.json`：corpus manifest
- `results/n3_feature_stats_v1.json`
- `results/n3_feature_stats_v1.npz`：16,384 feature 统计
- `results/n3_candidate_cohort_v1.json`
- `results/n3_contexts_v1.json`
- `results/n3_cohort_v1.json`
- `results/acts_L32_n3_v1.parquet`：200 rows
- `results/n3_analysis.json`

#### 脚本

`33_n3_build_corpus.py → 34_n3_feature_stats.py → 35_n3_resample_cohort.py → 36/37/38 QA`

#### 当前等级

“synthetic zero activation 是 coverage failure”是强诊断结论；readability predictor 和 120-feature cohort 仍是探索性。

### 4.15 2026-07-30 晚：N4，真实 content-token causal replication

#### 设计

- 200 rows / 101 documents；
- Pile 144、XNLI 56；
- 所有位置都是 natural content tokens；
- activation provenance bit-exact；
- identity KL=0；
- 2,301 base-model forwards；
- GPU约 30.4 分钟。

#### H1：text-channel localization

- `share(p3)=.932`：pass
- `share(p12)=.756`：未满足≤.50
- p3−p12 centered cosine：`+.1466`，CI `[.1274,.1663]`

形式上 H1 FAIL，但 p3 near-sufficient 的诊断很强；p12 仍保留大量几何信息。

#### H2：NLA vs SAE causal fidelity

预注册逐行 `1-KL_s/KL_zero` 因近零 denominator 病态，正式 verdict 只能是 FAIL/not established。

稳定的 ratio-of-sums sensitivity：

- NLA：`.94795`
- SAE-small：`.94417`
- SAE-big：`.96649`

NLA 在 80/101 文档上比 SAE-small 的 patched-position KL 更低，但少量 catastrophic failures 抹掉平均优势。对 SAE-big 没有优势。

#### H3：p3 causal utility

冻结状态是 NOT TESTABLE，因为主 endpoint 的 testability gate 被病态 denominator 破坏；但所有稳定 sensitivity endpoint 都支持 p3：

- p3 over p12 ratio-of-sums：`+.27865`
- raw KL(p12)−KL(p3)：`+3.333`
- p3/orig aggregate retention：`.99691`

这成为 N5 H5-B 的确认性前导证据。

#### 数据

- `results/n4_real_content_preregistration_v1.md`
- `results/n4_explanations_v1.json`
- `results/n4_variants_v1.json`
- `results/n4_av_checkpoint_v1.jsonl`
- `results/n4_recon_analysis_v1.json`
- `results/n4_recon_vectors_v1.npz`：200×3840及各文本变体
- `results/n4_causal_patch_v1.json`
- `results/n4_analysis_v1.json/.md`
- `results/N4_REAL_CONTENT_CAUSAL_AUDIT_2026-07-30.md`

本地 N4 artifact sidecar 通过；base model weight manifest 指向远端文件，当前无法本地重验模型权重本体。

#### 当前等级

N4 的 causal codec 数字是强探索性/事后稳定分析；不能改写成 NLA superiority。p3 机制证据后来由 N5/N6 正式确认。

### 4.16 2026-07-30–08-01：N5，selective hybrid 与 p3 replication

#### 设计

- 总计 600 independent content groups；
- discovery 200：Pile 150 / XNLI 50；
- held-out 400：Pile 300 / XNLI 100；
- discovery 只用来冻结 centered-cosine router threshold；
- held-out 禁止调参；
- v1 因 XNLI premise grouping 冲突在 tokenizer-only 阶段 abort；
- v2 在任何 model outcome 前修正并重新冻结。

冻结 threshold：`0.8361008239`。held-out 中 262/400 路由到 NLA，coverage `.655`。

#### H5-A：selective NLA + SAE-big

- `G=+.002419`
- one-sided 95% lower：`-.001326`
- raw gain：`+.032168 nat`
- 95% CI：`[-.026114,+.091261]`
- catastrophe：`9/400=2.25%`
- exact one-sided 95% upper：`3.893%`

gain lower bound 未大于 0，catastrophe upper bound 未低于 3%。

**正式裁决：`H5-A: NO SELECTIVE CLAIM`。**

这只否定冻结的单变量 centered-cosine router，不证明所有 router 都不可能成功。

#### H5-B：p3 channel

- `G_p3_p12=+.179490`
- 95% CI `[+.146504,+.215996]`
- raw `KL_p12-KL_p3=+2.386608 nat`
- 326/400 方向一致
- `R_orig=.964506`
- `R_p3=.952928`
- `T=.987996`
- one-sided lower `.983015`
- `quote_strip_p3 recovery=.402280`

**正式裁决：`H5-B: CHANNEL REPLICATED`。**

p3 是 causally dominant、near-sufficient，但不是唯一通道；p12 仍有 `.773438` aggregate recovery。

#### 数据

- 预注册：`results/n5_selective_hybrid_preregistration_v2.md/.sha256`
- cohort：`results/n5_cohort_plan_v2.json/.sha256`
- discovery activation：`results/acts_L32_n5_discovery_v2.parquet`
- held-out activation：`results/acts_L32_n5_heldout_v2.parquet`
- discovery：`n5_discovery_recon_v2.json`、`...vectors_v2.npz`、`n5_discovery_causal_v2.json`
- frozen gate：`results/n5_gate_v2.json`
- held-out：`n5_heldout_explanations_v2.json`、`n5_heldout_variants_v2.json`
- `n5_heldout_recon_v2.json`、`...vectors_v2.npz`
- `n5_heldout_causal_v2.json`
- `n5_analysis_v2.json`、`N5_ANALYSIS_V2.md`
- `n5_independent_audit_v2.json/.sha256`

所有本地 N5 artifact sidecar 通过；模型权重 manifest 指向远端文件，本地只验证了 manifest-of-manifest。

#### 当前等级

H5-A 是确认性 negative；H5-B 是确认性 positive。

### 4.17 2026-08-03：N6+，matched candidate-content substitution

#### 核心问题

N5 已证明 p3 quoted content 很重要，但它可能只利用 candidate-list 格式、词汇密度或熟悉表达。N6 直接问：

> 必须是当前 recipient 自己的 candidates，还是换成同格式、同来源、同数量的错误 candidates 也一样？

#### 冻结设计

- provisional rows：501；
- AV-format eligible：434，约 86.63%；
- final analysis：400 rows / 400 groups / 400 documents；
- 13 个 Pile sources；
- candidate count 4/5/6：105/192/103；
- donor 在 `(source,candidate_count)` cell 内 one-to-one derangement；
- recipient/donor 文档、group、row 不同；
- 无相同规范化 candidate 字符串；
- N4/N5 document/group/20-word shingles 全部 embargo；
- 50,000 shared bootstrap draws，`PCG64(20260803)`。

主要 conditions：

- `orig`
- `p3_true`
- `p3_cross_matched`
- `p3_candidate_strip`
- `p3_anchor_strip`
- `p3_all_quote_strip`
- `p12`
- `sae_big`
- `identity`
- `zero`

#### H6-A：sample-specific causal channel

- `G_specific=+.117954`
- 95% CI `[+.102860,+.133995]`
- raw `KL_cross-KL_true=+1.827883 nat/row`
- CI `[+1.591758,+2.074946]`
- `G_content=+.154382`
- CI `[+.136602,+.173507]`
- `T_p3=.995175`
- one-sided lower `.993194`

所有冻结 gate 通过：

**`H6-A: SAMPLE-SPECIFIC CHANNEL CONFIRMED`**

#### H6-B：predictive alignment

- `A_meanmass=+9.008470 log units`
- CI `[+8.444855,+9.572597]`
- true/cross candidate first-token hit@1：`66.5% / 3.75%`
- observed next token membership：`49.0% / 2.75%`

**`H6-B: PREDICTIVE ALIGNMENT CONFIRMED`**

它说明 true candidate first-token set 与 clean next-token distribution 强对齐，不代表完整 continuation 概率或 proposition truth。

#### secondary nuance

- `M_majority=+.040763`，CI `[+.028680,+.053252]`：recipient-specific identity 占 candidate-content gain 的显著多数；
- `G_candidate_anchor=-.045042`，CI `[-.078669,-.012779]`：不支持 candidate dominance。

Aggregate recovery：

| condition | recovery |
|---|---:|
| identity | 1.000000 |
| orig | .979493 |
| SAE-big | .978140 |
| p3_true | .974768 |
| p3_cross_matched | .856814 |
| p3_candidate_strip | .820385 |
| p12 | .795530 |
| p3_anchor_strip | .775343 |
| p3_all_quote_strip | .367074 |
| zero | 0 |

anchors 与 sample-specific candidates 是互补、非线性组合；不能把这些 ablation recovery 机械解释成可加的信息百分比。

#### 数据

唯一 canonical 本地目录：

`results/n6_pull_staging/n6_pull_20260803T061302Z/`

其中最关键的是：

- `n6_plus_preregistration_v2.md/.sha256`
- `n6_provisional_cohort_v1.json`
- `n6_av_checkpoint_v1.jsonl`
- `n6_av_explanations_v1.json`
- `n6_variants_donor_v1.json`
- `n6_recon_v1.json`
- `n6_recon_vectors_v1.npz`：400×3840及各变体
- `n6_causal_candidate_mass_v1.json`
- `n6_analysis_v1.json`
- `n6_independent_audit_v1.json`
- `N6_ANALYSIS_V1.md`
- code/model manifest、resource report、logs、exit markers 和 hash sidecars

`results/N6_FINAL_ANALYSIS_2026-08-03.md` 是 paper-facing summary；`results/n6_live_analysis_v1.json` 是重复/临时汇总，不应替代 staging frozen JSON。

Stage-50 activation parquet 没有被 N6 pull allowlist 拉回本地，最后记录仍在远端 persistent `activations/`；它不是 stage55/56 audit 的必需输入，但属于尚未本地归档的 upstream cache。

#### 运行与完整性

- supervisor wall time：`1:54:51`
- AV generation：约 58.4 分钟
- causal：4,400 forwards，约 393 秒
- stage49–56 与 supervisor exits 均为 0
- 15 个本地 SHA-256 sidecars 审计通过
- independent audit 的 56 个 numeric leaves 与主分析在 `1e-12` 内一致

#### 当前等级

N6 H6-A/H6-B 为当前最强 confirmatory positive。它不建立 NLA 对 SAE-big 的 superiority/equivalence，不建立 human-faithful explanation，也不建立 router 或 steering。

### 4.18 2026-08-05：27B 可行性审计

这不是新科学实验，而是下一阶段资产核查。

已确认：

- NLA AV：`kitft/nla-gemma3-27b-L41-av`
- NLA AR：`kitft/nla-gemma3-27b-L41-ar`
- extraction：Gemma-3-27B-IT block 41，`d_model=5376`
- Gemma Scope 2：`google/gemma-scope-2-27b-it`
- 严格层对齐 SAE：
  - `resid_post_all/layer_41_width_16k_l0_small`
  - `resid_post_all/layer_41_width_16k_l0_big`
  - 另有 262k small/big，可留给后续 capacity curve。

本地只有 27B datagen configs 和可复用代码，没有权重。

工程约束：

- AV 仓库约 108 GB、页面标记 F32；
- AR 约 38 GB；
- base BF16 约 54 GB；
- 还需 selected SAE、HF cache 和中间文件；
- 现有约 81 GB 余量不够，建议至少 250–350 GB 可用盘；
- 不得下载整个 Gemma Scope 2 27B repo，它是 TB 级，只取 L41 指定目录；
- 一张 80 GB GPU 可能通过 load-time BF16 cast、batch 1 和顺序加载完成，但必须先做 smoke；96 GB 更稳妥。

研究裁决：12B 保留为论文主线，27B 作为 N7 confirmatory replication，而不是整体迁移。

---

## 5. 数据资产总账

本节不重复所有日志，而是列出每个阶段真正需要保留的 raw/frozen/intermediate/analysis/audit 文件。

### 5.1 Legacy E1–E7 / B2

| 文件 | 内容与形状 | 地位 |
|---|---|---|
| `results/nla_results.json` | 40 行 explanation、token、position、NLA cos/MSE | E1 原始 NLA 输出 |
| `results/sae_results.json` | 40 行 SAE-small + summary | E1 原始 SAE |
| `results/sae_results_big.json` | 40 行 SAE-big + summary | E1 原始 SAE |
| `results/comparison.json/.md` | 对齐后的 baseline 汇总 | 历史 summary |
| `results/wdec_pilot.json` | 22 个 decoder/Gaussian 方向 | E2 raw+summary |
| `results/resid_pilot.json` | 40 个 residual probes | E3 raw+summary |
| `results/injection_pilot.json` | 80 个 injection rows | E4 raw+summary |
| `results/centered_rescore.json` | centered E1–E4 结果 | E5 canonical analysis |
| `results/recon_vectors.npz` | `x/pred_full/pred_resid/recon_sae_*=(40,3840)`；`feature_dirs=(4,3840)` | B2/N2 上游向量 |
| `results/retrieval_eval.json` | 40-way matrices、50k permutations、Top-k/MRR/margin | B2 canonical |

限制：

- 多数 legacy 文件没有 `.sha256` sidecar；
- 40 行有 13 个 template/blank token；
- `pred_resid` 不是 fixed generic floor；
- `server/02_extract_activations.py` 的旧 fallback 不应复用为正式 content-token cohort。

### 5.2 E8 / B6+B4

| 文件 | 内容与形状 | 地位 |
|---|---|---|
| `results/acts_L32_factorial_v1.parquet` | 1,365 个 L32 activations | raw activations |
| `results/b6b4_factorial_selection.json` | train-only frozen selection、45 directions | frozen design |
| `results/b6b4_factorial_feature_stats.npz` | 16,384 features；24×16,384 doc stats | upstream stats |
| `results/b6b4_factorial_vectors.npz` | 1,365×3,840 activations、45×3,840 directions | vectors |
| `results/b6b4_factorial_av_rows.jsonl` | 590 个有效 AV rows | raw generated text |
| `results/b6b4_factorial_result.json` | 590 scored rows、polarity/carrier results | primary result |
| `results/b6b4_factorial_recon_vectors.npz` | 590×3,840 input/reconstruction | audit vectors |
| `results/b6b4_factorial_analysis.json/.md` | 分层分析 | analysis |
| `results/b6b4_surface_audit.md` | post-hoc surface label audit | exploratory audit |

不要使用：

- `b6b4_factorial_av_rows.partial.jsonl`
- `debug_b6b4_result.json`
- `debug_b6b4_vectors.npz`

当前代码漂移：`b6b4_checksums.sha256` 中运行时的 `11_extract_factorial_activations.py` hash 与当前文件不符。

### 5.3 E9/C1

#### C1 pilot canonical

- `results/c1_pilot_benchmark.json`
- `results/c1_pilot_checkpoint_v2.jsonl`
- `results/c1_pilot_result.json`
- `results/c1_pilot_recon_vectors.npz`
- `results/c1_pilot_analysis.json/.md`
- `results/c1_pilot_secondary_analysis.json/.md`
- `results/c1_pilot_validation.json`
- `results/c1_pilot_checksums.sha256`

#### C1-confirmatory protocol failure canonical

- `activations/c1_confirmatory_all_v2.jsonl`
- `activations/c1_confirmatory_discovery_v2.jsonl`
- `activations/c1_confirmatory_heldout_v2.jsonl`
- `results/c1_confirmatory_corpus_report_v2.json`
- `results/c1_confirmatory_manual_audit_v1.json`
- `results/c1_confirmatory_scenario_anchor_audit_v3r2.json`
- `results/c1_confirmatory_scenario_anchor_v3r2_failure.md`

这里的 `activations/*.jsonl` 实际是 synthetic text manifests，不是 L32 activation tensors。因为 audit FAIL，后续真正 activation/evaluator 数据不存在。

### 5.4 F14 / 独立复算

- `results/local_recheck_opus.json`
- `results/local_recheck_b6b4_opus.json`
- `results/local_recheck_stratified_opus.json`
- `results/REVIEW_OPUS_2026-07-30.md`

F14 是对已有 frozen assets 的 derived audit，没有独立的 `f14_raw.*`。

### 5.5 N1/N2

| 文件 | 作用 |
|---|---|
| `results/c7b3_variants_v1.json/.sha256` | 冻结 40 行文本变体 |
| `results/c7b3_scores_v1.json` | AR scores |
| `results/c7b3_recon_v1.npz` | variant reconstructions |
| `results/causal_patch_v1.json` | 40 行 causal patch raw outcomes |
| `results/n1n2_analysis.json` | 当前 N1/N2 稳健汇总 |
| `results/n1n2_checksums.sha256` | 9 个文件的完整性清单 |

注意旧 `server/run_n1n2.sh` 的 N2 参数指向 legacy `recon_vectors.npz`，而不是前一步生成的 `c7b3_recon_v1.npz`；未来复现必须核对 NPZ keys，不能假设 29→30 是直接输出输入关系。

### 5.6 N3

- `results/n3_corpus_v1.jsonl`：10,404 valid lines，约 65.4 MB；
- `results/n3_corpus_v1.json`：manifest；
- `results/n3_feature_stats_v1.json/.npz`：8,240,945 tokens、16,384 features；
- `results/n3_candidate_cohort_v1.json`：120 exploratory candidates；
- `results/n3_contexts_v1.json`：真实 max-activating contexts；
- `results/n3_cohort_v1.json`；
- `results/acts_L32_n3_v1.parquet`：200 clean rows；
- `results/n3_analysis.json`。

部分 N3 文件没有独立 sidecar，只在 metadata 内嵌 corpus/parquet hash。

### 5.7 N4

- `results/n4_real_content_preregistration_v1.md`
- `results/n4_explanations_v1.json`
- `results/n4_variants_v1.json`
- `results/n4_av_checkpoint_v1.jsonl`
- `results/n4_recon_analysis_v1.json`
- `results/n4_recon_vectors_v1.npz`
- `results/n4_causal_patch_v1.json`
- `results/n4_analysis_v1.json/.md`
- `results/N4_REAL_CONTENT_CAUSAL_AUDIT_2026-07-30.md`

本地数据 sidecars 通过；`n4_base_model_full_weights.sha256` 中的远端权重路径当前不能在本地重验。

### 5.8 N5

最小 source-of-truth：

1. `results/n5_selective_hybrid_preregistration_v2.md`
2. `results/n5_cohort_plan_v2.json`
3. discovery/heldout activation Parquet；
4. discovery/heldout explanation、variant、recon JSON/NPZ；
5. discovery/heldout causal JSON；
6. `results/n5_gate_v2.json`
7. `results/n5_analysis_v2.json`
8. `results/n5_independent_audit_v2.json`

本地 artifact sidecars 通过；实际模型权重只在远端 manifest 中记录。

`n5_v1_*abort*` 是正式的 protocol abort 历史，不能与 v2 结果混用。部分 `.exit` 文件与最终日志状态不直观，判定应以 JSON status、完整 artifact 和 independent audit 为准。

### 5.9 N6

只采信：

`results/n6_pull_staging/n6_pull_20260803T061302Z/`

并以：

`results/n6_pull_staging/LATEST_VERIFIED_PULL.txt`

确认该目录是 verified pull。

核心 frozen chain：

`provisional cohort → AV checkpoint/explanations → variants/donor → recon JSON/NPZ → causal candidate mass → analysis → independent audit`

根目录的 `n6_live_analysis_v1.json` 和 launcher logs 只作运行历史；launcher 曾遇到 SSH reset/hash parser warning，但 staged payload 的 hash、stage logs 和 audit 均通过。

---

## 6. 主项目脚本目录：逐项作用

以下绝对路径前缀均为：

`D:\Projects\natural_language_autoencoder\server\`

### 6.1 基础 pipeline：00–05

| 脚本 | 计算 | 作用 | 当前状态 |
|---|---|---|---|
| `00_setup_env.sh` | CPU/网络 | 安装依赖、建立远端目录/HF cache；完整模式会安装 SGLang | 可用但会改变环境版本，谨慎 |
| `01_download.py` | 网络 | 只下载 base、AV、AR 和指定 SAE 文件，避免拉 TB 级仓库 | 可用 |
| `02_extract_activations.py` | GPU | 用 Gemma 抓 L32 raw activations → Parquet | 旧默认短 prompt fallback 有 F13 缺陷 |
| `03_run_nla.py` | GPU | activation → AV explanation → AR reconstruction → JSON | 可复现 baseline |
| `04_run_sae.py` | GPU | 直接加载 JumpReLU params，做 SAE encode/decode/FVE/L0 | 可复现 |
| `05_compare.py` | CPU | 按 doc/position 合并 NLA/SAE → JSON/MD | 可重算 |

辅助入口：

- `run_downloads.sh`：旧环境专用的 base 下载重试；
- `launch_av_server.sh`：SGLang AV server；
- `run_pipeline.sh`：等待下载 → 02 → 03 → 04 small/big → 05；
- 当前 `run_pipeline.sh` 无 shutdown，但旧 `pipeline.log` 来自不同版本，不能反推现行源码。

### 6.2 Legacy pilots 与 centered/B2：06–10

| 脚本 | 计算 | 作用 |
|---|---|---|
| `pilot_common.py` | helper | AVLocal、NLACritic、JumpReLU、load_acts 公共组件 |
| `06_pilot_wdec.py` | GPU | decoder directions → AV/AR |
| `07_pilot_residual.py` | GPU | SAE residual → AV/AR |
| `08_pilot_injection.py` | GPU | residual + α·feature direction 检测曲线 |
| `09_rescore_centered.py` | GPU/AR | 全部旧 explanation 去均值重算；保存向量 |
| `10_retrieval_eval.py` | CPU | 40-way centered retrieval、置换、margin |

Wrappers：

- `run_pilots.sh`：06→07；
- `run_injection.sh`：08；
- `run_rescore.sh`：09。

这些脚本没有关机逻辑，但其数据属于 legacy exploratory cohort。

### 6.3 B6+B4：11–14

| 脚本 | 计算 | 作用 |
|---|---|---|
| `11_extract_factorial_activations.py` | GPU | factorial JSONL → 1,365 L32 activations |
| `12_select_factorial_features.py` | GPU/SAE | train-only document-balanced feature selection；冻结 45 directions |
| `13_probe_factorial_polarity.py` | GPU | ±directions、stochastic generations、carrier interventions、AR/retrieval |
| `14_analyze_factorial_results.py` | CPU | ITT、heldout、domain/language、carrier 分层报告 |

`run_b6b4_factorial.sh` 主要运行 13，假设 11/12 已冻结。当前 11 号代码与运行 hash 漂移，不能改后重称原实验。

### 6.4 C1 pilot：15–19

| 脚本 | 计算 | 作用 |
|---|---|---|
| `15_build_c1_pilot.py` | CPU | 冻结 benchmark、correct/hard-negative/generic candidates |
| `16_run_c1_pilot.py` | GPU | base autointerp/paraphrase/judge + AR scoring |
| `17_analyze_c1_pilot.py` | CPU | 24-feature cluster bootstrap、主 contrast |
| `18_validate_c1_pilot.py` | CPU | 独立结构与数值 QA |
| `19_analyze_c1_secondary.py` | CPU | post-hoc retrieval/AUC/private-code diagnostics |

Wrappers：

- `run_c1_pilot.sh`
- `launch_c1_pilot.sh`
- `start_c1_pilot.sh`

均无关机逻辑；只应复核 frozen pilot，不应用它们构造“全新确认性结果”。

### 6.5 C1-confirmatory：20–24

20 号存在多个版本，因为语料设计不断修订：

| 脚本 | 作用 | 当前状态 |
|---|---|---|
| `20_validate_c1_inputs.py` | tokenizer/spec/denylist/prereg preflight | 可做静态 QA |
| `20_generate_c1_confirmatory_corpus.py` | v1 synthetic generation | failed/stale |
| `20_generate_c1_confirmatory_corpus_v2.py` | v2 144-doc generation | 已生成但被审计拒绝 |
| `20_make_c1_scenario_anchors_v3r2.py` | 构建 v3r2 anchors | 审计前置 |
| `20_make_c1_v3r2_audit_addendum.py` | 绑定 v3r2 addendum | 审计前置 |
| `20_freeze_c1_v3_inputs.py` | 冻结 v3 输入和 hashes | 未进入科学运行 |
| `20_aggregate_c1_manual_audits.py` | 合并 reviewer A/B | v2 FAIL |
| `20_aggregate_c1_manual_audits_v3.py` | 聚合 v3/v3r2 audit | FAIL |
| `21_validate_c1_activation_provenance.py` | activation manifest provenance | 下游未运行 |
| `21_select_c1_confirmatory_features.py` | discovery-only SAE feature selection | 未运行正式 v3 |
| `22_build_c1_confirmatory_benchmark.py` | 冻结 references/hard negatives | 未运行正式 v3 |
| `23_run_c1_confirmatory.py` | heldout AV/AR/SAE/evaluator | **从未执行** |
| `24_analyze_c1_confirmatory.py` | pair-cluster confirmatory analysis | **无输入、未执行** |

Wrappers：

- `run_c1_confirmatory_corpus.sh`
- `run_c1_confirmatory_corpus_v2.sh`
- `run_c1_confirmatory_corpus_v3.sh`
- `run_c1_confirmatory_discovery_selection_v2.sh`

v3r2 gate 已规定停止，不能直接重启并仍称同一 confirmatory run。

### 6.6 独立复核：25–27

| 脚本 | 作用 |
|---|---|
| `25_local_recheck.py` | 重算 E1–E7/B2，增加 nearest-other 与 centered FVE |
| `26_local_recheck_b6b4.py` | per-direction generic floor、q+ 分布与 surface 对照 |
| `27_local_recheck_stratified.py` | domain/language 分层 Spearman、dead/tied AUC 诊断 |

全部 CPU-only，可从 frozen assets 重跑。

### 6.7 N1/N2：28–32

| 脚本 | 计算 | 作用 |
|---|---|---|
| `28_build_text_variants.py` | CPU | 冻结 C7/B3/p1-p3/quote/word-shuffle variants |
| `29_score_text_variants.py` | GPU/AR | AR-only 重建与 centered score |
| `30_causal_patch.py` | GPU/base | 将 12 类向量 patch 回 L32，测 KL/CE |
| `31_analyze_n1n2.py` | CPU | 稳定 KL ratio-of-sums、paired/bootstrap汇总 |
| `32_diag_ce_window.py` | CPU | 诊断 CE window 被 template tokens 支配 |

`run_n1n2.sh` 可作历史 wrapper，但存在 NPZ 路径语义陷阱，并且需要看内部 `N1/N2_EXIT`，不能只信 shell 最后 exit。

### 6.8 N3：33–38

| 脚本 | 计算 | 作用 |
|---|---|---|
| `33_n3_build_corpus.py` | CPU/网络 | 冻结 Pile+XNLI real corpus |
| `34_n3_feature_stats.py` | GPU | 8.24M tokens，small/big 16k SAE 全 feature stats |
| `35_n3_resample_cohort.py` | GPU | 无 F13 fallback 的 200-row content cohort |
| `36_n3_dump_contexts.py` | CPU | 解码真实 max contexts，冻结 candidate cohort |
| `37_n3_analyze.py` | CPU | coverage 与 readability predictor 分析 |
| `38_n3_inspect.py` | CPU | 人类可读 stdout dump |

Wrappers/preflight：

- `run_n3.sh`
- `run_n3_smoke.sh`
- `n3_env_probe.sh`
- `n3_probe_corpora.sh`
- `n3_probe_parallel.sh`
- `n3_probe_xnli.sh`

### 6.9 N4：39–41

| 脚本 | 计算 | 作用 |
|---|---|---|
| `39_n4_real_recon.py` | GPU | 200-row AV explanation、文本 variants、AR+SAE recon |
| `40_n4_causal_patch.py` | GPU/base | exact-token L32 causal patch |
| `41_n4_analyze.py` | CPU | document-clustered prereg analysis |

Wrappers：

- `run_n4_recon.sh`
- `run_n4_causal.sh`

都做输入 hash/refuse-overwrite；N4 frozen outputs 不应覆盖。

### 6.10 N5：42–48

| 脚本 | 计算 | 作用 |
|---|---|---|
| `42_n5_freeze_cohort.py` | CPU/tokenizer | 冻结600-group plan与embargo |
| `42_n5_freeze_cohort_v1_aborted.py` | CPU | v1历史版本，已abort |
| `43_n5_extract_activations.py` | GPU/base | 按 immutable plan 抽 discovery/heldout L32 |
| `44_n5_reconstruct.py` | GPU | AV/AR + SAE small/big + paragraph variants |
| `45_n5_causal_patch.py` | GPU/base | discovery/heldout exact causal outcomes |
| `46_n5_freeze_gate.py` | CPU | 只用 discovery 冻结 centered-cosine threshold |
| `47_n5_analyze.py` | CPU | heldout H5-A/H5-B |
| `48_n5_independent_audit.py` | CPU | 不导入47，从 raw artifacts独立复算 |

Remote wrappers 分 discovery/heldout：

- `n5_freeze_cohort_remote.sh`
- `n5_discovery_extract_remote.sh`
- `n5_discovery_reconstruct_remote.sh`
- `n5_discovery_causal_remote.sh`
- `n5_freeze_gate_remote.sh`
- `n5_heldout_extract_remote.sh`
- `n5_heldout_reconstruct_remote.sh`
- `n5_heldout_causal_remote.sh`
- `n5_analyze_remote.sh`
- `n5_pipeline_after_recon_v2.sh`
- `n5_verify_model_manifest.sh`
- `n5_audit_xnli_shingle_conflict.py`

**危险/过时：**

`n5_resume_heldout_and_shutdown.sh` 在 exit 路径调用 `systemctl poweroff`。该容器不以 systemd 为 PID1，历史上产生过“日志显示关机、实例实际仍在线”的事故；不要再运行。

### 6.11 N6：49–56

| 脚本 | 计算 | 作用 |
|---|---|---|
| `49_n6_freeze_cohort.py` | CPU/tokenizer | Pile-only provisional plan、embargo、quota |
| `50_n6_extract_activations.py` | GPU/base | immutable plan → L32 activation parquet |
| `51_n6_generate_av.py` | GPU/AV | 501 provisional rows greedy AV，append checkpoint |
| `52_n6_freeze_variants.py` | CPU/text | parser gate、final400、matched donor derangement、byte audit |
| `53_n6_reconstruct.py` | GPU/AR+SAE | frozen variants → AR + SAE-big recon |
| `54_n6_causal_patch.py` | GPU/base | 10 substitutes、candidate-mass diagnostics、4,400 forwards |
| `55_n6_analyze.py` | CPU | H6-A/H6-B endpoints 与 labels |
| `56_n6_independent_audit.py` | CPU | 独立复算56 leaves和正式 decisions |
| `n6_common.py` | helper | hash/provenance/mechanical contracts，不定义 scientific endpoints |

Remote orchestration：

- `n6_stage_common.sh`
- `n6_freeze_cohort_remote.sh`
- `n6_extract_activations_remote.sh`
- `n6_generate_av_remote.sh`
- `n6_freeze_variants_donor_remote.sh`
- `n6_reconstruct_remote.sh`
- `n6_causal_candidate_mass_remote.sh`
- `n6_analyze_remote.sh`
- `n6_independent_audit_remote.sh`
- `n6_supervisor_template.sh`
- `n6_launch_monitor_pull.ps1`

N6 已完成，不应重跑。

运维风险：

- `n6_supervisor_template.sh` 的 EXIT trap 无论成功/失败都会 `sync; /usr/bin/shutdown -h now`；
- `n6_launch_monitor_pull.ps1` 默认 dry-run，只有 `-Execute` 才远程运行；正常路径不直接关机，`-EmergencyShutdownOnFailure` 才会额外关机；
- monitor 曾遇到短暂 SSH reset；
- pull-ready parser 对带数字的 key 有 bug；未来应新建版本和新 manifest 修复，不能修改 frozen N6 v2 后仍声称原 hash 有效。

### 6.12 项目根目录的本地运维/探测脚本

| 文件 | 作用 | 状态 |
|---|---|---|
| `connect.ps1` | 读取本机 SSH config 的 `autodl` alias，进入交互 shell或执行单条命令 | 可用，实际端点由SSH config决定 |
| `sync.ps1` | 在本地 `server/`、远端代码目录和 `results/` 间 push/pull | 可用，但大规模/frozen pull优先专用manifest |
| `remote.py` | Paramiko 辅助入口，用 SSH config 执行远端命令 | 可用 |
| `_probe_hf.py` | Hugging Face模型/文件探测 | 临时只读探测 |
| `_probe_hf.sh`、`_probe2.sh` | 远端/HF shell探测 | 临时脚本，不参与科学结果 |
| `results/nla_inference.remote_20260803.py` | N6运行时远端 `nla_inference.py` 快照 | 归档证据，不是默认导入位置 |

这些脚本只负责操作或探测，不能代替 preregistration、manifest 或结果审计。

---

## 7. 上游 `natural_language_autoencoders` 脚本地图

以下路径前缀：

`D:\Projects\nla-from-autodl\natural_language_autoencoders\`

### 7.1 推理/演示

| 脚本 | 作用 |
|---|---|
| `nla_inference.py` | standalone `NLAClient` + `NLACritic`；主项目03、pilots、N5/N6的关键依赖 |
| `demo/launch_av_server.sh` | 启 SGLang AV server |
| `demo/make_parquet.py` | base model → per-token activation parquet |
| `demo/roundtrip.py` | parquet → AV → AR → JSON |
| `demo/answer_probe.py` | 生成答案并抓 question/answer/full-context activations |
| `demo/answer_probe_tui.py` | answer_probe 交互终端前端 |

### 7.2 数据生成

推荐唯一总入口：

`python -m nla.datagen.run_pipeline --config <yaml>`

内部阶段：

| 脚本 | 作用 |
|---|---|
| `nla/datagen/stage0_extract.py` | GPU 抽 raw activation parquet + sidecar |
| `nla/datagen/merge_base.py` | 合并多GPU stage0 shards |
| `nla/datagen/stage1_split.py` | document-level 切分 AV-SFT/AR-SFT/RL |
| `nla/datagen/stage2_api_explain.py` | 调外部 provider 生成 explanation |
| `nla/datagen/stage2_join.py` | API-free join已有 explained cache |
| `nla/datagen/stage3_build.py` | 构造最终训练 parquets，保持 raw vectors |
| `nla/datagen/stage_shuffle.py` | deterministic row shuffle |
| `nla/datagen/shuffle_activations.py` | activation-only signal ablation |
| `nla/datagen/recover_explained.py` | 从 stage3 恢复 explanation cache |
| `nla/datagen/cast_to_fixed_size_list.py` | legacy parquet一次性迁移 |
| `scripts/datagen/stage0_multigpu.sh` | 多GPU stage0 分片+merge |

`configs/datagen/` 下有 Gemma-12B、Gemma-27B、Qwen、Llama 等 YAML。它们是新 checkpoint 的 datagen 配置，不是当前比较实验结果。

### 7.3 训练

| 脚本 | 作用 |
|---|---|
| `configs/critic_sft.sh` | AR/critic SFT |
| `configs/actor_sft.sh` | AV/actor SFT |
| `configs/rl.sh` | AV GRPO + AR supervised update |
| `nla/scripts/prepare_critic_checkpoint.py` | base HF → K+1-layer AR初始化 |
| `nla/scripts/rl_preflight.py` | reward/train/injection scale 全链烟测 |
| `nla/scripts/extract_rollout_samples.py` | rollout checkpoint → 可读报告 |
| `nla/scripts/fetch_parquet.py` | remote storage → local cache |
| `nla/scripts/push_checkpoint.py` | 断点安全上传 |
| `nla/scripts/pull_checkpoint.py` | manifest驱动恢复 |

这些默认参数主要对应 released Qwen7B 训练，不可直接当 Gemma-12B comparison runner。

### 7.4 转换与发布

`tools/` 包含：

- `convert_fsdp_to_hf.py`
- `convert_torch_dist_to_hf.py`
- `convert_hf_to_torch_dist.py`
- `convert_to_hf.py`
- `convert_hf_to_fp8.py`
- `convert_hf_to_mxfp8.py`
- `convert_hf_to_hf_int4.py`
- `fp8_cast_bf16.py`

`release/` 包含 checkpoint staging、scrub 和 model-card render。这些不参与现有科学结果，只在新训练/发布 checkpoint 时使用。

---

## 8. Fable 最初提出的 B/C 方向：完成与待办

| 方向 | 原问题 | 当前状态 | 现在如何理解 |
|---|---|---|---|
| B1 | 去均值/白化重跑 | ✅ E5 | 已成为强制评测纪律 |
| B2 | 40-way retrieval | ✅ E7 | Top-k 未胜 SAE；NLA margin 更大 |
| B3 | 实体替换敏感性 | ✅ N1 | 实体替换保留89.7%；实体身份不是主要通道 |
| B4 | ±方向极性恢复 | ✅ E8 | 正方向有异质信号，负方向弱；不能叫强双向语义 |
| B5 | 文本长度/bit budget rate-distortion | ❌ 未做 | 仍是公平比较 NLA/SAE 的重要缺口 |
| B6 | 多样语料重测 semantic features | ✅ E8，后由N3修正 | E8语料仍太小；N3证明coverage问题 |
| B7 | 多次生成自一致性作为置信度 | 部分E8随机重复，未形成正式router | N5证明简单cos router不安全 |
| C1 | AR评测 SAE feature labels/autointerp | pilot完成；confirmatory abort | 问题仍有价值，synthetic corpus路线失败 |
| C2 | NLA辅助 feature labeling pipeline | ❌ 未运行 | 只能作为 hypothesis generation/triage，不可自动验真 |
| C3 | NLA+SAE hybrid codec | 只做N4/N5 post-hoc/selective router | frozen router失败，混合架构尚未成立 |
| C4 | 相互残差迭代剥离 | ❌ 未做 | 因当前NLA off-manifold失败而降优先级 |
| C5 | hidden-signal audit工作点 | ✅ E4，负结果 | 当前AV对SAE residual注入不敏感 |
| C6 | model diffing / emergent misalignment | ❌ 未做 | 属于新项目，不应挤占当前论文 |
| C7 | AV/AR private-code风险 | ✅ N1 narrow test | 长度匹配第三方改写保留97.3%，降低狭义私有字符串风险；不等于完成人类忠实验证 |

另外，`Conclusion_complete.md` 中的 V1–V4（belief/strategy ablation、causal edit、position comparison、cross-model scaling）均是 2026-08-03 N6 前的路线草案，尚未执行。N6 已先回答 candidate identity/content，因此这些设想应重新排到 27B replication 和 paper consolidation 之后。

---

## 9. 当前正式 claim table

### 9.1 可以说

1. NLA 在旧40行队列的 centered geometry 上优于两只冻结 SAE；该结果是探索性的。
2. 真实语料证明旧 synthetic zero activation 是 coverage failure。
3. N5 在 fresh held-out cohort 上确认 p3 是 causally dominant、near-sufficient channel。
4. N6 确认 p3 candidate interiors 携带 recipient-specific incremental causal information。
5. true candidate first-token sets 与 clean next-token distribution 强对齐。
6. anchors 与 candidates 互补，candidate content 不是唯一或已证明的主导成分。
7. frozen centered-cosine router 没有建立 selective improvement/safe parity。
8. geometry、closed-loop decodability、causal fidelity 和 tail safety 必须分别报告。

### 9.2 不能说

- NLA 全面优于 SAE；
- NLA 是更好的 causal codec；
- NLA 与 SAE-big 已等价或可安全替代；
- N5 hybrid 已经成功；
- p3 是唯一通道；
- candidates 比 anchors 更重要；
- candidate first-token alignment 等于完整 continuation 或命题正确；
- AR round-trip 证明 SAE label 正确；
- C1 科学假设失败；
- synthetic non-activation 证明 feature dead；
- 已经实现 steering；
- 结论已跨层、跨模型或扩展到所有 activation positions；
- 现有文本已经通过 human proposition-level fidelity 验证。

---

## 10. 当前项目状况

### 10.1 科学状态

项目是**谨慎乐观**的：

- 原始的 NLA↔SAE 双向辅助目标尚未被直接、完整检验；
- 早期为校准协同工具而做的比较否定了“NLA 全面胜 SAE”这一过宽解读，但这不是
  对原始研究目标的否定；
- N6 给出了更窄、更有机制含量且预注册确认的正结果，证明 NLA 文本具有成为
  辅助信号所需的一项关键因果属性；
- N5 的 confirmatory negative 不是失败资产，而是论文方法论贡献的一部分；
- N5 同时说明简单 centered-cosine router 还不足以把 NLA 安全接入 SAE；
- 现在的证据为联合 Mech Interp 奠定了前提，但还不是联合方法本身的成功证明。

仅就 N1–N6 **现有数据能够支持的论文**，最稳的中心是：

> 自然语言 activation reconstruction 可以包含真实、样本特异且因果有效的 predictive content，但其内部几何分数不足以提供逐样本可靠性或人类语义忠实证书。

这条 paper-facing 结论不替代项目级的 NLA→SAE、SAE→NLA 和联合闭环目标。

### 10.2 写作状态

已有：

- claim table；
- N4/N5/N6 paper-facing reports；
- frozen JSON/NPZ；
- independent audits；
- preregistrations、hashes、resource reports。

尚无：

- 正式 paper manuscript；
- Methods/Results/Limitations 定稿；
- 论文主图；
- artifact appendix；
- reproducible release bundle；
- Git history。

### 10.3 计算与服务器状态

最新本地记录表明：

- N6 已完成、拉回、hash verify；
- 2026-08-03 已发送 `sync; /usr/bin/shutdown -h now`；
- 随后 SSH 两次不可连接；
- 但 AutoDL 控制面状态从未通过本地文件独立证明。

因此本文件不假设 2026-08-06 的实例开关状态；以用户当前 AutoDL 控制台为准。不要为了确认而无目的开机。

### 10.4 复现状态

最完整的是 N5/N6：

- 有 preregistration；
- frozen manifests；
- raw outcomes；
- hash sidecars；
- independent audit。

主要缺口：

1. 项目无 Git；
2. B2/N3 一部分文件缺 sidecar；
3. B6运行脚本发生hash漂移；
4. N4/N5实际模型权重只在远端manifest；
5. N6 Stage-50 activation parquet未拉回；
6. C1-confirmatory没有模型 outcome；
7. C2不存在；
8. 旧 E6 手工文本审计无独立脚本/标注表。

---

## 11. 未来路线：按信息增益排序

### P0：现在立刻做，CPU-only

#### 1. 写论文

优先完成：

- Abstract/Introduction；
- Methods：geometry/text/causal/tail 四层评测；
- N4–N6 Results；
- N5 router negative；
- Limitations；
- artifact/hash appendix。

建议四张主图：

1. N6 true vs cross matched paired causal effect；
2. p3 recovery decomposition；
3. geometry vs causal mismatch；
4. source/candidate-count robustness + router tail。

#### 2. CPU-only tail forensics

将 N4/N5/N6 只当 discovery data，分析 catastrophe 的 pre-outcome predictors：

- centered cosine；
- activation norm；
- base next-token entropy；
- p3 长度/quoted spans/candidate count；
- NLA–SAE disagreement；
- parser/format anomalies；
- source folds。

固定 tail label，例如 `KL_NLA-KL_SAEbig>1 nat`，做：

- grouped/leave-source-out CV；
- AUPRC；
- calibration；
- coverage-risk curve。

只有预测器跨 source 稳定，才值得新 held-out router confirmation；不得用 N4/N5 事后宣布新 router 成功。

#### 3. 复现卫生

- 将当前目录做只读快照，再初始化 Git；
- 保存 N6 activation parquet 和 metadata；
- 归档实际 model-weight hashes；
- 给 B2/N3补独立hash清单，但明确这是“归档时hash”，不是伪装成运行前冻结hash；
- 新版本修复 SSH bounded retry、pull-ready key parser 和危险 shutdown supervisor；
- 不改 frozen N5/N6 bytes。

### P1：下一项最值得开的 GPU 实验——N7 27B replication

不要把整个项目迁移到27B；只复制最强、最简洁的 N6 mechanism。

建议步骤：

1. 扩容/准备 250–350 GB 空闲盘；
2. 只下载：
   - Gemma-3-27B-IT base；
   - NLA L41 AV/AR；
   - Gemma Scope 2 L41 16k small/big；
3. 先跑 20–40 groups engineering smoke：
   - BF16加载；
   - L41 provenance；
   - parser success；
   - identity KL；
   - AR/SAE shape；
   - AV throughput；
4. 根据 smoke 估算方差和成本；
5. 冻结全新 N7 preregistration；
6. 用 fresh 200或400 groups复制：
   - `G_specific`
   - `G_content`
   - `T_p3`
   - candidate predictive alignment；
7. 独立 audit 后立即 pull+verify+shutdown。

成功：直接解除论文最大的 single-model/scale 审稿风险。  
失败：仍是预注册的 scale boundary，12B 主结论不被抹除。

### P2：回到原始主线——NLA↔SAE 联合 Mech Interp

N7 是对 NLA 辅助信号可迁移性的资格审查，不是项目最终目标。N7 之后最重要的
confirmatory work 应直接检验“交叉辅助是否比单独工具更好”。

#### J1：NLA → SAE feature interpretation（建议先做）

核心问题：

> 在相同 contexts、token/bit budget 和 feature cohort 下，加入 NLA 生成/筛选的
> hypothesis，能否让 SAE feature interpretation 比 SAE-only baseline 更准确、
> 更可检验或更省样本？

设计原则：

1. 使用真实大语料，按文档与来源切分；
2. feature cohort 在查看 NLA score 前冻结，并要求 heldout 有充分正激活；
3. 比较 matched-budget 条件：
   - SAE contexts/decoder-only baseline；
   - NLA-assisted hypothesis/label；
   - shuffled or mismatched NLA control；
4. primary endpoints 必须是外部端点：
   - heldout positive vs hard-negative activation prediction；
   - feature ablation/activation 对 logits 或行为影响的预测；
   - label-specific causal probe；
5. AUPRC、coverage 和 calibration 必须联合报告；完全未激活的 feature 记为
   not-testable，不能伪造 `AUC=0.5`；
6. human 或异构模型盲评可作校准，但同源 Gemma judge 不能成为唯一主端点；
7. q+、round-trip 和 retrieval 只允许做 triage，不允许自我验真。

J1 才是真正回答“**NLA 能否辅助 SAE/现有 Mech Interp**”的实验。成功门槛应是
相对 SAE-only baseline 的预注册外部增益，而不是 NLA 文本看起来合理。

#### J2：SAE → NLA explanation grounding

核心问题：

> 将真实 active SAE features、decoder directions 或 SAE residual evidence 提供给
> NLA explanation/audit，能否比 NLA-only 更准确地识别因果文本内容、降低
> hallucination 或预测 reconstruction failure？

至少比较：

- NLA-only；
- NLA + matched active-SAE evidence；
- NLA + shuffled/matched-wrong SAE evidence；
- SAE-only diagnostic。

可预注册端点：

- explanation proposition/candidate 与 clean logit/feature evidence 的一致性；
- 对 causal text-ablation effect 的预测；
- `KL_NLA-KL_SAEbig` catastrophe 的校准与 AUPRC；
- heldout intervention-to-behavior prediction；
- matched-budget human/异构模型 fidelity。

J2 才是真正回答“**SAE 能否辅助 NLA 做 Mech Interp**”。它不能只比较文本评分，
必须证明 SAE grounding 改善外部 activation、causal 或 behavioral endpoint。

### P3：candidate-semantic robustness / public-language vs paired protocol

N6 证明“正确 recipient candidates”重要，但尚未证明通道对表面改写稳定。

可做：

- candidate-preserving paraphrases；
- synonym/inflection/order changes；
- same semantic candidate、不同 surface form；
- false but lexically matched candidate；
- cross-family AR evaluator。

目的：区分真正公开的自然语言语义通道，与 paired AV/AR 因共同训练形成的特定表达协议。

### P4：capacity/rate-matched NLA–SAE operating curve

公平比较不能只取一只 NLA 和两只 SAE：

- SAE：width、top-k/L0、amplitude quantization、active-index bits；
- NLA：UTF-8 bytes、token count、独立LM cross-entropy bits、candidate truncation；
- primary distortion：causal KL recovery；
- secondary：centered geometry/retrieval；
- 另报 latency、model calls 和 generation cost。

当前 checkpoint 扫描只能称 post-training operating curve；真正 rate-distortion frontier 要在不同 bottleneck rate 下重训。

### P5：真正的 steering

本项目还没有做 steering。未来必须满足：

1. 从 NLA text edit 或 SAE feature 构造 direction；
2. 将 direction 写回 base model 的指定层和位置；
3. 测 logits、生成行为或任务成功率；
4. 有 neutral、matched wrong、random、norm-matched controls；
5. 报副作用与 off-target behavior；
6. 在新 prompts/documents 上确认。

有价值的方向：

- candidate edit steering；
- belief/strategy edit；
- NLA vs SAE steering efficiency；
- SAE feature + NLA-generated hypothesis 的闭环实验；
- cross-layer transfer。

只有 intervention 改变下游模型行为，才允许称 steering。

### P6：参数规模与“思维涌现”

只比较12B和27B不能证明“参数导致涌现”。

若另立 scaling 项目，至少应使用：

- 4B；
- 12B；
- 27B；
- 若资产允许再加1B。

统一：

- prompt/cohort；
- extraction depth比例；
- NLA/SAE operating point；
- causal endpoints；
- explanation functional annotation；
- token budget。

由于不同尺寸的训练 token、数据和优化也不同，稳妥表述应是 **scale-associated transition**，而不是 parameter-caused emergence。

---

## 12. 推荐的实际执行顺序

1. **冻结当前12B论文证据，不再增加同层重复实验。**
2. **CPU写作、主图和artifact appendix，同时起草 J1/J2 的联合实验预注册。**
3. **修复但另存新版launcher；不改N6 frozen code。**
4. **准备27B磁盘与20–40组smoke。**
5. **预注册N7，做27B N6 replication；把它明确写成联合路线的前置验证。**
6. **N7拉回、审计和关机后，优先执行 J1 NLA-assisted SAE feature
   interpretation，而不是继续追加 codec 排名实验。**
7. **随后根据 J1 结果执行 J2 SAE-grounded NLA，或修正联合接口。**
8. **candidate-semantic robustness、rate/capacity curve、steering和scaling
   emergence 均排在至少一项双向辅助 confirmatory test 之后。**

资源分配原则：

- CPU 时间可并行用于现有论文与 J1/J2 设计；
- 下一次已承诺的 GPU run 是 N7；
- N7 之后的优先 GPU 实验应回到 NLA↔SAE 联合主线；
- 暂不把所有旧实验整体迁移到 27B。

---

## 13. 用一段话概括整条研究史

Jason 与 Fable 5 一开始提出的就是 NLA↔SAE 双向辅助：让 NLA 帮助解释、标注和
筛选 SAE features，也让 SAE 为 NLA 的自然语言解释提供稀疏 grounding 与可干预
验证。为了判断这种协同是否有可信基础，早期实验先把二者放到
Gemma-3-12B-IT L32 的同一批激活和共同指标上校准；raw cosine 的表面优势随后被
残差、字典方向和注入 pilots 暴露出的 activation mean 混淆推翻，项目因此建立了
去均值与对照纪律。B2 发现 NLA 检索间隔更大但 Top-k 不胜 SAE，B6+B4 表明只有
一部分 SAE directions 能通过 AV/AR 稳定通信，而且内部 round-trip 不能验真自然
语言标签。C1 pilot 有 relative-axis 弱信号，但确认性 synthetic corpus 被自建
审计门禁卡死，迫使路线转向真实语料。Opus 复核指出 q+ 双峰、AUC 伪值和错误 gate
顺序，并推动第一次 causal patch。N1/N2 发现重建主要集中在 NLA 第三段的
continuation candidates，且几何优势不等于 causal superiority；N3 用 8.24M
真实 tokens 证明 synthetic non-activation 只是 coverage failure；N4 在真实
content tokens 上找到 p3 的强因果价值；N5 在 fresh held-out cohort 上复制 p3
channel，同时否定简单 centered-cosine router；N6 用 matched candidate
substitution 证明 recipient candidates 携带样本特异、next-token-aligned 的增量
因果信息。到这里，项目完成的是联合路线的一项关键“资格审查”：NLA 确实携带值得
用于 Mech Interp 的因果信息，但 NLA-assisted SAE、SAE-grounded NLA 和二者联合
闭环尚未正式建立，下一阶段必须回到这个原始主问题。

---

## 14. 2026-08-06：J1-D1 首次直接回到 NLA→SAE 辅助主线

在 Fable 5 对项目战略的复核后，执行优先级由“先做 N7 27B replication”改为：

1. 现有 N5/N6 四层评测框架继续作为可靠论文骨架；
2. 立即做一项直接回答原始问题的 `J1 / NLA→SAE` discovery pilot；
3. N7 作为买掉 single-model 风险的后续 replication，不再替代联合增益实验。

### J1-D1 研究问题

给定一个真实语料上高激活的 SAE feature，NLA 对四个真实
max-activating residual states 的自然语言描述，是否能帮助独立 interpreter
形成更能预测 held-out SAE activation 的 feature hypothesis？

这不是 NLA 与 SAE 的 codec 排名，也不是 confirmatory claim。它复用 N3 的冻结
corpus/cohort，只用于验证协议、测量效应与失败模式，并决定是否值得在 fresh corpus
上预注册正式 J1。

### 冻结设计

- 三个 N3 strata 各 15 个 feature，共 45；
- 每个 feature 四个不同文档的 discovery contexts、四个不同文档的 held-out
  positives；
- 每个 positive 配一个 target SAE activation exact-zero 的 hard negative；
- hard negatives 必须满足同 source+language、同 source 或同 corpus+language
  的固定严格层级；
- NLA 对 180 个真实 discovery residuals 生成 on-manifold snippets；
- SAE 与 NLA 的联合 contrastive arm 使用
  `x_minus = x - a_f w_dec[f]`，再分别 verbalize `x` 与 `x_minus`；
- 外部 truth 只来自 frozen SAE activation；
- Fable 5 生成 hypothesis，Terra 对 hypothesis × held-out context 做盲评；
- 所有结果保持 ITT，parser failure 不允许删行。

工程 smoke 在任何 AV outcome 前记录了两个 addendum：

- A1：BF16 不同 batch shape 下，360 个复算 activation 有 8 个超过最初
  `rtol=0.005`；无 sign mismatch、最大相对差 2.0167%，因此在未加载 AV 前把
  tolerance 冻结为 `rtol=0.025, atol=1.0`；
- A2：只用 45-feature top contexts 时，一个 Hindi-selective feature 缺少四个
  strict negatives；没有放宽 language matching，而是从同一 N3 真实 corpus
  决定性加入 background candidate pool。

### GPU 正式结果与审计

正式 AV：

- status：`EXPLORATORY_DISCOVERY_AV_COMPLETE`；
- freeze SHA：
  `8f7690f8b12842b32ce5cb32af7ee941b2ce2f71fcc0270768cf0f84edcb50d3`；
- result SHA：
  `d93d99e3c84b07a6f76b3b4549bb16fbe520f9c7aa59579f98e57bb2a85749a4`；
- 45 features、180 discovery contexts、180 held-out positives；
- 180 hard negatives，其中 80 来自其他 selected contexts、100 来自 A2
  background pool；
- 180/180 negatives 均为 tier 0、target activation exact zero；
- 360/360 AV plans、checkpoint rows 与 result rows 完整；
- 向量文件 1,050 rows，SHA：
  `fa9cc13508638417d05b21740b55ed8a14d6c6933bef6271b5800078bd6df71d`。

authoritative independent audit v3：

- `PASS`；
- 26/26 checks；
- 0 errors；
- activation recomputation 无 sign mismatch，最大相对差 `0.0201673`；
- SAE feature ablation 的向量、norm、cosine 重算最大误差全部为 `0`；
- AV checkpoint/result/vector/freeze/script bindings 全部一致。

v1/v2 audit failures 是审计器假设错误，不是数据失败：

- v1 错误要求所有 negatives 来自 background；
- v2 错误拒绝合法的 whitespace decoded tokens；
- 二者原始 FAIL artifacts 均保留，v3 修正后通过。

GPU 产物拉回本地后，远端执行了：

`sync; /usr/bin/shutdown -h now`

SSH 随即关闭，五秒复查在 banner exchange 超时。只有 AutoDL 控制面能最终证明
计费停止；不得为确认而重启。

### 下游标签协议修正

独立盲法审计在任何 Fable J1 outcome 前发现并修正：

- prompt 中误带的 numeric target activation；
- 同一 feature 五个 arms 放在同一次调用造成的直接 cross-arm leakage；
- cyclic donor 的最长→最短 wrap length confound；
- error checkpoint 无法续跑；
- raw output、usage/cost、model/version 与 context alignment 的 provenance 缺口。

最终冻结为 45 个 cross-feature Fable batches：

- 每批五种 arms 各一个；
- 每批来自五个不同 features；
- 每个 feature 的五个 arms 分布在五个不同 calls；
- 225 个全局唯一 opaque case IDs；
- private case→feature/arm map 不进入 prompt；
- mismatched donor 为 same-stratum、minimum-length-cost、bijective derangement。

### 当前外部阻塞

Fable 标签在 13/45 successful batches 后返回：

`403 insufficient balance`

截至中断：

- 13 个成功 batches：`0..12`；
- 11 个保留的 403 attempts：`13..21, 23, 24`；
- 无最终 label result；
- 未运行 Terra；
- 未查看或分析任何 partial arm outcome；
- 成功调用约 `$3.5433`，一个失败 envelope 另记 `$0.07027`；
- 剩余 32 calls 预计约 `$8.72`，建议补充至少 `$11`。

余额恢复后必须保留 frozen jobs 与 append-only checkpoint，原样续跑
`server/58_j1_discovery_labels.py`。45/45 完成前禁止 partial analysis 或模型替换。

## 2026-08-06：J2-P0 SAE projection × language-loop 顺序审计冻结

代码库逐脚本取证确认：此前做过 raw `x` 分别进入 NLA/SAE、单 feature decoder
direction 或 residual carrier 进入 AV→AR，以及 J1 的 raw/ablated residual→AV；
但从未完成

`x → D(E(x)) → AV → AR`

这一完整闭环。因而新实验不是重复 N4/N5，而是首个直接测试 `SAE→NLA grounding`
及两种工具组合顺序非交换性的实验。

在任何新 J2 AV 输出前冻结了四路径：

1. `NLA(x)`；
2. `SAE(x)`；
3. `NLA(SAE(x))`；
4. `SAE(NLA(x))`。

复用 N4 的 200 行、101 文档真实 cohort，以便逐行配对已有 direct-NLA、
native-SAE 与 causal patch。主要端点为 activation geometry、SAE sparse-code
fixed point/retention、KL/KL16/CE16，以及 9 类 metric-only case shortlist。
由于复用 cohort，J2-P0 永远是探索性机制审计；它只能生成 fresh J2 的假设。

冻结文件：

- `results/J2_SAE_PROJECTION_LANGUAGE_LOOP_PROTOCOL_2026-08-06.md`
- `results/J2_SAE_PROJECTION_LANGUAGE_LOOP_RUN_MANIFEST_2026-08-06.json`
- `server/66_j2_sae_projection_loop.py`
- `server/67_j2_sae_projection_causal.py`
- `server/68_j2_sae_projection_analyze.py`
- `server/69_j2_render_case_bundle.py`
- `server/run_j2_sae_projection_loop.sh`

截至 2026-08-06 冻结时，本地 sidecar、JSON 与 Python compile 已验证，J2 尚未
运行、没有 outcome、没有新文本被查看。该状态已被下方 2026-08-07 的正式运行覆盖。

## 2026-08-07：J2-P0 正式运行、独立审计与裁决

### Preflight correction

AutoDL 重开后，第一次正式 `--dry-run` 在任何 J2 output 生成前 fail closed：
manifest 中硬编码的上游 `n4_explanations` SHA 多抄了一个字符，成为 65 位字符串。
实际 upstream artifact 与 sidecar 一致，正确 SHA 为：

`b656ded845c8fd122e4dcb1391ba5d81e1a903f80a69c30575bf26910e200942`

修正 protocol、66/67/68、runner 与 manifest 后重新冻结并上传，第二次 dry run
通过。该事件完整记录于：

- `results/J2_PREFLIGHT_HASH_CORRECTION_2026-08-07.md`

最终 protocol SHA 为
`a41b7d89893a270218bf79e226c3e3d7a8726f71ca1fe6d41f40b583616a700f`，
run manifest SHA 为
`d3401ddc7180bd8dda2783ec73731da6e87c00f16acaf99f2f17d954351bd400`。

### 运行完成

- AV：400/400；
- frozen explanations SHA：
  `c7d1be1c81ee371d9a90945495d49ba6e0d7dc379b2c0dc8f56b8740b9f481db`；
- reconstruction result SHA：
  `946a399163fa9b5fc80088e1bede032fad26dca00331f2493e1b84028e1ed0f6`；
- vector NPZ SHA：
  `f08bd51344e9687f3f6885ce41e29374418ca4fb8b008cab829a704e3d5e3b0a`；
- causal：101/101 documents、200 positions、1301 forwards、约 111 秒；
- causal result SHA：
  `d021b5fd1cf89ea2a37be4b5ca74ee6f3f43c0442b8143ef3791607b31651bfe`；
- analysis SHA：
  `692f7dd0e7cf6eeb5b3e62c6cebb2afd3c0989bd54b76454f9c5d86b0e774642`；
- frozen shortlist：18 categories、35 unique positions，SHA：
  `b21a53bda1a6672b1327f303f978f67ac7eb1a0f3933e607c210cdb539fea405`；
- pipeline exit 0，拉回的 18 个 artifact sidecar 全部匹配。

结果拉回并写入 pull acknowledgement 后，supervisor 执行：

`sync; /usr/bin/shutdown -h now`

SSH 随即关闭。AutoDL 控制台仍是计费停止的最终独立证明；不得为确认而重启。

### 独立审计

两名 Luna Max 分离执行：

1. 数值审计不读 case 文本，从 raw N4/J2 NPZ、200 行 recon records 与 causal
   records 重算；
2. case 审计不改文件，验证预冻 metric shortlist 到后渲染 bundle 的绑定。

自动分析所有 vector bundles、paired geometry/causal contrasts 的最大绝对误差仅
`7.86e−08`；identity/zero/clean controls 与 N4 相差严格为 0。
18 categories × top3 = 54 memberships、35 unique positions 与 bundle 完全一致，
selection metric/rank/hash 零 mismatch。

权威审计：

- `results/J2_INDEPENDENT_AUDIT_2026-08-07.md`

### 主要结果

1. **几何改善。** loop 相对 native SAE 的 centered cosine 增加：
   - small `+.109487 [.094272,.124555]`；
   - big `+.095760 [.080511,.110902]`。
2. **仍逊于 direct NLA。**
   - small `−.139253 [−.153628,−.125607]`；
   - big `−.099128 [−.111160,−.087318]`。
3. **没有 sparse-code grounding。** loop 相对 direct NLA re-encoding：
   - code cosine `−.029295/−.031677`；
   - support Jaccard `−.035807/−.062617`。
4. **causal fidelity 明显恶化。** loop-minus-baseline KL：
   - small vs direct NLA `+.697680 [.467955,.988165]`；
   - small vs native SAE `+.652083 [.318883,1.043851]`；
   - big vs direct NLA `+.620861 [.320137,.980321]`；
   - big vs native SAE `+.844832 [.460211,1.295714]`。
5. **顺序优势未建立。** loop 与 reverse-order `SAE(NLA(x))` 的 KL 区间均跨零。
6. **文本不稳定。** `AV(SAE(x))` 与 `AV(x)` 的 token Jaccard 仅
   `.3001/.3330`，即使 residual raw cosine 约 `.99`。

### Case-study 线索

预冻 shortlist 中最有信息量的是：

- `idx75`：两只 SAE 的 loop 都在 code 与 causal 上局部 rescue；
- `idx168`：small code rescue 更大却 causal 恶化，big 反而 causal rescue；
- `idx185/186`：同一 Apache-license 文档的相邻位置发生 SAE width 成败翻转；
- `idx34`：Linux/dbus 主题文本与 code rescue 同 causal catastrophe 共存；
- `idx122`：日期/数字 identity 漂移导致 SAE-big loop catastrophe；
- `idx130`：causal rescue 与 propositionally dubious AV explanation 共存。

这些都是 post-hoc mechanism hypotheses，不能推广成总体增益或事后 router。

### 最终裁决

`DO NOT CONFIRM THE SAME SERIAL DESIGN`

J2-P0 对朴素 `SAE reconstruction → free-form AV → AR` grounding 给出清晰的探索性
负结果：language loop 像 activation-manifold prior 一样改善几何，却损失 sparse
code 并累积功能性错误。它不否定 SAE→NLA 的原始方向，但说明下一版必须改用
structured/conditional grounding 或 SAE counterfactual intervention，并在全新
held-out cohort 上保留外部 causal endpoint。

权威 paper-facing 总结：

- `results/J2_FINAL_ANALYSIS_2026-08-07.md`
