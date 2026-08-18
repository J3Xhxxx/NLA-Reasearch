# Continue — NLA × SAE 项目交接（N4 后历史记录）

> **最终执行覆盖（2026-08-07，J2-P0 exploratory complete）：**  
> 首个完整 `真实 activation → SAE reconstruction → AV → AR` 已完成：
> 400/400 AV、200/200 causal positions、101/101 documents。四路径为
> `NLA(x)`、`SAE(x)`、`NLA(SAE(x))`、`SAE(NLA(x))`。两名 Luna Max 已分别
> 从 raw NPZ/rows 重算数值及审查预冻 case；自动分析最大绝对差异
> `7.86e−08`，18 categories/35 unique cases 绑定零 mismatch。最终裁决：
> `DO NOT CONFIRM THE SAME SERIAL DESIGN`。loop 相对 native SAE 提升 centered
> geometry（small/big `+.1095/+.0958`），却相对 direct NLA re-encoding 损失
> 原始 SAE code cosine（`−.0293/−.0317`），并显著增加 causal KL：
> 相对 direct NLA `+.6977/+.6209`，相对 native SAE `+.6521/+.8448`，
> 四个 document-bootstrap 95% CI 均完全大于零。权威总结与审计见
> `results/J2_FINAL_ANALYSIS_2026-08-07.md` 和
> `results/J2_INDEPENDENT_AUDIT_2026-08-07.md`。这否定朴素串联，不否定
> structured/conditional SAE→NLA grounding。服务器已执行
> `shutdown -h now`、SSH 随即关闭；不要为复查而重启。
>
> **最终覆盖（2026-08-06，J1-D1 discovery complete）：**  
> 原有 Fable batches `0..12` 未重复；Luna Max 已完成 `13..44`。Terra 已完成
> 1,800/1,800 盲评分，完整 mixed-label 统计裁决为
> `REDESIGN_REPLICATE_BEFORE_CONFIRMATORY`。`NLA_ASSISTED` 的 ITT micro-AP
> 相对 `SAE_CONTEXT` 为 `+0.013281`，95% bootstrap CI
> `[-0.004941,+0.035954]`，并在 Luna–Luna / Fable–Fable 间发生
> `+0.009202 / −0.017494` 的方向翻转。因此不得立即启动 fresh confirmatory J1；
> 应先做非 OpenAI/人类全量重标和 capacity-matched 强 baseline。权威总结见
> `results/J1_MIXED_DISCOVERY_FINAL_2026-08-06.md`，权威数值见
> `results/j1_blinded_eval_analysis_mixed_v2.json`。服务器已发送
> `shutdown -h now`；不要为复查而重启。
>
> **最新覆盖（2026-08-06）：先读根目录 `Handoff.md` 的置顶 J1-D1 区块与
> `results/J1_DISCOVERY_LABEL_RUN_STATUS_2026-08-06.md`。**  
> J1-D1 GPU AV 已完成并通过 authoritative v3 独立审计；AutoDL 已发送
> `shutdown -h now`。Fable 标签因 `403 insufficient balance` 暂停于 13/45，
> 尚无 J1 科学结果。充值后保留 immutable jobs/checkpoint，原样续跑
> `server/58_j1_discovery_labels.py`；禁止 partial analysis、模型替换或重启 GPU。
>
> **状态更新（2026-08-03）：N6+ 已完成，本文件下文仍停留在 N4 后。**  
> 当前负责人必须先读最新交接 `RECOVERY_2026-08-03.md` 与
> `results/N6_FINAL_ANALYSIS_2026-08-03.md`；正式数值见
> `results/n6_pull_staging/n6_pull_20260803T061302Z/n6_analysis_v1.json`
> 和 `n6_independent_audit_v1.json`。  
> 下文关于“N5/N6 尚未运行”“AutoDL 保持开启”或旧实验优先级的表述均已过时，
> 仅用于恢复历史决策过程。当前 frozen verdict 是：N5 router 未建立，N5 p3 channel
> 已复制，N6 sample-specific candidate mechanism 与 predictive alignment 已确认。
>
> 原记录最后更新：2026-07-30（Asia/Shanghai）。

## 1. 当前一句话状态

N3 已在 8.24M-token 真实语料上完成 feature-activation 覆盖审计。旧 synthetic
test 中 8 个“完全不激活”的 feature 在真实文本上全部激活，24/24 旧 feature
无一死亡；因此 synthetic zero activation 是 coverage failure，不能作为淘汰门禁。
这不证明旧标签正确，也没有回答 readability Q2。旧 N3 120-feature cohort 因
raw source firing share 未按约 9,280 倍不等的 source exposure 归一化，不得用于
昂贵 AV benchmark。

N4 已在冻结的 200 个真实 content-token 位置、101 篇文档上完成 reconstruction
与 L32 causal patch：

- provenance 和 clean state 均与冻结 activation bit-exact；identity KL=0；
- H1 正式 FAIL：p3 share=.932，但 p12 share=.756，未满足 ≤.50；
- H2 逐行 `KL_recovered` 因 7 个近零 `KL_zero` 分母而病态，正式 gate 只能记
  FAIL/not established，不能把巨大负均值解释为 NLA 很差；
- 稳定 ratio-of-sums 为 NLA .94795、SAE-small .94417、SAE-big .96649；
  NLA–big 的 90% sensitivity CI 位于 ±.05 内，但这是事后 estimand，不能改判；
- NLA 在 80/101 文档上有较低 KL 于 SAE-small，但少量真实 catastrophic failures
  使平均优势未成立；
- H3 冻结状态应写 NOT TESTABLE；raw KL、KL16、CE16 和 ratio-of-sums 均强烈支持
  p3 比 p12 更有因果用处，p3 保留约 99.7% 的 orig aggregate recovery。

完整审计见 `results/N4_REAL_CONTENT_CAUSAL_AUDIT_2026-07-30.md`。最合理的新路线
是 held-out selective NLA + SAE-big：NLA 提供可读候选通道，SAE-big 提供 fidelity
与 tail fallback。推荐下一步先做 N5 新文档 confirmatory gate，再回到修正后的
real-corpus feature-level C1/Q2。

本轮 reconstruction + causal GPU 时间约 30.4 分钟；N4 后未再启动 GPU 实验。
AutoDL 实例保持开启，GPU 当前空闲。旧 C1 v3r2 的 corpus gate 失败记录和下面各节
仍有效。

## 2. C1 pilot 的设计边界

- 统计单位：24 个既往 B6 `semantic_new` feature。
- 历史冻结真值只有 7 个粗轴：4 个 domain + 3 个 language。
- 主对比：每个 feature 的
  `q_AR(axis_reference) - q_AR(axis_hard_negative)`。
- train reference、sibling mismatch、base autointerp、NLA paraphrase、
  Gemma blind judge、heldout-valid 子组均为探索性。
- 所有 autointerp 证据严格过滤为 train split；15/24 个旧 `top_contexts` 曾混入 test，
  这次没有沿用该泄漏。
- 细粒度 reference 是在查看 B6 后撰写的，不是独立人工真值。

## 3. 核心结果

### 3.1 冻结粗轴主结果

| 指标 | 结果 |
|---|---:|
| mean delta | **0.01550**，feature bootstrap 95% CI [0.00510, 0.02712] |
| median delta | **0.00897**，95% CI [−0.00016, 0.02402] |
| 正效应 | **16/24** |
| exact sign test（单侧） | **p=0.07579** |
| exact sign-flip mean test（单侧） | **p=0.003934** |

解释必须同时保留两面：平均幅度显著偏正，但多数一致性未过 0.05，且最大的四个正效应
贡献了约 70.6% 的净效应。7 个标签的 cluster mean 全部为正，任意删去一个标签后的总体
均值仍为正；但 feature 嵌套于重复标签，不能把 feature-level p 值包装成确认性证据。

对全部 7 个轴文本做事后敏感性分析：

- correct minus mean of six wrong axes：mean **0.01616**，17/24 为正；
- feature→7-axis-text Top-1：**50.0%**；
- correct minus best wrong：median **−0.00013**，仅 12/24 为正。

所以 AR 能做弱的粗轴排序，但离“稳定找出唯一正确轴”仍有距离。

### 3.2 细粒度 reference：强，但主要像 train 指纹

- train reference − train hard negative：mean **0.19468**，20/24 为正，
  sign `p=0.000772`；
- train reference − same-axis sibling mismatch：mean **0.10793**，
  19/24 为正，`p=0.00331`；
- 24-way retrieval：reference text→feature Top-1 **54.2%**，
  feature→reference text Top-1 **70.8%**。

但 context delta 在 heldout-invalid feature 上反而更大（0.248 vs valid 0.156），
区分 heldout-valid 的 AUC 只有 **0.379**。它更像重复词汇/训练主题匹配，不是外部泛化证明。

相比之下，粗轴的**相对 delta**与 heldout selectivity 有一个值得复现的描述性关系：

- heldout-valid 14 个：mean 0.02584，11/14 为正；
- invalid 10 个：mean 0.00102，5/10 为正；
- axis delta 预测 heldout-valid 的 AUC **0.764**；
- axis delta 与 test AUC 的 Spearman **0.501**。

绝对 `q(axis_reference)` 本身没有该关系（对 test AUC 的 rho 0.019），因此只能说
“正确−错误相对分数可能有信息”，不能说绝对 q 已校准。

### 3.3 NLA、autointerp 与 private-code 风险

- NLA original 的 24-way text→feature Top-1 **62.5%**，
  feature→text Top-1 **58.3%**；
- base autointerp 对应为 **45.8% / 37.5%**；
- NLA q 高于 base autointerp 仅 14/24，没有可靠多数优势；
- blind context judge 的 `score>=2`：NLA 仅 **2/24**，base autointerp **16/24**。

这再次说明内部身份检索与人类可接受解释不是同一目标。

NLA paraphrase 相对原文：

- mean Δq **−0.12776**，19/24 下降；
- private-code interaction mean **0.10429**、median 0.02627，
  17/24 为正，sign `p=0.03196`。

但 paraphrase 把 NLA 文本压缩到原字符数的 median **60.1%**，而 authored reference
paraphrase 为 **106.3%**；提示中还要求删除 meta phrase 和猜测列表。因此这只能支持
“AR 对 NLA 原始表述形式异常敏感”，不能识别为真正的 private code。下一版必须使用
多份人工确认、命题与长度匹配的盲 paraphrase。

### 3.4 base judge 失败

- generic controls 中 110/192 被判 `score>=2`；
- axis hard negatives 中 11/24 被判 `score>=2`；
- pooled feature-centered q–judge rho **−0.0418**；
- q 预测 `judge>=2` 的 pooled AUC **0.410**。

Gemma judge 只能保留作质性日志，不能进入主结论或方法排名。

## 4. 运行与 QA

成功的 v2 运行：

- wall time：**430.82 s（7 分 11 秒）**；
- base generation + 432 judges：411.84 s；
- AR：10.76 s，166 个 unique text；
- 432/432 judge 解析成功；
- 旧 B6 NLA 原文最大重算误差：**1.04e−9**；
- 独立向量复算误差：saved similarity 最大 **2.80e−8**，
  target score 最大 **5.86e−14**，rank 误差 0；
- peak VRAM：**24,669 MiB**；
- peak power：**407.41 W**；
- peak temperature：**62°C**；
- 成功运行监测能耗约 **25.45 Wh**。

第一次完整尝试在最终 QA 被正确拦截：benchmark 曾把旧 NLA 文本内部换行压成空格，
造成最大重算漂移 0.0744。修复后保留 24/24 精确原字符串并用全新 v2 checkpoint 重跑。
第一次日志和 GPU 日志保留为 `c1_pilot_attempt1.log` / `c1_pilot_gpu_attempt1.csv`。
含 smoke、失败尝试和成功尝试，本轮总 GPU 占用约 14 分 40 秒、约 49 Wh。

AR 加载时仍会显示 `model.norm.weight MISSING`；这是现有 `NLACritic` 路径的结构提示。
精确复现 B6 分数到 1e−9 已证明本次权重/索引口径没有漂移。

## 5. C1 pilot 脚本与结果

脚本：

- `server/15_build_c1_pilot.py`：冻结 benchmark，强制 train-only context；
- `server/16_run_c1_pilot.py`：base autointerp/paraphrase/judge + AR；
- `server/17_analyze_c1_pilot.py`：冻结主分析；
- `server/18_validate_c1_pilot.py`：独立结构与数值复算；
- `server/19_analyze_c1_secondary.py`：事后敏感性与双向检索；
- `server/run_c1_pilot.sh`、`launch_c1_pilot.sh`、`start_c1_pilot.sh`：无关机逻辑的断点 runner。

主要结果：

- `results/c1_pilot_benchmark.json`
- `results/c1_pilot_checkpoint_v2.jsonl`
- `results/c1_pilot_result.json`
- `results/c1_pilot_recon_vectors.npz`
- `results/c1_pilot_analysis.json` / `.md`
- `results/c1_pilot_secondary_analysis.json` / `.md`
- `results/c1_pilot_validation.json`
- `results/c1_pilot.log` / `c1_pilot_gpu.csv`
- `results/c1_pilot_checksums.sha256`

`c1_pilot_checksums.sha256` 是成功 v2 核心资产的固定清单。

两个扩样前的工程约束：

- `17_analyze_c1_pilot.py` 的 exact sign-flip 为 n=24 显式枚举 `2^n`，只适合本
  pilot；扩到 60/100 时必须改成预注册的 Monte Carlo sign-flip（分批采样并报告
  Monte Carlo SE），或保留 exact sign test + cluster bootstrap，不能直接复用。
- 最终 v2 checkpoint 是在修正 benchmark 后全新生成，当前结果干净；但当时行内尚未保存
  prompt hash。`16_run_c1_pilot.py` 现已为新行加入 input SHA 校验，runner 也不会再静默
  重建 frozen benchmark。当前旧 checkpoint 的绑定由 `c1_pilot_validation.json` 和
  checksum manifest 保证。

## 6. C1 confirmatory 的实际停止点

### 6.1 设计与门禁

- 科学设计保持 24 个新英文 concepts、6 个 superdomains、12 个 reciprocal
  within-domain hard-negative pairs、每 concept 4 discovery + 2 heldout、
  最多 4 SAE features/concept。
- 旧实验暴露过的 1282 个 feature IDs 全部 denylist。
- Feature selection 只允许 discovery activation；主门槛为至少 60 features、
  18 concepts、9 个完整 reciprocal pairs。
- 主终点是 centered AR cosine：
  `mean(correct references) - mean(reciprocal hard-negative references)`；
  feature 在 concept 内等权、两个 concept 在 pair 内等权，pair 是精确
  joint sign-flip 与 bootstrap 的统计单位。
- Heldout 在 corpus 语义审计后继续 embargo，直到 discovery feature 与 candidate
  benchmark 冻结。

### 6.2 V1/V2/V3/V3r2

- V1 在 activation 前因生成文本字数/格式可行性失败，证据保留。
- V2 成功机械生成 144 文档，但两名独立 reviewer 都判 `FAIL`，并同意同一组
  10 个文档失败；没有提取 v2 activation。
- V3 明确披露为 pre-activation adaptive corpus redevelopment。首版 144 anchors
  的独立审计因 20/24 concept 的 heldout 重组 discovery 机制/应用而 `FAIL`。
- 唯一允许的最终 pre-text 修订 v3r2 改写 33/144 anchors。独立全量审计结果：
  结构 PASS；直接 pair-level mapping、安全、难度与 framing 12/12 PASS；
  concept 为 **17 PASS / 7 FAIL**；计入 constituent 后 6/12 pairs PASS。
- 失败 concepts：
  `error_detecting_codes`、`protein_quality_control`、
  `microbial_quorum_sensing`、`microbial_cross_feeding`、
  `groundwater_contaminant_transport`、`quarantine_regimes`、
  `phonological_assimilation`。
- v3r2 anchors SHA256：
  `3b38876a663ea3a3a9a1623017242a06e0f51b667109cf60bb8de549cb21600a`。
- v3r2 audit SHA256：
  `23908a7784e3e49f96daf0437186cc9bc9d6c1f453e6d8c2c1c9405e1786b1ef`。
- 完整停止记录：
  `results/c1_confirmatory_scenario_anchor_v3r2_failure.md`。

冻结规则规定 v3r2 若失败则本轮停止。因此 **没有生成 v3 请求，没有执行 Stage 0，
没有 activation、feature selection、AV/AR 或 endpoint**。不能选择性再改 7 个
concept 并仍称同一 confirmatory run。

严格 generator/runner 已兼容真实 v3r2 schema 并通过离线与远端静态验证，但未启动：

- generator SHA256：
  `fad14b4cb01ca3789678b07d991a73ed1fa257a442b1f0c35712ec1fbf65803e`
- runner SHA256：
  `c54b02600c2eabb45d91da46df4af6b0ef544b92824daa01bf5f6ea3ca6a902f`

## 7. 下一步裁决与资源

下一步需要用户明确选择新设计，不能当普通 retry 自动继续。推荐优先重新定义
generalization estimand，而不是继续逐句修补：

1. **推荐：C1-v4 application-level holdout。** 每个 concept 仍共享定义机制，但
   discovery/heldout 的 application、perturbation 与 evidence source 三层都显式不重叠；
   先冻结 disjointness taxonomy，再一次性生成完整新 corpus。
2. **更严格但更贵：真实文献/教材语料。** 从预先分开的 source families 抽 discovery
   与 heldout，人工只做 target/hard-negative 标注，减少 synthetic scenario 的主观边界。
3. **不推荐但可做探索：放宽当前 rubric 后跑 v3r2。** 必须改名为 exploratory，
   不能再报告为这次预注册 confirmatory 结果。

当前单张 A800 80GB 足够后续执行；真正瓶颈是 corpus 设计/审计，不是显存：

| 阶段 | 预计墙钟时间 | A800 峰值显存/磁盘 |
|---|---:|---:|
| 新 estimand + anchors + 独立 pre-text audit | 1–3 小时设计/审计 | 不需 GPU |
| 144 文档全新 corpus generation | 20–35 分钟 | 约 24–25 GiB，<1 GiB |
| 两名独立全文 semantic audit | 30–90 分钟 | 不需 GPU |
| discovery activation + SAE selection | 20–45 分钟 | 约 25–40 GiB，<5 GiB |
| benchmark freeze + heldout + AV/AR C1 | 1–3 小时 | 约 25–40 GiB，<5 GiB |

若目标是顶会，单模型/单层还不够：至少需要跨层/SAE 或第二模型复现，粗估
5–15 A800 GPU-hours，并增加盲评预算。当前数据盘约余 81GB，足够当前 Gemma 内扩展；
若再放多个 12B checkpoint，需要扩盘或分批迁移。

### 推荐决策顺序

1. 先由用户选择 v4 application-level holdout、真实语料，或 exploratory v3r2。
2. 离线冻结新 cohort、disjointness taxonomy 和盲评协议；此阶段不需 GPU。
3. 冻结通过后才在当前服务器跑 confirmatory C1。
4. 若 AR delta 能预测盲评/heldout，再做 residual-stream read/write causal steering。
5. 若 AR 只偏好 NLA 原文，则把论文主张收敛为
   **closed-loop communication code ≠ human interpretability**，这仍有方法论价值。

## 8. 远端状态与硬约束

- SSH alias：`autodl`。
- Python：`/root/miniconda3/bin/python`。
- 代码：`/root/autodl-tmp/nla_compare`。
- 结果：`/root/autodl-tmp/results`。
- 模型：`/root/autodl-tmp/models`。
- v3r2 审计失败后未启动任何 GPU 进程；A800 当前空闲，实例仍保持开机。
- 远端已同步 v3/v3r2 anchors、audits、failure record 与严格 generator/runner；
  没有 `c1_confirmatory_stage0_freeze_v3.json`、v3 corpus、v3 activation 或 selection 输出。
- **除非用户明确给出“远程关机”命令，不得执行 shutdown、poweroff、halt、reboot，
  也不得恢复任何自动关机 runner。**
