# REVIEW — 对 GPT-5.6 Sol 第二轮分析的独立复核与下一步方案

> 2026-07-30 · 复核者：Claude Opus 5（接手第三轮）
> 方法：不采信任何既有结论文件的数字，直接从 `results/` 的 NPZ/JSON 原始资产重算。
> 复核脚本与产物：`server/25_local_recheck.py` → `results/local_recheck_opus.json`；
> `server/26_local_recheck_b6b4.py` → `results/local_recheck_b6b4_opus.json`；
> `server/27_local_recheck_stratified.py` → `results/local_recheck_stratified_opus.json`。
> 全部为零 GPU 本地计算，未新增任何生成、未覆盖任何既有文件。

---

## 0. 一句话裁决

Sol 的**数字全部可复现**，**方法论纪律（去均值、ITT、控制组、不把 carrier 叫 steering）全部正确且值得保留**；
但它的**三条核心论证之一是统计假象**，另有**一条路线设计自相矛盾**，
而**最致命的是流程**：整个 07-30 会话在自建的语料审计门禁上连续四次 FAIL，零新增数据，
GPU 空转计费；同时全项目至今**没有任何一个因果/行为端点**——而那正是最便宜、最能定生死的一步。

| 类别 | 判定 |
|---|---|
| F1 主线对比（NLA 0.859 vs SAE 0.658/0.725） | ✅ 复现，且在我新增的更严苛基线下依然成立 |
| F8 / B2（Top-k 未胜出、margin 更大） | ✅ 复现，且 margin 优势通过了尺度归一化攻击 |
| F9 “弱而系统的方向信号（median q+=0.114）” | ⚠️ **表述错误**：分布是双峰的，不是弱而均匀 |
| F9 “heldout AUC 与 q+ ρ=−0.015 ⇒ 内部分数不预测外部效度” | ❌ **论证失效**：Simpson 悖论 + 8/24 特征的 AUC 是并列伪值 |
| C2-v2 “先过 heldout activation gate 再标注” | ❌ **自相矛盾**：该门禁会剔掉本队列里最可读的三个特征 |
| C1 “闭环自洽 ≠ 可解释性” 论题 | ✅ 方向正确，但机制解释不对，且执行方案不可行（3 名盲评人不存在） |
| 因果/行为端点（KL / loss recovered） | ❌ **全项目缺失**，Fable 第一天就提过，从未做 |
| 成本 | ❌ A800 自 07-26 起开机空转（本次探测：利用率 0%、显存 0 MiB） |

---

## 1. Sol 正确的部分（我做了更严苛的复核，结论更稳）

### 1.1 F1 在新增基线下依然成立

去均值 cos 精确复现：NLA **0.85927**、SAE-small **0.65839**、SAE-big **0.72456**、
残差文本对照 **0.04095**（`local_recheck_opus.json → head_to_head_centered`，与
`centered_rescore.json` 完全一致）。

原报告缺两个基线，我补上了：

| 基线 | 去均值 cos | 最优单一尺度下的 centered FVE |
|---|---:|---:|
| NLA 重建 | **0.859** | **0.707** |
| SAE-big (L0≈75) | 0.725 | 0.543 |
| SAE-small (L0≈15) | 0.658 | 0.459 |
| **预言机：40 个真实激活里挑最接近的“另一个真实激活”** | **0.541** | 0.336 |
| 残差文本对照 | 0.041 | 0.001 |

（出处：`local_recheck_opus.json → activation_activation_centered_cos`、
`centered_fve_optimal_scale`。）

这个预言机基线很关键：**用另一条真实激活去猜，就已经能白拿 0.54**。SAE-small 的 0.658
只比这个作弊上界高 0.12，NLA 的 0.859 高 0.32。同时去均值后任意两条真实激活的平均
cos 是 **−0.025**（中位 −0.067），证明 Fable 的单方向投影确实把共享分量清干净了，
去均值口径本身是可信的。结论：F1 不但成立，而且比原报告更有说服力——建议今后一律
连带报告这条预言机基线和 centered FVE。

### 1.2 B2 的 margin 优势不是尺度假象

我怀疑 Sol 的 “NLA margin 更大” 可能只是因为 NLA 的相似度整体更大（尺度效应），
于是把 margin 换成对每行非对角分布做 z 标准化的 d′：

| 方法 | Top-1 | 原始 margin | **z-margin（以行内非对角 SD 为单位）** |
|---|---:|---:|---:|
| NLA | 92.5% | 0.3195 | **6.49** |
| SAE-big | 95.0% | 0.2531 | 5.51 |
| SAE-small | 95.0% | 0.2085 | 5.12 |
| 残差文本对照 | 17.5% | −0.0507 | 1.01 |

配对差：NLA − SAE-small = **+1.37 SD**，NLA − SAE-big = **+0.99 SD**
（`local_recheck_opus.json → b2_recheck, b2_paired_z_margin_delta`）。
优势幸存。Sol 的表述（“几何间隔更强但离散 Top-k 未胜出”）是准确的，可以照用。

### 1.3 值得保留的纪律

去均值双边投影、每个实验自带高斯/泛型对照、ITT 先于 post-selection 子组、
domain/language 必须分层、`−w_dec` 只是 OOD 轴、carrier readout 不叫 steering、
以及 SHA256 冻结与逐字段可回查——这套纪律是这个项目最值钱的资产，第三轮全部继承。

---

## 2. 需要修正的三点（均可从原始资产直接验证）

### 2.1 修正一：q+ 是**双峰**，不是“弱而系统”

`semantic_new` ITT 的 24 个 q+ 降序排列（`local_recheck_stratified_opus.json →
q_plus_distribution.sorted`）：

```
0.673 0.508 0.507 0.504 0.471 0.467 0.458 0.453 0.424 0.362 | 0.245 0.118 0.109
0.070 0.024 0.018 0.009 0.008 0.006 0.003 −0.004 −0.014 −0.035 −0.059
```

第 10 与第 11 名之间存在 **0.117 的断层**：10 个特征 ≥0.362，13 个特征 <0.15。
median 0.114 落在断层里，是两个峰之间的空隙，**不描述任何一个特征的行为**。

而且高分组不是噪声：这 10 个里 **9 个的 `+feature_rank`=1**（45 个候选方向中精确命中），
并且逐特征减掉**自己方向上的泛型文本地板**后依然成立——8 个泛型文本对每个方向单独算，
高分组的地板只有 0.014–0.079，全部 10 个高分特征**超过自己方向上最坏情况的泛型文本**
（`local_recheck_b6b4_opus.json → semantic_new_table.beats_worstcase_generic`）。
对照组同一口径：Gaussian 只有 1/8 超过，active-nonselective 3/8，semantic_new 14/24。
q+ 与逐方向泛型地板的 Spearman 仅 0.05，所以高分**不是**残余的均值/泛型混淆。

**为什么这条修正重要**：它把研究问题从“有没有微弱信号”（无聊、且注定要靠大 n 拼显著性）
换成“**什么样的 SAE 特征方向是可读的，什么样的不可读**”——一个 24 个样本就能看出结构、
且有明确机制假设可测的问题。同时它给 C2-v2 一个可用的工作点：结合已有 surface audit，
在 q+>0.3 的 10 个特征里，可判定的 7 个有 6 个至少粗粒度正确（4 strict + 2 coarse，
1 mismatch，3 因 AV 默认英文输出而无法判定）；q+<0.3 的 14 个里只有 6/14 至少粗正确。
即**阈值 0.3 处的精确率约 6/7 vs 6/14**。这仍不是验收证书（审计是非盲、单评审者、
且评审者看得到 q+），但它比“5/24 strict match”这个把 14 个无信号特征混进分母的
数字更有信息量。

### 2.2 修正二：`ρ=−0.015` 支持不了任何结论（Simpson 悖论 + 并列伪值）

Sol 把 “heldout AUC 与 q+ 的 Spearman ρ=−0.015” 当作 “最大的瓶颈…NLA 越容易重建某个
方向，并不代表该 SAE 特征越单义” 的主要证据。复核结果：

**(a) 分层后符号相反。** 我用 `b6b4_factorial_selection.json → selected_directions[].test.auc`
与逐特征 q+ 重算（`local_recheck_stratified_opus.json`）：

| 分层 | n | Spearman(q+, test AUC) | median test AUC |
|---|---:|---:|---:|
| 合并（Sol 报告口径） | 24 | −0.02 | — |
| **domain** | 15 | **+0.40** | 0.50 |
| language | 9 | −0.78 | **1.00** |

language 层 8/9 个特征的 test AUC 顶到 **1.0**（语言特征在译文文档上当然满分选择性），
是天花板饱和；在这种近常数变量上算相关系数没有意义。合并 ρ≈0 是两个异质分层相抵的
结果，不是“无关系”。

**(b) 更严重：8/24 个特征的 AUC 根本不是测量值。** 这 8 个特征在 test split 上
**完全不激活**（`pos_mean = neg_mean = 0`，`effect = 0`），AUC 按并列约定记为 0.5。
其中三个恰是全队列 q+ 最高的：f2725（q+ **0.673**）、f14470（0.507）、f10000（0.458）。

**(c) 而且这个 AUC 本身分辨率极低**：每个特征的 test 只有 3 正 / 9 负（domain）或
4 正 / 8 负（language）个文档，AUC 在整个队列里只取到 8 个离散值
（0.389/0.5/0.625/0.812/0.926/0.938/1.0 + 伪值 0.5）。

**正确表述**：这份数据**没有能力**检验“内部 round-trip 分数是否预测外部选择性”，
因为外部变量在 8/24 上是伪值、在 9 个语言特征上饱和、整体只有 12 个文档的分辨率。
ρ=−0.015 是**测量失败**，不是**已证明的零关系**。把“没测出相关”写成“证明了不相关”，
是这轮分析里唯一实质性的推理错误，而它恰好是 Sol 用来支撑
“Closed-loop self-consistency is not interpretability” 的主要定量支柱之一。

（该论题**本身**仍有其他证据支撑：surface audit 的 9/24 明显错配、f14470 的
rank=1 却解释成 Minecraft、盲 judge 对 NLA 原文只有 2/24 通过。论题不倒，倒的是这根柱子。）

### 2.3 修正三：C2-v2 的门禁顺序会剔掉方法唯一work的部分

Sol 推荐的产品形态是
`held-out activation/context gate → +w_dec AV → q+/retrieval 分诊 → 人工验收`。
但按 2.2(b)，heldout gate 淘汰的 10 个特征里包含 q+ 排名第 1、第 3、第 7 的三个
（f2725 疫苗、f14470 印刷史、f10000 消息队列），其中 f2725 与 f10000 还是 surface audit
判定“主题正确”的少数派。**门禁与可读性在本队列里是反相关的。**

原因不难理解：`w_dec` 方向的可读性是**嵌入几何属性**，而 heldout 激活率是**语料属性**。
一个完全单义的特征，在 12 篇文档里一次都不触发是完全正常的。用 12 篇合成文档去判定
“这个特征值不值得标注”，淘汰的是语料覆盖度，不是特征质量。

**修正方案**：门禁必须建立在**大规模自然语料**上的激活统计（这也是 Neuronpedia /
gemma-scope 的标准做法），而不是 24 篇合成文档的 test split；在小语料上，
“未触发”只能记为 *无证据*，不能记为 *未通过*。

---

## 3. 流程与优先级问题（比上面三条更值钱的教训）

### 3.1 语料自审门禁把整个会话吃掉了，零新增数据

07-30 14:07 → 15:48 的时间线（文件 mtime 可查）：
preregistration v1 → Stage0 freeze v1 → 生成 v1 **FAIL**（字数/格式）→ v2 amendment →
Stage0 freeze v2 → 144 条全新生成 → 两名 reviewer 双双 **FAIL**（10 个文档问题）→
v3 anchors → 独立审计 **FAIL**（20/24 概念 heldout 机制重叠）→ v3r2（改写 33/144）→
独立审计仍 **FAIL**（17 PASS / 7 FAIL）→ 按冻结规则停止。

**没有抽取任何 activation，没有选任何 feature，没有跑任何 AV/AR。** 而卡在门禁上的
判据是自己定的——“同一个窄概念内 heldout 必须用不同的实质机制”。Sol 自己在停止记录里
诊断出了这一点（“可能把概念泛化与过强的机制独立性混在一起”），这个诊断是对的。

**根因不是 rubric 写歪了，而是把“自己生成评测语料”放在了关键路径上。**
合成语料必然要审计，审计必然要 rubric，rubric 必然会在边界上失败；再改第 5 版 rubric
只会得到第 5 次 FAIL。SAE 特征评测的正确底座是**真实语料**（Wikipedia / C4 / 代码 /
多语言真实文本），heldout 按**文档来源**切分——这既不需要审计，也天然消除
“hard negative 污染”“非实例”“禁用专名”这些全部由生成过程引入的问题。

### 3.2 推荐方案有一个不存在的资源依赖

Sol 的 C1-confirmatory 要求 “至少 60–100 个 feature + **至少 3 名盲评人类评审**”。
这个项目的人力是**一个人 + 一台按时计费的 GPU**。把 3 名盲评人写进主终点，等于
把论文主线挂在一个不存在的资源上——这也是为什么执行时会不由自主地退化成
“让模型互相审计语料”的替代品。

可行替代（都不需要人类评审团）：
1. **异构模型盲评**：用**非 Gemma 家族**的多个模型（Claude / GPT / Qwen 各一）做盲评，
   彼此不同源，可算 inter-rater agreement。Sol 判定 judge 失效的那个 judge 是 Gemma
   自己——同源自偏是它失效的主要嫌疑，换家族是最直接的修法。
2. **客观端点代替评分**：因果 patch（见 3.3）与行为干预不需要任何评审者。
3. **人类只做小样本抽检**：让 Jason 本人盲评 20–30 条，用于校准模型评审者，而不是充当主终点。

### 3.3 全项目缺一个因果端点——而它是最便宜的那个实验

Fable 在第一天就把 “loss recovered / KL” 列为方向 4，此后**从未被执行**
（全仓库搜索 `KL|loss_recovered|patch` 只命中 Prompt.md 的纪律条款）。
现状是：所有证据都是激活空间里的余弦几何 + 一个同源 critic 打的分。

这正是那个能一次性回答“这些 cos 到底有没有意义”的测量：把重建向量 patch 回
base model 的 L32 继续前向，测下一 token 分布相对干净前向的 KL 与 loss recovered。
它是 SAE 文献的标准指标（审稿人一定会问），**不需要人类评审、不需要新语料、
不需要 AV 生成**——40 个位置、5 条 prompt、重建向量都已在服务器
`/root/autodl-tmp/results/recon_vectors.npz`，token 位置在 `nla_results.json → rows[].position`，
prompt 在 `server/02_extract_activations.py → DEFAULT_PROMPTS`。一次开机 20–30 分钟。

它还能顺手判掉一个悬着的问题：**如果 NLA 的 0.859 在 KL 上并不比 SAE 的 0.658 更好，
那 F1 这个唯一的正面结果就要重新解释；如果更好，路线①就拿到了领域通用货币的背书。**

### 3.4 成本

本次只读探测：实例在线，GPU **利用率 0%、显存 0 MiB**。按 07-26 的 keep-alive 指令，
A800 已空转约 4 天，而这期间真实 GPU 工作量是 C1 pilot 的几十分钟。
AutoDL 的数据盘关机不丢，且支持**无卡模式**保住实例与环境。
建议规则改为：**需要跑批时开卡；跑完转无卡模式或关机**——而不是长期带卡待机。
（此项需 Jason 决定，本次未执行任何关机动作。）

---

## 4. 下一步方案（按“证据增量 / 成本”排序）

### N1 · C7+B3：第三方等义改写与受控实体替换 —— ✅ 已执行，见 §6
- 本地由**我**（非 Gemma 家族的第三方）为 40 条主线解释各写：
  (a) 命题与长度匹配的等义改写 ×2，(b) 受控最小编辑版（Eiffel→London Eye 等实体替换、
  否认句、体裁替换），全部在打分前冻结并存盘。
- 服务器上只跑 AR 重打分（33 层截断、秒级/条，**无 AV 生成**），去均值口径。
- **判定**：等义改写保留大半分数 → 文本的人类可读成分承载信息，循环性风险排除；
  分数崩 → 所有“自然语言可读性”主张降级为 private code，论文主线随之确定。
  实体替换 Δq≈0 → F6 的相关性警告升级为因果证据。
- 成本：本地 1–2 小时（我写改写）+ **GPU 约 10 分钟**。Sol 指出的 pilot paraphrase
  “只保留 60.1% 字符”的混杂，正好由“长度匹配 + 非同源作者”一次修掉。

### N2 · E10：因果 patch-in（KL / loss recovered）—— ✅ 已执行，见 §6
- 对 40 个位置，分别把 NLA 重建、SAE-small/big 重建、残差文本重建、
  数据集均值向量、随机同范数向量 patch 回 L32，测下一 token 分布 KL、
  后续 token 的 loss recovered，并与 zero-ablation / mean-ablation 归一化。
- **判定**：给出五种方法在领域标准货币下的排序；同时检验 centered cos 0.859 与
  KL 的相关性——若强相关，去均值 cos 作为廉价代理指标被验证（本身就是方法论成果）。
- 成本：**GPU 20–30 分钟**，零新增生成。可与 N1 合并成一次开机。

### N3 · 真实语料底座（替换合成语料，修掉 2.3 与 3.1 的根因）
- 从 HF 拉 5–10 万 token 的真实文本（多语言 Wikipedia + 代码 + 论坛），对 16k 特征
  算激活统计，得到**真实的 max-activating contexts、触发频率、跨来源选择性**。
- 用它重建 gate：`有足够真实激活证据` 而非 `在 12 篇合成文档里触发`；heldout 按
  **文档来源**切分（不需要任何 rubric 审计）。
- 顺带回答 2.1 的新主问题：**什么预测可读性**——候选自变量有触发频率、激活稀疏度、
  `w_dec` 与均值方向夹角、`w_dec` 与 embedding/unembedding 空间的对齐、特征所属语义类别。
- 成本：抽激活 **GPU 1–2 小时**（可断点续跑），之后的分析全部本地零 GPU。
- **这是 C1/C2 想做成任何正经东西的前提**，也是唯一能把 n 从 24 提到 100+ 而不需要
  自己生成语料的路径。

### N4 · C1-confirmatory（只有 N1–N3 都过了才启动）
- Cohort 来自 N3 的真实语料统计，≥60 个从未查看的特征，按语义簇分层冻结。
- 主终点改为双端点：(i) `q+` / matched delta 能否预测**异构模型盲评**（≥2 个非 Gemma
  家族，报 agreement）；(ii) 能否预测 N2 的**因果端点**。人类只做抽检校准。
- 统计单位是特征，按语义簇聚类推断；一切在看 AV/AR 输出前冻结。
- 若 (i)(ii) 都不成立 → 论文收敛为 `closed-loop communication code ≠ interpretability`，
  此时已有 N1（private code 直接证据）+ N2（因果对照）+ 2.1（双峰 + 可读性预测因子）
  三块独立证据，比现在这版更硬。

**顺序建议**：N1+N2 合成一次开机（≈40 分钟 GPU）→ 看结果决定 N3 规模 → N4。
在 N1/N2 出结果之前，不要再动 C1 语料设计。

**执行结果（2026-07-30 晚）**：N1+N2 已在一次开机内跑完（GPU 约 6 分钟，见 §6）。
两项判定都出了明确答案，且都改写了上面的方案：N1 的 C7 **通过**（不再是论文分岔点，
而是"通道拆解"这条正面主线的起点），N2 显示 **cos 不是可靠的因果代理**（N2 的第二个
判定给出否定答案：跨方法相关但方法内不一致），且 **NLA 的 cos 优势不转化为因果优势**。
N4 的双端点设计因此更重要：因果端点已就位，可直接作主终点之一。

---

## 5. 本次复核的产物与出处

| 文件 | 内容 |
|---|---|
| `server/25_local_recheck.py` / `results/local_recheck_opus.json` | F1 复现、预言机基线、centered FVE、B2 的 z-margin |
| `server/26_local_recheck_b6b4.py` / `results/local_recheck_b6b4_opus.json` | 逐方向泛型地板校正、双峰分布、方向几何、逐特征表 |
| `server/27_local_recheck_stratified.py` / `results/local_recheck_stratified_opus.json` | 分层 Spearman、test-dead 特征识别、q+ 分布断层 |

§1–§5 阶段未修改任何既有 `results/` 文件；未新增任何 GPU 作业；未执行关机。
§2 的修正已回灌为 **F14**，§6 的新实验回灌为 **F11–F13**（见 `POSSBILITY.md`）。

---

## 6. N1+N2 执行结果（2026-07-30 晚，一次开机 ≈6 分钟 GPU）

### 6.1 冻结与自证

- `server/28_build_text_variants.py` 本地生成 40 行 × 11 条件 + 8 条固定泛型文本，
  **打分前冻结** → `c7b3_variants_v1.json`（sha256 `b30a74be…`，含逐条撰写规则、
  作者声明、诊断字段）。C7 改写由 Claude Opus 5 手写（`c7b3_paraphrases_opus_v1.txt`），
  字符比 1.06–1.12、命题与引号逐字保留；B3 替换表 `b3_entity_swap_map_v1.json`
  共触发 214 处替换。
- `29_score_text_variants.py` 只调 `NLACritic.reconstruct()`，420 条文本 17 秒。
  **原文复算 0.8593，与 E5 的 `head_to_head_centered.nla_mean` 逐位相同** → 口径无误。
- `30_causal_patch.py` 的 `identity` KL **0.0000**、provenance cos **1.000000**（40/40）、
  `gaussian` 31.07、`zero` 16.34 → 阴性与阳性对照都干净。

### 6.2 结果（详表见 `POSSBILITY.md` F11/F12）

| 判定项 | 结果 |
|---|---|
| C7（私有编码风险） | ✅ **通过**，第三方长度匹配等义改写保留 **97.3%** |
| B3（指称身份是否在通道里） | ❌ 不在，替换后只掉 **10.3%**，检索命中率反而升到 95% |
| 通道定位（新） | 承载信号的是第 3 段"下一 token 候选串"：只留它 96.1%，抹掉它 20.8% |
| cos 是否可作因果廉价代理 | ❌ 跨方法 ρ=−0.402，但方法内不一致（NLA **+0.188**） |
| NLA vs SAE 的因果排序 | **NLA ≈ SAE-big > SAE-small**；KL-recovered 0.757/0.771/0.713 |
| CE 版 loss-recovered | ❌ 作废，评测窗口被 chat 模板 token 支配（干净 CE 21 nats） |
| 附带发现 | E1–E7 队列 **13/40 是模板/空白 token**（F13） |

### 6.3 这三个发现如何改写项目定位

1. **正面主线换了**：不再是"NLA 重建比 SAE 好"（F12 否掉），而是
   **"NLA 学到的是一个可读的下一 token 预测器"**——F11 用 11 个冻结条件把这件事测出来了，
   并一次性解释 F6（编造实体不掉分）、F7（离流形模板化）、F10（抽象轴标签 q 为负而
   含真实词汇的 train reference 高达 0.1925）。
2. **Sol 的"closed-loop self-consistency is not interpretability"方向正确但归因不足**：
   闭环分数确实不测语义忠实度，**但原因不是"私有编码"（C7 已排除），而是通道里装的
   是候选续写而不是语义**。这个归因比原命题强，因为它可测、可迁移、有正面内容。
3. **成本纪律的教训**：N1+N2 合计 6 分钟 GPU，产出的证据量超过前两轮 200+ 分钟 GPU 的
   任何单项。真正贵的从来不是算力，是没先想清楚测什么。

### 6.4 新增产物

| 文件 | 内容 |
|---|---|
| `server/28_build_text_variants.py` | 变体生成与冻结（本地，零 GPU） |
| `server/29_score_text_variants.py` | AR 重打分（含原文复算校验、逐变体检索） |
| `server/30_causal_patch.py` | 因果 patch-in（12 个替换源、norm-matched、含 provenance） |
| `server/31_analyze_n1n2.py` | 稳健汇总：KL-recovered、配对 bootstrap CI、文档聚类、token 分层 |
| `server/32_diag_ce_window.py` | CE 窗口诊断（证明 loss-recovered 端点作废的原因） |
| `results/c7b3_variants_v1.json` + `.sha256` | 冻结文本变体 |
| `results/c7b3_scores_v1.json` / `c7b3_recon_v1.npz` | N1 分数与重建向量 |
| `results/causal_patch_v1.json` | N2 原始结果 |
| `results/n1n2_analysis.json` | N1+N2 稳健分析汇总 |
| `results/n1n2.log` / `n1n2_checksums.sha256` | 运行日志与校验和 |

### 6.5 下一步（在 §4 基础上重排）

1. **N3 真实语料底座**（1–2h GPU）——不变，仍是一切正经工作的前提，且现在多了一个
   必须的任务：**重抽 E1–E7 级队列**（F13 的模板 token 缺陷）。
2. **通道消融 + 因果 patch 在真实语料/多层上复现**（每层 ≈10 分钟 GPU）——F11/F12 目前
   都建立在 n=40 且 32.5% 是模板 token 的队列上，这是当前最大的效度风险。
   若 F11 在真实长文本上仍然成立（"下一 token 候选串"承载 >90%），这就是可发表的主结果。
3. **N4 重新设计的 C1**：主终点改为"异构模型盲评 + 因果端点"双终点，人类只做抽检校准。
4. **不要再重启 C1 合成语料自审**（§3.1 的根因未修）。
