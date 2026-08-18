# Conclude — NLA vs SAE 项目现状详尽总结

> 初稿:2026-07-11。作者:Claude(Fable 5),与 Jason 协作完成。B2 与 E8 由 Codex 于
> 2026-07-26 补充并完成正式统计审查；C1 protocol pilot 于 2026-07-30 完成。
> 本文档面向**接手本项目的下一位研究者/模型**,自足可读,不依赖任何对话历史。
> 配套文件:`Prompt.md`(接手者操作指南)、`results/POSSBILITY.md`(可行性评估正文)、
> `THOUGHT.md`(研究方向发散)、`README.md`(操作与踩坑)。

---

## 1. 项目是什么

对 **NLA(Natural Language Autoencoder,kitft/natural_language_autoencoders)** 与
**SAE(Gemma Scope 2 JumpReLU)** 在同一组激活上做往返重建对比,并延伸探索
"NLA 能否辅助 SAE 做审计与可解释性研究"。

- **对象**:google/gemma-3-12b-it 第 32 层 resid_post 激活(抽取点 `model.layers[32]` 块输出,
  与 NLA L32 检查点的训练点、gemma-scope-2 layer_32 SAE 的 hook 点完全一致)
- **NLA 线**:激活向量 → AV(verbalizer,12B 微调模型,向量经嵌入注入)生成 2-3 段英文解释
  → AR(reconstructor,33 层截断模型 + value head)从文本重建向量
- **SAE 线**:JumpReLU 编码/解码,w16k,两档稀疏度(config L0=20 实测 15.2;config L0=120 实测 75.5)
- **公共指标**:方向余弦 cos 与 mse_nrm=2(1−cos);SAE 另有原生 FVE/L0
- **E1–E7 数据**:5 条英文 instruct prompt(埃菲尔铁塔/光合作用/印刷术/猫狗宠物/议会立法),
  每条取 position≥50 起均匀 8 个位置,共 **40 个激活向量**(d=3840,原始未归一化,
  均值 L2≈67024)
- **E8 factorial 数据**:4 个 domain(software/geography/biology/history)×2 个
  topic(train/test)×3 种语言(en/zh/es),共 **24 个文档、8 个独立内容 topic、1365 个
  token 激活**。选择只看 train split;test split 只用于记录/分层 heldout label-selectivity。

## 2. 已完成的全部实验(时间序)

| # | 实验 | 脚本 | 结果文件 | 状态 |
|---|---|---|---|---|
| E1 | 主线对比(抽激活→NLA→SAE×2→合并) | server/02–05 | nla_results / sae_results(_big) / comparison.* | ✅ |
| E2 | Pilot 1:NLA 读 SAE w_dec 方向(top12+rand6+gauss4) | 06_pilot_wdec.py | wdec_pilot.json | ✅ |
| E3 | Pilot 2:NLA 读 SAE 残差(暗物质) | 07_pilot_residual.py | resid_pilot.json | ✅ |
| E4 | Pilot 3:残差注入敏感性(4特征×4残差×α∈{0,.25,.5,1,2}) | 08_pilot_injection.py | injection_pilot.json | ✅ |
| E5 | **修正实验:去均值空间重打分全部解释** | 09_rescore_centered.py | centered_rescore.json + recon_vectors.npz | ✅ |
| E6 | 解释文本统计(模板化/编造普查/特异性),本地纯标准库 | (子代理完成,无脚本存档) | 数字已写入 POSSBILITY.md F6/F7 与附录 A | ✅ |
| E7 | **B2 centered 40-way 检索**(Top-k/MRR/margin/置换/文档内诊断) | 10_retrieval_eval.py | retrieval_eval.json + b2_retrieval.log | ✅ |
| E8 | **B6+B4 factorial 语义选择、±方向、随机稳定性与 carrier readout** | 11–14 + run_b6b4_factorial.sh | b6b4_factorial_selection/result/analysis + NPZ/JSONL | ✅ |
| E9 | **C1 external-validity protocol pilot**（正确/错配/generic/autointerp/NLA/paraphrase） | 15–19 + run_c1_pilot.sh | c1_pilot_benchmark/result/analysis/secondary/validation + NPZ/JSONL | ✅ pilot |
| E10 | **因果 patch-in（KL / KL-recovered，领域标准指标）** | 30 + 32 + run_n1n2.sh | causal_patch_v1.json | ✅ |
| E11 | **N1：C7 第三方改写 + B3 实体替换 + 文本通道 11 条件消融** | 28 + 29 + 31 | c7b3_variants/scores/recon + n1n2_analysis.json | ✅ |

**E5 是全项目的转折点**:E4 发现 α=0 假阳性地板高达 0.92,追查出贯穿 E1–E4 的系统性混淆——
高频 SAE 特征的 w_dec 方向与数据集均值方向对齐达 ±0.89~0.96,而 AR 对任何泛型文本的重建都落在
均值方向附近,导致一切未去均值 cos 虚高。E5 用 `NLACritic.reconstruct()`(文本→向量,单次前向)
对全部 182 条已有解释重打分(投影掉均值方向后再算 cos),**零新增生成**,10 分钟完成修正。

**E8 是路线②/④的正式压力测试**:先用 train 文档级 SAE 激活冻结 24 个新 semantic
feature,再加入 4 legacy、1 structural、8 active-nonselective 与 8 Gaussian,共 **45 个方向**。
每方向测 ±v,以 1 次 greedy 为主结果、4 次 temperature=0.7 为稳定性估计;另对 28 个
semantic 方向做 high-carrier amplify/ablate 与 low-carrier insert,总计 **590 个 AV job
(450 direction + 140 carrier)**。选择 manifest 在 AV/AR 前冻结;正式结果、重建向量和分层
分析均已回传本地。

**E10/E11 是第三轮(Opus 5)的转折点**:E11 用 11 个冻结条件把解释文本拆开,证明承载
重建的是第 3 段里那串"下一 token 候选"(只留第 3 段保留 96%,抹掉其引号内容只剩 21%,
而对原文的逐字引用贡献 ≈0),同时第三方长度匹配等义改写保留 97%(**C7 通过,私有编码
风险排除**)、受控实体替换只掉 10%(**B3:指称身份不在通道里**)。E10 首次把重建 patch 回
基座模型:`identity` KL=0、provenance cos=1.000000、gauss 31.07、zero 16.34 对照干净,
但 **NLA 的 KL-recovered 0.757 与 SAE-big 0.771 打平**,内容 token 上还略输
(0.776 vs 0.865),相对 SAE-big 的 KL 配对 CI 跨零。**去均值 cos 上"NLA 误差不到 SAE 一半"
的优势在因果口径下消失。** 另外 E10 顺带暴露 E1–E7 的取样缺陷:13/40 是 chat 模板 token
(详见 POSSBILITY F11–F13)。

**E9 是被既往查看污染的 protocol pilot，不是确认性 C1**：24 个 feature 继承 E8，
只有 7 个历史冻结粗轴；细 reference 是事后撰写的。主检验仅比较冻结粗轴正确文本与
受控错误轴，其他指标均为探索性。成功 v2 含 432 个候选，432/432 judge 可解析，
旧 B6 NLA 原文的 AR 重算误差为 1.04e−9；独立数值验证全部通过。

## 3. 核心数字(去均值 = 可信口径)

| 指标 | 未去均值 | 去均值 | 出处(results/ 下) |
|---|---|---|---|
| NLA 重建真实激活 | 0.996 | **0.859** | nla_results;centered_rescore.head_to_head_centered |
| SAE-small 重建 | 0.9925 | 0.658 | sae_results;同上 |
| SAE-big 重建 | 0.9936 | 0.725 | sae_results_big;同上 |
| 残差解释弱信号对照(vs 全向量) | 0.975 | 0.041 | resid_pilot.cos_rx;centered_rescore.resid_centered |
| w_dec top 特征往返 | 0.80 (median\|cos\|) | 0.189 (mean\|cos\|) | wdec_pilot;centered_rescore.wdec_centered |
| w_dec rand / gauss 对照 | 0.30 / 0.013 | 0.105 / 0.011 | 同上 |
| 残差可读性 cos_rr | 0.036 | −0.029 | resid_pilot;centered_rescore.resid_centered |
| 注入检测 α=2(信号=2×残差范数) | 0.919(地板 0.919,平坦) | 0.092(地板 0.072,平坦) | injection_pilot.summary;centered_rescore.injection_centered_curve |
| 特征-均值对齐(4 个注入特征) | — | −0.958 / −0.950 / +0.911 / +0.893 | centered_rescore.feature_mean_alignment |
| SAE FVE(small/big) | 0.608 / 0.675 | — | sae_results*.summary.fve |
| **固定泛型文本地板(直接测)** | — | **−0.005** | c7b3_scores_v1.summary_by_variant.__generic_fixed__ |
| **第三方等义改写保留率(C7)** | — | **97.3%** | n1n2_analysis.n1_variant_summary.para_tp |
| **实体替换保留率(B3)** | — | **89.7%** | 同上 .entity_swap |
| **只留"下一 token 预测"段** | — | **96.1%** | 同上 .p3_only |
| **抹掉该段引号内候选串** | — | **20.8%** | 同上 .quote_strip_p3 |
| **KL-recovered:NLA/SAE-b/SAE-s** | — | **0.757 / 0.771 / 0.713** | n1n2_analysis.n2_kl_recovered |
| **mean KL@pos:NLA/SAE-b/SAE-s/zero/gauss** | — | **2.41 / 3.25 / 4.14 / 16.34 / 31.07** | 同上 |
| **队列取样缺陷** | — | **13/40 是 chat 模板 token** | n1n2_analysis.cohort_token_composition |

B2 的判别式结果另报:Top-1 为 NLA **92.5%**、SAE-small/big **95.0%/95.0%**,
Top-5 三者均 100%;mean margin 为 NLA **0.3195**、SAE-small/big **0.2085/0.2531**。
因此 B2 没有支持“NLA Top-k 高于 SAE”,但支持“NLA 正确配对与最佳错配之间的连续间隔更大”。

E8 的正式数字如下。`q+` 是 `+w_dec` 解释经 AR 重建后与正方向的 centered cos;
`r−` 是 `−w_dec` 的符号校正分数;`polarity=(q+ + r−)/2`。括号内为 feature-bootstrap
median 95% CI;重复生成不增加统计 n。

| E8 指标 | 正式结果 | 出处 |
|---|---|---|
| train-only 选择 → heldout-valid | 全部 **14/24=58.3%**;domain **6/15=40.0%**;language **8/9=88.9%** | b6b4_factorial_analysis.selection_yield |
| semantic-new ITT q+ / r− / polarity median | **0.114** [0.009,0.453] / **0.031** / **0.071** [0.020,0.245] | b6b4_factorial_result.summary_by_cohort_greedy |
| heldout-valid q+ / r− / polarity median | **0.181** [0.018,0.446] / **0.042** / **0.102** [0.024,0.260] | 同上 |
| domain heldout-valid(条件子组,n=6) | q+ **0.407**;r− 0.039;polarity 0.224;sign accuracy 100% | b6b4_factorial_analysis.direction_summary |
| language heldout-valid(n=8) | q+ **0.066**;r− 0.046;polarity 0.052;sign accuracy 75% | 同上 |
| active / Gaussian q+ median | −0.004 / 0.005;semantic-minus-active 冻结配对差 median **0.045**(6/8 为正) | result + analysis.active_control_frozen_pairs |
| signed Top-1(90 个 signed candidates) | ITT **33.3%**;heldout-valid **39.3%**;active 18.8%;Gaussian 0% | b6b4_factorial_analysis.direction_summary |
| 固定 generic text 地板 | 所有方向 mean \|cos_c\| **0.034**;semantic-new **0.037**;单项最大 0.330 | b6b4_factorial_result.generic_control |
| heldout AUC 与 q+ | Spearman **−0.015**(几乎无单调关系) | b6b4_factorial_analysis |
| stochastic 稳定性(每 feature 4 次抽样) | ITT greedy-vs-stochastic q+ Spearman 0.943;heldout-valid 0.987;平均 sign consistency 66.7% / 78.6% | b6b4_factorial_analysis.stochastic_summary |
| carrier(ITT / nonzero / heldout-valid) | ITT 有 **9/24 no-op**;nonzero/valid 的 ablate median cos **0.210/0.206**、insert **0.097/0.113**、amplify **0.042/0.019** | b6b4_factorial_analysis.carrier_summary |
| `+w_dec` 表面语义审计 | strict 5/24;coarse 7/24;mismatch 9/24;indeterminate 3/24;至少 22/24 含证据未支持的具体化 | b6b4_surface_audit.md |

C1 pilot 的关键数字：

| C1 指标 | 结果 | 含义 |
|---|---:|---|
| 冻结 axis reference − hard negative | mean **0.0155**；median **0.0090**；16/24 正 | sign-flip mean p=.00393，但 sign test p=.0758；小而非多数稳健 |
| 对全部六个错误轴 | correct−mean-wrong 17/24 正；feature→7-axis Top-1 **50%** | 有粗轴排序信息，correct−best-wrong 仅 12/24 |
| axis delta → heldout-valid | AUC **0.764**；vs test AUC rho **0.501** | 事后描述性、值得在新 cohort 预注册复现 |
| train reference − hard negative / sibling | mean **0.195 / 0.108** | 强 train 指纹；heldout-invalid 反而更强 |
| NLA original 24-way text→feature Top-1 | **62.5%** | 身份检索强，但 blind context judge 仅 2/24 通过 |
| private-code interaction | mean **0.104**；17/24 正 | NLA paraphrase 只保留 60.1% 字符，不能排除压缩混杂 |
| pooled q → base judge | rho **−0.042**；AUC **0.410** | 同源 judge 不能作 quantitative oracle |

## 4. 十四个核心发现(F1–F8 详见 results/POSSBILITY.md;F9/F10 以 E8/E9 资产为准;F11–F14 以 E10/E11 与第三轮复算为准)

> **读 F1 前必读 F12**:F1 的优势在因果口径下不成立。

1. **F1** 在真实激活上 NLA 的去均值 cos 显著高于 SAE(0.859 vs 0.658/0.725);
   **但 F12 证明这个差距不转化为因果优势**,故 F1 只能作为方向保真度陈述。
2. **F2** 未去均值 cos 存在 0.975 的"泛型文本地板";去均值 + 报告地板是必须的评测卫生。
3. **F3** Pilot 1 的"字典方向强可读"大部分是均值混淆;真实可读性仅 0.2–0.45,且只在
   有清晰语义足迹的特征上(f1491 光合作用 0.45、f5389 百科体 0.37);
   E8 进一步表明读数高度不对称:`+w_dec` 有可重复信号,`−w_dec` 的 r− 虽常为正确符号,
   但幅度接近 generic floor,且存在乱码仍获较高 AR alignment 的案例。因此是弱 signed-axis
   discrimination,不是高保真“反义 feature”解释。
4. **F4** SAE 残差(暗物质)对 NLA 完全不可读(去均值 −0.03)。
5. **F5** 往残差里注入已知特征方向(最强 2× 残差范数≈激活范数 25%)也检不出——检测曲线平坦;
   机制:AV 把整个离流形邻域 collapse 成同一句 "Article structure…" 模板(解释文本不随 α 变化)。
6. **F6** 解释会编造具体实体(3/40:London Eye、Chicago Museum of Illusions、Chinese Crested Dog),
   且编造行与正确行的 cos 完全相同(0.9962=0.9962)——**高 cos ≠ 解释忠实**,AR 主要消费
   体裁/话题大类/局部句法期望。
7. **F7** 解释高度模板化,越离流形越模板化(残差解释的 4-gram Jaccard 是全向量解释的 3.1 倍)
   ——模板化程度可反过来当"离流形检测器"。
8. **F8** B2 的离散 Top-k 已饱和且 SAE 略高(NLA 92.5% vs SAE 95%),没有复现预设的
   “NLA 检索胜出”;但 NLA mean margin 在 5 个文档上都高于两档 SAE。只有 5 个文档,
   该 margin 优势是探索性证据,不能夸成确认性显著。
9. **F9** E8 证明了三件必须分开的事:
   - **方向可通信**:`+w_dec` 的 q+、检索与 stochastic ranking 明显高于 Gaussian,也高于
     active control 的中位数;自然 carrier 中 ablate/insert 的 AR 差向量方向一致,说明存在
     可重复的 AV→AR 读出信号。
   - **语义正确性不由 AR 自证**:heldout AUC 与 q+ 的 Spearman=−0.015;存在 heldout-invalid
     但 q+ 很高的 feature,也存在 valid 但 q+ 很低/为负的 feature。非盲表面审计仅 5/24
     strict match、9/24 obvious mismatch,精确方向检索也可能对应错误的自然语言表面解释。
   - **选择是独立瓶颈**:domain 的 heldout yield 仅 40%,虽其 6 个 survivor 的条件 q+
     median 达 0.407;language yield 88.9%,但 survivor q+ median 仅 0.066。正确架构必须把
     activation/context generalization gate 与 NLA/AR round-trip gate 串联,不能用后者替代前者。
10. **F10** C1 pilot 没有救回“AR 是 standalone correctness metric”，但发现
    matched `正确轴−错误轴` delta 可能预测 heldout selectivity；绝对 q、同源 judge 和
    NLA 原文身份检索均不能验真。private-code 方向有提示，但与文本压缩严重混杂。
11. **F11(E11)** 解释文本的通道被拆开了:承载重建的是第 3 段里那串"下一 token/短语候选"
    (只留第 3 段 96.1%,抹掉其引号内容剩 20.8%,逐字引用贡献 0.6%,体裁段 28.3%,
    词序打乱 66.9%);第三方长度匹配等义改写保留 **97.3%**(**C7 通过**),受控实体替换只掉
    **10.3%**(**B3:指称身份不在通道里**);真实固定泛型地板 **≈0**。
    **NLA 的"解释"实际是被语言化的下一 token 分布,不是语义解释。**
12. **F12(E10)** 因果 patch:NLA 的 KL-recovered **0.757** 与 SAE-big **0.771** 打平,
    内容 token 上 SAE-big 更好(0.865 vs 0.776),NLA 只在模板 token 占优;相对 SAE-big 的
    KL 配对 CI **[−0.21, 1.98]** 跨零,相对 SAE-small 成立([0.50, 3.21])。
    **cos 能排方法不能排样本**(方法内 Spearman 不一致甚至反向)。
13. **F13** E1–E7 队列 **13/40 是 chat 模板 token**(`--min-position 50` 对 28–38 token 的
    prompt 从未生效,走了取后半段的回退分支)。这重新解释了"结构 token 上 SAE 反超"这一
    早期观察;并使基于同窗口的 loss-recovered 端点作废(干净 CE 21 nats > 均匀 12.48)。
14. **F14** 对 F9 的勘误:`q+` 是**双峰**(10 个 ≥0.362 且 9 个 rank=1,13 个 <0.15;
    median 0.114 落在 0.117 的断层里);`heldout AUC 与 q+ 的 ρ=−0.015` 是 Simpson 悖论
    加并列伪值(domain 层 +0.40,language 层 AUC 饱和于 1.0,8/24 特征在 test 上完全不激活
    致 AUC=0.5 为伪值),**只能说这份数据没能力检验该关联**;并因此推出 C2-v2 的
    heldout gate 顺序有误(会剔掉最可读的三个特征)。

## 5. 路线裁决(当前共识)

| 路线 | 裁决 |
|---|---|
| ① NLA 读真实激活(分诊漏斗深读层) | ⚠️ **降级(F12)**:因果口径下与 SAE-big 相当、仅强于 SAE-small;每向量一次 12B 生成的成本要为此付账 |
| ② NLA 标注 SAE 特征(+AR 往返质检) | ⚠️ **条件可行**。C2-v2 必须先过 activation/context heldout gate,默认只把 `+w_dec` 当语义方向;q+/retrieval 仅作 triage,不得自动采信文本 |
| ③ NLA 审计暗物质/藏匿信号 | ❌ 当前判负;翻案唯一路径 = off-manifold 课程训练 AV(训练级立项) |
| ④ AR / 外部 evaluator 评测 autointerp | 🔬 **C1 pilot 已完成、确认性 C1 仍是最优先论文问题**；relative delta 有信号，但 standalone q/judge 不成立 |
| ⑤ 评测方法论(去均值+泛型地板+对齐混淆+因果口径+通道消融) | ✅ 已成立并被 F11–F13 大幅加强,独立成果,可直接写作 |
| ⑥ 研究对象重定义:"NLA 学到的是可读的下一 token 预测器" | 🆕 **F11+F12 直接支持,当前最有论文价值的一条** |
| ⑦ AV/AR 私有编码风险(C7) | ✅ **已排除**(第三方长度匹配改写保留 97.3%) |

这里要区分“**研究问题优先级**”与“**现有方法已获支持**”:E9 只完成 C1 的协议 pilot。
“什么 evaluator 能预测人类/heldout/因果正确性”仍是最值得发表的问题；当前同源 AR
只是一个有用 baseline，不是 ground truth。C2-v2 只支持外部 gate 后的辅助质检。

## 6. 排好序的后续实验队列(编号沿用 THOUGHT.md)

> **2026-07-30 晚重排(第三轮)**。已完成:T4 的 C7+B3(E11)、因果端点(E10)。
> 新队列见 `results/REVIEW_OPUS_2026-07-30.md` §4:
> **N3 真实语料底座(1–2h GPU,替换合成语料,修掉 C2-v2 门禁与 C1 语料自审的根因)
> → 通道消融 + 因果 patch 在真实语料/多层上复现 → 重新设计的 C1(异构模型盲评 +
> 因果端点作双主终点,不再依赖 3 名人类评审)**。
> 下面的原队列条目保留作背景,但 C1-confirmatory 在 N3 之前不应重启。

1. **C1-confirmatory · 外部 evaluator benchmark(原论文主线,现排在 N3 之后)**:新冻结至少 60、优先
   100 个 feature，覆盖 ≥15–20 个 label cluster；在看 NLA/AR 输出前冻结 truth、
   hard negatives、generic 与统计方案。至少 3 名盲评者基于 contexts 评价 correctness、
   specificity 和 unsupported claims；主 target 是 matched AR delta/AUC 能否预测人类、
   heldout 与因果效度，并按 label/model 聚类。
2. **C7-v2 + 盲表面评测**:对 NLA/reference/autointerp 做多份命题与长度匹配的独立改写。
   E9 的一次 Gemma 压缩式 paraphrase 不能证明 private code。
3. **C2-v2 · activation/context gate 后的标注流水线**:扩大到每个 domain 多个独立 topic,
   先用 max-activating contexts 与 heldout activation selectivity 决定 feature 是否可进入
   标注;再读 `+w_dec`,用 q+/retrieval/stochastic consistency 排序并送人工复核。`−w_dec`
   只保留为 OOD 诊断,禁止用“±取最大”自动接受标签。
4. **B3 · 实体/事实受控替换**:Eiffel Tower→London Eye 等最小改写测 Δq/检索/外部 evaluator,
   把 F6/F9 的相关性警告升级为因果敏感性证据。
5. **跨模型/层/SAE 复现 E8**,并补充更严格的 matched controls、预注册阈值和行为因果 read/write
   测试;这是把 pilot 推向论文证据的必要扩展。
6. **B5 · 率失真曲线**、**B7 · 自一致性**、**C3 · 混合自编码器**、**C4 · 迭代剥离**:
   作为次级方向,见 THOUGHT.md。
7. **off-manifold 课程训练 AV**(救路线③):独立训练项目,等外部 evaluator 与 C2-v2 结果后再决定立项。

**B2、E8 与 C1 protocol pilot 均已完成**。C1-confirmatory 尚未开始；E9 产物见 §7。

## 7. 基础设施与资产

### 7.1 服务器(AutoDL,按时计费;当前按用户指令保持开启)
- 实例:A800-SXM4-80GB,bjb1 区,**当前为克隆实例**,`ssh autodl`
  (= connect.bjb1.seetacloud.com,端口 **11813**,root,密钥 `~/.ssh/autodl_nla` 已授权)。
  旧实例(端口 33325)已弃用;不要在项目文档中保存明文密码。
- 开机只能由用户在 AutoDL 控制台操作。**2026-07-30 用户再次明确要求保持实例开启;
  不得执行 `shutdown` 或设置自动关机链,直到用户另行改变指令。**
- 数据盘 `/root/autodl-tmp`(150G,已用 ~70G),关机不丢:
  - `models/`:gemma-3-12b-it(24G)、nla-gemma3-12b-L32-av(22G)、nla-gemma3-12b-L32-ar(16G)、
    gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_{l0_small,l0_big}(4 文件 961M)
  - `activations/acts_L32.parquet`(E1–E7,40×3840)与
    `activations/acts_L32_factorial_v1.parquet`(E8,24 文档/1365 token 激活)
  - `nla_repo/`(nla_inference.py,**已打 2 处 transformers 5 补丁**,见 §8)
  - `nla_compare/`(所有编号脚本 **00–19**、`semantic_prompts_factorial.jsonl`、
    `pilot_common.py` 与 runner)
  - `results/`(与本地 results/ 同步;含 E8/E9 JSON/JSONL/NPZ/MD)、各 *.log
- 环境:miniconda Python 3.12,torch 2.5.1+cu124,transformers 5.12.1,pyarrow 24。
  HF token 在 `/root/.hf_token`;下载须 shell 层 `export HF_ENDPOINT=https://hf-mirror.com`。

### 7.2 本地(Windows,D:\Projects\natural_language_autoencoder\)
- `connect.ps1` / `sync.ps1`(push|pull)/ `remote.py`(三者均读取 `Host autodl`;
  密钥认证,仓库不保存密码)
- `server/`:00 环境、01 下载、02 抽激活、03 NLA、04 SAE、05 合并、06–09 pilots、
  10 B2、11 factorial 抽激活、12 冻结 feature 选择、13 ±方向/随机重复/carrier 批处理、
  14 分层统计;另有 `semantic_prompts_factorial.jsonl`、`pilot_common.py` 与
  `run_b6b4_factorial.sh`/其他 runner
- `results/`:全部原始数据 + 日志 + POSSBILITY.md。E8 的 source-of-truth 资产:
  - `b6b4_factorial_selection.json` + `b6b4_factorial_feature_stats.npz`
    (AV/AR 前冻结的选择与全 feature 统计)
  - `b6b4_factorial_av_rows.jsonl`(590 个可恢复 AV job 的原始 completion)
  - `b6b4_factorial_result.json` + `b6b4_factorial_recon_vectors.npz`
    (AR 重建、全相似度与主结果)
  - `b6b4_factorial_analysis.json/.md`(ITT/heldout/domain/language/control/carrier 分层)
  - `b6b4_factorial_vectors.npz`、`b6b4_factorial.log`、`b6b4_surface_audit.md`
- `README.md`(顶部 = 状态与踩坑)、`THOUGHT.md`(方向发散)、本文件、`Prompt.md`

### 7.3 关键代码接口(复用,勿重写)
- `pilot_common.AVLocal(av_dir).generate(v)`:向量→解释文本(注入数学与训练配方一致;
  内部会归一化,输入尺度不敏感,但习惯上缩放到 target_norm≈67024)
- `pilot_common.JumpReLUSAE(sae_dir)(X)` → (recon, acts)
- `nla_inference.NLACritic(ar_dir).score(text, v)` → (mse_nrm, cos)(未去均值!)
- `NLACritic.reconstruct(text)` → 原始预测向量(**去均值分析全靠它**;AR 只有 33 层,
  秒级加载,不需要 AV 时实验极便宜)
- 去均值口径:`a_perp = a − (a·m̂)m̂` 双边投影后再 cos。E1–E7 的 m̂ 是 40 激活均值方向,
  存在 `recon_vectors.npz['m_hat']`;E8 的 m̂ 是 **train split 的 per-document mean
  再等权平均**所得冻结方向,存在 `b6b4_factorial_vectors.npz['m_hat']`,两者不可混用。
  AR 预测尺度 ≈ mse_scale=61.97,与原始激活尺度不同,**只能做方向投影,不能做仿射减均值**。
- `13_probe_factorial_polarity.py` 的 carrier 只是把人工改变后的 activation 交给 AV/AR
  读出;没有把向量写回 base model 并测下游行为,所以 **carrier readout 不是 steering**。

## 8. 踩坑全录(操作层,违反任意一条都可能烧钱或丢会话)

1. **AutoDL 的 `/usr/bin/shutdown` 可能忽略 `+N` 延时参数并立即断电**。当前用户要求
   保持实例开启,所以禁止调用它,也禁止创建任何 sleep-shutdown 链。
2. **用户明确偏好**:自治跑批**不设硬时限定时器**("很容易跑一半死掉")。长任务用
   `setsid nohup`、checkpoint 和明确退出标记保证可恢复;runner 成功或失败后只退出,不关机。
   硬时限只用于交互式调试期间。
3. 若发现历史遗留的自动关机链,先用只读命令解析父子 PID 并确认精确目标;清除后再次验证,
   避免误触立即关机。当前不得恢复该链。
4. **远程 pkill/pgrep 的模式若以明文出现在 ssh 命令里会自匹配**(pkill 会杀掉自己的会话,
   exit 255)。永远写成方括号形式 `'[r]un_pipeline'`,且同一条命令里不要出现明文目标名。
5. **transformers ≥5:`apply_chat_template(tokenize=True)` 返回 BatchEncoding 而非 list/Tensor**。
   服务器 `nla_repo/nla_inference.py` 187/403 两处和 `02_extract_activations.py` 已打补丁
   (`if not isinstance(ids, list): ids = ids["input_ids"]`);新写代码要防同类坑。
6. **本地 Clash Verge(TUN+fake-IP)会劫持 SSH**:"Connection closed by 198.18.0.x" 先查代理
   (`Get-Process | ? {$_.Name -match 'clash|mihomo'}`),让用户关 TUN 或加 `*.seetacloud.com`
   直连;**该报错无法区分"代理拦截"与"实例已关机"**,连接偶发抖动重试即可。
7. **HF 下载**:`HF_ENDPOINT` 必须 shell 层 export(hub 在 import 时读取);hf-mirror 带 token
   可下 gated 仓库(~20MB/s);gemma-scope-2 是 8TB 仓库,**禁用 snapshot_download**
   (树列举会挂),用 `hf_hub_download` 按精确路径取文件。
8. **上传的 .sh/.py 一律 `sed -i 's/\r$//'` 去 CRLF** 再执行。
9. AR 加载会报 `model.norm.weight MISSING` ——**良性**(final LN 被设计性剥离,critic 用
   raw residual;见 nla_inference.py 注释),勿当错误处理。
10. 历史事故备忘:曾因调试命令卡在权限弹窗,实例空转 7 小时(~40 元)。当前在“不关机”
    指令下,用后台 runner、checkpoint、GPU/日志监控和及时汇报控制风险,不得自行改写用户指令。
11. 判断长任务是否崩溃前先看日志时间戳与 checkpoint——09 曾被误判为"崩溃",实际 5 分钟
    正常跑完。**status=0 + 对应 `*_COMPLETE` 字样才是唯一真相来源。**

## 9. 效度边界(写论文/汇报时必须带上)

- **E1–E7** 仍只有 n=40 token 激活、5 条同质英文 instruct prompt;统计独立单位至多是
  5 个文档,不能把 40 行当 40 个独立样本。
- **E8** 虽有 24 文档/1365 token,实质只有 8 个独立内容 topic,每个 topic 是 en/zh/es
  三份翻译。domain 在每个 split 每类只有 1 个独立 topic;“3 个正文档”不是 3 个独立主题。
  language 结果则混有文字系统、tokenization 与文档长度效应。
- E8 从 16,384 feature×7 个 label 中在 train 上择优;train AUROC=1 是选择后的
  winner's-curse,不能当确认性证据。heldout-valid 是条件子组,必须先报告 ITT yield
  (全部 58.3%、domain 40%、language 88.9%),不能只报 survivor 的高分。
- E8 的统计单位是 **feature**;每 feature 的 4 次随机生成只估计生成不确定性,不把 n 放大
  5 倍。bootstrap CI 只是冻结 feature 集内部的条件区间,不能外推到 feature 总体。
- 对照仍有限:structural n=1;active-nonselective 仅 8 个近似匹配且匹配距离不均;
  Gaussian n=8 是完全 OOD 地板。当前没有完成严格的组间 permutation/matched-pair 推断。
- 单模型(Gemma-3-12B-IT)单层(L32)单 SAE 配置为主;NLA 公开检查点仅此一层,尚无跨模型/
  跨层复现。
- AV/AR 同源,round-trip 可能包含私有通信;heldout AUC 与 q+≈零相关已证明 AR 分数不能自动
  升格为 correctness。C7 未完成,也没有独立盲多评者或行为因果 ground truth。
- `b6b4_surface_audit.md` 是 post-hoc、非盲、单评者诊断,无 inter-rater reliability;
  其 5/24 strict、9/24 mismatch 只能作风险证据,不能当正式人类基准。
- `−w_dec` 对 JumpReLU SAE 是 isolated OOD signed-axis 测试,不是语义 antifeature。
  E8 的 5 次 explanation-tag 失败全部集中在一个 heldout-valid feature 的负方向,且输出乱码,
  进一步限制 r−/polarity 的自然语言解释。
- carrier 的 amplify/ablate/insert 只测试 AV/AR 对被改 activation 的读出变化;
  **没有运行 base model 行为、没有测任务成功/副作用,不是 steering 或因果行为证据**。
- 因此 E8 的结论级别仍是高信息量 pilot:支持“外部 activation gate + NLA/AR 辅助质检”
  的立项,不支持自动标签 correctness、双向语义极性或 steering 主张。
