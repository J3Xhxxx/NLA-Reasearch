# POSSBILITY — NLA 辅助 SAE 审计路线:可行性与合理性评估

> 初稿 2026-07-10；B2、B6+B4(F9) 更新 2026-07-26；C1 pilot(F10) 更新 2026-07-30；
> **N1/N2(F11–F14:文本通道拆解、因果 patch、取样缺陷、F9 勘误)更新 2026-07-30 晚** · Gemma-3-12B-IT L32 resid_post
> 旧主线:40 激活向量(5 条英文 instruct prompt)；正式 B6+B4:24 文档、1,365 激活、45 个冻结方向
> NLA = kitft/natural_language_autoencoders 的 L32 AV/AR 检查点
> SAE = gemma-scope-2-12b-it, resid_post, width 16k, L0≈15(small)与 L0≈75(big)
>
> **本文档每条结论都标注了数据出处(本目录内文件 → 字段),可逐条回查。F9 是正式 B6+B4、F10 是 C1 protocol pilot 的叙事事实源；数值以 JSON/NPZ 原始资产为准。**

---

## 0. 一句话裁决

| 路线 | 裁决 | 依据 |
|---|---|---|
| ① NLA 读实际激活(分诊/实例级审计) | **降级:cos 上的大幅优势在因果指标上消失,NLA ≈ SAE-big > SAE-small** | F1, **F12** |
| ②/C2-v2 NLA 读 SAE 字典方向(特征自动标注) | **条件性可行:仅作候选分诊,不可自动验收标签** | F3, F9, §3.2 |
| ③ NLA 读 SAE 残差(暗物质审计) | **当前不可行(干净的阴性结果)** | F4, F5 |
| ④/C1 AR 解释评分 | **standalone correctness metric 不成立；相对正确−错配 delta 有弱信号，需全新盲评 cohort 确认** | F6, F9, F10, §3.4 |
| ⑤ 评测方法论(去均值 + 泛型基线) | **确定成立,立刻可用,独立成果**;地板由 0.041 修正为 **≈0** | F2, F11 |
| ⑥ 解释文本到底在传什么(新) | **已定量拆解:传的是"下一 token 候选串",不是语义描述** | F11 |
| ⑦ AV/AR 私有编码风险(C7) | **已排除:第三方等义改写保留 97%** | F11 |

---

## 1. 数据资产清单(本目录)

| 文件 | 内容 | 关键字段 |
|---|---|---|
| `nla_results.json` | 主线 NLA:40 向量 → AV 解释 → AR 重建打分 | `rows[].explanation, nla_cos`;`summary.mean_cos=0.996` |
| `sae_results.json` / `sae_results_big.json` | 主线 SAE(L0≈15 / L0≈75)重建打分 | `summary.mean_cos=0.9925/0.9936, fve=0.6076/0.6747, mean_l0=15.2/75.5` |
| `comparison.json` / `.md` | 主线逐行对照表 | — |
| `wdec_pilot.json` | Pilot 1:NLA 读 w_dec 方向(top12/rand6/gauss4) | `rows[].group, feature, cos, contexts` |
| `resid_pilot.json` | Pilot 2:NLA 读 SAE 残差 | `rows[].cos_rr, cos_fr, cos_rx, resid_frac` |
| `injection_pilot.json` | Pilot 3:残差注入敏感性(4特征×4残差×α∈{0,.25,.5,1,2}) | `rows[].cos_sig, cos_res, overlap`;`summary.detection_curve` |
| `centered_rescore.json` | **修正实验**:全部解释在"投影掉数据集均值方向"空间重打分 | `head_to_head_centered, resid_centered, wdec_centered, injection_centered_curve, feature_mean_alignment` |
| `recon_vectors.npz` | 40 原向量 + 全部 AR 重建向量 + SAE 重建 + 残差 + 特征方向(本地可复算一切) | `x, m_hat, pred_full, pred_resid, recon_sae_small/big, resid, feature_dirs` |
| `retrieval_eval.json` | **B2 判别式检索**:4 组 40×40 centered 相似度矩阵、逐行 rank/margin、置换零假设、按文档检验 | `methods.*.summary/permutation_null/rows/similarity_matrix; paired_comparisons` |
| `b6b4_factorial_selection.json` | **B6+B4 冻结选择与因子语料**:24 文档、4 域×3 语言×train/test；24 新语义方向 + 控制 | `protocol, dataset, summary, selected_directions[].train/test/top_contexts` |
| `b6b4_factorial_result.json` | **B6+B4 正式原始结果**:45 方向、590 个 AV 作业、greedy 主结果、随机重复与 carrier | `protocol_notes, summary_by_cohort_greedy, polarity_rows, carrier_summary, carrier_effects, scored_generation_rows` |
| `b6b4_factorial_analysis.json` / `.md` | B6+B4 分层与派生汇总；其中 heldout-valid 只作选择后描述 | `selection_yield, direction_summary, frozen_primary_summary, heldout_auc_vs_q_plus_spearman, carrier_summary, interpretation_limits` |
| `b6b4_factorial_vectors.npz` / `b6b4_factorial_recon_vectors.npz` / `b6b4_factorial_av_rows.jsonl` | 冻结方向、输入与 AR 重建向量、逐次 AV 生成记录 | 由 `b6b4_checksums.sha256` 固定 |
| `b6b4_surface_audit.md` | 对 24 个 `semantic_new` greedy `+w_dec` 文本的事后表面语义审计 | `Rubric and counts, Relationship to round-trip score, Unsupported specificity` |
| `c1_pilot_benchmark/result/analysis.json` + `.md` | **C1 protocol pilot**：24 feature、正确/错配/generic/autointerp/NLA/paraphrase、冻结主检验与分层结果 | `primary_axis_reference_minus_axis_hard_negative, kind_summary, heldout_valid_descriptive, external_associations` |
| `c1_pilot_recon_vectors.npz` / `checkpoint_v2.jsonl` / `validation.json` | C1 全部 AR 向量、480 条断点记录与独立数值复算 | `semantic_similarity, reconstruction_vectors`;`status=all_checks_passed` |
| `c1_pilot_secondary_analysis.json` / `.md` | 事后敏感性：全轴、双向检索、label cluster、heldout 与 paraphrase 混杂 | `all_axis_text_sensitivity, heldout_prediction, paraphrase_diagnostics` |
| `c7b3_variants_v1.json` (+`.sha256`) | **N1/E11 冻结文本变体**:40 行 × 11 条件 + 8 条固定泛型文本,含撰写规则与作者声明 | `protocol, rows[].variants, diagnostics` |
| `c7b3_scores_v1.json` / `c7b3_recon_v1.npz` | N1 的 AR 重打分(420 条文本)、逐条件汇总、逐变体 40-way 检索、全部重建向量 | `verification, summary_by_variant, retrieval_by_variant, rows` |
| `causal_patch_v1.json` | **N2/E10 因果 patch**:40 位置 × 12 个替换源的 KL/CE,含 provenance 与对照 | `provenance, summary, rows[].results, cos_vs_causal_association` |
| `n1n2_analysis.json` | N1+N2 的稳健汇总:KL-recovered、配对 CI、文档聚类、token 类型分层、CE 端点作废说明 | `n1_variant_summary, n2_kl_recovered, n2_paired, cohort_token_composition, n2_ce_endpoint_invalid` |
| `local_recheck_opus.json` / `local_recheck_b6b4_opus.json` / `local_recheck_stratified_opus.json` | 第三轮独立复算:预言机基线与 centered FVE、逐方向泛型地板校正、F9 分层勘误 | 见 `REVIEW_OPUS_2026-07-30.md` |
| `REVIEW_OPUS_2026-07-30.md` | 对第二轮分析的独立复核与下一步方案 | — |
| `pipeline.log / pilot.log / injection.log / rescore.log / n1n2.log` | 服务器运行日志(复现审计用) | — |
| `b6b4_selection.log` / `b6b4_factorial.log` | B6+B4 选择与正式运行日志 | `B6B4_FACTORIAL_COMPLETE, EXIT(status=0)` |

---

## 2. 核心发现(F1–F10)

### F1 · 在真实激活上,NLA 显著强于 SAE——去均值后优势放大 3 倍
- 未去均值(被共享均值方向抬高):NLA 0.996 vs SAE-s 0.9925 vs SAE-b 0.9936
  (`nla_results.json summary.mean_cos`,`sae_results*.json summary.mean_cos`)
- **去均值(干净指标)**:**NLA 0.859 vs SAE-s 0.658 vs SAE-b 0.725**
  (`centered_rescore.json → head_to_head_centered.nla_mean/sae_small_mean/sae_big_mean`)
- 换算误差(1−cos):NLA 0.141,SAE-s 0.342,SAE-b 0.275——NLA 的方向误差不到 16k 宽 SAE 的一半,与 SAE 自报 FVE 0.61/0.67 量级一致。
- **含义**:约 100 词的生成式自然语言瓶颈,在保真度上实质性超过 JumpReLU 稀疏编码。这是路线①(分诊架构:SAE/探针便宜全量初筛 → NLA 对可疑激活深读)的直接依据。

### F2 · 模板化弱信号对照与均值混淆:未去均值的 cos 全部虚高(方法论成果)
- 对残差逐样本生成的高度模板化解释经 AR 重建,与**全向量**的未去均值 cos 达 **0.975**(范围 0.959–0.991)(`resid_pilot.json rows[].cos_rx`);去均值后同一批分数跌到 **0.041**(`centered_rescore.json → resid_centered."cos_rx_c(generic floor)"`)。
- 即:未去均值口径下,"随便说点像样的话"就能白拿 0.975,主线里 NLA 0.996 / SAE 0.9925 的表面差距其实是 3.2× vs 1.7× 的基线误差压缩比。
- **B2 后的术语修正**:`pred_resid` 来自 40 个不同残差的逐样本 AV 输出,不是严格的固定泛型文本。它在 centered 40-way 检索中仍有 Top-1 **17.5%**、MRR **0.288**,显著高于 2.5% / 0.107 的随机基线(`retrieval_eval.json → methods.residual_text_control`)。因此它应称为“残差解释/模板化弱信号对照”;真正的固定 generic-text 检索地板尚未由现有向量资产直接测量。
- **含义**:凡以未去均值 cos 评价激活重建的工作都高估绝对性能;正确姿势 = 报告去均值分数 + 明确定义且不与样本配对泄漏的阴性文本对照。这一条不依赖 NLA 是否"好",独立可发表/可复用。

### F3 · Pilot 1 的"字典方向强可读"大部分是假象;真实可读性微弱但存在
- 表面结果:top 特征 median|cos|=0.80(7/12 达 0.78–0.95)(`wdec_pilot.json rows[].cos`)。
- 真相:这 4 个最强特征的 w_dec 方向与数据集均值方向对齐度 **±0.89–0.96**(`centered_rescore.json → feature_mean_alignment`:f166 −0.958,f443 −0.950,f239 +0.911,f490 +0.893)。AR 对任何泛型文本的输出都落在均值方向附近 → 高 |cos| 是白拿的。
- 之前报告的"42% 符号为负(符号盲)"同样被解释:**往返符号 = 特征与均值对齐的符号**(f166:−0.946↔−0.958;f239:+0.922↔+0.911),不是 AV 读出了什么。
- 去均值后:top 组 mean|cos_c| **0.189**,rand 0.105,gauss 0.011(`centered_rescore.json → wdec_centered`)。排序仍成立但幅度小一个量级。
- **残存的真实信号**:去均值后最高的特征恰是描述与触发足迹语义吻合的——f1491=0.453(触发于光合作用文本,描述"biology textbook pattern for photo…")、f5389=0.374(百科体)、f239/f276=0.34(换行/结构特征,描述"structured article/Q&A")(`centered_rescore.json → wdec_centered.top.rows` 对照 `wdec_pilot.json rows[].contexts/explanation`)。
- **含义**:路线②不判死刑,但 pilot 级证据只支持"微弱、且仅对有清晰足迹的特征成立";在多样语料上重选语义特征重测(THOUGHT.md B6)是升级为可用工具的前提。

### F4 · SAE 残差(暗物质)对 NLA 完全不可读
- 未去均值:mean cos_rr = 0.036,仅 3/40 超 0.3,最高 0.495(`resid_pilot.json summary.mean_cos_rr, rows[].cos_rr`);
- 去均值:cos_rr_c = **−0.029**(`centered_rescore.json → resid_centered.cos_rr_c`)——与零无异;
- 残差与均值方向近正交(alignment 0.041,同文件 `resid_mean_alignment`),所以这一条不受 F2 混淆影响,是干净的阴性。

### F5 · 注入已知信号也检不出:暗物质审计的敏感性为零(当前 AV)
- 检测曲线(去均值,|cos_sig_c| 对注入强度 α):α=0 → 0.072(假阳性地板),α=0.25/0.5/1.0 → 0.044/0.044/0.056,α=2.0 → 0.092,frac>0.3 在 α=2 仅 6%(`centered_rescore.json → injection_centered_curve`)。α=2 意味着植入信号是整个残差的 2 倍、约占激活范数 25%——仍然读不出。
- 机制证据:解释文本几乎不随 α 变化(同一残差在 α=0→2 下开头句式相同,`injection_pilot.json rows[].explanation`;cos_res 恒 ≈0.13,`rows[].cos_res`)。**AV 把整个残差邻域(离流形区域)collapse 成同一句"Article structure…"模板**——失败发生在 AV(读数端),不是 AR(打分端)。
- 未去均值口径下曲线同样平坦但地板高达 0.88–0.95(`injection_pilot.json summary.detection_curve`)——那是 F2/F3 的混淆,不是检测。
- **含义**:路线③(藏匿信号审计)在现成 AV 上不成立。要救,只能改造 AV(在残差/方向/合成混合向量上继续训练,即"off-manifold 课程"),这是训练级研究,不是评测级 pilot。

### F6 · 解释会编造事实,而重建保真度对此完全不敏感
- 40 条主线解释中 3 条含主题错误实体(London Eye、Chicago Museum of Illusions ← 正确主题是埃菲尔铁塔;Chinese Crested Dog ← 泛猫狗主题),错误行与正确行的 nla_cos 均值**完全相同(0.9962 vs 0.9962)**(本地文本统计,数据源 `nla_results.json rows[].explanation/nla_cos`,普查明细见 §附录 A)。
- **含义**:AR 重建主要消费"体裁 + 话题大类 + 局部句法期望",不消费具体事实——**高 cos ≠ 解释忠实**。审计场景中 NLA 文本的具体指称不能直接采信,必须配独立核查(这正是路线④要提供的)。

### F7 · 解释高度模板化,且越离流形越模板化
- 全向量解释:两两 4-gram Jaccard 均值 0.0043,最高频 4-gram "final token it' opens" 出现 11/40;残差解释:Jaccard 均值 0.0134(**3.1 倍**),头部 4-gram 覆盖 25/40(本地文本统计;数据源 `nla_results.json` 与 `resid_pilot.json` 的解释字段)。
- 特异性(非模板实义词占比)与 nla_cos 相关 r=+0.53,但主要由 token 类型驱动:cos 最低 3 行全是 `<end_of_turn>`(特异性 0.27),最高 3 行为内容/结构 token(0.44)。
- **含义**:模板化程度本身可以当"离流形检测器"用——输入越出分布,AV 输出越退化为模板。这反向支持:NLA 解释的多样性/特异性可作为审计置信度信号(THOUGHT.md B7 自一致性实验的静态版)。

### F8 · B2 检索没有复现“NLA Top-k 胜过 SAE”,但 NLA 的配对间隔更大
- 在双边投影掉 `m_hat` 后,对每个重建向量检索 40 个原激活。严格 Top-1 / Top-5 / MRR:
  - NLA:**92.5% / 100% / 0.9583**;
  - SAE-small:**95.0% / 100% / 0.9708**;
  - SAE-big:**95.0% / 100% / 0.9708**;
  - residual-text control:**17.5% / 40.0% / 0.2883**。
  (`retrieval_eval.json → methods.*.summary`;随机期望 2.5% / 12.5% / 0.107)
- 因而预设判定“NLA Top-1 显著高于 SAE”**没有成立**。Top-5 已完全饱和;NLA 错 3 行(idx 6/13/29),两档 SAE 错 2 行(idx 13/29),其中共同错误主要是 `<end_of_turn>`。
- 连续间隔指标给出不同但互补的结果:`margin = sim(pred_i,x_i) − max_{j≠i}sim(pred_i,x_j)`。NLA mean margin **0.3195**,高于 SAE-small **0.2085**、SAE-big **0.2531**;差值分别 **+0.1110 / +0.0664**,且 5 个文档的聚合差均为正(`paired_comparisons.*.mean_margin`)。
- 5 个文档上的 exact cluster sign-flip 单侧 `p=1/32`,只是最小分辨率的探索性证据,且本实验同时查看了多个指标;不能把它表述成大样本确认性显著。
- 文档内 8-way 检索同样饱和:NLA Top-1 97.5%,两档 SAE 100%。所以 B2 支持的精确表述是:**NLA 的 centered 方向保真与配对几何间隔更强,但在这个小而同质的数据集上没有转化为更高的离散 Top-k 命中率。**

### F9 · 正式 B6+B4:字典方向有弱而系统的内部可读性,但生成标签的外部语义忠实性仍不足

**设计与主分析口径。** 因子语料含 24 个文档、1,365 个 L32 激活,覆盖 software/geography/biology/history 四域、en/zh/es 三语及 train/test 两个 topic split。特征只用 train 选择:24 个新语义特征全部通过冻结 strict gate；另含 4 个 legacy、1 个 structural、8 个 active-nonselective、8 个 Gaussian 方向,共 45 个方向。正式运行完成 590 个 AV 作业(450 direction + 140 carrier),每个符号 5 次生成；`sample_index=0` 的 greedy 生成是冻结主结果,其余温度重复只表征生成不确定性(`b6b4_factorial_selection.json → protocol, summary, dataset`;`b6b4_factorial_result.json → inputs, protocol_notes, summary_by_cohort_greedy`)。

最重要的分析规则是:**24 个冻结 `semantic_new` 特征的 intention-to-test(ITT) 是主结果。** `heldout-valid` 会再用 test AUROC≥0.75、正 heldout effect、正样本支持≥2 过滤；虽然 test 未参与原始特征选择,这个子集本身仍是 test-dependent 的选择后结果,故以下一律标作 **post-selection descriptive**,不能替代 ITT(`b6b4_factorial_result.json → protocol_notes.heldout_valid_gate`;`b6b4_factorial_analysis.json → selection_yield`)。

**方向与极性主结果。** 这里 `q+` 是 `+w_dec` 解释经 AR 重建后对正轴的 centered cosine；`r−` 是对 `-w_dec` 结果按负轴纠正符号后的读数。`polarity=(q+ + r−)/2`。ITT 中 `q+` 明显强于 `r−`,说明 AV 更善于读正向 SAE 特征,不能把 `-w_dec` 当作同等自然的“反语义特征”。

| 分层 | n | median q+ | median r− | median polarity | sign accuracy | signed Top-1 / Top-5 | feature Top-1 / Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **semantic_new ITT(主结果)** | 24 | **0.1136** | 0.0306 | **0.0708** | 75.0% | 33.3% / 41.7% | 33.3% / 43.8% |
| domain ITT | 15 | 0.1181 | 0.0276 | 0.0728 | 80.0% | 30.0% / 40.0% | 30.0% / 40.0% |
| language ITT | 9 | 0.1090 | 0.0416 | 0.0688 | 66.7% | 38.9% / 44.4% | 38.9% / 50.0% |
| heldout-valid,全体(描述性) | 14 | 0.1815 | 0.0423 | 0.1017 | 85.7% | 39.3% / 50.0% | 39.3% / 53.6% |
| heldout-valid,domain(描述性) | 6 | 0.4075 | 0.0388 | 0.2242 | 100% | 41.7% / 58.3% | 41.7% / 58.3% |
| heldout-valid,language(描述性) | 8 | 0.0664 | 0.0460 | 0.0516 | 75.0% | 37.5% / 43.8% | 37.5% / 50.0% |

出处:`b6b4_factorial_result.json → summary_by_cohort_greedy.semantic_new_intention_to_test/semantic_new_heldout_valid`;`b6b4_factorial_analysis.json → direction_summary.domain_itt/language_itt/domain_heldout_valid/language_heldout_valid`。ITT 的 `q+` median bootstrap 95% 区间为 [0.0091, 0.4527],polarity 为 [0.0199, 0.2455]；23/24 polarity 为正(one-sided sign test `p=1.49e−6`)。这是有选择、单模型且同时查看多个指标的研究性实验,该 p 值不能包装成跨设定或经多重校正的确认性结论。作为参照,active-nonselective(n=8)的 median `q+`/polarity 为 −0.0036/0.0148、Gaussian(n=8)为 0.0045/0.0043；Gaussian signed/feature Top-1 均为 0(`b6b4_factorial_result.json → summary_by_cohort_greedy.active_nonselective/gaussian`)。

**heldout 转移并不等于 AV 可读。** test-dependent gate 通过率为 14/24=58.3%；domain 为 6/15=40.0%,language 为 8/9=88.9%。但冻结特征的 heldout AUC 与 `q+` 的 Spearman 相关仅 **ρ=−0.015**；而且 language heldout-valid 的 median `q+` 反低于 domain(0.066 vs 0.407)。所以“SAE 特征跨 topic 保持选择性”与“AV 能正确 verbalize 该方向”是两项不同能力,不能用前者替后者背书(`b6b4_factorial_analysis.json → selection_yield, heldout_auc_vs_q_plus_spearman, direction_summary.*`)。

**表面语义审计给出更严格的外部警报。** 对 24 个冻结新语义特征的 greedy `+w_dec` 文本做事后、非盲、单审阅者审计:与 label 和主要 context 都严格相符 **5/24**；只保留粗域/语言信号 **7/24**；明显错配 **9/24**；英语方向因 AV 默认用英语输出而无法判断 **3/24**。至少 **22/24** 含冻结 context 不支持的具体实体或技术细节；这应称为“证据不支持的具体化”,不等价于这些世界知识本身必然为假。即使 direction retrieval 正确,表面解释也可能错:例如 f14470 的 `q+=0.507` 且 feature rank=1,却把印刷史方向解释成 Minecraft/HP LaserJet。`q+` 与表面质量有粗相关(strict median≈0.458,mismatch≈0.008),适合分诊而不是验收证书(`b6b4_surface_audit.md`;原始文本与分数见 `b6b4_factorial_result.json → scored_generation_rows, polarity_rows`,冻结 context 见 `b6b4_factorial_selection.json → selected_directions[].top_contexts`)。

**carrier 只证明 readout sensitivity,不是 steering。** 24 个 ITT 特征中 9 个 carrier activation 为零；对其余 **15 个 nonzero** 特征:

| carrier 对比 | median centered effect | 正效应比例 |
|---|---:|---:|
| high − ablate | **0.2102** | **15/15** |
| insert − low | **0.0975** | 13/15 |
| amplify − high | **0.0415** | 8/15 |

(`b6b4_factorial_analysis.json → carrier_summary.semantic_new_nonzero`;逐特征见 `b6b4_factorial_result.json → carrier_effects`)。这说明在自然 carrier 附近消融/插入/放大 SAE 分量会改变 **AV→AR 的解释重建读数**。实验没有把解释或重建向量重新注入基础模型、没有测下游生成或任务行为,因此不能称为因果行为控制、activation steering 或 steering 成功。

**裁决。** B6+B4 把 F3 的“微弱残存信号”升级为跨 domain/language ITT 可复现的内部方向信号,但没有把它升级为可靠自动标注。当前唯一稳妥产品形态是 **C2-v2 条件性分诊**:`+w_dec` 为主、冻结特征证据与外部文本审计共同把关、禁止按高 `q+` 自动采纳标签；任何 steering 主张必须另做基础模型下游行为干预。与此同时,F6 与本节表面审计共同说明 C1 不能再问“AR 能否给统一忠实分”,而应问“内部 round-trip 分数能否预测独立的人类/context/行为效度”。

### F10 · C1 protocol pilot：相对粗轴分数有弱外部信号，但绝对 q、同源 judge 与原文身份检索都不能验真

**设计边界。** 用 F9 已查看过的 24 个 `semantic_new` feature 做 protocol pilot；历史冻结真值只有 7 个粗粒度 domain/language 轴。所有 autointerp 证据严格过滤为 train split。细粒度 reference 是查看 F9 后撰写的，base judge 与 paraphrase 又来自同族 Gemma，因此只有 `axis_reference − fixed axis_hard_negative` 是预先指定的主 contrast，整个批次仍标作 non-confirmatory(`c1_pilot_benchmark.json → scope, protocol`)。

**主结果是“小幅度、非多数稳健”的正信号。** 24-feature ITT 的 mean delta 为 **0.01550**，feature bootstrap 95% CI [0.00510, 0.02712]，exact sign-flip mean `p=0.00393`；但 median 仅 **0.00897**、区间跨零，只有 **16/24** 为正，exact sign test `p=0.07579`。7 个 label cluster mean 全部为正，任意 leave-one-label-out mean 仍为正；对全部六个错误轴的事后敏感性中，correct-minus-mean-wrong 为 17/24 正、feature→7-axis-text Top-1 **50%**，但 correct-minus-best-wrong 只有 12/24 正。因此可表述为“AR 对粗轴正确−错配的相对排序含信息”，不能表述为“AR 已能稳定识别正确标签”(`c1_pilot_analysis.json → primary_axis_reference_minus_axis_hard_negative, label_summary`;`c1_pilot_secondary_analysis.json → all_axis_text_sensitivity, cluster_sensitivity`)。

**真正值得复现的是 delta，而不是绝对 q。** heldout-valid 14 个 feature 的 axis delta mean 为 **0.02584**、11/14 为正；invalid 10 个仅 0.00102、5/10 为正。事后计算 axis delta 预测 heldout-valid 的 AUC **0.764**，与 test AUC 的 Spearman **0.501**。相反，绝对 `q(axis_reference)` 对 test AUC 的 rho 只有 0.019、heldout AUC 0.436；NLA-original q 也近乎无关。故下一版应检验 matched contrast/calibration，不能再用绝对 q 阈值(`c1_pilot_analysis.json → heldout_valid_descriptive, external_associations`;`c1_pilot_secondary_analysis.json → heldout_prediction`)。

**细粒度强结果主要像 train 指纹。** train reference−hard negative mean **0.19468**、20/24 正；train reference−same-axis sibling mean **0.10793**、19/24 正；24-way reference text→feature / feature→text Top-1 为 **54.2% / 70.8%**。但该 context delta 在 heldout-invalid 上反而比 valid 更大(0.248 vs 0.156)，预测 heldout-valid 的 AUC 仅 0.379。这与“重复词汇/训练主题可被 AR 配对”一致，不是外部语义泛化(`c1_pilot_analysis.json → exploratory_deltas, heldout_valid_descriptive`;secondary analysis 的 `bidirectional_24way_retrieval, heldout_prediction`)。

**内部身份与表面正确性再次分离。** NLA original 的 24-way text→feature Top-1 **62.5%**、feature→text **58.3%**，但 blind context judge 只有 **2/24** 达 `score>=2`；base autointerp 为 45.8%/37.5%，却有 16/24 达 judge 阈值。judge 本身也失效：generic 110/192、axis hard negative 11/24 被判为正，pooled feature-centered q–judge rho **−0.0418**、AUC **0.410**。同一 Gemma-family 生成并裁判还引入风格自偏，故 judge 只保留作质性日志(`c1_pilot_analysis.json → kind_summary, provisional_blind_base_judge`)。

**private-code 证据只是提示。** NLA paraphrase 相对原文 mean Δq **−0.12776**，19/24 下降；相对 matched reference paraphrase 的 interaction mean **0.10429**、17/24 为正。但 NLA paraphrase 只保留原字符数 median **60.1%**，而 authored reference paraphrase 为 106.3%，且提示显式删除 meta phrase/猜测列表，长度与信息删减是严重混杂。当前只支持“AR 对 NLA 原始表述形式异常敏感”，不能识别为人类不可读 private code(`c1_pilot_analysis.json → exploratory_deltas.private_code_interaction`;`c1_pilot_secondary_analysis.json → paraphrase_diagnostics`)。

**QA 与裁决。** 成功 v2 共 432 candidate、432/432 judge 可解析；AR 重算旧 B6 NLA 原文的最大误差 **1.04e−9**，独立 similarity 复算误差 **2.80e−8**。本 pilot 不支持 AR standalone correctness，也不足以支持顶会主张；它支持立项一个全新、预注册、至少 60–100 feature、按 label/model 聚类且有多名盲人工评者与长度匹配 paraphrase 的 C1。若 AR delta 能预测人类/heldout/因果效度，再进入 steering；若只偏好 NLA 原文，则论文应收敛为 **closed-loop communication code ≠ human interpretability**(`c1_pilot_validation.json`;`c1_pilot_checksums.sha256`)。

### F11 · 解释文本的通道被拆开了:承载重建的是"下一 token 候选串",不是语义描述

**设计(N1 / E11)。** 40 条主线解释被拆成 11 个条件,全部在打分前冻结
(`c7b3_variants_v1.json`,sha256 `b30a74be…`,含逐条撰写规则与作者声明),只用 AR
`reconstruct()` 重打分(420 条文本 17 秒,零 AV 生成)。原文复算得 **0.8593**,与 E5
的 `head_to_head_centered.nla_mean` 逐位相同,口径确认无误
(`c7b3_scores_v1.json → verification`)。

| 条件 | 去均值 cos | 占原文 | 40-way 检索 Top-1 | 含义 |
|---|---:|---:|---:|---|
| 原文 | 0.8593 | 100% | 92.5% | 基准 |
| **只留第 3 段**(下一 token 预测) | **0.8259** | **96.1%** | **95.0%** | 几乎就是全部信号 |
| 只抹掉第 3 段里的引号内候选串 | 0.1755 | 20.8% | 22.5% | 抹掉它信号就没了 |
| 抹掉全部引号内内容 | 0.1112 | 12.9% | 7.5% | 同上 |
| 只抹掉第 2 段里对原文的逐字引用 | 0.8540 | 99.4% | 95.0% | **逐字引用几乎不贡献** |
| 只留第 1 段(体裁/格式判断) | 0.2403 | 28.3% | 15.0% | 体裁只值三成 |
| 第 1+2 段(去掉下一 token 预测) | 0.6087 | 71.0% | 75.0% | — |
| 打乱全部词序 | 0.5731 | 66.9% | 85.0% | 三分之二是词袋 |
| **受控实体替换(B3)** | **0.7711** | **89.7%** | **95.0%** | 指称身份只值一成 |
| **第三方等义改写(C7)** | **0.8511** | **97.3%** | — | **私有编码风险排除** |
| 固定泛型文本(8 条) | **−0.0046** | ≈0 | — | **真实地板是 0,不是 0.041** |

(出处:`c7b3_scores_v1.json → summary_by_variant / retrieval_by_variant`;
`n1n2_analysis.json → n1_variant_summary, n1_channel_share_above_generic_floor`。
所有配对 delta 的 bootstrap CI 与逐行分数在同文件内。)

**四条独立结论。**
1. **C7 通过(正面结果)。** 由本项目之外的模型族(Claude Opus 5)撰写、长度匹配
   (字符比 1.06–1.12)、命题与引号逐字保留的等义改写,保留 **97.3%** 的分数
   (配对 delta −0.0238,19/20 为负但幅度极小)。C1 pilot 里 Δq=−0.128 的"私有编码"
   嫌疑,主要来自那次改写只保留 60.1% 字符;长度匹配后嫌疑消失。**AR 消费的是
   人类可读的内容,不是同源私有码。**
2. **B3 给出因果版的 F6。** 只替换具体指称(埃菲尔铁塔→悉尼歌剧院等,共 214 处替换)
   代价仅 10.3%,检索命中率反而从 92.5% 升到 95.0%。F6 的"编造不影响 cos"从相关性
   证据升级为受控操纵证据:**指称身份基本不在通道里。**
3. **通道被定位了。** 承载信号的是第 3 段里那串引号内的"下一 token / 下一短语候选"。
   注意这不是长度效应:`p3_only` 删掉的字符比 `quote_strip_p3` 更多,却保留 96% 而不是
   21%;`quote_strip_p2` 删字符但保留 99.4%。
4. **词序只值三分之一。** 打乱词序仍保留 66.9% 与 85% 检索命中,说明相当大一部分是词袋级
   词汇匹配。

**这条发现统一解释了此前的一堆现象**:F6(编造实体不影响分数)、F7(越离流形越模板化)、
F10 里"正确但抽象的粗轴标签 q 中位数为负(−0.0133)而含真实词汇的 train reference 高达
0.1925"——因为抽象标签里没有候选续写串。**因此"自然语言解释"在 NLA 里的实际功能是
一个被语言化的下一 token 分布,而不是语义解释。** 一切把 AR 分数当"解释忠实度"的用法
都应改述为"候选续写重合度"。

### F12 · 因果 patch(领域标准指标):cos 上的大幅优势在因果上消失

**设计(N2 / E10)。** 把每个重建向量按原范数缩放后 patch 回 `model.layers[32]` 输出的
原 token 位置,继续前向,测该位置下一 token 分布相对干净前向的 KL。抽取复算的
provenance cos = **1.000000**(40/40),`identity` 的 KL = **0.0000**,`gaussian` 31.07、
`zero` 16.34,阴性/阳性对照都干净(`causal_patch_v1.json → provenance, summary`)。

| 替换源 | mean KL@pos | median KL@pos | **KL-recovered**(1=原向量,0=零消融) | 内容 token | 模板 token |
|---|---:|---:|---:|---:|---:|
| NLA | **2.41** | 0.280 | **0.757** | 0.776 | **0.717** |
| SAE-big (L0≈75) | 3.25 | **0.239** | **0.771** | **0.865** | 0.577 |
| SAE-small (L0≈15) | 4.14 | 0.572 | 0.713 | 0.738 | 0.660 |
| 残差解释对照 | 7.64 | 6.31 | −0.943 | −1.593 | 0.407 |
| 数据集均值 | 14.47 | 9.92 | −1.781 | −2.784 | 0.303 |
| 另一条真实激活 | 17.93 | 18.34 | −3.035 | −4.618 | 0.252 |
| 高斯同范数 | 31.07 | 28.16 | −9.411 | −13.31 | −1.321 |

(出处:`n1n2_analysis.json → n2_kl_recovered, n2_paired, cohort_token_composition`。)

- **配对检验**:NLA 相对 SAE-small 的 KL 优势成立(mean +1.73,bootstrap CI
  [0.50, 3.21],27/40 行、5/5 文档为正);相对 **SAE-big 不成立**(mean +0.83,
  CI **[−0.21, 1.98]** 跨零,median −0.000,仅 18/40 行为正)。
- **按范数归一的 KL-recovered 三者基本打平(0.757 / 0.771 / 0.713)**;在内容 token 上
  SAE-big(0.865)反而优于 NLA(0.776),NLA 只在模板 token 上占优(0.717 vs 0.577)。
- **含义(对 F1 的实质修正)**:去均值 cos 上 NLA 的方向误差不到 SAE 的一半
  (0.141 vs 0.342/0.275),但换到"模型自己真正在用的信息"这一口径,优势收缩为
  与 SAE-big 打平。**路线①不能再写成"NLA 显著强于 SAE",只能写成"在因果指标上与
  16k 宽、L0≈75 的 SAE 相当,并显著强于 L0≈15 的 SAE"。** 这是本项目第一个领域标准
  指标,应作为对外报告的主口径。
- **cos 不是因果保真的可靠行级预测器**:跨方法合并 Spearman(cos, KL@pos)=−0.402
  (方向正确),但方法内部不一致甚至反向(NLA **+0.188**、SAE-small −0.480、
  SAE-big −0.338)。cos 能排方法,不能排样本。

### F13 · E1–E7 队列有一个此前未被发现的取样缺陷:13/40 是 chat 模板 token

`02_extract_activations.py` 以 `--min-position 50` 记录为"仓库不变量",但 5 条 prompt
套上 chat template 后只有 **28–38 个 token**,该不变量从未生效,实际走的是
`cand = range(len(seq)//2, len(seq))` 的回退分支——也就是取序列后半段,而后半段正是
`<end_of_turn> / \n / <start_of_turn> / model` 这段模板尾部。

结果:**40 个"激活"里 13 个(32.5%)落在模板或纯空白 token 上**
(`n1n2_analysis.json → cohort_token_composition`)。这解释了 Fable 最初把
"`<end_of_turn>` 上 SAE 反超 NLA"读成"两种方法失败模式互补"的现象——那不是语义/结构
方向之分,而是取样落到了模板边界。F12 的分层也印证:NLA 只在模板 token 上占优。

**副作用**:基于同一窗口的 loss-recovered 端点不可用。干净前向对该窗口真实后继 token 的
CE 高达 **21 nats**(均匀分布只有 12.48),因为窗口里的 `<end_of_turn>`(36 nats)、
`<start_of_turn>`(53 nats)本来就不可能被基座模型预测
(逐 token 明细见 `32_diag_ce_window.py` 输出,记录于 `n1n2.log`)。故 F12 一律以
KL 与 KL-recovered 为准,`causal_patch_v1.json` 里的 `loss_recovered_*` 字段作废
(`n1n2_analysis.json → n2_ce_endpoint_invalid` 已标注原因)。

### F14 · 对 F9 的两处勘误(纯统计,零新增实验)

1. **`q+` 是双峰,不是"弱而系统"。** 24 个 `semantic_new` 的 q+ 降序在第 10/11 名之间有
   **0.117 的断层**:10 个 ≥0.362(其中 9 个方向检索 rank=1,且全部超过自己方向上
   最坏情况的泛型文本地板),13 个 <0.15。median 0.114 落在断层里,不描述任何特征。
   同口径下 Gaussian 只有 1/8、active-nonselective 3/8 超过最坏泛型地板,semantic_new
   14/24。q+ 与逐方向泛型地板的 Spearman 仅 0.05,故高分不是残余混淆
   (`local_recheck_b6b4_opus.json`,`local_recheck_stratified_opus.json`)。
   **正确的研究问题是"什么方向可读",不是"有没有弱信号"。**
2. **`heldout AUC 与 q+ 的 ρ=−0.015` 支持不了"内部分数不预测外部效度"。** 该合并系数是
   Simpson 悖论加并列伪值的产物:domain 层 ρ=**+0.40**(n=15),language 层 ρ=−0.78 但
   8/9 个特征的 test AUC 顶到 **1.0**(天花板饱和);更关键的是 **8/24 个特征在 test split
   上完全不激活**(`pos_mean = neg_mean = 0`),其 AUC=0.5 是并列约定的伪值,而这 8 个里
   恰含 q+ 最高的三个(f2725=0.673、f14470=0.507、f10000=0.458);该 AUC 本身只有
   3 正/9 负共 12 篇文档的分辨率。**结论应为"这份数据没有能力检验该关联",不是"已证明零关联"。**
3. **由 2 直接推出 C2-v2 的门禁顺序有误**:先过 heldout activation gate 会剔掉本队列里
   最可读的三个特征——`w_dec` 可读性是嵌入几何属性,heldout 激活率是语料属性。小语料上
   "未触发"只能记为*无证据*,不能记为*未通过*。

---

## 3. 对"NLA×SAE 审计"四条路线的裁决与理由

### 3.1 分诊式实例审计(SAE/探针全量在线 → NLA 对可疑激活深读)——**降级为"相当",不是"胜出"**
F1 的 centered cosine 与 F8 的 margin 支持 NLA 保留更强的实例方向信号,而其成本(每向量一次 12B 生成)决定了它只能放在漏斗下游。但 F8 的 Top-k 已饱和且 SAE 略高,所以不能再笼统写成“NLA 在所有判别指标上胜过 SAE”。**F12 进一步把这条路线降级**:在因果 patch 口径下 NLA 的 KL-recovered 0.757 与 SAE-big 0.771 打平,内容 token 上还略输(0.776 vs 0.865),只在模板 token 上占优;相对 SAE-big 的 KL 配对 CI 跨零。因此对外表述只能是"与 L0≈75 的 SAE 相当、显著强于 L0≈15 的 SAE",而 NLA 每向量一次 12B 生成的成本要为此付账。F6/F11 另加一条使用规范:NLA 文本里的具体指称要当"线索"而非"证词"——F11 已证明指称身份只占通道的 10%。

### 3.2 C2-v2:SAE 特征自动标注(NLA 读 w_dec + AR 往返质检)——**条件性可行,仅作分诊**
F9 已完成 F3 要求的冻结语义特征、去均值、`±w_dec`、matched controls、domain/language 分层与 carrier 检验。ITT 的 median `q+=0.114` 高于 `r−=0.031`,polarity 23/24 为正,且两类检索 Top-1 均为 33.3%,证明存在弱而系统的内部方向信号；但人工表面审计只有 5/24 strict match、9/24 mismatch、3/24 indeterminate,至少 22/24 有证据不支持的具体化。故 C2-v2 的合理形态是:**优先读 `+w_dec` → 用 `q+` 排序候选 → 强制对照冻结 activating contexts/外部标签 → 人工或独立模型验收**。它不能按 `q+` 阈值自动接纳标签,更不能把 `-w_dec` 描述当作真实 antifeature。

### 3.3 暗物质审计(NLA 读残差/藏匿信号)——**判负,除非改造 AV**
F4+F5 是一致的阴性:残差不可读,注入 2 倍强度的已知信号也不可读,且失败模式明确(AV 对离流形输入 collapse 成模板,F7)。合理性上这不意外——AV 只在自然激活分布上训练过,没有理由外推。翻案路径唯一:**off-manifold 课程训练 AV**(混入方向、残差、合成叠加向量的解释监督),属于训练级研究计划。

### 3.4 C1:AR 外部效度审计——**protocol pilot 已完成；确认性 benchmark 仍是最高优先**
F10 已把正确粗轴、受控错配、generic、事后人写标签、base autointerp 与 paraphrase 放进同一 centered AR 评测。结论不是简单的全阴性：正确−错配的**相对 axis delta**有小而可复现的信号，并与 heldout selectivity 呈描述性关系；但绝对 q 不校准，细标签强度主要跟 train 指纹走，同源 base judge 失效，NLA 原文的高身份检索也不能预测表面正确性。下一步必须换成从未查看的新 feature cohort，在任何 NLA/AR 输出前冻结 truth/negatives/统计方案，并以盲多人类 context judgment 或基础模型因果行为为主 target。AR 仍只能是待验证 predictor；若它不能跨长度匹配 paraphrase 预测外部效度，就应把研究对象明确写成 closed-loop communication code，而不是 fidelity metric。

### 3.5 方法论成果(顺产)——**已成立,并被 F11–F13 大幅加强**
去均值指标 + 泛型文本地板 + 特征-均值对齐混淆的识别(F2/F3),适用于一切用 cos/重建评价可解释性方法的工作。加上本轮:真实固定泛型地板 ≈ **0**(F11,原估 0.041 偏高)、**cos 能排方法但不能排样本**(F12 的方法内 Spearman 不一致)、**必须报因果 patch 口径**(F12)、以及**文本通道拆解协议**(F11 的 11 条件消融,可直接搬到任何"用自然语言解释激活"的方法上)。这套东西现在足够撑起一篇独立的评测方法论文章,且不依赖 NLA 是否"好"。

### 3.6 新增:研究对象的重新定义——**这才是最有论文价值的一条**
F11 把"NLA 的解释在传什么"从猜测变成了测量:传的是**被语言化的下一 token 候选分布**。
配上 F12(该通道在因果指标上与稀疏字典相当而非更优)与 C7 通过(F11,不是私有编码),
可写的主张不再是"闭环自洽≠可解释性"这种纯否定,而是一个有正有负、机制清楚的判断:
**"自然语言激活自编码器学到的是一个可读的下一 token 预测器,而不是一个语义解释器;
它的重建优势来自局部预测信息,而不是语义忠实度。"** 这个主张同时解释 F6、F7、F10,
并给出可迁移的评测协议(F11 的通道消融)。

---

## 4. 边界与效度声明

0. **E1–E7 队列有取样缺陷(F13)**:40 个位置里 13 个是 chat 模板/空白 token,因为
   `--min-position 50` 对 28–38 token 的 prompt 从未生效,走了"取后半段"的回退分支。
   所有 F1/F6/F7/F8/F11/F12 的数字都建立在这个队列上,报告时必须声明,并优先看
   content-token 分层。任何后续 E1–E7 级实验都应改用更长的自然文本重抽。
1. **两套样本不能混写**:F1–F8 的旧主线是 40 激活/5 个同质英文 prompt；F9 是 24 文档/1,365 激活/24 个新语义特征。两者均为 pilot 规模,不能把 feature 数当独立人类样本数,也不能直接外推到其他模型、层或 SAE。
2. **单模型单层、同源编码器**:全部结果只覆盖 Gemma-3-12B-IT L32；AV/AR 均来自同族模型微调。AR 可能解码人类不可读的私有编码,C7 第三方 paraphrase 检验尚未完成。
3. **F9 的 ITT/heldout 边界**:冻结 24 特征 ITT 是主结果。`heldout-valid` 使用 test AUROC/effect/support 再筛,只能作 post-selection descriptive；不得用 14 个通过者替代 24 个 ITT 报主结论。
4. **因子语料的“泛化”有限**:domain 的 train/test 是不同 topic,但每个 split 内三语是同一 topic 的翻译,不是独立主题；language 分层同时混入 script、tokenization 与长度。domain/language 数字必须分层报告,不能只看 14/24 总通过率。
5. **控制组覆盖有限**:structural control 仅 n=1；active-nonselective 只有 8 个近似配对,并非每个新特征都有一一 matched control；Gaussian 只检验随机方向地板。
6. **`-w_dec` 是 OOD 轴检验**:它不一定对应自然、可命名的语义反特征。因此 `q+ > r−` 支持正向可读性强于负向,不支持关于“负语义”的本体论结论。
7. **内部重建不等于外部标签忠实**:F9 的数值主指标仍由同源 AR 给出；`b6b4_surface_audit.md` 是事后、非盲、单审阅者诊断,无独立盲评与 inter-rater reliability。5/24 strict 与至少 22/24 unsupported specificity 是警报,不是最终人类效度估计。
8. **carrier 不等于 steering**:nonzero carrier 的 ablate/insert/amplify 只测 AV→AR readout 的局部敏感性；没有把方向注入基础模型生成并测行为、任务成功率或副作用。所有“控制/steering”结论仍在本实验支持范围之外。
9. **选择与多重查看**:新语义特征由 train 上的 SAE 选择性筛得；方向、极性、检索、carrier 与表面审计同时被查看。小 sign-test p 值只能作为研究性证据,不构成跨设定确认性显著。
10. **F10 仍不是确认性 C1**:24 个 feature 与 AV 文本此前已查看；只有 7 个重复粗轴，细 reference 是事后撰写；base autointerp、paraphrase 与 judge 均非独立人工真值。所有数值必须称 protocol pilot。
11. **paraphrase 未匹配长度/命题**:NLA 改写的字符保留率 median 60.1%，不能把 Δq 直接归因于 private code。必须在新 cohort 上用多份盲、长度与命题匹配的等义改写复现。

## 5. 建议的下一步(按性价比)

> **2026-07-30 晚更新**:原第 1–2 项(C1-confirmatory、C7-v2)已被 N1/N2 的结果改写。
> C7 已通过(F11),不再需要 C7-v2 作为前置;C1-confirmatory 在换成真实语料底座之前
> 不应重启(合成语料自审门禁连续四次 FAIL 的根因见 `REVIEW_OPUS_2026-07-30.md` §3.1)。
> 当前顺序:**N3 真实语料底座 → 通道消融跨方法复现 → 重新设计的 C1**。
> 具体设计见 `REVIEW_OPUS_2026-07-30.md` §4 的 N3/N4,以及下方保留的原始条目作为背景。

1. **C1-confirmatory(原最高优先,现改为 N3 之后)**:新冻结至少 60、优先 100 个 feature，覆盖 ≥15–20 个 label cluster；在查看输出前冻结 truth/hard negatives/generic/统计方案，由至少 3 名盲评者评价 correctness、specificity 与 unsupported claims。主终点是 matched AR delta/AUC 能否预测盲评人类正确性，推断按 label/model 聚类。
2. **C7-v2(长度与命题匹配的第三方改写)**:NLA/reference/autointerp 各做多份人工确认等义、长度匹配 paraphrase；当前一次性压缩改写只构成 form-sensitivity 警报，不能证明 private code。
3. **C2-v2(仅在 C1/C7 通过后扩展)**:以 `+w_dec` 和 `q+` 做候选排序,加盲评、多审阅者 agreement、真实 Neuronpedia/自动解释基线；预注册 ITT 与外部标签准确率,heldout-valid 只保留为描述性分层。
4. **真正 steering 实验(另立项)**:把 SAE 方向在基础模型 residual stream 中做剂量化注入/消融,测目标行为、非目标能力与文本质量,并与随机/active control 比较。F9 carrier 只能为剂量与候选特征提供先验,不能作为 steering 结果。
5. **off-manifold 课程训练(救路线③)**:若仍要读残差/合成方向,需对 AV 混入方向、残差与叠加向量的解释监督,属于训练级项目。

**B2、正式 B6+B4、C1 protocol pilot、N1(C7+B3 通道拆解)与 N2(因果 patch)均已完成；
C1-confirmatory 尚未开始,且在真实语料底座就位前不应开始。**

---

## 附录 A · 编造普查明细(F6 数据)

| idx | doc | 编造实体 | 正确主题 | nla_cos |
|---|---|---|---|---|
| 2 | 0 | Chicago Museum of Illusions | Eiffel Tower | 0.9977 |
| 5 | 0 | London Eye | Eiffel Tower | 0.9946 |
| 25 | 3 | Chinese Crested Dog(域内凭空具体化) | cats & dogs | 0.9964 |

分类计数:主题正确实体 19 行 / 编造 3 行 / 无实体 18 行;三组 nla_cos 均值 0.9962 / 0.9962 / 0.9958(源:`nla_results.json`,统计脚本为纯标准库文本处理)。

## 附录 B · 关键数字速查

| 指标 | 未去均值 | 去均值 | 出处 |
|---|---|---|---|
| NLA 全向量重建 | 0.996 | **0.859** | `nla_results` / `centered_rescore.head_to_head_centered` |
| SAE-small 重建 | 0.9925 | 0.658 | `sae_results` / 同上 |
| SAE-big 重建 | 0.9936 | 0.725 | `sae_results_big` / 同上 |
| 残差解释弱信号对照(vs 全向量) | 0.975 | 0.041 | `resid_pilot.cos_rx` / `centered_rescore.resid_centered` |
| B2 Top-1 (NLA / SAE-s / SAE-b / residual-text control) | — | 92.5% / 95.0% / 95.0% / 17.5% | `retrieval_eval.methods.*.summary` |
| B2 mean margin (同上) | — | 0.3195 / 0.2085 / 0.2531 / −0.0507 | 同上 |
| w_dec top 特征往返 | 0.80(median\|cos\|) | 0.189(mean\|cos\|) | `wdec_pilot` / `centered_rescore.wdec_centered` |
| w_dec gauss 对照 | 0.013 | 0.011 | 同上 |
| 残差可读性 cos_rr | 0.036 | −0.029 | `resid_pilot` / `centered_rescore.resid_centered` |
| 注入检测 α=2 | 0.919(地板 0.919) | 0.092(地板 0.072) | `injection_pilot.summary` / `centered_rescore.injection_centered_curve` |
| SAE FVE | 0.608 / 0.675 | — | `sae_results*.summary.fve` |

### N1 / N2(F11–F13)速查

| 指标 | 值 | 出处 |
|---|---|---|
| 固定泛型文本地板(E1–E7,直接测) | **−0.005**(单对最大 0.248) | `c7b3_scores_v1.summary_by_variant.__generic_fixed__` |
| 第三方等义改写保留率(C7) | **97.3%**(delta −0.024,n=20) | `n1n2_analysis.n1_variant_summary.para_tp` |
| 受控实体替换保留率(B3) | **89.7%**,检索 Top-1 95.0% | 同上 `.entity_swap` |
| 只留第 3 段(下一 token 预测) | **96.1%**,检索 Top-1 95.0% | 同上 `.p3_only` |
| 抹掉第 3 段引号内候选串 | **20.8%**,检索 Top-1 22.5% | 同上 `.quote_strip_p3` |
| 抹掉第 2 段逐字引用 | 99.4% | 同上 `.quote_strip_p2` |
| 词序打乱 | 66.9%,检索 Top-1 85.0% | 同上 `.word_shuffle` |
| KL-recovered:NLA / SAE-big / SAE-small | **0.757 / 0.771 / 0.713** | `n1n2_analysis.n2_kl_recovered` |
| 内容 token 上的 KL-recovered | 0.776 / **0.865** / 0.738 | 同上 |
| mean KL@pos:NLA / SAE-big / SAE-small / zero / gauss | 2.41 / 3.25 / 4.14 / 16.34 / 31.07 | 同上 |
| NLA−SAE-big 的 KL 配对 CI | **[−0.21, 1.98]**(跨零) | `n1n2_analysis.n2_paired` |
| NLA−SAE-small 的 KL 配对 CI | [0.50, 3.21] | 同上 |
| identity KL / provenance cos | 0.0000 / 1.000000 | `causal_patch_v1.provenance, summary` |
| 队列构成 | 13/40 是模板或空白 token | `n1n2_analysis.cohort_token_composition` |

### B6+B4(F9) 正式结果速查

| 指标 | 主结果 / 分层值 | 口径 | 出处 |
|---|---|---|---|
| 语料与方向 | 24 文档,1,365 激活；24 semantic_new / 45 total directions | 4 domain×3 language×train/test | `b6b4_factorial_selection.json → summary,dataset`;`b6b4_factorial_result.json → inputs` |
| semantic_new `q+` | median **0.1136**,bootstrap 95% [0.0091,0.4527] | **ITT n=24,主结果** | `b6b4_factorial_result.json → summary_by_cohort_greedy.semantic_new_intention_to_test` |
| semantic_new `r−` | median **0.0306** | ITT n=24；弱于 `q+` | 同上 |
| polarity | median **0.0708**,95% [0.0199,0.2455]；23/24>0 | ITT n=24；研究性 sign-test p=1.49e−6 | 同上 |
| sign / retrieval | sign accuracy 75.0%；signed Top-1/5 33.3%/41.7%；feature Top-1/5 33.3%/43.8% | ITT；90 signed / 45 feature candidates | 同上；`protocol_notes` |
| domain ITT | median `q+`/`r−`/polarity=.1181/.0276/.0728 | n=15 | `b6b4_factorial_analysis.json → direction_summary.domain_itt` |
| language ITT | median `q+`/`r−`/polarity=.1090/.0416/.0688 | n=9 | `b6b4_factorial_analysis.json → direction_summary.language_itt` |
| heldout-valid yield | all 14/24；domain 6/15；language 8/9 | **post-selection descriptive** | `b6b4_factorial_analysis.json → selection_yield` |
| heldout AUC ↔ `q+` | Spearman **ρ=−0.015** | 冻结新语义特征 | `b6b4_factorial_analysis.json → heldout_auc_vs_q_plus_spearman` |
| active / Gaussian control | median `q+` −0.0036 / 0.0045；feature Top-1 18.8% / 0% | n=8 / 8 | `b6b4_factorial_result.json → summary_by_cohort_greedy.active_nonselective/gaussian` |
| nonzero carrier | ablate **0.2102(15/15+)**；insert **0.0975(13/15+)**；amplify **0.0415(8/15+)** | median centered effect,n=15；**readout,非 steering** | `b6b4_factorial_analysis.json → carrier_summary.semantic_new_nonzero` |
| `+w_dec` surface audit | **5 strict / 7 coarse / 9 mismatch / 3 indeterminate**；≥22/24 unsupported specificity | post-hoc,非盲,单审阅者 | `b6b4_surface_audit.md`;原始字段见 `b6b4_factorial_result.json → scored_generation_rows` |
