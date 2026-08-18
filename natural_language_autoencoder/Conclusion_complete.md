# Conclusion Complete — 从 F11 到精确表述的实验路线图

> **目的**：把 F11 的发现从"NLA 的解释是下一 token 候选列表"精确化为一个能同时解释
> 本项目（Gemma-3-12B L32）和 Anthropic 论文（Claude Opus 4.6）所有案例的表述，
> 并提供可直接执行的实验设计与提示词。

---

## 1. 当前发现与精度不足

### 我们的 F11（E11 实验结论）

> NLA 的"解释"实际上是被语言化的下一 token 分布预测，不是语义解释。
> 承载重建信号的是第 3 段里那串"下一 token 候选"。

**数据支撑**（[n1n2_analysis.json](file:///d:/Projects/natural_language_autoencoder/results/n1n2_analysis.json)）：

| 条件 | 去均值 cos | 保留比例 |
|---|---:|---:|
| 完整解释 | 0.859 | 100% |
| 只留第 3 段 | 0.826 | 96.1% |
| 抹掉第 3 段引号内候选串 | 0.176 | 20.4% |
| 第三方改写（C7） | 0.851 | 97.3% |
| 实体替换（B3） | 0.771 | 89.7% |

### 与 Anthropic 论文的对照差距

Anthropic 的 7 个 Case Study 中，有 3 个无法被"下一 token 候选列表"简单覆盖：

| Case | NLA 读出内容 | 是否超出"候选词列表" |
|---|---|---|
| 2. 俄语回复 | "用户可能是俄语母语者" | ✅ 这是**信念**，不是候选词 |
| 4. 奖励规则 | "偶数获得奖励，模型故意选 42" | ✅ 这是**策略意图** |
| 5. 评估感知 | "这像是一个安全测试" | ✅ 这是**元认知** |

### 提出的精确化表述（待验证）

> **NLA 的解释语言化的是模型的"预测状态"（predictive state）——包括：**
> 1. **候选层**：下一 token/短语的具体候选（最直接，在 Gemma-12B 上承载 ~96% 信号）
> 2. **信念层**：驱动候选选择的关于世界/用户/上下文的信念
> 3. **策略层**：模型关于"我应该怎么回答"的意图和推理
>
> **这不同于"语义摘要"——它不是在总结"文本说了什么"，而是在描述"模型接下来打算做什么以及为什么"。**

---

## 2. 验证实验设计

### 实验 V1：信念层消融（Belief-Layer Ablation）— gating experiment

**目的**：测量 AV 解释中"信念/策略描述"部分对 AR 重建的贡献。

**原理**：E11 把解释按段落拆分（p1/p2/p3）。但这种拆法把"体裁描述"和"信念描述"混在了一起（都可能出现在 p1/p2 中）。V1 改为按**语义功能**拆分。

**方法**：

1. 设计 prompt，使模型产生丰富的内部信念/策略状态。例如：
   - **信念触发 prompt**：翻译腔英语（触发"用户不是英语母语者"的信念）
   - **策略触发 prompt**：包含隐藏规则的代码块（触发"利用规则"的策略）
   - **元认知触发 prompt**：明显的安全测试场景（触发"这是测试"的感知）
   - **对照 prompt**：普通事实问答（仅触发 token 预测，不触发复杂信念）

2. 对每条解释，由人工或 LLM 标注每个句子的**语义功能**：
   - `TOKEN_PRED`：具体的下一 token/短语预测
   - `BELIEF`：关于世界/用户/上下文的信念陈述
   - `STRATEGY`：关于模型意图/推理策略的描述
   - `GENRE`：体裁/格式描述
   - `QUOTE`：对上文的逐字引用

3. 构造消融变体：
   - `only_token_pred`：只保留 TOKEN_PRED 句
   - `only_belief`：只保留 BELIEF 句
   - `only_strategy`：只保留 STRATEGY 句
   - `no_token_pred`：删除 TOKEN_PRED 句，保留其余
   - `no_belief`：删除 BELIEF 句，保留其余

4. 用 AR 重打分，计算每种功能的 share_of_orig。

**判定标准**：
- 若 `only_token_pred` 的 share ≈ `p3_only`（~96%）且 `only_belief` share ≈ 0 → F11 原始表述足够
- 若 `only_belief` 或 `only_strategy` 的 share 显著 > 0 → 需要精确化表述
- 比较信念触发 prompt vs 对照 prompt 的 BELIEF share 差异

**GPU 预算**：~20 分钟（只需 AR 重打分，不需 AV 生成）

---

### 实验 V2：因果编辑验证（Causal Edit Test）

**目的**：直接测试"编辑信念/策略描述"能否因果地改变 AR 重建向量的方向。

**原理**：Anthropic 论文的 Case 4 展示了通过编辑 NLA 文本（"偶数有奖励" → "奇数有奖励"）来生成 steering vector。我们可以复现类似的做法来测试信念层。

**方法**：

1. 选取 AV 解释中包含明确信念/策略的样本（从 V1 的标注中挑选）
2. 做受控编辑：
   - **信念编辑**：改变关于用户/上下文的信念（如 "user is asking about chemistry" → "user is asking about history"）
   - **策略编辑**：改变策略描述（如 "model will provide a factual answer" → "model will provide a creative story"）
   - **候选词编辑**：改变具体候选词（如 "likely 'chlorophyll'" → "likely 'mitochondria'"）
   - **体裁编辑**：改变体裁描述（如 "educational Q&A" → "casual conversation"）
3. 用 AR 重建编辑前后的向量
4. 计算 delta = AR(edited) - AR(original)
5. 把 delta patch 回基座模型，测下一 token 分布的变化

**判定标准**：
- 若候选词编辑的因果效应 >> 信念编辑 → token 候选是主要通道
- 若信念编辑也有显著因果效应（KL 变化 > 0.1）→ 信念层也在通道里
- 若策略编辑能定向改变模型行为 → 策略层是独立的信息源

**GPU 预算**：~30 分钟（AR 重建 + 因果 patch）

---

### 实验 V3：位置对比（Cross-Position Channel Localization）

**目的**：测试"预测状态的丰富度"是否随 token 位置变化。

**原理**：假说——prompt 早期位置的预测状态以 token 级信息为主，后期位置包含更多积累的信念/策略。

**方法**（不需要多层 NLA 检查点）：

1. 从 N4 的真实语料队列（`results/acts_L32_n3_v1.parquet`，200 个内容 token）中，按 position 分 3 组：
   - `early`：position < 100
   - `mid`：100 <= position < 300
   - `late`：position >= 300

2. 对每组各随机抽 30 个位置（不足则全取），用 AV 生成解释。

3. 对每条解释做 V1 的语义功能标注（TOKEN_PRED / BELIEF / STRATEGY / GENRE / QUOTE）。

4. 比较三组的 BELIEF/STRATEGY/TOKEN_PRED 句子比例。

5. 对每组做 p3_only 和 quote_strip_p3 的消融，比较 share_of_orig。

**判定标准**：
- 若 `late` 组的 BELIEF share > `early` 组 → 模型在积累上下文后 AV 会读出更多
- 若无差异 → AV 的三段结构是 AV 训练塑造的，与位置无关

**GPU 预算**：~15 分钟（AV 生成 + AR 打分）

---

### 实验 V4：跨模型验证（Cross-Model Replication）

**目的**：测试 F11 的信念/候选分离比例是否因模型规模而变化。

**前提**：Anthropic 发布了多模型 NLA 检查点（论文提到 "we release trained NLAs for popular open models"）。

**方法**：

1. 搜索 Anthropic 发布的 NLA 检查点列表（GitHub 或 HuggingFace）。

2. 如果有不同规模的检查点（如 Gemma-3-4B vs Gemma-3-12B vs Gemma-3-27B）：
   - 在相同 prompt 集上提取激活 + 用各自 AV 生成解释
   - 做 V1 标注 + 消融
   - 比较不同规模模型的 BELIEF/TOKEN_PRED share

3. 如果有不同家族的检查点（如 Llama-3 的 NLA）：
   - 跨家族重复对比

**判定标准**：
- 若大模型 BELIEF share 显著高 → 精确化表述成立，揭示 scaling 规律
- 若无差异 → F11 原始表述对所有规模适用

---

## 3. 执行提示词

> [!IMPORTANT]
> 以下提示词设计为可直接复制粘贴给 GPT-5.6-Luna-Max 或同等模型在本仓库中执行。
> 每个提示词都是自包含的，包含完整的背景、目标、方法和纪律要求。

---

### 提示词 P1：V1 信念层消融实验

````
# 任务：V1 信念层消融实验

## 背景

你在 `D:\Projects\natural_language_autoencoder` 仓库中工作。这是一个 NLA（Natural
Language Autoencoder）研究项目，对比 NLA 与 SAE 在 Gemma-3-12B-IT L32 激活上的
重建质量。

E11 实验（脚本 `server/28_build_text_variants.py` + `server/29_score_text_variants.py`）
已经把解释按**段落**拆分并测了每段的信号贡献。本实验改为按**语义功能**拆分。

## 你需要做的

### 阶段 1：本地（零 GPU）

1. 读 `results/nla_results.json`（encoding='utf-8'），提取 40 条解释
   （`rows[].explanation`）。

2. 对每条解释的每个句子，标注语义功能标签：
   - `TOKEN_PRED`：预测下一 token/短语的具体候选（通常在第 3 段，以 "likely"、
     "immediately expects"、引号列表为标志）
   - `BELIEF`：关于世界/用户/上下文的信念（如 "user is asking about..."、
     "signals a question about..."）
   - `STRATEGY`：关于模型意图的描述（如 "establishing..."、"sets up a..."）
   - `GENRE`：体裁/格式描述（如 "Academic/cultural question format"、"Q&A format"）
   - `QUOTE`：对原文的逐字引用（如 'The phrase "..." sets up...'）

3. 对每条解释构造 5 个消融变体：
   - `only_token_pred`：只保留标注为 TOKEN_PRED 的句子
   - `only_belief`：只保留 BELIEF 句
   - `no_token_pred`：删除 TOKEN_PRED，保留其余
   - `no_belief_strategy`：删除 BELIEF 和 STRATEGY，保留其余
   - `only_genre`：只保留 GENRE 句

4. 把所有变体写入 `results/v1_belief_ablation_variants.json`，格式参考
   `results/c7b3_variants_v1.json`：
   ```json
   {
     "protocol": "V1 belief-layer ablation",
     "frozen_before_scoring": true,
     "rows": [
       {
         "idx": 0,
         "variants": {
           "orig": "...",
           "only_token_pred": "...",
           "only_belief": "..."
         },
         "sentence_labels": [
           {"sentence": "...", "label": "TOKEN_PRED"}
         ]
       }
     ]
   }
   ```

5. 计算 SHA-256 并存为 `.sha256` sidecar。

### 阶段 2：远端 GPU（ssh autodl）

6. 上传 variants JSON 和一个打分脚本到服务器。打分脚本应：
   - 加载 AR（`NLACritic(ar_dir).reconstruct(text)` → 向量）
   - 加载原始激活（`results/recon_vectors.npz` 的 `x` 数组）
   - 加载均值方向（同文件的 `m_hat`）
   - 对每个变体用双边投影去均值后算 cos
   - 同时做 40-way centered 检索（Top-1, MRR）
   - 输出格式参考 `results/c7b3_scores_v1.json`

7. 关键代码接口（**不要重写**，直接 import）：
   ```python
   from pilot_common import AVLocal, JumpReLUSAE, load_acts
   import sys; sys.path.insert(0, str(REPO))
   from nla_inference import NLACritic
   critic = NLACritic(ar_dir)
   vec = critic.reconstruct(text)  # text → 向量，秒级
   ```

8. 结果写入 `results/v1_belief_ablation_scores.json`。

### 阶段 3：分析

9. 计算每种语义功能的 share_of_orig（= variant_cos / orig_cos）。
10. 做配对 bootstrap 95% CI。
11. 主要报告：TOKEN_PRED vs BELIEF vs STRATEGY 的 share 比较。

## 重要纪律

- **不修改任何既有 results/ 文件。** 新结果用新文件名。
- 远端任务用 `setsid nohup`，日志写 `results/v1_belief_ablation.log`。
- 脚本名用 `50_v1_build_belief_variants.py`（本地）和
  `51_v1_score_belief_variants.py`（远端）。
- 参考 `server/28_build_text_variants.py` 和 `server/29_score_text_variants.py`
  的架构风格。
- AR 加载会报 `model.norm.weight MISSING`——这是良性的，不要当错误处理。
- 上传的 .py 先 `sed -i 's/\r$//'` 去 CRLF。
- 远端路径：模型在 `/root/autodl-tmp/models/`，NLA 仓库在
  `/root/autodl-tmp/nla_repo/`，结果在 `/root/autodl-tmp/results/`。
- SSH 连接用 `ssh autodl`（已配置在 `~/.ssh/config`）。
````

---

### 提示词 P2：V2 因果编辑实验

````
# 任务：V2 因果编辑验证实验

## 背景

同 V1 的仓库和基础设施。本实验测试"编辑解释中的信念/策略描述"能否因果地改变
模型的下一 token 预测。这复现了 Anthropic NLA 论文 Case 4（Reasoning about
Rewards）的方法论。

## 前提

V1 的 `v1_belief_ablation_variants.json` 已经完成，包含句级语义功能标注。

## 你需要做的

### 阶段 1：本地（零 GPU）

1. 从 V1 的标注结果中，挑选包含明确 BELIEF 或 STRATEGY 句子的样本（至少 20 条）。

2. 对每条做 4 种受控编辑：
   - `belief_edit`：改变信念内容但保持句式
     （如 "question about photosynthesis" → "question about Roman history"）
   - `strategy_edit`：改变策略描述
     （如 "structured explanation" → "creative poem"）
   - `candidate_edit`：改变候选词但保持结构
     （如 "chlorophyll, carbon dioxide" → "centurions, gladiators"）
   - `genre_edit`：改变体裁描述
     （如 "Educational Q&A" → "Social media post"）

3. 冻结所有编辑到 `results/v2_causal_edit_variants.json`，含 SHA-256。

### 阶段 2：远端 GPU

4. 对每条原文和每个编辑版本：
   - 用 `NLACritic.reconstruct()` 得到重建向量
   - 计算 delta = reconstruct(edited) - reconstruct(original)
   - 把 delta 缩放到与原始激活同量级后 patch 回基座模型 L32
   - 测下一 token 分布的 KL 散度

5. 代码结构参考 `server/30_causal_patch.py`（因果 patch 的完整实现）。

6. 输出：`results/v2_causal_edit_results.json`，含每种编辑类型的 mean KL 变化。

### 判定标准

- 若 `candidate_edit` 的 mean KL >> `belief_edit` 的 mean KL → 候选词是主通道
- 若 `belief_edit` 的 mean KL 显著 > 0（即编辑信念能改变模型预测）→ 信念层是独立信息源
- 记录每种编辑的 KL 的 95% doc-clustered bootstrap CI

## 脚本命名

- `52_v2_build_causal_edits.py`（本地）
- `53_v2_score_causal_edits.py`（远端）

## 纪律同 P1。
````

---

### 提示词 P3：V3 位置对比实验

````
# 任务：V3 位置对比实验（轻量版多层代理）

## 背景

同 V1 的仓库。没有多层 NLA 检查点，所以用不同 token 位置做代理：prompt 早期
（模型积累的上下文少）vs 后期（上下文丰富）。

## 你需要做的

1. 从真实语料激活 `results/acts_L32_n3_v1.parquet` 中，按 position 分 3 组：
   - `early`：position < 100
   - `mid`：100 <= position < 300
   - `late`：position >= 300

2. 对每组各随机抽 30 个位置（不足则全取），用 AV 生成解释。

3. 对每条解释做 V1 的语义功能标注（TOKEN_PRED / BELIEF / STRATEGY / GENRE / QUOTE）。

4. 比较三组的 BELIEF/STRATEGY/TOKEN_PRED 句子比例。

5. 对每组做 p3_only 和 quote_strip_p3 的消融，比较 share_of_orig。

## 假说

- 若 `late` 组的 BELIEF share > `early` 组 → 模型在积累上下文后 AV 会读出更多
- 若无差异 → AV 的三段结构是 AV 训练塑造的，与激活的信息丰度无关

## GPU 预算

AV 生成 ~90 条 x ~3s = ~5 分钟；AR 打分 + 消融 ~10 分钟。总计 ~15 分钟。

## 脚本命名

`54_v3_position_analysis.py`（远端一体脚本）

## 纪律同 P1。
````

---

### 提示词 P4：跨模型验证

````
# 任务：V4 跨模型验证（前提探测 + 条件执行）

## 背景

Anthropic 论文提到 "we release training code and trained NLAs for popular open
models"。本实验首先确认有哪些开源 NLA 检查点可用，然后做跨规模对比。

## 你需要做的

### 阶段 0：可行性探测（本地/网络，零 GPU）

1. 搜索以下位置的 NLA 检查点：
   - GitHub: https://github.com/anthropics/natural-language-autoencoders
   - HuggingFace: 搜索 `kitft` 或 `anthropic` 的 NLA 相关仓库
   - 论文附录中的检查点列表

2. 记录：有哪些模型/层的 AV 和 AR 检查点？文件大小？下载方式？

3. 如果有至少 2 个不同规模的检查点：继续阶段 1。
   如果没有：写一份简短报告说明找到了什么，然后停止。

### 阶段 1：跨规模对比（远端 GPU）

4. 在**相同的 prompt 集**（用 N4 的真实语料）上：
   - 用各模型提取 L32（或等价层）激活
   - 用各自的 AV 生成解释
   - 做 V1 的语义功能标注和消融

5. 比较不同规模模型的 BELIEF/TOKEN_PRED share。

6. 输出：`results/v4_cross_model_results.json`

## 脚本命名

- `55_v4_probe_checkpoints.py`（阶段 0，本地）
- `56_v4_cross_model_analysis.py`（阶段 1，远端）

## 纪律同 P1。
````

---

## 4. 推荐执行顺序

```
V1（信念层消融）  ← 最便宜、最直接回答问题；这是 gating experiment
       |
V3（位置对比）    ← 如果 V1 发现信念层有信号
       |
V2（因果编辑）    ← 如果 V1+V3 支持信念层，用因果证据加固
       |
V4（跨模型）      ← 如果前三个都成立，推广到不同规模
```

> [!IMPORTANT]
> **V1 是 gating experiment。** 如果 V1 发现在 Gemma-12B 上 BELIEF share 约等于 0
> （意味着小模型的解释确实只有候选词），那精确化表述应改为：
>
> "在中等规模开源模型上，NLA 的通道以 token 候选为主；Anthropic 论文中
> Opus 4.6 上的信念/策略读出可能需要更大的模型容量和更长的 NLA 训练。
> 这是一个 scaling 现象。"
>
> 这本身也是一个有发表价值的发现。

---

## 5. 最终论文框架草案

无论 V1-V4 结果如何，论文都有一个可写的故事：

| V1 结果 | 论文主线 |
|---|---|
| BELIEF share 约为 0 | "NLA 在中等模型上学到的是 token 级预测器；信念/策略读出是大模型特有的 scaling 现象" |
| 0 < BELIEF share < TOKEN_PRED | "NLA 的通道分层：候选词为主、信念为辅，符合预测状态假说" |
| BELIEF share 约等于 TOKEN_PRED | "NLA 的通道是多层次的预测状态描述，F11 原始表述需要大幅修正" |

**所有三种结果都有论文价值。没有浪费 GPU 的场景。**
