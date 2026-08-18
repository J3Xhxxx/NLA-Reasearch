# Fable 5 对 NLA vs SAE 项目的研究分析：从“向量很像”到“解释真的忠实”

> 写给刚进入机械可解释性、表征学习与模型评测领域的研究者。
>
> 本文的目标不是简单复述七条建议，而是解释：**这些建议分别在怀疑什么、为什么原指标可能骗人、每个实验到底能证明什么、不能证明什么，以及结合本仓库后续进展，现在最值得继续做什么。**

---

## 目录

1. [先说结论：Fable 5 真正在提醒什么](#1-先说结论fable-5-真正在提醒什么)
2. [这个项目究竟在比较什么](#2-这个项目究竟在比较什么)
3. [必须先掌握的概念](#3-必须先掌握的概念)
4. [证据不是一个分数，而是一座阶梯](#4-证据不是一个分数而是一座阶梯)
5. [方向一：去均值、白化、范数与可比 FVE](#5-方向一去均值白化范数与可比-fve)
6. [方向二：用判别性评测代替单纯相似度](#6-方向二用判别性评测代替单纯相似度)
7. [方向三：自然语言解释的忠实度敏感性分析](#7-方向三自然语言解释的忠实度敏感性分析)
8. [方向四：因果评测——把重建向量放回模型](#8-方向四因果评测把重建向量放回模型)
9. [方向五：率失真曲线与容量匹配](#9-方向五率失真曲线与容量匹配)
10. [方向六：混合自编码器，以及为什么事情没有想象中简单](#10-方向六混合自编码器以及为什么事情没有想象中简单)
11. [方向七：扩大样本、分布、层和模型](#11-方向七扩大样本分布层和模型)
12. [结合仓库后续实验，原七条建议哪些已经完成](#12-结合仓库后续实验原七条建议哪些已经完成)
13. [如何重新理解发现 1、发现 3、发现 4](#13-如何重新理解发现-1发现-3发现-4)
14. [现在的研究方向应该怎样重新排序](#14-现在的研究方向应该怎样重新排序)
15. [一套适合写论文的实验框架](#15-一套适合写论文的实验框架)
16. [新手最容易犯的推理错误](#16-新手最容易犯的推理错误)
17. [术语表](#17-术语表)
18. [最后应该记住的十句话](#18-最后应该记住的十句话)

---

# 1. 先说结论：Fable 5 真正在提醒什么

Fable 5 的分析可以浓缩成一句话：

> **不要把“重建向量与原向量的余弦相似度很高”，直接解释成“自然语言准确表达了模型内部真正使用的语义”。**

这句话包含四层警告：

1. **几何指标可能被共享方向污染。**  
   所有激活都有一个很强的共同背景方向，重建器只要复现这个背景，余弦就可能很高。

2. **能重建样本身份，不等于文字表面语义正确。**  
   文本可能携带某种闭环编码、措辞指纹，或者只描述了主题和候选续写，却没有忠实表达具体事实。

3. **激活空间里“看起来接近”，不等于模型行为保持不变。**  
   模型对不同方向的敏感度高度不均匀。一个很小但落在关键方向上的误差，可能大幅改变下一 token 分布。

4. **NLA 与 SAE 的信息预算不同，默认配置的单点比较可能不公平。**  
   一段自然语言和 15 个稀疏 feature 并没有使用相同的容量。

因此，原始七条研究建议不是七个随意的扩展实验，而是一条逐步升级的证据链：

```text
清理几何指标
    ↓
验证样本可区分性
    ↓
验证文本具体语义是否重要
    ↓
验证重建是否保持模型行为
    ↓
控制信息容量进行公平比较
    ↓
探索组合方法
    ↓
检查结论能否跨数据、层和模型推广
```

这也是本文最重要的思想：

> **研究不是不断堆更多指标，而是让每一个新实验排除前一个结论的替代解释。**

---

# 2. 这个项目究竟在比较什么

## 2.1 什么是 activation？

大语言模型每读入一个 token，都会在每一层产生一个高维向量。这个向量不是一个人类可直接读懂的标签，而是模型当时用于后续计算的内部状态。

本项目抽取的是 Gemma-3-12B-IT 第 32 层的 `resid_post` 激活：

```text
x ∈ R^3840
```

也就是说，每个激活是一个包含 3840 个浮点数的向量。

可以把它想成模型在某个 token 位置上的“脑电图”。但这个比喻也有局限：激活不是静态知识条目，而是由以下因素共同决定的瞬时计算状态：

- 前文上下文；
- 当前 token；
- 句法角色；
- 主题与实体；
- 模型正在预测的下一 token；
- 该层需要传给后续层的各种中间信息。

抽取位置对应 `model.layers[32]` 的 block output，见：

- `server/02_extract_activations.py:8-15`
- `server/02_extract_activations.py:55-69`

## 2.2 NLA 做什么？

NLA 是 Natural Language Autoencoder，自然语言自编码器。它由两个模型组成：

```text
原始激活 x
   │
   │ AV：Activation Verbalizer
   ▼
自然语言解释 t
   │
   │ AR：Activation Reconstructor
   ▼
重建激活 x_hat
```

- **AV** 做 `vector → text`：把激活写成自然语言。
- **AR** 做 `text → vector`：从自然语言恢复激活。

上游仓库对 NLA 的定义见：

- `../nla-from-autodl/natural_language_autoencoders/README.md:10-21`

本项目使用的 L32 检查点与维度见：

- `../nla-from-autodl/natural_language_autoencoders/README.md:46-50`

NLA 的诱人之处在于：它的瓶颈是自然语言。理想情况下，压缩后的表示不仅能重建向量，还能被人读懂。

但这里有一个非常重要的逻辑区别：

> **“中间经过了自然语言”不自动等于“自然语言的普通人类语义就是重建成功的原因”。**

文字还可能通过以下方式携带信息：

- 特定词序；
- 候选 token 列表；
- 句子长度；
- 标点与格式；
- AR 熟悉但人类不重视的措辞模式；
- 与具体事实无关的主题、体裁或预测分布信息。

所以，NLA 同时有两个不同目标：

1. **压缩目标**：文字能否帮助 AR 重建向量？
2. **解释目标**：人类读到的具体命题是否忠实表达了激活的功能内容？

第一个目标比第二个目标弱得多。原项目最容易混淆的就是这两者。

## 2.3 SAE 做什么？

SAE 是 Sparse Autoencoder，稀疏自编码器。

它把 3840 维激活编码成一个更大的 feature 空间，但每个样本只允许少数 feature 非零：

```text
原始激活 x
   │
   │ SAE encoder
   ▼
稀疏特征 z
   │
   │ SAE decoder
   ▼
重建激活 x_hat
```

本项目使用 Gemma Scope 2 的 JumpReLU SAE。其核心公式写在：

- `server/04_run_sae.py:8-9`

可以写成：

```text
pre  = x W_enc + b_enc
z_i  = ReLU(pre_i) · 1[pre_i > threshold_i]
x_hat = z W_dec + b_dec
```

SAE 的解释性来自另一条路线：

- 每一个激活的稀疏 feature 可以被收集其最大激活上下文；
- 研究者据此给 feature 命名；
- decoder direction `w_dec[i]` 可以被看作该 feature 对 residual stream 的写入方向。

SAE 的优点不是天然生成完整自然语言，而是：

- 表示稀疏；
- feature 可以跨样本重复使用；
- 线性结构更容易做干预；
- 已有成熟的 feature activation 与 causal intervention 工具。

## 2.4 项目原始问题

最初看起来，研究问题很简单：

> 在相同的 Gemma L32 激活上，NLA 和 SAE 谁重建得更好？

但随着实验推进，这个问题必须拆成至少五个问题：

1. 谁在原始几何空间中更接近原向量？
2. 谁更好地保存样本之间的差异？
3. NLA 文字是否忠实表达具体语义？
4. 谁更好地保存模型后续计算真正使用的信息？
5. 在相同信息容量下，谁更高效？

这五个问题的答案可以不同。事实上，本仓库后来的结果正是如此：

- NLA 的 centered cosine 更好；
- 但 SAE-big 的 aggregate causal recovery 并不比 NLA 差；
- NLA 文本主要传递候选续写信息；
- 对具体实体的替换并不会造成同等程度的得分下降。

---

# 3. 必须先掌握的概念

## 3.1 余弦相似度 cosine similarity

对两个向量 `x` 和 `y`：

```text
cos(x, y) = (x · y) / (||x|| ||y||)
```

它衡量两个向量的夹角：

- `1`：方向完全相同；
- `0`：正交；
- `-1`：方向完全相反。

余弦忽略长度。例如：

```text
y = 100x
```

仍然有：

```text
cos(x, y) = 1
```

因此余弦回答的是：

> “方向是否相似？”

而不是：

> “向量是否完全重建？”

本项目把向量 L2 normalize 后再计算误差，此时：

```text
||x_norm - y_norm||² = 2(1 - cos(x, y))
```

见：

- `server/03_run_nla.py:2-5`
- `server/04_run_sae.py:96-103`

## 3.2 范数 norm

向量的范数是它的长度：

```text
||x|| = sqrt(x_1² + ... + x_d²)
```

在神经网络中，向量长度可能影响：

- 后续 RMSNorm/LayerNorm 前后的相对比例；
- residual stream 中该分量相对其他分量的强度；
- logits 变化幅度；
- 某些非线性或 gating 行为。

如果评测只计算 cosine，就可能出现：

```text
方向完全正确，但强度严重错误
```

这对“方向重建”也许没问题，对“把向量 patch 回模型”则可能有严重影响。

## 3.3 去均值 centered representation

设数据集的平均激活为：

```text
μ = (1/n) Σ_i x_i
```

最直接的中心化是：

```text
x_i_centered = x_i - μ
```

本仓库的 centered rescore 更精确地针对平均方向做投影移除：

```text
m_hat = μ / ||μ||
x_perp = x - (x · m_hat)m_hat
```

这相当于从每个向量中删除“沿共同平均方向的分量”。

为什么需要这样做？因为 transformer residual activation 往往是高度各向异性的：大量样本共享一个或几个很强的公共方向。一个只会输出“典型激活”的重建器，在 raw cosine 上也可能表现很好。

直觉例子：

```text
真实 A = 公共背景 + A 的独特信息
真实 B = 公共背景 + B 的独特信息
预测   = 公共背景
```

如果公共背景的范数远大于独特信息，那么：

```text
cos(真实 A, 预测) 很高
cos(真实 B, 预测) 也很高
```

但预测实际上无法区分 A 和 B。

## 3.4 白化 whitening

去均值只删除共同中心，白化还要处理不同方向的方差不均衡。

设协方差矩阵为：

```text
Σ = E[(x - μ)(x - μ)^T]
```

理想白化变换是：

```text
x_white = Σ^(-1/2)(x - μ)
```

经过白化后，各主方向的方差都变成 1。

为什么这很重要？假设激活有两个方向：

- 方向 A 的自然标准差为 100；
- 方向 B 的自然标准差为 1。

未经白化时，欧氏误差几乎全由 A 决定。即使 B 对模型行为更关键，它也可能在总体误差中被忽略。

但本项目的旧 pilot 有一个必须强调的技术限制：

```text
样本数 n = 40
维度 d = 3840
```

样本协方差矩阵的秩最多只有 `n - 1 = 39`。因此不能天真地求一个稳定的 3840 维完整逆协方差矩阵。真正可行的方案包括：

- 在更大的 N3/N4/N5 activation cohort 上估计协方差；
- PCA whitening，只对白化后的有效主成分评分；
- diagonal whitening，只按每个维度方差缩放；
- shrinkage covariance，例如向单位阵收缩；
- low-rank + diagonal 模型；
- 在训练集估计变换，在独立测试集评分，避免数据泄漏。

所以原建议里“去均值加十几行代码”是准确的；但“完整白化也只要十几行且一定可靠”并不准确。

## 3.5 MSE

均方误差：

```text
MSE = (1/d) ||x - x_hat||²
```

它同时受到方向和范数影响。

问题是，raw MSE 的数值依赖激活的天然尺度，单看绝对值难以解释。因此通常需要相对于基线归一化。

## 3.6 FVE

FVE 是 Fraction of Variance Explained，方差解释比例：

```text
FVE = 1 - Σ_i ||x_i - x_hat_i||²
          / Σ_i ||x_i - μ||²
```

它的基线是“永远输出数据均值”。

- `FVE = 1`：完美重建；
- `FVE = 0`：和永远猜平均值一样；
- `FVE < 0`：比猜平均值还差。

SAE 的 FVE 已在：

- `server/04_run_sae.py:115-128`

中实现。

为什么说要给 NLA 算一个“可比的 FVE”？因为旧 NLA 主指标偏重规范化方向，而 SAE 原生 FVE 使用 raw vector。若一边看方向、一边看 raw variance，不能直接宣布谁更好。

要得到真正可比的 FVE，必须保证：

- 使用同一批原始 activation；
- 两边都输出 raw-scale reconstruction；
- 不在评分前分别任意 normalize；
- 均值基线只用训练或评估协议规定的数据估计；
- 报告 per-sample error 以及 aggregate FVE；
- 同时报告方向与范数，方便解释失败来自哪里。

## 3.7 L0

本项目中的 SAE L0 指每个样本上非零 feature 的数量：

```text
L0 = #{j : z_j > 0}
```

代码见：

- `server/04_run_sae.py:103-112`

例如 L0≈15 表示一个 activation 平均使用约 15 个非零 SAE feature 重建。

注意：

> L0 不是完整的信息容量。

因为每个非零 feature 还需要编码：

- feature 的编号；
- feature 的激活幅值；
- 幅值的量化精度；
- 稀疏集合的编码方法。

## 3.8 KL divergence

KL divergence 用于比较两个概率分布。原模型下一 token 分布为 `p`，patch 后为 `q`：

```text
KL(p || q) = Σ_v p(v) log[p(v) / q(v)]
```

- KL 越小，两个分布越接近；
- KL 为 0，说明完全相同；
- KL 不对称，`KL(p||q)` 和 `KL(q||p)` 不一样。

在本项目中，原始模型分布是参照，因此自然使用原分布到重建分布的方向。

## 3.9 rate–distortion

Rate–distortion 理论问的是：

> 在只允许使用 R bit 的情况下，最少能把失真 D 降到多少？

- **Rate**：编码长度、信息预算；
- **Distortion**：重建误差。

单点比较只告诉你两个默认配置谁得分高。率失真曲线告诉你：

- 低容量时谁更高效；
- 高容量时谁的性能上限更高；
- 为多一点保真度需要付出多少额外 bit；
- NLA 的优势究竟来自表示形式，还是来自更大的容量。

---

# 4. 证据不是一个分数，而是一座阶梯

理解整个项目最好的方式，是把不同实验放在一座证据阶梯上。

## 第 1 层：raw cosine

问题：

> 重建向量是否大致朝同一个方向？

它容易被共同均值方向污染，是最弱的一层。

## 第 2 层：centered / whitened geometry

问题：

> 删除公共背景后，重建是否保存样本特有的方向与方差结构？

它比 raw cosine 强，但仍只描述几何关系。

## 第 3 层：retrieval / discrimination

问题：

> 重建能不能从候选中认出它对应的原样本？

它验证样本身份信息，但还不验证文字的普通人类语义。

## 第 4 层：semantic sensitivity

问题：

> 改错文字中的具体实体、关系或命题，重建会不会明显恶化？

它开始直接检验自然语言解释是否在传递具体内容。

## 第 5 层：causal fidelity

问题：

> 把重建向量放回模型后，下一 token 分布、loss 或内部计算是否恢复？

这是“模型真的使用了这些信息吗”的直接检验。

## 第 6 层：capacity-matched comparison

问题：

> 在相同 bit 预算下，谁能保存更多几何、语义和因果信息？

它才是 NLA 与 SAE 的公平效率比较。

## 第 7 层：external validity

问题：

> 结论能否跨样本、语料、语言、层和模型成立？

只有到了这一层，才适合讨论普遍性。

可以把它记成：

```text
高 raw cosine
  不推出高 centered cosine

高 centered cosine
  不推出高 retrieval

高 retrieval
  不推出文本命题忠实

文本命题敏感
  不推出 causal fidelity

高 causal fidelity
  不推出 rate efficiency

单层单模型上有效
  不推出普遍成立
```

---

# 5. 方向一：去均值、白化、范数与可比 FVE

## 5.1 原建议在怀疑什么？

它怀疑发现 1——“NLA 比 SAE 重建得更好”——可能主要由评测口径造成。

原始 raw cosine 很高：NLA、SAE 都在 0.99 左右。这看起来像几乎完美，但在高度各向异性的 activation space 中，0.99 未必意味着保存了大量样本特异信息。

如果所有向量都有：

```text
x_i = μ + δ_i
```

并且：

```text
||μ|| >> ||δ_i||
```

那么只预测：

```text
x_hat_i = μ
```

就能获得很高 cosine。

这里：

- `μ` 是共享方向；
- `δ_i` 才是区分不同样本的信息。

因此，第一步不是训练更大的模型，而是先问：

> NLA 是否真的重建了 `δ_i`，还是主要重建了 `μ`？

## 5.2 去均值结果告诉了我们什么？

仓库后来已经做了 centered rescoring，结果约为：

- NLA：0.859；
- SAE-small：0.658；
- SAE-big：0.725。

见：

- `results/POSSBILITY.md:61-67`

这意味着：

> 删除公共平均方向后，NLA 仍然明显更好地保存了样本特异的方向信息。

所以发现 1 并没有完全被推翻，而是被精确化了：

错误的强表述：

> NLA 全面重建得比 SAE 好。

更准确的表述：

> 在旧 n=40 pilot 的 centered direction metric 上，NLA 明显优于两个 SAE。

这两句话差别很大。第二句明确限定了：

- 数据集；
- 指标；
- 几何口径；
- 不外推到因果功能。

## 5.3 为什么还需要范数？

若 AR 训练和评分主要关注规范化方向，它可能没有压力恢复原始范数。

一个更完整的向量重建模型至少需要拆分：

```text
方向 u = x / ||x||
长度 r = ||x||
```

然后分别预测：

```text
u_hat, r_hat
x_hat = r_hat · u_hat
```

范数预测可以采用：

- 回归 `log ||x||`，通常比直接回归 norm 更稳定；
- 将 norm 分桶，做分类或 ordinal prediction；
- 在 AR value head 中联合预测向量与标量 norm；
- 用 raw-vector loss 与 directional loss 的加权组合。

例如：

```text
L = λ_dir · (1 - cos(x, x_hat))
  + λ_norm · (log||x|| - log||x_hat||)²
  + λ_raw · ||x - x_hat||² / d
```

但不同 loss 权重会改变训练目标，因此必须在 held-out 数据上选择，不能只挑让最终结论最好看的设置。

## 5.4 为什么可比 FVE 很重要？

FVE 的优点是它自动与“猜均值”比较。

假如某数据集的激活主要由公共方向构成，raw cosine 可能都很高，但如果重建没有解释样本间方差，FVE 不会因此虚高。

不过 FVE 也不是终点。它仍然按欧氏几何平均所有方向，不能知道哪些方向对后续模型行为更重要。因此：

```text
FVE 是比 raw cosine 更完整的描述性指标，
但不是 causal fidelity 的替代品。
```

## 5.5 新手应怎样理解“白化后再算 cosine”？

可以用“考试标准分”理解。

未经白化：

- 某些高方差方向像满分 1000 分的科目；
- 低方差方向像满分 10 分的科目；
- 总误差被大科目支配。

白化后：

- 每个统计主方向都变成标准差约 1；
- 比较不再被天然波动巨大的方向垄断。

但请注意，白化不是“绝对客观”。它改变了你认为哪些误差重要：

- raw geometry 重视大能量方向；
- whitening 重视相对于自然方差的异常；
- causal metric 重视后续网络 Jacobian 敏感的方向。

三者回答不同问题，都应该报告，而不是选一个最有利的分数。

## 5.6 这一方向能证明和不能证明什么？

能证明：

- NLA 是否只复现公共均值；
- NLA 是否保存样本特异方向；
- NLA 是否恢复范数；
- 在欧氏方差意义上，NLA 和 SAE 谁解释得更多。

不能证明：

- 文字具体命题是否正确；
- AR 是否依赖人类可读语义；
- 重建是否保持模型下一 token 行为；
- 相同 bit 预算下谁更有效率。

---

# 6. 方向二：用判别性评测代替单纯相似度

## 6.1 从“像不像”改成“认不认得”

相似度评测问：

> `x_hat_i` 与 `x_i` 有多像？

检索评测问：

> 在 `x_1, ..., x_N` 中，哪个最像 `x_hat_i`？它是不是正确的 `x_i`？

最简单的 top-1 retrieval：

```text
prediction(i) = argmax_j sim(x_hat_i, x_j)
accuracy = mean[prediction(i) = i]
```

若 N=40，随机猜测 top-1 accuracy 是：

```text
1/40 = 2.5%
```

如果模型只输出数据均值，那么它对很多候选都差不多，检索表现会接近随机或被少数中心样本占据。

## 6.2 为什么 retrieval 能抵抗共享方向污染？

假设所有样本都共享 `μ`：

```text
x_i = μ + δ_i
```

模型只输出 `μ`。

raw cosine 对每个样本都很高，但因为预测不包含 `δ_i`，它无法判断应该匹配 A、B 还是 C。

所以 retrieval 迫使重建器保存“谁是谁”的差异。

## 6.3 应该做哪些 retrieval 指标？

不只报告 top-1：

- top-1 accuracy；
- top-5 accuracy；
- Mean Reciprocal Rank；
- median rank；
- 每个 prompt/document 内的检索；
- 跨文档检索；
- centered cosine retrieval；
- whitened-space retrieval；
- hard-negative retrieval。

其中 hard negatives 很重要。随机候选可能主题差异很大，过于容易。更有价值的候选是：

- 同一篇文档的相邻位置；
- 相同 token 在不同上下文中的位置；
- 同一主题但不同实体；
- 相似下一 token 分布但不同真实状态；
- 同语言、同体裁的样本。

如果模型只能做主题级识别，它在随机候选中可能很强，但在同主题 hard negatives 中会失败。

## 6.4 “解释↔向量配对准确率”是什么意思？

有两种方向：

### 文本到向量

```text
解释 t_i → AR → x_hat_i
```

在所有原始向量中寻找最近邻，看是否找到 `x_i`。

### 向量到文本

给定原始激活 `x_i`，对多个解释 `t_j` 经 AR 得到的向量评分，看正确解释是否排名最高。

也可以把问题写成匹配矩阵：

```text
S_ij = sim(AR(t_i), x_j)
```

理想情况下，对角线 `S_ii` 应明显高于非对角元素。

## 6.5 retrieval 高为什么仍不等于解释忠实？

这是最容易误解的一点。

设 AV 为每个样本生成一段不同的文字。即使人类认为这些文字内容很模糊，只要文字里存在某些稳定差异，AR 就可能恢复样本身份。

例如：

```text
样本 A 的文字总是使用 “notably”
样本 B 的文字总是使用 “in particular”
```

如果 AR 学会把这种措辞当条形码，retrieval 会很高，但普通读者并不会认为这些词解释了激活。

更现实的情况是：NLA 文本包含候选下一 token 列表。这些候选确实高度区分当前模型状态，也确实是人类可读的；但它们描述的更像预测分布，而不是一段完整的抽象语义解释。

所以 retrieval 证明：

> 文本通道保留了实例级可恢复信息。

它不单独证明：

> 文本里的每个自然语言命题都正确并具有因果意义。

## 6.6 仓库状态

仓库已实现 centered 40-way retrieval：

- `server/10_retrieval_eval.py`

因此这一方向已从建议变成实际证据。但它应当被放在证据阶梯的中间，而不是最终答案。

---

# 7. 方向三：自然语言解释的忠实度敏感性分析

## 7.1 “自然语言解释”到底要忠实于什么？

“忠实”不是一个单一概念，至少可以拆成：

1. **主题忠实**：旅游、光合作用、法律等大类是否正确；
2. **体裁忠实**：百科体、说明文、候选续写列表等形式是否正确；
3. **实体忠实**：Eiffel Tower、Paris 等具体指称是否正确；
4. **关系忠实**：谁对谁做了什么，因果与属性是否正确；
5. **预测忠实**：文字是否准确表达模型下一 token 的候选与不确定性；
6. **因果忠实**：文本所表达的信息是否正是后续模型行为所依赖的信息。

NLA 可能在第 1、2、5 层很强，却在第 3、4 层较弱。若只说“解释忠实”，就把这些不同含义混在了一起。

## 7.2 受控替换实验的核心逻辑

假设原解释是：

> The Eiffel Tower is associated with Parisian tourism and landmark visits.

构造多个最小修改版本：

### 实体替换

> The London Eye is associated with London tourism and landmark visits.

### 关系反转

> Parisian tourism is caused by visitors explaining the Eiffel Tower.

### 属性替换

> The Eiffel Tower is an underwater biological structure.

### 主题保持、具体事实错误

> A famous European landmark attracts tourists to a major city.

### 只保留体裁

> This passage discusses a notable entity and why people encounter it.

### 删除候选续写段

保留解释散文，但移除引号中的下一 token/短语候选。

每个版本尽量控制：

- token 长度；
- 句法结构；
- 标点；
- 词频；
- 文体；
- 信息量。

然后测：

```text
score(original) - score(perturbed)
```

这叫 sensitivity：评测器是否对我们关心的语义变化敏感。

## 7.3 为什么必须“受控”？

如果错误版文本同时更短、更不流畅、格式不同，那么得分下降可能来自长度或语言自然度，而不是具体语义。

例如：

```text
原文：120 token 的详细解释
错文：20 token 的简略句子
```

AR 得分下降不能说明它在意“事实错误”，也可能只是原文携带更多编码容量。

所以良好干预要做到：

> 只改变要检验的语义因素，尽可能固定其他因素。

这就是因果实验中的“控制变量”思想在文本上的应用。

## 7.4 Δcos≈0 应怎样解释？

若把 Eiffel Tower 换成 London Eye，重建得分几乎不变，那么至少说明：

- 具体实体不是 AR 重建的主要信息来源；
- 或者这两个实体在当前激活/候选分布中太相似；
- 或者 AR 对该文本位置不敏感；
- 或者文本中其他信息高度冗余，掩盖了实体变化。

因此不能只凭一次替换直接说“文本完全没有语义”。更严谨的设计应该：

- 对很多实体、属性、关系做批量替换；
- 同时构造相近和相远替换；
- 报告 paired distribution，而不是只看均值；
- 测试不同文本段落；
- 配合 causal patch 与独立 evaluator；
- 检查是否有少数样本高度敏感、均值却接近 0。

## 7.5 仓库后续发现

仓库的通道消融结果表明：

- 候选续写段 `p3` 单独保留大部分 centered score；
- 删除或破坏该段后，分数大幅下降；
- 具体实体替换后仍保留很高比例；
- 第三方等义改写在长度匹配后几乎不掉分。

旧 n=40 结果见：

- `results/POSSBILITY.md:166-211`

N4 在真实 content-token 上复现了“p3 是主要通道”，但也发现 `p1+p2` 仍保留大量冗余信息：

- `results/N4_REAL_CONTENT_CAUSAL_AUDIT_2026-07-30.md:53-68`

因此当前最准确的结论不是：

> NLA 文本没有语义。

也不是：

> NLA 文本完整忠实地解释激活。

而是：

> **对于这个 Gemma L32 NLA，文本中最稳定、最可恢复、最具因果作用的内容主要集中在候选续写信息；完整散文中的具体实体和命题并不是同等重要的信息通道。**

这使 NLA 更像：

> 人类可读的预测状态编码器。

而不是已经被证明的：

> 对内部抽象语义逐命题忠实的解释器。

## 7.6 “体裁级忠实”是什么意思？

如果替换具体事实几乎不影响重建，但把旅游类文字改成化学实验类文字影响较大，那么文本传递的可能是：

- 大主题；
- 文体；
- 词汇分布；
- 可能的下一 token 家族；
- 生成阶段。

这可以叫 topic-level 或 genre-level fidelity。

它不是毫无价值。模型内部状态本来就可能主要编码预测相关的分布信息。但论文中的主张必须跟证据匹配：

不应写：

> NLA produces faithful natural-language explanations of internal representations.

可以更保守地写：

> NLA produces a natural-language bottleneck that preserves substantial instance-specific and next-token-predictive information.

---

# 8. 方向四：因果评测——把重建向量放回模型

## 8.1 为什么这是“真正的考试”？

前面的指标都在 activation space 内部比较。

但机械可解释性最终关心的是：

> 这个表示是否保留了模型后续计算真正使用的信息？

模型后半段可以视为一个函数：

```text
logits = F(x)
```

原激活是 `x`，重建是 `x_hat`。

真正重要的是：

```text
F(x_hat) 是否接近 F(x)
```

而不只是：

```text
x_hat 是否在欧氏空间接近 x
```

## 8.2 为什么高 cosine 仍可能产生大 KL？

在 `x` 附近做一阶近似：

```text
F(x + ε) ≈ F(x) + J ε
```

其中 `J` 是后续网络对 L32 activation 的 Jacobian。

如果 `ε` 很小，但刚好落在 `J` 放大很强的方向上，输出变化仍会很大。

反过来，一个欧氏范数很大的误差，如果落在后续网络不敏感的方向上，可能几乎不改变 logits。

因此 activation space 的几何与功能空间的几何不是同一个东西。

可以把后续模型想成一台对某些方向极度敏感、对另一些方向几乎失明的仪器。

## 8.3 patch 实验怎么做？

对每个样本和目标位置：

1. 正常前向，保存 L32 原始 activation `x`；
2. 正常继续前向，得到原始下一 token 分布 `p_original`；
3. 把 L32 activation 替换为 NLA 重建 `x_hat_NLA`；
4. 继续后半段，得到 `p_NLA`；
5. 替换为 SAE 重建，得到 `p_SAE`；
6. 替换为零向量或均值向量，得到破坏基线 `p_zero`；
7. 计算 KL、loss difference、top-token 保持率等指标。

仓库中的相关实现包括：

- `server/30_causal_patch.py`
- `server/40_n4_causal_patch.py`
- `server/45_n5_causal_patch.py`

## 8.4 identity patch 为什么必须做？

先把原始 activation 自己 patch 回原位置。

理论上应得到：

```text
KL_identity = 0
```

如果不是 0，可能有：

- hook 位置错误；
- dtype 不一致；
- position 对齐错误；
- attention cache 不一致；
- 模型处于 dropout/train mode；
- 激活保存与重放过程有损；
- tokenization 或 prompt provenance 不一致。

identity control 是实验管线的单元测试。没有它，所有 causal 结果都可能是基础设施错误。

N4 报告 identity KL=0 且 provenance bit-exact，这是后续结论可信的重要原因：

- `results/N4_REAL_CONTENT_CAUSAL_AUDIT_2026-07-30.md:24-37`

## 8.5 zero ablation 的作用

只报告 `KL(original || reconstruction)` 很难判断数值大不大。

zero ablation 提供一个“完全破坏该位置状态”的参照：

```text
x_zero = 0
```

于是可以定义恢复比例：

```text
recovered = 1 - KL_recon / KL_zero
```

直觉：

- reconstruction KL = 0 → recovered = 1；
- reconstruction KL = zero KL → recovered = 0；
- reconstruction 比 zero 更差 → recovered < 0。

## 8.6 为什么逐样本 KL recovered 可能病态？

若某个样本的 `KL_zero` 本来就非常小：

```text
KL_zero = 0.000001
KL_recon = 0.000002
```

则：

```text
recovered = 1 - 2 = -1
```

比例看起来灾难性，但绝对差异只有百万分之一。

因此对很小分母逐样本取比值会产生重尾和极端值。

更稳定的 aggregate 指标是 ratio of sums：

```text
aggregate_recovered
= 1 - Σ_i KL_recon_i / Σ_i KL_zero_i
```

同时应报告：

- raw paired KL difference；
- median KL；
- quantiles；
- catastrophic tail；
- 按 `KL_zero` 大小分层；
- bootstrap confidence interval。

仓库 N4/N5 后续分析已经采用更稳定的 aggregate 口径。

## 8.7 N4 的真实结果怎样改变了发现 1？

在无模板污染的 200 个真实 content-token 位置上，aggregate causal recovery 约为：

- NLA：0.948；
- SAE-small：0.944；
- SAE-big：0.966。

见：

- `results/N4_REAL_CONTENT_CAUSAL_AUDIT_2026-07-30.md:94-135`

这说明：

1. 三种方法总体上都恢复了 zero ablation 所破坏信息的大部分；
2. NLA 在 centered cosine 上的明显优势，没有转化成相对于 SAE-big 的明显因果优势；
3. SAE-big 在 aggregate causal recovery 上数值最高；
4. NLA 有少量严重失败，因此均值之外必须研究 failure tail。

所以当前不能写：

> NLA causally preserves more information than SAE-big.

可以写：

> NLA achieves stronger centered activation-space reconstruction, while causal patching places it near SAE baselines and does not establish superiority over SAE-big.

## 8.8 除 KL 外还能测什么？

### 下一 token loss

对真实下一 token `y`：

```text
loss = -log p(y)
```

比较：

- 原始 loss；
- reconstruction loss；
- zero-ablation loss。

### Top-1 token agreement

重建后 argmax token 是否与原模型相同。

### Top-k overlap

原模型与重建模型的 top-k token 集合重叠率。

### Logit correlation

比较整个词表 logits，但要注意大量低概率 token 可能主导某些相关指标。

### Downstream internal state

比较 L33、L34 或最终 residual 的变化，定位误差在哪些后续层被放大。

### Sequence-level continuation

从 patch 状态继续生成多个 token。但生成是随机且误差会递归放大，适合做补充，不适合作为唯一指标。

---

# 9. 方向五：率失真曲线与容量匹配

## 9.1 为什么默认配置比较不公平？

NLA 的瓶颈可能是一段几十到上百 token 的文本；SAE-small 的瓶颈可能平均只有约 15 个非零 feature。

如果 NLA 使用更多信息，自然可能重建得更好。

类比：

- 选手 A 可以写 200 字答案；
- 选手 B 只能提交 15 个编号和强度；
- 最后比较谁恢复原文更准确。

这不能区分：

- 表示方法更有效率；
- 还是只是用了更多 bit。

## 9.2 SAE 的 bit 数怎么计算？

假设 SAE 有 16,384 个 feature：

```text
log2(16384) = 14 bit
```

若直接逐个编码 15 个 feature ID，需要粗略：

```text
15 × 14 = 210 bit
```

但还没有包括：

- 每个 feature 的幅值；
- 稀疏集合的边界；
- feature ID 是否排序；
- 幅值量化精度；
- 熵编码收益。

更准确的集合编码可用：

```text
log2 C(16384, 15)
```

而不是简单的 `15 × 14`。幅值则可用 4、8、16 bit 等不同量化设置。

因此 SAE rate 可以定义为：

```text
R_SAE = bits(active index set) + bits(quantized values)
```

## 9.3 文本的 bit 数怎么计算？

也不能简单地用：

```text
token 数 × 固定 bit
```

因为不同 token 的概率不同。

可选择的定义包括：

### 固定词表编码

若 tokenizer 词表大小为 `V`：

```text
R_text = length × ceil(log2 V)
```

简单但不够紧。

### 语言模型交叉熵编码

用一个固定、独立的语言模型估计：

```text
R_text = -Σ_t log2 p(token_t | prefix)
```

这更接近算术编码后的实际 bit 数。

### 实际压缩长度

使用明确的无损编码器，但通用压缩器对短文本、tokenization 和语料分布可能不理想。

论文中关键不是只有一种正确定义，而是：

- 预先规定；
- 对所有方法一致；
- 报告敏感性分析；
- 不根据结果临时换定义。

## 9.4 怎样画 rate–distortion 曲线？

### NLA 侧扫容量

- 最大解释长度：8、16、32、64、128 token；
- 候选续写个数：1、2、4、8、16；
- 是否保留 p1/p2/p3；
- 压缩式 paraphrase；
- 显式限制 bit budget；
- 训练不同瓶颈长度的 AV/AR，或至少在现有模型上做截断实验。

### SAE 侧扫容量

- 改变 JumpReLU threshold；
- 扫实际 L0；
- top-k feature 截断；
- 幅值量化为 2/4/8/16 bit；
- small 与 big SAE；
- 不同 width 与稀疏度。

### 每个 rate 点测多种 distortion

不要只画一条 cosine 曲线。至少包括：

- centered cosine；
- whitened MSE；
- raw FVE；
- retrieval accuracy；
- semantic sensitivity；
- causal KL；
- catastrophic failure rate。

这会形成多个不同的 rate–distortion frontier。

## 9.5 为什么这可能是一篇论文的骨架？

因为它把争论从：

> “自然语言好还是稀疏特征好？”

升级成：

> “在相同通信预算下，不同表示形式分别保留多少几何信息、样本身份、可读语义和因果功能？”

这更像一个可推广的科学问题，而不是两个具体 checkpoint 的跑分对比。

可能出现的有趣结果包括：

- 低容量时 SAE 更高效，高容量时 NLA 上限更高；
- NLA 对 retrieval 有优势，但对 causal KL 无优势；
- SAE 在 causal fidelity/bit 上占优，NLA 在 human-readable information/bit 上占优；
- p3 候选列表比完整散文具有更好的 rate–distortion；
- 混合表示形成更好的 Pareto frontier。

## 9.6 公平比较中的隐藏陷阱

### 文本可读性本身也是价值

即使 NLA 使用更多 bit，它输出的是人可以读的文字。SAE feature ID 本身不具可读性，需要额外标注成本。

所以可以同时画：

- 纯 rate–distortion；
- 加上 human annotation cost 的系统成本；
- 给定人工阅读时间的效用。

### 训练成本与推理成本不同

rate 是表示容量，不等于计算成本。还需分别报告：

- AV 推理 FLOPs；
- AR 推理 FLOPs；
- SAE encoder/decoder FLOPs；
- 延迟；
- 显存；
- 存储；
- 人工解释成本。

### 默认自然语言并不一定是最有效的码

NLA 可能为了语言流畅性使用很多冗余 token。真正高效的表示可能是：

- 简短候选 token 列表；
- 结构化 JSON；
- 自然语言 + 少量连续向量；
- SAE feature + 自动标签；
- 分层码。

率失真研究会把“自然语言是否是好瓶颈”变成可测量的问题。

---

# 10. 方向六：混合自编码器，以及为什么事情没有想象中简单

## 10.1 原始串行 hybrid 设想

Fable 5 的原始直觉是：SAE 与 NLA 的失败模式可能互补。

先让 SAE 重建：

```text
x_SAE = SAE(x)
```

计算残差：

```text
r = x - x_SAE
```

再让 NLA 编码残差：

```text
r_hat = NLA(r)
```

组合：

```text
x_hybrid = x_SAE + r_hat
```

理想解释：

- SAE 吸收常见、结构化、可复用的方向；
- NLA 描述 SAE 漏掉的实例级或高层信息；
- 两者相加比任何单一方法更好。

## 10.2 “免费的午餐”是什么意思？

这里不是数学意义上真的免费，而是说：

> 如果两个模型的误差向量负相关或落在不同子空间，组合可能在不发明全新基础模型的情况下显著降低误差。

设：

```text
ε_SAE = x - x_SAE
ε_NLA = x - x_NLA
```

如果二者失败位置不同，就可能通过 selector、residual correction 或 ensemble 获益。

但“平均分不同”并不足以证明互补。必须直接检查：

- per-sample error correlation；
- failure overlap；
- regret distribution；
- selector 能否在 held-out 上预测谁更好；
- 组合是否超过 oracle-free baselines。

## 10.3 仓库为什么给串行 residual hybrid 泼了冷水？

仓库的发现 4 表明，现有 NLA 对 SAE residual 几乎不可读：

- residual round-trip cosine 接近 0；
- centered 后也没有恢复；
- 注入很强的已知 SAE direction 后，AV 仍基本不响应；
- 输出发生模板化 collapse。

见：

- `results/POSSBILITY.md:83-92`

这说明 residual 不是 AV 熟悉的输入分布。

原始 activation 来自真实模型流形，可以粗略写成：

```text
x ~ P_natural_activation
```

而 SAE residual 来自：

```text
r = x - SAE(x)
```

它可能具有完全不同的：

- 范数；
- 均值；
- 协方差；
- 方向分布；
- 高阶结构；
- 与 token/context 的对应关系。

AV 只在前一种分布上训练，却被要求解释后一种 off-manifold 输入，失败并不奇怪。

## 10.4 如果仍想救 residual hybrid，需要什么？

不能只调一个 cosine 阈值。至少需要：

1. 收集真实 activation 及对应 SAE residual；
2. 设计 residual-specific AV/AR 训练数据；
3. 控制 residual norm 和方向分布；
4. 加入合成 feature injection curriculum；
5. 使用已知注入方向做 sensitivity unit test；
6. 先证明模型能读取 residual，再做自然语言解释；
7. 最终做 causal patch，检查 `x_SAE + r_hat` 是否真的改善 logits。

可能的训练目标：

```text
L = L_reconstruct_residual
  + λ_injection L_detect_known_direction
  + λ_causal L_match_downstream_logits
```

但这已经是一个新的训练项目，不再是“拿现有 NLA 免费接到 SAE 后面”。

## 10.5 选择式 hybrid：仓库现在更现实的路线

另一种 hybrid 不相加，而是做路由：

```text
if selector predicts NLA is safe:
    use NLA reconstruction
else:
    use SAE-big reconstruction
```

这叫 selective hybrid 或 fallback routing。

它利用的是：

- NLA 很多样本很好；
- NLA 有少量 catastrophic failures；
- SAE-big 更稳定；
- 若能提前识别 NLA 风险，就可以兼得部分 NLA 优势与 SAE 的可靠性。

N5 正在验证这条路线。当前 discovery gate 已冻结，但 held-out 结果尚未形成最终本地结论。

因此必须遵守研究纪律：

> discovery 上选出的 gate 只能叫候选策略；只有 frozen held-out cohort 上成功，才能叫验证成功。

## 10.6 selector 应该看什么？

可使用但需要严格防止泄漏的特征：

- centered cosine 的代理预测；
- AR reconstruction confidence；
- 解释长度；
- p3 候选熵；
- NLA 与 SAE reconstruction disagreement；
- activation norm；
- token 类型；
- base-model next-token entropy；
- 到训练分布的 Mahalanobis distance；
- AV 多次采样的一致性。

不能在实际部署 selector 中使用真实 causal KL，因为那是要预测的答案。真实 KL 只能作为训练标签或评估指标。

---

# 11. 方向七：扩大样本、分布、层和模型

## 11.1 为什么 n=40 只能算 pilot？

原始设计是 5 条英文 instruct prompt，每条最多 8 个位置，共 40 个 activation。

见：

- `server/02_extract_activations.py:33-39`
- `Conclude.md:23-25`

这适合：

- 验证代码管线；
- 快速发现巨大效应；
- 生成研究假说。

它不适合：

- 稳定估计小效应；
- 研究罕见失败；
- 比较长尾；
- 证明跨主题泛化；
- 训练或选择 selector 后再无偏评估。

更严重的是，原脚本要求 position≥50，但 prompt 太短，于是静默回退到后半段：

- `server/02_extract_activations.py:109-115`

最终 13/40 是 chat template 或 blank token：

- `results/POSSBILITY.md:246-263`

这说明原 pilot 不只是样本少，还存在系统性取样污染。

## 11.2 为什么 token 类型必须分层？

模型在不同 token 位置的内部任务不同：

- 模板 token：编码对话协议和角色边界；
- 标点：编码句法与段落结构；
- 实体 token：可能编码身份和事实；
- 普通 content token：混合局部语义与预测；
- 长上下文深处：需要跨距离整合；
- 高熵位置：下一 token 不确定性高；
- 低熵位置：预测几乎确定。

如果把它们混在一起取均值，模型之间的差异可能由某一类 token 主导。

因此扩样本不是只把 40 变成 400，而是需要预注册分层：

- content vs template；
- punctuation vs lexical；
- entity vs non-entity；
- entropy bins；
- position bins；
- language；
- domain。

## 11.3 为什么要扩到代码、多语言和事实回忆？

### 代码

代码具有：

- 强结构；
- 低熵局部模式；
- 长距离变量绑定；
- 精确符号身份。

SAE 可能在重复结构 feature 上更有优势，NLA 也可能更容易生成候选 token 列表。结果未必与自然语言相同。

### 多语言

若 AV/AR 主要在英文分布训练，多语言 activation 可能发生 domain shift。还可检验：

- 解释语言是否影响重建；
- 中文激活能否用英文解释后重建；
- 跨语言 paraphrase 是否保留同一 activation 信息；
- NLA 保存的是语言无关语义还是表面 token 预测。

### 事实回忆

事实任务可以做更明确的实体与关系替换，是测试命题忠实度的好场景。

### 长上下文

长上下文深处的 activation 可能包含：

- 远距离引用；
- 文档结构；
- 多实体状态；
- 检索结果；
- 当前局部候选之外的信息。

这能检验 L32 NLA 是否仍主要压缩成短期下一 token 候选。

## 11.4 为什么不同层可能完全不同？

一种常见但不能过度绝对化的直觉是：

- 早层更偏词形、局部模式；
- 中层更偏组合特征、实体与关系；
- 晚层更靠近最终预测分布。

如果 L32 已经是较晚层，那么 NLA 文本主要传递下一 token 候选并不意外。

真正有价值的问题是：

```text
NLA 与 SAE 的差距是否随层深呈系统变化？
```

可能出现：

- 早层 SAE 更占优，因为局部方向更线性、可复用；
- 中层 NLA 的自然语言语义最丰富；
- 晚层 NLA 主要变成预测候选编码；
- causal fidelity 与 centered cosine 的关系随层变化。

但跨层研究成本很高，因为：

- SAE 需要对应层 checkpoint；
- NLA 也需要对应层训练或 checkpoint；
- 激活尺度和分布不同；
- 每层 causal patch 的后续深度不同；
- rate 与 latency 也会变化。

## 11.5 仓库后来怎样修复了分布问题？

N3 使用真实 Pile 和多语言 XNLI 语料，目的就是修复旧 synthetic prompt 的模板污染和 feature coverage 假门禁：

- `server/33_n3_build_corpus.py:1-30`

N3 还证明旧 synthetic test 中所谓“死亡”的 8 个 feature，在真实语料中 8/8 都会激活。这说明旧门禁测到的是语料覆盖不足，而不是 feature 本身死亡：

- `results/n3_analysis.json:9-45`

N3/N4 重抽了 200 个真实 content-token 位置：

- 101 篇文档；
- template/blank=0；
- position≥64；
- Pile + XNLI。

N5 又冻结了 600 个独立 content groups，其中 200 discovery、400 held-out。

因此，仓库已经从“40 个被模板污染的短 prompt pilot”推进到“真实语料上的因果复现”。但尚未完成：

- 系统代码域；
- 更广多语言；
- 多层；
- 多基础模型；
- N5 最终 held-out 结论。

---

# 12. 结合仓库后续实验，原七条建议哪些已经完成

原始建议是一个较早阶段的路线图。不能在今天把它原样当成待办清单。

| 原方向 | 当前状态 | 当前含义 |
|---|---|---|
| 去均值/白化、范数、NLA FVE | centered 已完成；完整白化和完全同口径 FVE 仍可加强 | 发现 1 在 centered direction 上保留，但不能外推到 causal superiority |
| Retrieval | 已实现 | 证明文本通道保留强实例信息，但不证明命题忠实 |
| 具体指称替换 | 已大幅实施 | 具体实体不是主要信息通道；p3 候选续写更关键 |
| Causal KL patch | 已在旧 pilot、N4、N5 discovery 中推进 | NLA 未证明优于 SAE-big，并存在少量严重失败 |
| Rate–distortion | 尚未形成完整曲线 | 仍是最有论文骨架价值的未完成方向之一 |
| Hybrid | residual hybrid 有负面证据；selective hybrid 正在 N5 验证 | 必须区分串行 residual correction 与选择式 routing |
| 扩分布 | N3/N4/N5 已显著推进 | 真实 content-token 已复现；跨层、跨模型仍不足 |

## 12.1 为什么“已经做过”不等于“问题彻底解决”？

例如，仓库已经做过实体替换，但仍可以进一步加强：

- 更大样本；
- 自动生成最小对；
- 人工验证替换质量；
- 更多关系、属性、否定、数量修改；
- hard negative；
- 与 causal effect 联合分析。

因此合理的说法是：

> 某方向已获得第一轮实证，不再是完全空白；下一步应从“是否存在”升级到“边界、机制和泛化”。

---

# 13. 如何重新理解发现 1、发现 3、发现 4

## 13.1 发现 1：NLA 的 centered direction reconstruction 更强

旧结果：

- raw cosine：NLA、SAE 都非常高；
- centered cosine：NLA≈0.859，SAE-small≈0.658，SAE-big≈0.725。

因此发现 1 的可靠部分是：

> NLA 在旧队列中保留了更多去公共方向后的 activation direction 信息。

但 N4 causal patch 显示：

- NLA aggregate recovery≈0.948；
- SAE-small≈0.944；
- SAE-big≈0.966；
- 方法间均值差没有得到“明确 NLA 胜过 SAE-big”的支持；
- NLA 有少量严重 failure tail。

所以发现 1 不能再表达为：

> NLA 总体重建优于 SAE。

它必须表达为：

> **NLA 在 centered activation geometry 上明显占优；但在真实下游 causal fidelity 上没有证明优于 SAE-big。**

这不是把结果“说弱了”，而是把结论绑定到真正被测量的对象。

## 13.2 发现 3：SAE direction 的可读性不是普遍强，而是存在子群

早期 `w_dec` 方向的 AV/AR 得分看起来很高，但很多最强方向与数据均值高度对齐。

去均值后，大量得分下降，说明原高分的一部分是：

```text
SAE direction 与共享 mean direction 对齐
    +
AR 对 generic-like 文本也输出 mean-like vector
```

这是一种混淆。

但后续结果也不能简单概括为“所有 SAE feature 都只有一点弱信号”。`q+` 分布更像双峰：

- 一部分 feature 高度可读；
- 另一部分很弱；
- 中位数落在两个峰之间，反而不代表典型 feature。

见：

- `results/POSSBILITY.md:265-282`

所以发现 3 应更新为：

> **SAE decoder directions 的自然语言可读性具有强异质性。早期普遍高分主要受 mean-direction confound 影响，但仍存在一个可重复的高可读 feature 子群。**

研究问题应从：

> 平均 SAE feature 是否可读？

转向：

> 哪些 feature 可读，什么统计或因果属性预测可读性？

## 13.3 发现 4：当前 NLA 不能读取 SAE residual

这一发现目前仍是稳定负面结果：

- residual 本身 round-trip 很弱；
- centered 后没有改善；
- 强注入已知 feature 后，AV 仍不敏感；
- 输出模板化。

因此：

> 现有 AV 不能被直接当作 SAE 暗物质审计器。

这个负结果很有价值，因为它排除了一个诱人的简单故事：

> “SAE 负责结构，NLA 自动解释剩余语义。”

当前数据不支持这个故事。

如果以后重训 residual-specific NLA 并成功，那将是一个新结果；不能把它当作现有方法的自然能力。

---

# 14. 现在的研究方向应该怎样重新排序

原始七条按当时的边际成本排序是合理的。但结合仓库现状，今天的优先级应该更新。

## 第一优先级：完成 N5 frozen held-out selective hybrid

### 为什么性价比最高？

- discovery cohort 已完成；
- gate 已冻结；
- held-out cohort 已预留；
- 主要代码已经存在；
- 这是最接近一个“可确认或否定”的完整研究问题；
- 不完成 held-out，所有 selector 结果都可能只是 discovery overfitting。

### 必须回答

- held-out 上 normalized gain 是否仍为正；
- catastrophic regret 是否受控；
- selector 相对 always-NLA、always-SAE-big、随机路由是否显著更好；
- gate coverage 与风险的曲线如何；
- discovery threshold 是否原样应用，没有重新调参。

## 第二优先级：在真实 cohort 上做容量匹配的 rate–distortion

### 为什么？

这是当前最核心的公平性缺口。

已有：

- NLA reconstruction；
- SAE-small/big reconstruction；
- centered 指标；
- causal patch 管线；
- 真实 content-token cohort。

缺的是系统扫：

```text
文本 bit / SAE bit → geometry / retrieval / KL
```

这能把项目从“两个 checkpoint 的比较”升级为“表示效率框架”。

## 第三优先级：研究 NLA failure tail，而不只比较平均值

N4 显示 NLA 有少量严重 causal failure。

对真实系统而言，尾部风险可能比平均提升更重要。

需要分析：

- 哪些 token 类型失败；
- 是否与 activation norm、entropy、语言、position 相关；
- AV 文本是否有可识别异常；
- 多次 AV sampling disagreement 能否预警；
- SAE disagreement 是否能预测灾难；
- selector 是否真正降低 conditional risk。

## 第四优先级：建立更严格的语义忠实度 benchmark

当前已有 p1/p2/p3 消融和实体替换，应进一步系统化：

- 实体；
- 属性；
- 关系；
- 否定；
- 数量；
- 时间；
- 因果方向；
- 下一 token 候选；
- 文体与长度 controls。

每个干预同时测：

- AR geometry；
- retrieval；
- causal KL；
- 独立人类/异构 evaluator 判断。

这样才能精确描述 NLA 文本究竟在哪种 fidelity 上有效。

## 第五优先级：在更大真实数据上完成规范化 FVE / whitening audit

这项成本不高，但已不再像最初那样是决定整个项目生死的第一问题，因为 centered 结论和 causal patch 都已有结果。

仍值得补齐：

- raw norm reconstruction；
- NLA FVE；
- PCA/shrinkage whitening；
- 训练/测试分离估计协方差；
- 几何指标与 causal KL 的相关性。

## 第六优先级：跨层研究

科学价值高，但成本也高。需要对应层 NLA/SAE 资产，否则不是简单改一个 hook index。

一个经济方案是先选择：

- 早层；
- 中层；
- L32 晚层；

做三个代表点，而不是一开始扫所有层。

## 第七优先级：off-manifold residual NLA 重训

这是高风险、高成本的新项目。

当前已有负面证据，所以除非主要目标就是“SAE residual 可读性”，否则应排在：

- N5 held-out；
- rate–distortion；
- failure-tail；
- semantic benchmark

之后。

---

# 15. 一套适合写论文的实验框架

## 15.1 核心研究问题

可以把论文问题写成：

> Natural-language bottlenecks and sparse feature codes preserve different aspects of transformer activations. Under matched information budgets, which representation better preserves geometry, instance identity, human-readable semantics, and downstream causal function?

这比“谁 cosine 高”更完整。

## 15.2 预注册假说

### H1：几何

在去均值和白化空间中，NLA 保留的样本特异方向信息高于相近默认稀疏度的 SAE。

### H2：功能

NLA 的几何优势不会等比例转化为 causal KL 优势。

### H3：文本通道

NLA 的主要可恢复信息集中在候选续写段，而非散文中所有具体命题。

### H4：容量

在低 bit 预算下 SAE 可能更高效；在较高 bit 预算下 NLA 可能达到更高几何或 retrieval 上限。

### H5：选择性组合

冻结的 selector 能在 held-out 上以受控 catastrophic regret 获得正 routing gain。

## 15.3 数据划分

必须分成：

- training：训练任何新 AR/norm head/selector；
- validation/discovery：选 threshold、bit points、超参数；
- held-out test：只运行一次最终协议。

如果在 test 上看到结果后再改 threshold，test 就不再是 test。

## 15.4 主要终点与次要终点

### 主要终点

建议选择少量、明确的 primary endpoints：

- aggregate causal KL recovery；
- rate-matched causal distortion；
- held-out selector regret。

### 次要终点

- centered cosine；
- whitened MSE；
- FVE；
- retrieval；
- semantic sensitivity；
- top-token agreement；
- explanation length；
- compute cost。

不要把二十个指标都当 primary，否则容易产生多重比较与选择性汇报问题。

## 15.5 对照组

必须包含：

- identity patch；
- zero ablation；
- mean-vector baseline；
- generic text；
- shuffled explanation；
- mismatched explanation；
- Gaussian/random vector；
- p1/p2/p3 ablation；
- length-matched paraphrase；
- semantically wrong but surface-matched text；
- always-NLA / always-SAE selector baselines。

## 15.6 统计报告

至少报告：

- paired bootstrap confidence intervals；
- mean、median、quantiles；
- effect size；
- catastrophic failure rate；
- 分层结果；
- 样本级散点图；
- error correlation；
- 不只报告 p-value。

尤其要避免：

> 均值略高，就写成方法全面更好。

如果两个方法平均接近，但一个有更重的失败尾部，对部署和科学解释都很重要。

## 15.7 建议图表

1. raw cosine vs centered cosine 对比；
2. centered cosine vs causal KL scatter；
3. 每个方法的 causal recovery violin/ECDF；
4. NLA 与 SAE per-sample error correlation；
5. semantic intervention delta 分布；
6. rate–distortion Pareto frontier；
7. selector coverage–risk curve；
8. 按 token/domain/language/position 分层；
9. 跨层趋势；
10. 最严重失败案例的定性审计。

---

# 16. 新手最容易犯的推理错误

## 错误 1：把高 cosine 当成“解释正确”

高 cosine 只说明方向接近，甚至可能主要说明公共均值方向接近。

## 错误 2：把 centered cosine 当成最终真相

centered cosine 清理了一个混淆，但仍然是 activation-space geometry，不是模型行为。

## 错误 3：把 retrieval 高当成文字命题忠实

retrieval 说明文本带有身份信息。身份信息可能来自候选 token、格式或措辞，不一定来自每个表面命题。

## 错误 4：把 AR 当作独立裁判

AV 与 AR 是闭环系统。AR 分数天然偏向 AV 熟悉的编码方式。

AR 更像：

> “这个文本能否被本系统解码回目标向量？”

而不是：

> “这个文本对人类而言是否是正确解释？”

要评估语义正确性，需要独立证据：

- 人类标注；
- 异构模型；
- 原上下文；
- 受控干预；
- causal endpoint。

## 错误 5：只看平均值

平均值可能掩盖少量灾难性失败。NLA 的 N4 结果正提示要研究 tail risk。

## 错误 6：用 discovery 结果当 final result

N5 gate 在 discovery 上可行，不等于 held-out 一定成功。任何阈值、coverage、feature 选取都必须冻结后再测试。

## 错误 7：把“没有检测到”写成“证明不存在”

residual 不可读可以说明现有 AV 在当前协议下没有读出信号，但不能证明 residual 中没有任何语义信息。

更准确的是：

> 当前测量工具和训练分布下未检出。

## 错误 8：把 feature 没激活当成 feature 死亡

旧 synthetic corpus 太小。N3 证明旧测试中 8 个“死亡” feature 在真实语料中全部会激活。

所以：

```text
not activated on this tiny corpus
≠
dead feature
```

## 错误 9：默认两个瓶颈容量相等

自然语言 token 与 SAE L0 不是相同单位。必须定义 bit 编码。

## 错误 10：看见负结果就认为项目失败

发现现有 NLA 不能读 residual，是有价值的机制边界；发现 p3 才是主要通道，也使“NLA 到底是什么”更清楚。

好研究的目标不是维护最初故事，而是越来越准确地描述真实现象。

---

# 17. 术语表

## Activation

模型在某层某 token 位置上的内部向量状态。

## Residual stream

Transformer 各层持续读写的主隐藏状态通道。

## L32

本项目中指 Gemma-3-12B-IT 第 32 个 decoder block 的 `resid_post` 输出。

## NLA

Natural Language Autoencoder，使用自然语言作为中间瓶颈的向量自编码器。

## AV

Activation Verbalizer，`activation → text`。

## AR

Activation Reconstructor，`text → activation`。

## SAE

Sparse Autoencoder，将 activation 编码为少量非零 feature，再线性解码。

## JumpReLU

带学习阈值的稀疏激活方式，只有超过阈值的 preactivation 才保留。

## `w_dec`

SAE decoder matrix 中某个 feature 对应的写入方向。

## L0

单样本非零 SAE feature 数量。

## Cosine similarity

两个向量方向的相似程度，不关心长度。

## Centered cosine

删除均值或平均方向后计算的 cosine，更强调样本特异信息。

## Whitening

按协方差结构重新缩放方向，使不同统计主方向具有可比方差。

## MSE

原向量与重建向量的均方误差。

## FVE

Fraction of Variance Explained，相对“永远猜均值”解释了多少数据方差。

## Retrieval

从多个候选中识别正确配对，用于检验样本身份信息。

## Hard negative

与正确样本高度相似、因此更难区分的错误候选。

## Faithfulness

解释与模型内部实际信息或因果机制的一致程度。必须明确是主题、实体、关系、预测还是因果忠实。

## Fidelity

保真度。可指几何、语义或因果保真，不能不加限定地使用。

## Causal patch

把重建向量替换回模型内部位置，继续前向并测量行为变化。

## KL divergence

两个概率分布的差异指标。本项目用于比较下一 token 分布。

## Zero ablation

把目标 activation 替换为零，用作破坏基线。

## Identity patch

把原 activation 自己 patch 回去，用于验证实验管线无误。

## Rate

表示使用的编码 bit 数。

## Distortion

重建误差，可以是几何、检索错误或 causal KL。

## Pareto frontier

在无法同时进一步降低 rate 和 distortion 的情况下，由最佳折衷点组成的边界。

## Off-manifold

不属于模型自然 activation 分布的向量，例如某些 SAE residual 或合成 direction。

## Discovery set

用于探索、选阈值和提出规则的数据。

## Held-out set

冻结规则后才使用的独立测试数据，用于确认结论能否泛化。

## Catastrophic regret

选择器选错方法后造成特别严重损失的情况。

---

# 18. 最后应该记住的十句话

1. **NLA 是 `activation → text → activation`；SAE 是 `activation → sparse features → activation`。**

2. **raw cosine 很容易被所有激活共有的 mean direction 抬高。**

3. **去均值后 NLA 的方向重建优势仍在，但这只是几何结论。**

4. **高 retrieval 说明文本保留了样本身份，不说明每个自然语言命题都正确。**

5. **具体实体替换影响较小，而候选续写段影响很大，说明 NLA 主要传递预测相关信息。**

6. **把重建向量 patch 回模型测 KL，才直接检验是否保留模型真正使用的信息。**

7. **N4 中 NLA 没有证明在 causal fidelity 上优于 SAE-big，并且有少量严重失败。**

8. **NLA 文本和 SAE 稀疏码容量不相等；真正公平的比较需要 rate–distortion 曲线。**

9. **现有 NLA 读不懂 SAE residual，因此串行 residual hybrid 不是现成的免费提升。**

10. **当前最重要的工作不是重复旧 n=40 pilot，而是完成 N5 held-out、做容量匹配、研究 failure tail，并继续跨层与跨分布验证。**

最后，用一句最精炼的话概括整个项目目前的认识：

> **高 cosine 只证明向量看起来像；高 retrieval 证明能够识别样本；文本干预证明文字中的哪些内容真正参与解码；低 causal KL 才证明重建保住了模型实际使用的信息；而只有容量匹配和跨分布复现，才能支持 NLA 与 SAE 谁更好的普遍结论。**
