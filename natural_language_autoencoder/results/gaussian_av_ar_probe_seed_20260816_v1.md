# 单个高斯 activation 的 AV → AR 结果

> 生成时间（UTC）：`2026-08-16T09:48:33.140663+00:00`  
> 性质：**探索性 OOD 随机对照**；不能把 AV 文本当成该随机向量的“真实语义”。

## 输入张量

| 项目 | 数值 |
|---|---:|
| shape / dtype | `(3840,)` / `float32` |
| 随机种子 | `20260816`（NumPy PCG64） |
| 分布 | 每维独立 `N(0, 1064.509137²)` |
| L2 norm | 66576.226348 |
| 冻结真实 activation 平均 norm | 65965.218574 |
| AV 内部实际注入 norm | 80000.000000 |
| 相对真实 cohort 的 norm percentile | 53.0% |
| 最近真实向量 raw cosine | 0.041137 |
| 最近真实向量 centered cosine | 0.051765 |

前 24 个元素（完整 3840 维张量保存在 NPZ 中）：

```text
[  296.16666,  -334.38367,  1366.1914 ,  1500.8011 ,  -405.54324,  1619.549  ,   286.91406,  -512.31323,
  -858.47614, -1046.3463 ,   338.4264 ,   443.18338,  1425.271  ,  1280.2482 ,   542.2355 ,  -883.73865,
  1248.1898 ,  -783.9634 ,  -705.4497 ,  -504.83054,  -550.4197 ,   610.3702 ,  -857.00806,  -288.39566]
```

读取方式：

```python
import numpy as np, torch
x = np.load('gaussian_av_ar_probe_seed_20260816_v1.npz')['input_activation']
tensor = torch.from_numpy(x)  # shape: [3840]
```

## AV 原文

```text
Article structure: a technical explainer with a Q&A format, following a structured essay pattern of listing and analyzing the problem.

The phrase "The following table" signals a citation/reference pattern, likely a truncated URL or publication detail for the "The Role of the..."

Final token "it'" opens a repeated phrase ("it.'") — likely a truncated word or phrase like "it." or "The 1999" or "The article" continuing the citation/footnote pattern, or "The text" or "The phrase '..." — likely "The article is" or "The answer is" referencing a specific academic/medical context.
```

## AR 得分

| 指标 | 数值 | 含义 |
|---|---:|---|
| Direction cosine | **0.033285** | AV 文本经 AR 重建后，与输入方向的余弦 |
| Direction MSE | **1.933431** | 官方口径 `2(1-cos)`；0 最好、2 约为正交 |
| Centered cosine | 0.006455 | 投影掉冻结 N4 均值方向后的辅助诊断 |
| Raw relative L2 error | 1.076226 | 未做方向归一化的补充量，不是官方主分 |
| AR reconstruction norm | 28793.970691 | AR 输出向量的原始 L2 norm |

## 如何解释

- 这是维度合法、可直接输入 AV 的高斯张量；AV 会把任何非零输入重标到 checkpoint 的注入尺度。
- 它只匹配真实 activation 的平均范数，不匹配均值、协方差或模型流形，因此仍是强 OOD 对照。
- AV 即使输出流畅、具体的文本，也不代表随机向量客观携带那些命题；AR 分数只衡量同一 NLA 系统的 round-trip 方向重建。

## 文件绑定

- 输入 tensor SHA-256（float32 bytes）：`e30e502e794c9dea1d748db651d0c36985225bc3bbe5657deaad3f05aed5c884`
- AV 文本 SHA-256（UTF-8）：`dcc83aa5b6d66d0a2533fd01868d08f027a6e262f939de8789dd78b52c824b65`
- AR tensor SHA-256（float32 bytes）：`dd3b3f4b351291812ccd4483970d2924c575562c80084f8047ae41721a7f1164`
- 真实 reference SHA-256：`e9d89713dc64381a52f05224d6522abb0ec547777a8c6a7f08b841a72a339967`
