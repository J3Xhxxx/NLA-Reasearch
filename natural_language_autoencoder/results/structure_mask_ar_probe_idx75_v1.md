# 真实 activation：AV 结构语言 X-mask → AR 单案例报告

> 冻结样本：N4/J2 `idx=75`，`OpenSubtitles`，doc `5849`，position `276`。  
> 性质：**探索性单案例**；样本在查看本次 masked-AR 输出之前固定。

## 1. 真实语料

### 目标位置之前的上下文

```text
Yes." "I will exercise caution." "shall I kill GennosukeNif given the chance?" "Do not assume
```

### 真实后续文本

```text
 heNcan be slain so easily!" "His eyes..." "Those twoNun
```

## 2. 冻结的 AV 自然语言解释（原版）

```text
Structured narrative format with ongoing dialogue: a formal anime synopsis, guiding a character's moral lesson through a battle encounter.

The sentence "Do not assume" establishes a directive warning against rash action, with the protagonist's emotional state of grief and uncertainty framing a cautionary statement about the fight.

Final token "assume" ends a prohibition clause ("Do not assume...Do not assume"), requiring a verb phrase — likely "that you can decide" or "such authority" or "my role." or "you are acting preemptively." or "that the outcome is yours." or "such liberties." — a specific overreach or assumption about taking the initiative or killing.
```

## 3. 结构语言替换为 X 后

规则：双引号外的每个 Unicode 字母逐个替换为一个 `X`；引号内内容、空格、标点、数字、段落和字符总长度保持不变。

```text
XXXXXXXXXX XXXXXXXXX XXXXXX XXXX XXXXXXX XXXXXXXX: X XXXXXX XXXXX XXXXXXXX, XXXXXXX X XXXXXXXXX'X XXXXX XXXXXX XXXXXXX X XXXXXX XXXXXXXXX.

XXX XXXXXXXX "Do not assume" XXXXXXXXXXX X XXXXXXXXX XXXXXXX XXXXXXX XXXX XXXXXX, XXXX XXX XXXXXXXXXXX'X XXXXXXXXX XXXXX XX XXXXX XXX XXXXXXXXXXX XXXXXXX X XXXXXXXXXX XXXXXXXXX XXXXX XXX XXXXX.

XXXXX XXXXX "assume" XXXX X XXXXXXXXXXX XXXXXX ("Do not assume...Do not assume"), XXXXXXXXX X XXXX XXXXXX — XXXXXX "that you can decide" XX "such authority" XX "my role." XX "you are acting preemptively." XX "that the outcome is yours." XX "such liberties." — X XXXXXXXX XXXXXXXXX XX XXXXXXXXXX XXXXX XXXXXX XXX XXXXXXXXXX XX XXXXXXX.
```

### 遮罩审计

| 项目 | 数值 |
|---|---:|
| 总字符数（两版相同） | 669 |
| 替换成 X 的字母数 | 397 |
| 引号内保留的字母数 | 132 |
| 原版 explanation tokens | 135 |
| masked explanation tokens | 211 |
| 原版完整 AR prompt tokens | 150 |
| masked 完整 AR prompt tokens | 226 |

## 4. AR 重建得分

| 指标 | 原始 AV | X-mask AV | masked − original |
|---|---:|---:|---:|
| Direction cosine ↑ | **0.996121** | **0.990748** | -0.005372 |
| Direction MSE ↓ | **0.007758** | **0.018503** | +0.010745 |
| Centered cosine ↑ | 0.834406 | 0.547987 | -0.286419 |
| Raw relative L2 error ↓ | 0.174025 | 0.340147 | +0.166123 |

AR 重建向量彼此的相似度：

- raw cosine：`0.994423`
- centered cosine：`0.658324`
- 本次原版 AR 与冻结 N4 原版 AR cosine：`1.000000000`

## 5. 解释边界

- 这个差值是该样本的局部结构遮罩效应，不是总体平均效应。
- 等字符数不等于等 token 数；连续 `X` 会改变 AR tokenizer 分词，因此结果同时包含结构语义移除与 OOD/tokenization 扰动。
- 引号内候选和内容词被刻意保留，所以测试的是 unquoted scaffold 的增量贡献，不是移除全部文本信息。
- AR 分数衡量内部 activation round-trip，不等价于人类命题级忠实度。

## 6. 文件绑定

- row activation SHA-256（x[75] float32 bytes）：`014d980b8b8dff6099a80fa0bf521bd892a56823875c0e1e32a67689266b182c`
- source activation cohort SHA-256：`eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66`
- original text SHA-256：`38a853ac6a50f2dfa2d696ead4592e38f4b99d33dd95536c514757adf12635a4`
- masked text SHA-256：`1a10a278b70655832eecaa7f90882f63b72021762e9eece0b95f0f65c2ae5528`
- protocol SHA-256：`d9880ea86c288dcae9ad6364dcff1cacbd66eb2ee2cbc6509dccefd3a27ed1c9`
