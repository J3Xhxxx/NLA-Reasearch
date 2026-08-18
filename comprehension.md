# natural_language_autoencoder 项目理解

> 阅读范围：`d:\Projects\natural_language_autoencoder`。本文重点记录当前仓库的整体研究脉络，以及 `server/run_pipeline.sh` 的实际调用链。项目代码、结果和交接文档中同时存在旧主线与 N1–N6 后续实验，因此必须区分“基线 pipeline”与“后续实验流水线”。

## 1. 项目目标与核心对象

这个项目研究 Gemma-3-12B-IT 第 32 层 `resid_post` 隐状态能否通过自然语言进行压缩/重建，并将 NLA（Natural Language Autoencoder）与 SAE（Sparse Autoencoder）放在同一批激活上比较。

核心数据流是：

```text
Gemma-3 base model
        │
        └─ 第 32 层 block output / resid_post 激活向量 x
             ├─ NLA：x ──AV──> explanation text ──AR──> x_hat
             └─ SAE：x ──JumpReLU encode/decode──> x_hat
                                      │
                                      └─ NLA/SAE 指标与逐 token 结果
```

- 激活抽取点固定为 `model.layers[32]` 的输出，模型维度为 3840；NLA 与 Gemma Scope 2 SAE 因而接收同一表示空间的向量。
- NLA 由两个 checkpoint 组成：AV（activation verbalizer）把向量注入固定 prompt 后生成解释文本，AR（activation reconstructor/critic）从文本恢复向量并评分。
- SAE 使用 Gemma Scope 2 的 JumpReLU 参数直接编码、解码，不依赖 `sae-lens`。
- 由于 NLA 的 AR 训练/评分主要关注方向，主线用 cosine 及 `mse_nrm = 2(1-cos)` 做公平的方向比较；SAE 另外报告原生的 FVE、L0 和 raw MSE；自然语言解释是 NLA 独有输出。

## 2. 目录和实验演进

- `server/00_setup_env.sh`：在远程 AutoDL 环境安装核心依赖，可选安装 SGLang；把 Hugging Face cache、模型、激活和结果放到 `/root/autodl-tmp` 数据盘。
- `server/01_download.py`：按需下载 base、AV、AR 和指定层/宽度的 SAE。base Gemma 是 gated model，需要 token；开放模型可走镜像。
- `server/02_extract_activations.py` 至 `05_compare.py`：早期/基线 NLA-vs-SAE 主线，正是 `run_pipeline.sh` 调用的五个 Python 脚本。
- `server/06_pilot_wdec.py` 至 `14_analyze_factorial_results.py`：对 NLA 文本通道、SAE 残差、注入及 B6+B4 因子实验的 pilot/分析。
- `server/15_build_c1_pilot.py` 至 `24_analyze_c1_confirmatory.py`：C1 pilot 和 confirmatory 相关的冻结、生成、运行、验证、分析脚本；`18_validate_c1_pilot.py` 是独立结构/数值审计，而不是基线 pipeline 的一步。
- `server/25_*` 至 `38_*`：N1/N2/N3 等文本变体、因果 patch、语料与分析实验。
- `server/39_*` 至 `48_*`：N4/N5 的真实内容重建、因果审计、held-out 以及选择性路由相关实验。
- `server/49_*` 至 `56_*`：N6+ 的 cohort 冻结、激活抽取、AV 生成、变体冻结、重建、因果 candidate mass、分析和独立审计。
- `results/`：各阶段结果、日志、checkpoint、向量归档、哈希和审计报告。
- `README.md`、`continue.md`、`RECOVERY_2026-08-03.md`：实验背景、历史结果和交接信息。最新事实以 `RECOVERY_2026-08-03.md`、`results/N6_FINAL_ANALYSIS_2026-08-03.md` 以及冻结 JSON/独立审计为准；README 和 continue 中部分运行状态是历史记录。

## 3. `run_pipeline.sh` 的实际运行方式

入口文件：`server/run_pipeline.sh`。它是一个面向远程 Linux/AutoDL 机器的 Bash 编排脚本，不是本地 Windows 入口。它假设以下路径和环境已经存在：

```text
Python       /root/miniconda3/bin/python
代码目录     /root/autodl-tmp/nla_compare
模型目录     /root/autodl-tmp/models
激活文件     /root/autodl-tmp/activations/acts_L32.parquet
结果目录     /root/autodl-tmp/results
HF cache     /root/autodl-tmp/hf
```

### 3.1 初始化与失败处理

1. 通过 `exec > /root/autodl-tmp/pipeline.log 2>&1` 把标准输出和错误输出全部写入远程日志。
2. `set -x` 打开 Bash 命令回显，方便按日志重建执行顺序。
3. 设置 Python、代码、模型、激活和结果路径，切换到代码目录，并设置 `HF_HOME`。
4. 定义 `finish(status)`：打印带退出码和时间的结束标记，执行 `sync`，再以该状态退出。
5. 用 `pgrep -f '[r]un_downloads'` 每 60 秒轮询下载脚本；下载进程仍存在时不开始 GPU 计算。
6. 下载进程结束后，在一个 Python heredoc 中检查 base 模型的 `model.safetensors.index.json` 和 index 中列出的每一个 shard。没有 index 或有 shard 缺失时，打印失败原因并以状态 1 结束。
7. 创建结果目录和激活目录。

这里的等待只是“等下载进程结束”，不是重新下载；下载由 `run_downloads.sh`/`01_download.py` 负责，pipeline 只做完整性门禁。

### 3.2 共享激活抽取

执行：

```text
02_extract_activations.py \
  --base-model /root/autodl-tmp/models/gemma-3-12b-it \
  --out /root/autodl-tmp/activations/acts_L32.parquet
```

这个步骤加载 base Gemma，默认使用内置的 5 个 prompt、chat template、第 32 层、默认 `min-position=50` 和每个 prompt 最多 8 个位置。它注册 forward hook，抓取第 32 层 block output，将选定 token 位置的向量和 `token/position/doc_id/prompt/norm` 元数据写成 Parquet。

这份 Parquet 是后续两条比较线的唯一共享输入：它避免 NLA 和 SAE 因输入文本、层位置或采样位置不同而失去可比性。

脚本失败时调用 `finish 2`，后续步骤不执行。

### 3.3 NLA 线：向量 → 文本 → 向量

执行：

```text
03_run_nla.py \
  --av /root/autodl-tmp/models/nla-gemma3-12b-L32-av \
  --ar /root/autodl-tmp/models/nla-gemma3-12b-L32-ar \
  --activations /root/autodl-tmp/activations/acts_L32.parquet \
  --out /root/autodl-tmp/results/nla_results.json
```

主要步骤：

1. 读取 Parquet 中的 activation vectors 和 token 元数据。
2. 默认走 `AVLocal`，用普通 Transformers 直接加载 AV checkpoint；只有传 `--sglang-url` 才走 SGLang 服务。
3. AVLocal 调用项目的 `nla_inference.py` 中 `NLAClient._build_embeds`，按训练时的 sidecar 配置、注入 token、邻接 token 和 embedding scale 构造输入 embedding。这一点保证推理注入数学与训练 recipe 一致。
4. AV 生成文本，并用 `EXPLANATION_RE` 提取 `<explanation>...</explanation>` 内容。
5. `NLACritic` 加载 AR checkpoint，调用 `critic.score(text, v)`，得到重建误差和 cosine。
6. 对每个向量写一行：原始 norm、解释文本、`nla_mse_nrm`、`nla_cos` 以及位置元数据；最后写 `nla_results.json` 和总体 summary。

这一步失败时以状态 3 结束。它是主线中最重的文本生成步骤之一，会加载完整 AV 与 AR 模型并使用 GPU。

### 3.4 SAE 线：JumpReLU 编码/解码

第一次执行：

```text
04_run_sae.py \
  --sae /root/autodl-tmp/models/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_small \
  --activations /root/autodl-tmp/activations/acts_L32.parquet \
  --out /root/autodl-tmp/results/sae_results.json
```

`04_run_sae.py` 直接加载 `params.safetensors` 中的 `w_enc/b_enc/w_dec/b_dec/threshold`，计算：

```text
pre   = x @ w_enc + b_enc
acts  = relu(pre) * (pre > threshold)
x_hat  = acts @ w_dec + b_dec
```

随后针对每个向量计算方向 cosine、方向 MSE、激活特征数 L0 和 raw MSE，并在数据集层面计算 FVE，写入 `sae_results.json`。该步骤失败时以状态 4 结束。

随后脚本用同一个 `04_run_sae.py` 再跑一次 `l0_big` SAE，输出 `sae_results_big.json`。这次命令没有 `|| finish 4`，因此 big 是 bonus：失败不会阻止后续 small-SAE comparison；但由于 shell 没有 `set -e`，失败后的行为仍由后续命令继续执行这一事实决定。

需要注意：当前 `05_compare.py` 只接收 small SAE 文件，所以 `sae_results_big.json` 只是额外产物，不会进入本次 `comparison.json/.md`。

### 3.5 合并与最终退出

执行：

```text
05_compare.py \
  --nla /root/autodl-tmp/results/nla_results.json \
  --sae /root/autodl-tmp/results/sae_results.json \
  --out /root/autodl-tmp/results/comparison
```

`05_compare.py` 以 `(doc_id, position)` 作为 NLA/SAE 对齐键，把两边的逐 token 行合并，输出：

- `comparison.json`：NLA/SAE summary 与合并后的逐位置结构化结果；
- `comparison.md`：方向指标摘要、逐 token 表格以及最多前 8 行 NLA explanation 示例。

合并失败时以状态 5 结束；成功后列出结果目录，打印 `PIPELINE_COMPLETE`，调用 `finish 0` 正常退出。

## 4. `run_pipeline.sh` 直接涉及的 Python 文件

### 4.1 直接作为独立进程执行的文件

| 文件 | 作用 | 输入 | 输出 |
|---|---|---|---|
| heredoc 内联 Python | 校验 base safetensors index 与全部 shard 是否存在；不是项目中的独立 `.py` 文件 | `/root/autodl-tmp/models/gemma-3-12b-it/model.safetensors.index.json` | 成功返回 0；缺失则退出 |
| `server/02_extract_activations.py` | 用 base Gemma 抽取第 32 层 `resid_post` 向量，保存共享 Parquet | base checkpoint、可选 prompt 文件 | `activations/acts_L32.parquet` |
| `server/03_run_nla.py` | AV 生成 explanation，AR 将 explanation 重建回向量并评分 | AV/AR checkpoint、激活 Parquet | `results/nla_results.json` |
| `server/04_run_sae.py` | 用 JumpReLU SAE 对每个激活编码/解码并计算指标 | SAE 参数、激活 Parquet | `results/sae_results.json` 或 `sae_results_big.json` |
| `server/05_compare.py` | 以 `doc_id + position` 对齐 NLA 和 SAE，生成总表 | 两个结果 JSON | `results/comparison.json`、`comparison.md` |

### 4.2 NLA 线的间接 Python 依赖

`03_run_nla.py` 通过 `NLA_REPO`（默认 `/root/autodl-tmp/nla_repo`）导入 `nla_inference.py` 中的：

- `NLAClient`：加载 NLA sidecar、tokenizer 和 embedding，并构造 activation injection；AV 服务模式还负责请求 SGLang。
- `NLACritic`：加载 AR 模型，从解释文本恢复/评分 activation。
- `EXPLANATION_RE`：提取解释标签内的文本。

本地项目当前没有名为 `server/nla_inference.py` 的源文件；`results/nla_inference.remote_20260803.py` 是一个远端推理文件快照/归档，不能据此断言当前 pipeline 会自动从 `results` 导入。要在远端成功运行 `03_run_nla.py`，`/root/autodl-tmp/nla_repo/nla_inference.py` 必须实际存在，或者通过 `NLA_REPO` 指向包含该模块的目录。

### 4.3 不属于 `run_pipeline.sh` 调用链的相邻 Python 文件

`01_download.py` 不由 `run_pipeline.sh` 调用；它负责下载模型。`run_downloads.sh` 通常负责等待/重试 base 下载，pipeline 通过进程检测等待它完成。

`06`–`56` 系列脚本也不由当前这个旧基线入口自动调用。它们属于 pilots、C1、N3、N4、N5、N6 后续实验，各自由单独的 `run_*.sh`、remote launcher 或 supervisor 编排。尤其 N6+ 已经是独立的冻结 cohort、因果 patch 和审计链，不应理解为 `run_pipeline.sh` 的隐式后续步骤。

## 5. 产物关系图

```text
models/gemma-3-12b-it
        │
        └─ 02_extract_activations.py
             └─ activations/acts_L32.parquet
                    ├─ 03_run_nla.py
                    │    └─ results/nla_results.json
                    └─ 04_run_sae.py (small)
                         └─ results/sae_results.json
                    └─ 04_run_sae.py (big, bonus)
                         └─ results/sae_results_big.json

nla_results.json + sae_results.json
        └─ 05_compare.py
             ├─ results/comparison.json
             └─ results/comparison.md
```

## 6. 运行时的重要边界和风险

1. **平台边界**：脚本是 Bash + Linux 绝对路径，不能直接在当前 Windows 工作站用 PowerShell 执行；通常通过 SSH/AutoDL 远程运行，再用 `sync.ps1` 上传 `server/*` 或拉回 `results/*`。
2. **下载门禁不等于下载**：`run_pipeline.sh` 不会修复缺失模型；base shard 缺失时直接退出，须先运行下载流程。
3. **下载竞争条件**：它只用 `pgrep` 判断 `run_downloads` 是否还在。若下载脚本异常退出但文件不完整，后面的 index/shard 检查负责拦截；若下载脚本进程名不匹配，等待保护可能失效。
4. **big SAE 是非阻断 bonus**：small SAE 才是比较输入；big 结果虽然产生，但当前 pipeline 不合并它。
5. **激活采样陷阱**：`02_extract_activations.py` 的短 prompt 可能触发后半序列 fallback，历史记录指出这会让结构/chat token 混入样本。因此阅读旧 `pipeline.log` 的 40 行结果时，不能把它自动当成高质量 content-token 评测。
6. **指标边界**：高 cosine 只说明重建方向接近；不能单独推出自然语言解释在人类语义上忠实，也不能推出因果安全、steering 能力或 NLA 全面优于 SAE-big。
7. **源码与历史日志可能漂移**：当前 `run_pipeline.sh` 的 `finish()` 只 `sync` 后退出；历史 `results/pipeline.log` 的末尾还出现 `sleep 600` 和 `shutdown`，说明日志对应过旧版本/旧 runner。分析当前行为应以当前 `server/run_pipeline.sh` 为准，不应把旧日志中的关机动作当作现行源码行为。
8. **项目最新状态**：N6+ 已完成独立审计并形成正式结论；最新结果不能通过重新运行这个早期 pipeline 得到。`run_pipeline.sh` 只能复现早期 NLA-vs-SAE 基线。

## 7. 一句话总结

`run_pipeline.sh` 是一个“等待 base 下载完成 → 抽取共享 L32 激活 → 分别跑 NLA 和 SAE → 用位置键合并比较 → 记录结果并退出”的远程基线编排器；它直接运行 `02_extract_activations.py`、`03_run_nla.py`、`04_run_sae.py`、`05_compare.py`，并间接依赖 NLA repo 中的 `nla_inference.py`，但不负责下载，也不负责 N1–N6 后续实验。
