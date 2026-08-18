# NLA vs SAE 对比实验 — 远程操作工具包

## ✅ 主线、B2、B6+B4、C1 protocol pilot、N1（文本通道拆解）与 N2（因果 patch）已完成（更新至 2026-07-30 晚）

> **接手先读这三条,它们推翻/限定了下面很多旧数字**：
> 1. **F12**：因果 patch 口径下 NLA 的 KL-recovered **0.757** 与 SAE-big **0.771 打平**
>    （内容 token 上 SAE-big 更好），"NLA 误差不到 SAE 一半"只在方向 cos 上成立。
> 2. **F11**：解释文本里承载重建的是第 3 段那串"下一 token 候选"（只留它保留 96%，
>    抹掉它剩 21%）；第三方等义改写保留 **97.3%**（**C7 通过**）；实体替换只掉 10%。
>    **NLA 的"解释"实际是被语言化的下一 token 分布。**
> 3. **F13**：E1–E7 队列 **13/40 是 chat 模板/空白 token**（`--min-position 50` 从未生效）。
>    所有 n=40 结论都要带这个声明，优先看 content-token 分层。

主线与 centered 检索评测均已跑通。当前结果在本地 `results\`。**2026-07-30 用户
明确要求 AutoDL 保持开启,不得自动或手动关机,直至用户另行改变指令。**

核心结论（n=40，Gemma-3-12B-IT L32 resid_post，方向指标）：
- **NLA**：mean cos **0.996**（mse_nrm 0.008）
- **SAE w16k L0≈15**：cos 0.9925（mse_nrm 0.0149），FVE 0.61
- **SAE w16k L0≈75**：cos 0.9936（mse_nrm 0.0127），FVE 0.67
- NLA 方向误差约为 SAE 的一半（**仅方向 cos 口径；因果口径下打平，见 F12**）；
  `<end_of_turn>` 等结构 token 上 SAE 反超（L0=5 即可）——**F13 显示这类位置占 13/40，
  是取样缺陷而非"失败模式互补"**
- 注意：未去均值的 cos 被共享方向抬高（SAE FVE 仅 0.61 说明真实方差解释低得多）

Pilot 实验（06/07/08/09，`results\` 下 wdec/resid/injection_pilot.json + centered_rescore.json，
完整结论见 **`results\POSSBILITY.md`**）：
- **去均值后的真实 head-to-head**：NLA **0.859** vs SAE-s 0.658 vs SAE-b 0.725——NLA 优势放大
- **泛型文本基线**：未去均值 0.975（白拿），去均值 0.041——一切未去均值 cos 都被均值方向抬高
- **w_dec 方向可读性大部分是均值混淆**：top 特征未去均值 median|cos|=0.80 → 去均值 0.19；
  "负号往返"= 特征与均值对齐的符号（±0.9），不是 AV 读出的极性
- **SAE 残差（暗物质）不可读**：去均值 cos_rr=−0.03；注入 2×残差强度的已知信号也检不出
  （检测曲线平坦 0.07→0.09）——AV 把离流形输入 collapse 成同一句模板
- 解释会编造实体（3/40）且 cos 完全不受影响（0.9962=0.9962）——高 cos ≠ 解释忠实
- **B2 centered 40-way 检索**：NLA / SAE-small / SAE-big 的 Top-1 =
  **92.5% / 95.0% / 95.0%**,Top-5 均 100%;预设的“NLA Top-k 胜出”没有成立。
  但 mean margin = **0.3195 / 0.2085 / 0.2531**,NLA 的正确配对间隔在 5 个文档上均更大。
  完整矩阵、置换零假设和逐行错误见 `results\retrieval_eval.json`。
- 原先称“泛型文本地板”的 `pred_resid` 实际是 40 条逐样本残差解释;其 B2 Top-1 仍有
  17.5%,故应改称 **residual-text weak-signal control**,不能当固定 generic text。
- **B6+B4 因子实验**：24 个冻结 semantic feature 的 heldout-valid yield 为
  **14/24**（domain 6/15，language 8/9）。Greedy `q+ / r− / polarity` median =
  **0.114 / 0.031 / 0.071**，sign accuracy 75%，45-way feature Top-1 33.3%；
  Gaussian 对照为 0.005 / 0.001 / 0.004、Top-1 0%。这支持正方向存在可重复、可检索
  的内部信号，但不支持强双向语义极性。
- **内部一致性不等于标签忠实**：heldout AUC 与 `q+` Spearman≈−0.015；事后
  surface audit 仅 5/24 strict match、9/24 明显错配。C1 应作为外部效度/反例审计
  的论文核心，C2 只能采用“activation/context gate + NLA/AR 辅助质检”的 v2 形态。
- 非零 carrier 上 ablate / insert / amplify median centered cos =
  **0.210 / 0.097 / 0.042**。向量只经过 AV/AR，未注回 Gemma 后续层，**不是 steering**。
  详见 `results\POSSBILITY.md` F9 与 `results\b6b4_factorial_analysis.md`。
- **C1 protocol pilot**：冻结粗轴正确−错配 mean delta **0.0155**，16/24 为正；
  sign-flip mean `p=.00393`，但 sign test `p=.0758`，是小而非多数稳健的信号。
  axis delta 预测 heldout-valid 的事后 AUC **0.764**，绝对 q 则不预测。
- 细 train reference 的 AR 区分很强，但 heldout-invalid 反而更强，主要像训练词汇/主题指纹。
  NLA original 24-way text→feature Top-1 **62.5%**，blind context judge 却只有 2/24 通过。
- 同源 base judge 校准失败（generic 110/192 被判正；pooled q→judge AUC 0.410）。
  NLA paraphrase 大幅降分提示 form/private-code 风险，但只保留原文约 60% 字符，尚未排除压缩混杂。
  完整结果、独立复算与资源日志见 `results\c1_pilot_*`，结论见 `POSSBILITY.md` F10。
- **N1 文本通道拆解（28/29/31，`results\c7b3_*`）**：40 行 × 11 冻结条件，AR 重打分 17 秒。
  原文复算 0.8593 与 E5 逐位一致。保留率：只留第 3 段 **96.1%** / 抹掉其引号内候选串
  **20.8%** / 抹掉第 2 段逐字引用 99.4% / 只留体裁段 28.3% / 词序打乱 66.9% /
  **实体替换 89.7%** / **第三方等义改写 97.3%**；固定泛型文本地板 **−0.005（≈0，
  原估 0.041 偏高）**。
- **N2 因果 patch（30/32，`results\causal_patch_v1.json`）**：identity KL 0.0000、
  provenance cos 1.000000、gauss 31.07、zero 16.34（对照干净）。KL-recovered
  NLA/SAE-b/SAE-s = **0.757 / 0.771 / 0.713**；NLA−SAE-big 的配对 CI **[−0.21, 1.98] 跨零**。
  centered cos 跨方法 ρ(cos,KL)=−0.402 但**方法内不一致**（NLA +0.188）→ cos 能排方法、
  不能排样本。CE 版 loss-recovered **作废**（窗口被模板 token 支配，干净 CE 21 nats）。
- **F14（纯统计勘误）**：`q+` 是**双峰**（10 个 ≥0.362 且 9 个方向检索 rank=1，13 个 <0.15，
  median 0.114 落在 0.117 的断层里）；`heldout AUC vs q+ 的 ρ=−0.015` 是 Simpson 悖论
  加并列伪值（domain 层 +0.40；8/24 特征在 test 上完全不激活致 AUC=0.5 为伪值，
  且这 8 个含 q+ 最高的三个）→ **这份数据没能力检验该关联**，也因此 C2-v2 的
  heldout gate 放在最前是错的。复算见 `results\local_recheck_*_opus.json` 与
  `results\REVIEW_OPUS_2026-07-30.md`。

已踩过的坑（勿重复）：
- `HF_ENDPOINT` 必须在 **shell 层 export** 再启 python（hub 导入时读取）
- base 走 hf-mirror **带 token 可下 gated**（20MB/s）；network_turbo 官方源 <1MB/s
- SAE 别用 snapshot_download（8TB 仓库列举会挂），用 `hf_hub_download` 按路径取 4 个文件
- 远程 pkill 的模式若出现在 ssh 命令明文里会自杀，模式写成 `'[r]un_...'` 且命令里别再出现明文
- transformers 5.x：`apply_chat_template(tokenize=True)` 返回 BatchEncoding 不是 list——
  02 与服务器上 `nla_repo/nla_inference.py`（187/403 两处）都已打补丁
- **AutoDL 的 `shutdown` 可能忽略 `+N` 延时参数并立即断电**；当前用户明确要求保持
  实例开启,所有 runner 都不得调用 `shutdown` 或设置自动关机链
- 本地 Clash TUN 会劫持 SSH（fake-IP 198.18.x），先关代理或加 `*.seetacloud.com` 直连；
  被劫持时连 `Resolve-DnsName -Server 223.5.5.5` 也返回 198.18.x，可用此判定
- **PowerShell 会吞掉传给 ssh 的反斜杠**：`ssh autodl 'sed -i "s/\r$//" x.py'` 实际执行的是
  `s/r$//`，把每行行尾的字母 `r` 删掉（曾一次改坏 4 个刚上传的脚本，报错表现为
  `cannot import name 'Counte'` / `AutoTokenize`）。**上传的脚本本来就是 LF，先用 `file`
  确认，不要盲目 sed**
- PowerShell 引号：**外层双引号 + 内层单引号**最可靠；外层单引号里出现 `|` 或 `\` 都可能被
  重写。复杂命令一律写成 `.sh` 脚本 scp 上去执行
- 远程 `pkill -f 'pat'` 若 `pat` 出现在本次 ssh 命令明文里会杀掉自己的会话（用 `N1_EXI[T]`
  这类字符类写法规避）——本轮又踩了一次
- `02_extract_activations.py` 的 `--min-position 50` 对短 prompt 静默失效并回退到"取序列
  后半段"，导致 13/40 落在 chat 模板尾部（F13）。**新抽激活时必须打印实际 position 分布
  与 token 文本自查**，并优先用长自然文本而非 5 条短 instruct prompt
- 中心化投影的 `perp()` 若写成 `a - (a @ m) * m`，对 2D 数组会静默广播出错，
  必须按维度分支用 `np.outer`（29 号脚本踩过）
- patch 回模型测 CE 时，评测窗口若含 `<end_of_turn>` / `<start_of_turn>`，
  这些 token 的干净 CE 高达 36–53 nats，会让任何 loss-recovered 归一化失去意义——
  **要么按 token 类型分层，要么只报 KL**


在 AutoDL（seetacloud bjb1）服务器上，对 **Gemma-3-12B-IT 第 32 层（resid_post）** 做
**NLA（自然语言自编码器）** 与 **Gemma Scope 2 SAE** 的往返重建对比实验。

- NLA：`激活向量 → AV 生成自然语言解释 → AR 重建向量`，看方向 cos / `mse_nrm=2(1-cos)`
- SAE：`激活向量 → JumpReLU 编码 → 解码重建`，看同样的方向 cos，外加 SAE 原生的 FVE / L0
- 两条线读取**同一份**第 32 层激活（抽取点 `model.layers[32].output`，NLA 与 SAE 完全对齐）

> 当前现状（2026-07-30）：A800-SXM4-80GB 实例已开启，四类模型均已下载，主线、pilot、
> B2、B6+B4 与 C1 protocol pilot 资产均在数据盘。下文下载/部署步骤保留作历史复现说明，
> 不代表当前待办。

## 一键连接

```powershell
.\connect.ps1                      # 进入交互式 shell
.\connect.ps1 "df -h /root/autodl-tmp"   # 跑一条命令就返回
```

也可直接 `ssh autodl`。当前主机、端口和密钥只维护在 `~/.ssh/config` 的 `Host autodl`
条目中；`connect.ps1`、`sync.ps1` 和 `remote.py` 都读取该条目，避免克隆实例后端口漂移。
推送脚本 / 拉取结果：

```powershell
.\sync.ps1 push     # server\*  ->  /root/autodl-tmp/nla_compare/
.\sync.ps1 pull     # /root/autodl-tmp/results/  ->  results\
```

Python 辅助入口：`python remote.py "<命令>"`（需本地装 paramiko，同样使用 SSH config
与密钥；仓库不保存密码）。

## 服务器现状（已探明）

| 项 | 值 |
|---|---|
| 系统盘 `/` | 30 GB（用 1 GB） |
| **数据盘 `/root/autodl-tmp`** | **150 GB（已用约 70 GB，可用约 81 GB）** |
| GPU | **NVIDIA A800-SXM4-80GB（当前已开启）** |
| CPU / 内存 | 144 核 / 1 TB |
| OS / CUDA | Ubuntu 22.04 + CUDA 12.4 |
| Python | miniconda3 base，3.12 |
| torch | 2.5.1+cu124（保留不动） |
| transformers | 5.12.1（原生支持 Gemma-3 ✓） |
| 加速 | `source /etc/network_turbo`（仅用于 github/hf；**装 pip 包时不要开**，会走通的 aliyun 源更快） |

## 下载清单与磁盘账

| 文件 | 大小 | 状态 |
|---|---|---|
| base `google/gemma-3-12b-it` | 24.4 GB | ✅ 已下载 |
| AV `kitft/nla-gemma3-12b-L32-av` | 23.6 GB | ✅ 已下载 |
| AR `kitft/nla-gemma3-12b-L32-ar` | 16.9 GB | ✅ 已下载 |
| SAE `gemma-scope-2-12b-it` L32 resid_post w16k（small+big） | ~1 GB | ✅ 已下载所需文件（整库约 8 TB，**切勿下整库**） |
| **合计** | **~66 GB** | ✅ 已落盘；当前 150 GB 数据盘可用约 81 GB |

## 流程（环境已就绪后）

脚本都在服务器 `/root/autodl-tmp/nla_compare/`，用 `/root/miniconda3/bin/python` 跑。

```bash
# 0. 环境（已执行核心部分；sglang 非必需，见下）
bash 00_setup_env.sh core

# 1. 下载（先扩容数据盘！开放模型走镜像，base 需 token 走官方源）
python 01_download.py --only av ar sae
HF_TOKEN=hf_xxx python 01_download.py --only base

# 2. 抽取第 32 层激活（base 模型）—— 两条线共用
python 02_extract_activations.py \
    --base-model /root/autodl-tmp/models/gemma-3-12b-it \
    --out /root/autodl-tmp/activations/acts_L32.parquet

# 3. NLA 线（默认本地 transformers，无需 sglang）
python 03_run_nla.py \
    --av /root/autodl-tmp/models/nla-gemma3-12b-L32-av \
    --ar /root/autodl-tmp/models/nla-gemma3-12b-L32-ar \
    --activations /root/autodl-tmp/activations/acts_L32.parquet \
    --out /root/autodl-tmp/results/nla_results.json

# 4. SAE 线（自包含 JumpReLU，无需 sae-lens）
python 04_run_sae.py \
    --sae /root/autodl-tmp/models/gemma-scope-2-12b-it/resid_post_all/layer_32_width_16k_l0_small \
    --activations /root/autodl-tmp/activations/acts_L32.parquet \
    --out /root/autodl-tmp/results/sae_results.json

# 5. 合并对比 -> comparison.json + comparison.md
python 05_compare.py \
    --nla /root/autodl-tmp/results/nla_results.json \
    --sae /root/autodl-tmp/results/sae_results.json \
    --out /root/autodl-tmp/results/comparison
```

NLA 与 SAE 可**分开跑**（3、4 互不依赖；5 有谁的结果就比谁）。

### 关于 sglang（可选）

NLA 的 AV 默认用 **本地 transformers** 生成（`03_run_nla.py` 不带 `--sglang-url`），
跑在现有 torch 2.5.1 上，省去依赖折腾。只有当你要**大规模扫特征字典**追求吞吐时，
才需要 sglang：`bash 00_setup_env.sh`（完整）会装 `sglang[all]`，然后
`bash launch_av_server.sh` 起服务，再给 03 加 `--sglang-url http://localhost:30000`。
注意：sglang 0.5.6+ 会把 torch 升到 2.8/cu128，**上卡后需确认驱动支持 CUDA 12.8**，否则保持本地路径即可。

## 对比口径（为什么公平）

- **方向保真**（cos、`mse_nrm=2(1-cos)`）是唯一可直接 PK 的指标：NLA 往返天生只比方向
  （两端都 L2 归一化），所以 SAE 也按同一 `√d` 归一化后比方向。
- **FVE / L0** 是 SAE 的主场指标（含幅度的重建质量、稀疏度），NLA 没有对应物，单列展示。
- **自然语言解释**是 NLA 的独有产物，单列展示。
- 抽取点严格对齐：`model.layers[32].output`（SAE config 的 `hf_hook_point` 同点），无 off-by-one。

## 当前研究队列

1. ✅ B6+B4 与 C1 protocol pilot 已完成并归档（产物与哈希见 `continue.md`）。
2. 优先做 **C1-confirmatory**：新冻结 60–100 feature、≥15–20 个 label cluster，
   先冻结盲人工真值、长度匹配改写和统计方案，再查看 NLA/AR 输出。
3. 只有 matched AR delta 能预测盲评、heldout activation 与人类/因果效度后，
   才把 C2 扩成自动标签流水线，并另行设计真正的 downstream
   steering 实验。当前 carrier-conditioned AV/AR readout 不等于 steering。
4. AutoDL 按用户当前指令保持开启；不得设置自动关机链。
