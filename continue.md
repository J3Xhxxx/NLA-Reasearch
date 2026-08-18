# Continue — NLA 项目接手总览

> 更新：2026-07-30  
> 目的：给下一位接手者一份经过本地核对、去除明文凭据的宏观状态说明。  
> 2026-07-26 已完成 B2 检索评测与 B6+B4 因子语料扩展实验。用户明确要求
> **保持 AutoDL 开机，不要自动关机**，直至用户另行改变该指令。
> 2026-07-30 C1 confirmatory 在任何 v3 文本生成或 activation 提取之前被
> 冻结的场景锚点审计截停；服务器仍保持开机。

## 1. 先认清三个目录

本项目实际由三部分组成，不能只看其中一个：

1. `D:\Projects\.claude\`
   - 只有 `settings.local.json`，本质是 Claude 会话积累的本地命令权限白名单。
   - 它记录了旧 AutoDL 端口和历史上传/下载命令，不是项目设计文档，也不应当被当作当前基础设施真相来源。

2. `D:\Projects\natural_language_autoencoder\`（单数）
   - Jason 与 Claude Fable 5 完成的 **NLA vs SAE 实验工作区**。
   - 包含 `server/00`–`24` 实验/分析脚本、远端 runner、本地结果、日志，以及前任的 `Conclude.md`、`Prompt.md`、`THOUGHT.md`。
   - 这里不是 Git 仓库，但它是本轮研究结论与接手工作的主要现场。

3. `D:\Projects\nla-from-autodl\natural_language_autoencoders\`（复数）
   - `kitft/natural_language_autoencoders` 的完整训练仓库副本，带 `.git`。
   - 包含 NLA 数据生成、AV/AR SFT、GRPO RL、Miles/SGLang 集成、推理客户端、checkpoint 转换和本地探针 demo。
   - 实验工作区调用的是这里公开的 NLA 思路与 `nla_inference.py` 接口，但服务器上另有一份打过热修补的 `nla_repo/` 副本。

建议的文档阅读顺序：本文件 → `natural_language_autoencoder/Conclude.md` → `results/POSSBILITY.md` → `Prompt.md` → `THOUGHT.md` → 核心仓库 `CLAUDE.md`、`README.md`、`docs/design.md`、`docs/inference.md`。

## 2. 项目在研究什么

NLA（Natural Language Autoencoder）由两个模型组成：

- **AV / actor（Activation Verbalizer）**：把残差流激活向量缩放后替换固定 prompt 中一个 token 的 embedding，再自回归生成自然语言解释，即 `vector → text`。
- **AR / critic（Activation Reconstructor）**：截断到目标层的语言模型加一个 `Linear(d,d)` value head，从解释文本最后一个 token 的隐状态重建激活，即 `text → vector`。

训练仓库的主链路是：

1. Stage 0 从语料和基座模型抽取原始激活；
2. Stage 1 按 `doc_id` 做文档级 AV-SFT / AR-SFT / RL 切分；
3. Stage 2 调用解释模型生成训练解释；
4. Stage 3 构造三个 parquet 与 `nla_meta.yaml` sidecar；
5. AR SFT、AV SFT，最后以 `-mse_nrm` 为奖励做 AV 的 GRPO，并继续监督训练 AR。

工程上依赖 Miles 做分布式训练、SGLang 做 `input_embeds` rollout。最重要的契约是 sidecar：token ID、相邻 token、prompt 模板、`d_model`、`injection_scale`、`mse_scale` 都必须从它读取并在运行时校验，不能硬编码。数据生成永远保存原始向量；归一化只在注入和损失阶段发生。

本地实验没有重训 NLA，而是使用公开的 Gemma-3-12B-IT L32 AV/AR 检查点，与 Gemma Scope 2 的 L32 resid_post、width 16k JumpReLU SAE 做同输入对比，并继续探索 NLA 能否辅助 SAE 审计。

## 3. 已完成实验与数据口径

E1–E7 对象：Gemma-3-12B-IT 第 32 层 `resid_post`，5 条英文 instruct prompt，每条抽
8 个位置，共 40 个 `d=3840` 原始激活。E8 另用 24 个因子文档、1365 个激活与 45 个
冻结方向；两套样本不能混写。

已完成并有结果文件的实验：

| 编号 | 内容 | 主要结果 |
|---|---|---|
| E1 | 主线 NLA vs SAE-small / SAE-big | `nla_results.json`、`sae_results*.json`、`comparison.*` |
| E2 | AV 读取 SAE `w_dec` 方向 | `wdec_pilot.json` |
| E3 | AV 读取 SAE 重建残差 | `resid_pilot.json` |
| E4 | 在残差中注入已知特征方向 | `injection_pilot.json` |
| E5 | 投影掉数据集均值方向后重打分 | `centered_rescore.json`、`recon_vectors.npz` |
| E6 | 解释模板化、编造与特异性统计 | 数字写入 `results/POSSBILITY.md`，原统计脚本未留存 |
| E7 / B2 | centered 40-way 检索、MRR、margin、置换零假设 | `retrieval_eval.json`、`b2_retrieval.log` |
| E8 / B6+B4 | 4 领域 × 3 语言 × train/test;冻结特征、`±w_dec`、carrier-conditioned readout | `b6b4_factorial_result.json`、recon NPZ、analysis JSON/MD、surface audit、checkpoint/log |

本地核验结果：

- `pipeline.log`、`pilot.log`、`injection.log`、`rescore.log` 均存在 `*_COMPLETE` 和 `EXIT (status=0)`。
- 结果目录中的 JSON 均可解析。
- `recon_vectors.npz` 可正常读取压缩成员，包含 `x`、`m_hat`、`pred_full`、`pred_resid`、两档 SAE 重建、残差和 4 个特征方向；主体数组形状均与 `40×3840` 对齐。
- B2 已由本地与远端独立复算且摘要、逐行 rank、置换结果一致。
- B6+B4 新抽取 `1365×3840` 激活;冻结 24 个新 semantic feature(均通过 discovery gate),
  heldout-valid 14 个。正式批次完成 590/590 job、590 个唯一 checkpoint key,日志含
  `B6B4_FACTORIAL_COMPLETE` 与 `EXIT(status=0)`;JSON/NPZ 均已本地/远端哈希核对。

## 4. 当前可信结论

本项目最重要的认识是：**未去均值的余弦相似度严重虚高，必须把数据集均值方向从预测与目标两侧都投影掉，再报告余弦和泛型文本地板。** AR 预测向量与原激活尺度不同，因此这里只能做方向投影，不能做仿射减均值。

关键数字：

| 指标 | 未去均值 | 去均值（可信主口径） |
|---|---:|---:|
| NLA 重建真实激活 | 0.9960 | **0.8593** |
| SAE-small，L0≈15 | 0.9925 | 0.6584 |
| SAE-big，L0≈75 | 0.9936 | 0.7246 |
| 泛型文本对全激活 | 0.9749 | 0.0410 |
| SAE 残差可读性 | 0.0356 | -0.0287 |
| `w_dec` top 组 mean `|cos|` | 表面约 0.80（median） | 0.1892 |
| 高斯方向对照 | 0.0129 | 0.0112 |
| 残差注入检测，α=2 | 0.9189（与地板相同） | 0.0915（α=0 地板 0.0720） |
| B2 Top-1：NLA / SAE-small / SAE-big | — | **92.5% / 95.0% / 95.0%** |
| B2 mean margin：NLA / SAE-small / SAE-big | — | **0.3195 / 0.2085 / 0.2531** |
| B6 selection yield：all / domain / language | — | **58.3% / 40.0% / 88.9%** |
| B6 `q+` median：semantic ITT / heldout-valid / Gaussian | — | **0.114 / 0.181 / 0.005** |
| B4 `r−` median：semantic ITT / heldout-valid / Gaussian | — | **0.031 / 0.042 / 0.001** |
| B4 polarity / sign accuracy / feature Top-1，ITT | — | **0.071 / 75.0% / 33.3%** |
| carrier nonzero median：amplify / ablate / insert | — | **0.042 / 0.210 / 0.097** |

由此得到的路线判断：

- **实例级激活分诊：可行。** 在真实激活上，NLA 去均值方向保真明显高于两档 SAE，适合放在廉价探针/SAE 初筛之后做少量深读。
- **B2 给出限定。** NLA 没有在已饱和的离散 Top-k 指标上胜过 SAE,但正确匹配相对最佳
  错配的 margin 更大;只能写成“方向保真/几何间隔更强”,不能写成“所有判别指标全面胜出”。
- **NLA 标注 SAE 特征：条件性可行,不能自动验真。** B6+B4 证明 `+w_dec` 有非随机、
  可重复、可检索的内部 round-trip 信号,但 `r−` 很弱;heldout AUC 与 `q+` 的
  Spearman≈−0.015。事后单评审者 surface audit 仅 5/24 strict match、9/24 明显错配,
  所以 AR 分数只能作 triage/一致性信号,不能自动接受标签。
- **读取 SAE 残差/暗物质：当前判负。** 残差本身不可读，注入相当于两倍残差范数的已知信号仍基本检不出。失败点在 AV 对离流形输入退化成模板，不是简单调阈值能修复。
- **C1 仍是最值得做的论文核心问题,但要重构。** 不是预设“AR 是统一正确性标尺”,而是
  直接检验 closed-loop reconstruction/retrieval 是否预测正确标签、错配标签、第三方改写、
  heldout activation 与人类/因果效度。B6+B4 已给出“内部一致性不等于语义忠实”的反例。
- **解释不能当事实。** 40 条解释中发现 3 条具体实体编造，而编造行与正确行的原始 round-trip cos 几乎相同；高重建分只表示文本保留了某些方向信息，不等于表面语义忠实。

效度边界必须始终附带：旧主线只有 40 个激活；B6+B4 虽扩到 24 文档/1365 激活,
但 domain 每个 split 每类只有一个独立 topic,语言结果混有 script/tokenization/长度效应。
全部仍是单模型单层、AV/AR 同源、无盲式多人评测/C7/下游因果验证的 pilot,数字不可外推。

### C1 最新状态（2026-07-30）

C1 pilot 已跑完，但 confirmatory 主实验尚未进入 activation 阶段：

- Pilot 主效应（AR 对正确 reference 相对 reciprocal hard negative）：
  mean difference `0.01550`，pair bootstrap 95% CI
  `[0.00510, 0.02712]`；16/24 concepts 为正，concept sign test
  `p=0.0758`，pair-joint exact sign-flip `p=0.00393`。信号方向可见，但效应集中，
  不能把 pilot 当 confirmatory 结论。
- Confirmatory v1 在 activation 前因生成文本字数/格式可行性失败而停止，证据保留。
- Confirmatory v2 完成 144 个全新请求的机械生成；两名独立 reviewer 都给出
  `FAIL`，且共同发现同一组 10 个文档问题（hard-negative contamination、
  非实例、禁用专名及 train/heldout 场景重复）。因此没有抽取 v2 activation。
- V3 被明确登记为 pre-activation adaptive corpus redevelopment；首版 144 个
  scenario anchors 的独立审计因 20/24 concept 的 heldout 机制/应用重叠而失败。
- 唯一允许的最终 pre-text 修订 v3r2 改写 33/144 anchors。结构、reciprocal
  mapping、安全性、难度及 pair framing 均通过，但独立审计仍为
  **17 concept PASS / 7 FAIL**。失败 concept 是
  `error_detecting_codes`、`protein_quality_control`、
  `microbial_quorum_sensing`、`microbial_cross_feeding`、
  `groundwater_contaminant_transport`、`quarantine_regimes` 和
  `phonological_assimilation`。
- v3r2 audit SHA256 为
  `23908a7784e3e49f96daf0437186cc9bc9d6c1f453e6d8c2c1c9405e1786b1ef`；
  anchors SHA256 为
  `3b38876a663ea3a3a9a1623017242a06e0f51b667109cf60bb8de549cb21600a`。
  完整停止记录见
  `natural_language_autoencoder/results/c1_confirmatory_scenario_anchor_v3r2_failure.md`。
- 冻结规则明确规定 v3r2 若失败则本轮停止。因此 **没有生成任何 v3 请求，
  没有 discovery/heldout activation，没有 feature selection、AV/AR 或 endpoint**。
  继续需要用户明确选择新设计，不能把它伪装成同一 confirmatory run 的普通 retry。
- 严格 provenance 版 generator/runner 已完成真实 v3r2 schema 兼容和离线验证，
  但因审计门禁未启动。Generator SHA256：
  `fad14b4cb01ca3789678b07d991a73ed1fa257a442b1f0c35712ec1fbf65803e`；
  runner SHA256：
  `c54b02600c2eabb45d91da46df4af6b0ef544b92824daa01bf5f6ea3ca6a902f`。

下一次继续 C1 的关键不是再改 7 个句子，而是先决定 estimand：若目标是概念内
跨应用泛化，应冻结 application/perturbation/evidence-source 的不重叠层级；当前 rubric
要求狭窄 concept 内“不同 substantive mechanism”，可能把真正想测的概念泛化与过强的
机制独立性混在一起。若开 v4，应全量新生成、完整披露适应过程，并重新冻结 Stage 0。

## 5. 核心仓库当前代码状态

`D:\Projects\nla-from-autodl\natural_language_autoencoders` 当前为：

- 分支 `main`，相对本地记录的 `origin/main` **ahead 5**。
- 5 个本地提交依次加入：answer-token AV probe、question-token probe、全上下文安静导出、交互式 TUI、TUI 层选择。
- 未跟踪文件：`demo/launch_av_server.sh`、`demo/make_parquet.py`、`demo/roundtrip.py`、`demo/运行指南.md` 和 `remote_results/*.json`。
- 5 个已跟踪 `.sh` 显示修改，但 diff 只有执行位从 `100755` 变为 `100644`，没有文本改动；这是 Windows 工作区的 mode-bit 漂移，不要把它误判成业务修改，也不要未经确认覆盖用户工作。
- 仓库没有正式测试目录/测试套件。此次只做了静态核验：核心仓库加实验工作区共 75 个 Python 文件均能通过 `ast.parse`，12 个 JSON 资产均可解析；这不等于依赖、GPU、Miles、SGLang 的运行时测试通过。
- `remote_results` 证明探针曾跑通：其中带 AR 的 answer probe 有 3 条评分，mean cos 约 0.924；其余多为 AV-only 或上下文选择功能验证。

## 6. 状态漂移与处理记录

以下问题中，基础设施卫生项已于 2026-07-26 处理；代码兼容项仍待正式回灌：

1. **明文凭据泄漏：代码侧已清除，凭据侧仍建议轮换**
   - `natural_language_autoencoder/remote.py` 与 `Conclude.md` 原含明文 AutoDL 密码，本文件不复述。
   - 两处明文已删除；`remote.py` 现只读取 `~/.ssh/config` 与密钥，并已连接实测通过。
   - 因旧文件可能曾被同步或备份，仍应把旧密码按已泄漏处理并在控制台轮换。

2. **SSH 端口不一致：已修复**
   - 当前 `~/.ssh/config` 的 `Host autodl` 指向端口 **11813**。
   - `connect.ps1`、`sync.ps1`、`remote.py` 和 `README.md` 已统一读取/调用 `autodl` 别名，
     不再硬编码 host/port；PowerShell 与 Paramiko 两条路径均已实测。
   - `.claude/settings.local.json` 仍保留旧端口的历史权限条目；它不是连接配置或当前真相来源。

3. **Transformers 5 热修补没有回灌到核心仓库**
   - 实验脚本 `server/02_extract_activations.py` 已兼容 `apply_chat_template(tokenize=True)` 返回 `BatchEncoding`。
   - 服务器的 `/root/autodl-tmp/nla_repo/nla_inference.py` 据交接文档也修了两处。
   - 但本地 Git 仓库的 `nla_inference.py` 约第 187/403 行及 `nla/schema.py` 的相同调用仍假定返回 list；在服务器的 Transformers 5.12.1 环境下存在直接失败风险。应正式补丁化并加回归测试，不能继续依赖服务器热改副本。

4. **文档历史段落冲突：已修复**
   - `natural_language_autoencoder/README.md` 已把“无卡、未下载、待扩容/待 token”标为
     历史部署说明，并同步当前 A800、150G 数据盘与模型已落盘状态。
   - `README.md`、`Prompt.md`、`Conclude.md` 及历史 runner 中的自动关机规则已删除；
     当前统一执行用户的 keep-alive 指令。

5. **后续优先级状态**
   - B2 已完成;其预设“NLA Top-k 高于 SAE”未成立,但发现 NLA margin 更大。
   - B6+B4 已完成;支持正向 direction readout,不支持 standalone AR 标签验真或 steering。
   - 当前顺序为：**C1 外部效度审计 → C7+B3 → 据此决定 C2-v2 / 真正 steering**。

## 7. 推荐的下一阶段执行顺序

### P0：基础设施卫生（大部分已完成）

- ✅ 已从代码/文档移除明文凭据；仍建议在控制台轮换旧密码。
- ✅ 已统一 SSH 配置，修正运行脚本和文档中的旧端口引用。
- 把 Transformers 5 兼容修复回灌到 Git 仓库，并为 list / tensor / BatchEncoding 三种返回形态补最小测试。
- 清点并分类 Git 工作树：5 个本地提交、未跟踪 demo/结果、Windows mode-bit 漂移分别处理，禁止一把重置。
- ✅ 已整理 README 的历史/当前状态，并清除旧自动关机指令。

### P1：B2 检索评测，已完成

已使用 `results/recon_vectors.npz` 构造四组 centered `40×40` 矩阵并完成置换评测。
NLA Top-1 92.5%,两档 SAE 95%;Top-5 均 100%。NLA mean margin 0.3195,高于
SAE-small 0.2085 与 SAE-big 0.2531。`pred_resid` Top-1 17.5%,表明它不是严格
generic floor,而是保留弱样本信息的 residual-text control。

建议产物：

- `natural_language_autoencoder/server/10_retrieval_eval.py`
- `natural_language_autoencoder/results/retrieval_eval.json`
- `POSSBILITY.md` F8、`Conclude.md` 与 README 已同步。

### P2：B6+B4 因子语料扩展，已完成

- 新语料为 4 领域 × 3 语言 × train/test 的 24 个长 prompt,保存所有 position≥50 激活。
- 特征按 discovery 文档级 top-3 activation、AUROC、跨文档 support、方向去重规则冻结;
  AV/AR 结果不参与选择。
- 24 个新 semantic、4 个 legacy、1 个 structural、8 个 active-nonselective、
  8 个 Gaussian,共 45 个方向;全部测 `±v` 的 greedy+4 stochastic generations。
- 另对 28 个 semantic 特征做 carrier-conditioned amplify/ablate/insert;合计 590 个 AV job,
  AV 73.8 分钟、AR 25.9 秒,解释标签成功 585/590。
- ITT `q+`/`r−`/polarity median = 0.114/0.031/0.071;sign accuracy 75%,
  feature Top-1 33.3%。正方向可读,负方向只有弱几何信号。
- 24 个新 semantic 中 9 个 carrier coefficient 为 0(no-op)。非零 15 个的
  amplify/ablate/insert median cos = 0.042/0.210/0.097;只证明 AV/AR 对激活空间增删敏感,
  未重新注入 Gemma 后续层,不是 steering。
- 完整结果见 `results/b6b4_factorial_analysis.md`、`b6b4_surface_audit.md` 与
  `POSSBILITY.md` F9。
- `results/b6b4_checksums.sha256` 已逐项验证；正式 result / recon NPZ / checkpoint / log
  SHA256 分别为 `ce866a…` / `d011ee…` / `43d9e4…` / `e2a9ef…`。

### P3：C1，外部效度审计

Pilot 已完成；confirmatory 在 v3r2 pre-text anchor audit 处按冻结规则停止，详见
“C1 最新状态”。继续前必须显式新建 v4 设计，或修改 heldout generalization estimand；
不能在本轮继续选择性改写。原科学判据不变：只有正确 label/reference 稳定胜过
reciprocal hard negative，且 AR 分数能够预测独立的 surface/human/causal validity，
AR 才能称 evaluator，而不只是同源 closed-loop consistency measure。

### P4：C7+B3，循环性与实体敏感性

对现有解释做第三方语义等价改写和受控实体替换，再让 AR 重打分。若改写后分数崩溃，说明 AV/AR 可能依赖同源措辞或私有编码；若实体替换几乎不影响分数，则应继续降低对解释具体指称的信任。

只有上述路线跑清楚后，才考虑 B5/B7、混合 NLA×SAE、迭代剥离或 off-manifold 课程训练。后者属于训练级新项目，不应作为普通 pilot 顺手启动。

## 8. 远端运行纪律

- AutoDL 开机只能由用户在控制台操作；没有明确需要时不要启动 GPU。
- 每个远端长任务必须用独立于 SSH 会话的 `setsid nohup` runner，并记录明确退出状态。
- **当前覆盖规则（2026-07-26 用户明确指令）：不要关机，也不要设置自动关机链。**
  该规则覆盖旧交接中的“完成后 shutdown”习惯，直到用户另行改变指令。
- 用户偏好不设可能中途杀任务的硬超时。
- 结果出现后立即 `scp` 回本地，但当前保持实例开启供后续 GPU 实验。
- 远端 `pkill/pgrep` 模式用方括号打断自匹配；上传脚本先去 CRLF。
- 只信日志中的 `*_COMPLETE`、`EXIT (status=N)` 与时间戳，不用 SSH 断连推断任务成功/失败。
- `HF_ENDPOINT` 在 shell 层设置；大型 SAE 仓库只按精确文件下载，禁止整库 snapshot。
- 新结果用新文件名，禁止覆盖既有 `results/` 和 `activations/` 资产。

## 9. 一句话接手判断

这是一个已完成第一轮 pilot、B2、B6+B4 与 C1 pilot 的研究项目；C1 confirmatory
在 activation 前因严格 corpus/anchor 门禁停止。当前证据支持 NLA 读取
真实激活、提供更大的 centered 配对 margin,并从一部分 SAE 正方向提取可重复的内部信息;
不支持它在已饱和 Top-k 上胜过 SAE,不支持读取 SAE 暗物质,也不支持把高 round-trip、
direction retrieval 或 carrier readout 直接解释成自然语言标签忠实或 behavioral steering。
