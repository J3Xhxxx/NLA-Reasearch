# N4 real-content causal audit — 2026-07-30

## Executive verdict

Opus 的 N3 Q1 结论成立：旧 synthetic test 中 8 个“完全不激活”的
feature 在 8.24M-token 真实语料上全部激活，24/24 旧 feature 无一死亡。
因此 synthetic zero activation 是 coverage failure，不能再作为 feature
淘汰门禁。但“真实语料上激活”不等于旧标签正确，也没有回答 feature
readability 的 Q2。

N4 随后在冻结的 200 个真实 content-token 位置、101 篇文档上补做了项目
一直缺失的 causal endpoint。实验与数据闭环通过，主要科学结果是：

1. NLA 的 centered-cosine 优势不能直接推出 causal 优势。
2. 三种 reconstruction 相对 zero ablation 都恢复约 94%–97% 的 aggregate
   causal effect；描述性结果为 SAE-big 最好、NLA 次之、SAE-small 最后。
3. NLA 在多数文档上优于 SAE-small，但有少量真实的 catastrophic causal
   failures，因而尚未证明平均效用优势。
4. NLA 文本的第三段几乎保留完整文本的 aggregate causal effect，并显著
   强于 p12；稳定指标一致支持“第三段是主要 causal channel”。
5. 预注册的逐行 `KL_recovered` 均值因近零 `KL_zero` 分母而病态。冻结结果
   不能事后改判，但巨大负均值不能被解释为方法效应。

## Frozen run and QA

- Cohort: 200 rows / 101 documents; 144 Pile rows + 56 XNLI rows.
- All positions are natural content tokens; template/blank count is zero.
- Layer-32 provenance versus frozen activation: bit-exact, max abs error 0.
- Full-sequence clean evaluation state versus frozen activation: bit-exact.
- Identity patch: max KL 0.
- Reconstruction JSON versus NPZ: every centered/raw/LODO cosine was recomputed;
  maximum discrepancy was `3.33e-16`.
- Causal run: 2,301 full-sequence batch-size-one forwards.
- Reconstruction wall time: 1,641 s; causal patch wall time: 182 s; total GPU
  work for N4 was about 30.4 minutes, excluding smoke/provenance hashing.
- Causal stage observed about 24.4 GiB VRAM on one A800-80GB.
- The AutoDL instance remains on; GPU is idle after completion.

Frozen artifact hashes:

| Artifact | SHA-256 |
|---|---|
| activation parquet | `eb9a686a4cdea1f97134b2367d7c8a74a35351678d3c043eddae2f993e17ab66` |
| preregistration | `44bd48998f6436616347bb2d74d9b8a569a85a3639c314ce9711a252c8075f1c` |
| reconstruction JSON | `747f92e024a7aeb5a47e05e1d006b0403760f908229c6451f7a844c33223cdcd` |
| reconstruction NPZ | `e9d89713dc64381a52f05224d6522abb0ec547777a8c6a7f08b841a72a339967` |
| causal JSON | `8dd532f65d8c9c153f04ba433cc6f160798598fbbcbee388c15fb4a75a366233` |
| frozen analysis JSON | `3c8a4d87d7289ac6c41b58e2bbdd6955585db46eaaa5306822d9d802259943cc` |

The five base-model weight shards are independently hashed in
`n4_base_model_full_weights.sha256`.

## H1: text-channel localization

Formal verdict: **FAIL**.

| Gate | Result | Verdict |
|---|---:|---|
| `share(p3_only) >= .80` | .932 | pass |
| `share(p12) <= .50` | .756 | fail |
| `p3 - p12` centered cosine | +.1466, 95% doc-CI [.1274, .1663] | pass |

The correct interpretation is not that paragraph 3 is unimportant. It is nearly
sufficient, but not necessary: paragraph 2 alone retains a share of .688, so
geometric information is distributed or redundant across p2/p3. Removing quoted
candidate strings from p3 collapses its share to about .229, while word shuffling
retains about .539. These are reconstruction diagnostics, not human-semantic
fidelity claims.

## Why the preregistered KL-recovered mean broke

The frozen endpoint was

`KL_recovered(s) = 1 - KL_s / max(KL_zero, 1e-6)`.

Only one row was below `1e-6`, but 7/200 rows had `KL_zero < .01` and 10/200 had
`KL_zero < .1`. Thus values such as `.000294` and `.001021` remained valid
divisors and amplified individual failures by thousands.

Two important rows were not mere arithmetic noise:

- idx 185: `KL_zero=.000294`, NLA KL=16.375, CE16 degradation `+1.023`;
- idx 87: `KL_zero=.001021`, NLA KL=7.920, CE16 degradation `+.495`, while
  both SAEs were nearly unchanged.

Their NLA centered cosines were still about .689 and .736. High cosine therefore
does not guarantee local causal fidelity; the downstream Jacobian is strongly
anisotropic.

The frozen H2 mean became roughly -313 and the frozen H3 ratio became undefined.
Those values must remain in the immutable preregistered result, but they are not
scientifically interpretable estimates of method quality.

## H2: NLA versus the two SAEs

Formal preregistered verdict: **FAIL / not established**. NLA–SAE-big equivalence
and NLA superiority over SAE-small were not confirmed under the frozen row-wise
endpoint. This does not prove non-equivalence or inferiority.

### Stable aggregate sensitivity

Document-clustered ratio-of-sums:

| Condition | KL@position recovered | 95% CI | KL16 recovered | 95% CI |
|---|---:|---:|---:|---:|
| NLA | .94795 | [.91797, .97074] | .94548 | [.91613, .96799] |
| SAE-small | .94417 | [.92599, .95924] | .93850 | [.92057, .95380] |
| SAE-big | .96649 | [.95313, .97719] | .96231 | [.94807, .97394] |

Paired ratio-of-sums contrasts:

- NLA minus SAE-small: `+.00377`, 95% CI `[-.02757, +.02941]`.
- NLA minus SAE-big: `-.01854`, 90% CI `[-.04275, +.00219]`.

The latter sensitivity interval lies inside the preregistered ±.05 equivalence
margin, but ratio-of-sums was selected after seeing the denominator failure and
therefore cannot turn H2 into a confirmatory PASS.

Denominator-free document-clustered contrasts (positive means NLA has lower KL):

| Contrast | KL@position | 95% CI | KL16 | 95% CI | CE16 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| SAE-small − NLA | +.0501 | [-.3247, +.3641] | +.00561 | [-.01777, +.02515] | +.01552 | [-.01342, +.04343] |
| SAE-big − NLA | -.2210 | [-.5679, +.0640] | -.01273 | [-.03469, +.00543] | -.00743 | [-.03386, +.01567] |

No mean contrast is resolved. Distributionally, however, NLA beats SAE-small in
80/101 documents at the patched position and 81/101 over KL16; the median
advantages are positive. Rare NLA catastrophes erase this typical-case advantage
in the mean. Against SAE-big, sign counts are approximately balanced.

The defensible summary is:

> NLA is typically better than SAE-small and approximately in the same aggregate
> causal regime as SAE-big, but it has a heavier failure tail and has not
> established mean superiority over SAE-small.

## H3: is the localized text channel causally useful?

Frozen status: **NOT TESTABLE**, not an ordinary scientific failure. The frozen
renderer labels it `FAIL`, but the preregistered testability gate failed because
the pathological row-wise recovered mean for `orig` was negative.

All stable sensitivity endpoints agree:

| Endpoint | p3 advantage over p12 | 95% doc-CI |
|---|---:|---:|
| ratio-of-sums recovered @ position | +.27865 | [.20413, .37111] |
| ratio-of-sums recovered over KL16 | +.27490 | [.20143, .36601] |
| raw `KL(p12)-KL(p3)` @ position | +3.333 | [2.481, 4.309] |
| raw `KL16(p12)-KL16(p3)` | +.2087 | [.1553, .2697] |
| `CE16(p12)-CE16(p3)` | +.1715 | [.1131, .2364] |

- 89/101 documents favor p3 on raw KL; 91/101 favor it on KL16.
- Aggregate p3/orig causal-retention is .99691, 95% CI [.98810, 1.00771].

This is strong exploratory evidence that the dense candidate paragraph is the
main causally useful text subchannel even though geometric information is also
present in p2. It requires a fresh confirmatory replication with a stable frozen
estimand.

## Implication for NLA × SAE

The result does not say that NLA should replace an SAE. It gives a more promising
combination:

- NLA supplies a readable, candidate-level causal channel and wins on many
  ordinary examples.
- SAE-big supplies lower aggregate KL and better tail reliability.
- A selective NLA-with-SAE-fallback system can use NLA when its reconstruction
  passes a frozen reliability gate and otherwise retain the SAE representation.

Post-hoc upper bounds show real complementarity:

- choosing the better of NLA and SAE-big per row would recover .98076 of the
  zero-ablation effect versus .96649 for SAE-big alone;
- choosing the best of NLA, SAE-small and SAE-big would recover .98248;
- a simple five-fold, document-separated cosine-threshold gate reached .96931
  with SAE-big fallback, but its paired CI still crossed zero.

The gate was designed after inspecting N4 and is hypothesis-generating only.

## Recommended next experiment

Run **N5: held-out selective NLA + SAE-big** before returning to the expensive
feature-level C1/Q2 benchmark.

1. Freeze fresh discovery and held-out real documents before any causal result.
2. Learn one NLA reliability threshold only on discovery documents.
3. On held-out documents compare always-NLA, always-SAE-big, the frozen hybrid,
   and an oracle upper bound.
4. Primary endpoints: document-clustered ratio-of-sums and raw paired KL;
   secondary endpoints: KL16, CE16, and catastrophic-failure rate.
5. Keep a minimum-ablation-effect eligibility rule or avoid row-wise division
   entirely.
6. If the hybrid improves SAE-big and preserves readable p3 text, then connect
   it to corrected real-corpus feature-level C1: NLA explains/steers the readable
   subset, SAE remains the high-fidelity carrier and fallback.

An A800-80GB is sufficient. A fresh 400–600-position N5 should require roughly
1–1.5 GPU-hours, with causal patching itself only a few minutes; AV generation is
the dominant cost. No further GPU experiment was launched after N4.

