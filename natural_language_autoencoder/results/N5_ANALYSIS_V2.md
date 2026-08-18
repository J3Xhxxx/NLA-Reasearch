# N5 held-out selective hybrid — results

- ITT cohort: **400 independent content groups** (Pile 300 / XNLI 100).
- Frozen gate: **FEASIBLE**; held-out NLA coverage **0.655** (one-sided 95% lower 0.615).
- H5-A: **NO SELECTIVE CLAIM**.
- H5-B: **CHANNEL REPLICATED**.

## H5-A

- `G = +0.002419`; one-sided 95% lower `-0.001326`; 90% CI `[-0.001326, +0.006147]`.
- `Delta_raw = +0.032168` nat; 95% CI `[-0.026114, +0.091261]`.
- Catastrophic regret: **9/400** (2.250%); one-sided exact 95% upper **3.893%**.
- Failed gates: G one-sided 95% lower bound <= 0, catastrophic-regret exact one-sided 95% upper bound >= .03.

## H5-B

- `G_p3_p12 = +0.179490`; 95% CI `[+0.146504, +0.215996]`.
- `T = R_p3/R_orig = 0.9879956847092126`; one-sided 95% lower `0.9830146346758943`.

## QA

- Frozen mean-direction match max abs: `0`.
- Identity KL maxima: position `0`, KL16 `0`.
- Negative numerical KL values clamped under the frozen rule: **0**.
- Result SHA-256: `1043424d1e21440a5bef3e581c01b002a86e573cbb91f35f9a319b5cde212602`.
