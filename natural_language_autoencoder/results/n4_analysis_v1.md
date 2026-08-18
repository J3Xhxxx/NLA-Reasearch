# N4 real-content replication — results

- Cohort: **200 rows / 101 documents**, all content tokens.
- Provenance bit-exact: **True**; identity max KL: **0**.
- Reconstruction JSON/NPZ semantic closure: **True**.
- H1 channel localization: **FAIL**.
- H2 causal ranking: **FAIL**.
- H3 causal channel: **FAIL**.

## H1 — centered-cosine channel localization

| Condition | Equal-doc mean cos | Share above generic | Retrieval Top-1 |
|---|---:|---:|---:|
| orig | +0.8357 | 1.000 | 0.995 |
| p3_only | +0.7793 | 0.932 | 0.995 |
| p12 | +0.6327 | 0.756 | 0.950 |
| quote_strip_p3 | +0.1934 | 0.229 | 0.355 |
| p1_only | +0.1972 | 0.233 | 0.290 |
| p2_only | +0.5755 | 0.688 | 0.950 |
| word_shuffle | +0.4522 | 0.539 | 0.990 |

`p3_only - p12` = +0.1466, 95% doc-bootstrap CI [+0.127, +0.166].

## H2 — causal reconstruction

| Condition | KL recovered @ pos | KL @ pos | KL16 | CE16 delta |
|---|---:|---:|---:|---:|
| orig | -313.8704 | 0.6238 | 0.0415 | +0.0399 |
| sae_small | +0.8561 | 0.6740 | 0.0471 | +0.0554 |
| sae_big | -4.5678 | 0.4028 | 0.0287 | +0.0325 |
| p3_only | -254.3812 | 0.6599 | 0.0456 | +0.0448 |
| p12 | -156099.6242 | 3.9926 | 0.2543 | +0.2163 |
| quote_strip_p3 | -153247.7474 | 9.0995 | 0.5795 | +0.5239 |
| dataset_mean | -113889.5461 | 6.6791 | 0.4279 | +0.3677 |
| zero | +0.0050 | 12.1449 | 0.7709 | +0.7351 |

NLA−SAE-big recovered = -309.3026; 90% CI [-854.711, +5.421]; equivalence ±0.05: **False**.

NLA−SAE-small recovered = -314.7265; 95% CI [-905.267, +0.024]; superiority: **False**.

## H3 — causal paragraph mechanism

`p3_only - p12` recovered = +155845.2430, 95% CI [+5761.721, +403007.145].
`p3_only/orig` recovered retention = **NA**.

## Decision

H1+H2 fail: restrict F11/F12 to the original contaminated pilot

Result SHA-256: `3c8a4d87d7289ac6c41b58e2bbdd6955585db46eaaa5306822d9d802259943cc`
