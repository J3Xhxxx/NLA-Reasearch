# J2-P0 SAE projection → language loop

Status: **EXPLORATORY_ANALYSIS_COMPLETE**

## Vector geometry

| Condition | target | centered cos | LODO cos | centered FVE |
|---|---|---:|---:|---:|
| `sae_small_to_x` | `x` | 0.586929 | 0.592222 | 0.354053 |
| `sae_small_loop_to_sae_small` | `sae_small` | 0.664820 | 0.665924 | 0.445477 |
| `sae_small_loop_to_x` | `x` | 0.696417 | 0.699892 | 0.514317 |
| `sae_big_to_x` | `x` | 0.640781 | 0.645166 | 0.421003 |
| `sae_big_loop_to_sae_big` | `sae_big` | 0.646140 | 0.647100 | 0.426048 |
| `sae_big_loop_to_x` | `x` | 0.736541 | 0.739626 | 0.565950 |

## Causal KL at patched position

| Condition | mean KL | median KL | aggregate recovery |
|---|---:|---:|---:|
| `nla_direct` | 0.628851 | 0.117300 | 0.947946 |
| `sae_small` | 0.674448 | 0.211190 | 0.944171 |
| `small_loop` | 1.326531 | 0.407320 | 0.890194 |
| `direct_small` | 1.314217 | 0.337486 | 0.891213 |
| `sae_big` | 0.404879 | 0.099284 | 0.966485 |
| `big_loop` | 1.249711 | 0.247168 | 0.896553 |
| `direct_big` | 1.082186 | 0.259916 | 0.910420 |

## Scope

This reused-cohort result is a mechanism audit, not confirmatory evidence that SAE-grounded NLA is superior.

Frozen metric-only case shortlist SHA-256: `b21a53bda1a6672b1327f303f978f67ac7eb1a0f3933e607c210cdb539fea405`.
