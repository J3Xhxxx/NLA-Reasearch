# C1 External-Validity Protocol Pilot

## Scope

- ITT unit: 24 previously inspected B6 semantic features.
- Only the seven coarse domain/language axes were historically frozen.
- Fine-grained references and the Gemma context judge are exploratory, not human ground truth.

## Primary frozen-axis contrast

| Metric | Result |
|---|---:|
| Mean delta | 0.0155 [0.0051, 0.0271] |
| Median delta | 0.0090 [-0.0002, 0.0240] |
| Positive fraction | 0.6667 [0.4583, 0.8333] |
| Exact sign test | 16/24 wins, p=0.0757948 |
| Exact sign-flip test on mean | p=0.00393438 |

The delta is `q_AR(axis reference) - q_AR(fixed deranged axis)` for each feature.

## Candidate-source summaries

| Kind | n | q median | axis Top-1 | feature Top-1 | judge>=2 | unsupported |
|---|---:|---:|---:|---:|---:|---:|
| axis_reference | 24 | -0.0133 | 33.3% | 8.3% | 100.0% | 0.0% |
| axis_paraphrase | 24 | -0.0200 | 33.3% | 8.3% | 58.3% | 0.0% |
| axis_hard_negative | 24 | -0.0335 | 16.7% | 0.0% | 45.8% | 0.0% |
| train_reference | 24 | 0.1925 | 70.8% | 54.2% | 62.5% | 25.0% |
| train_reference_paraphrase | 24 | 0.2009 | 66.7% | 50.0% | 62.5% | 20.8% |
| train_hard_negative | 24 | -0.0087 | 20.8% | 16.7% | 12.5% | 45.8% |
| sibling_mismatch | 24 | 0.0869 | 70.8% | 12.5% | 54.2% | 29.2% |
| base_autointerp | 24 | 0.1230 | 62.5% | 45.8% | 66.7% | 16.7% |
| nla_original | 24 | 0.1136 | 70.8% | 62.5% | 8.3% | 12.5% |
| nla_paraphrase | 24 | 0.0263 | 70.8% | 54.2% | 20.8% | 29.2% |
| generic | 192 | -0.0346 | 16.1% | 4.2% | 57.3% | 0.0% |

## Exploratory contrasts

| Contrast | Median [95% bootstrap] | Positive fraction | sign p |
|---|---:|---:|---:|
| train_reference_minus_train_hard_negative | 0.1932 [0.0354, 0.3122] | 83.3% | 0.00077194 |
| train_reference_minus_sibling_mismatch | 0.0575 [0.0427, 0.1466] | 79.2% | 0.00330538 |
| axis_reference_minus_generic_feature_mean | 0.0091 [0.0000, 0.0214] | 70.8% | 0.0319573 |
| train_reference_minus_generic_feature_mean | 0.2191 [0.0477, 0.3964] | 87.5% | 0.000138581 |
| axis_paraphrase_minus_axis_reference | -0.0038 [-0.0115, 0.0045] | 37.5% | 0.924205 |
| train_paraphrase_minus_train_reference | -0.0058 [-0.0319, 0.0014] | 33.3% | 0.968043 |
| nla_paraphrase_minus_nla_original | -0.0310 [-0.1874, -0.0102] | 20.8% | 0.999228 |
| private_code_interaction | 0.0263 [0.0049, 0.1487] | 70.8% | 0.0319573 |

## External-association diagnostics

| Candidate | q vs test AUC ρ | heldout-valid AUC | q vs blind judge ρ |
|---|---:|---:|---:|
| axis_reference | 0.019 | 0.436 | -0.383 |
| train_reference | -0.260 | 0.300 | -0.296 |
| nla_original | -0.015 | 0.514 | 0.261 |
| nla_paraphrase | 0.259 | 0.643 | 0.437 |
| base_autointerp | -0.343 | 0.307 | -0.331 |

## Interpretation limits

- The 24 B6 features and AV outputs were inspected before this pilot.
- Only seven coarse axis labels were historically frozen; fine-grained references were authored later.
- The base judge and base autointerpreter are provisional model signals, not blind human ground truth.
- Features are nested in seven labels and come from one model, layer, SAE setting, and synthetic prompt design.
- The heldout-valid subgroup is post-selection descriptive and never replaces ITT n=24.
- AR discrimination or retrieval does not establish monosemanticity, causal validity, or behavioral steering.
