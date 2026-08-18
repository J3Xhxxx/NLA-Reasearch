# C1 Pilot Secondary Diagnostics

> Exploratory only; computed after inspecting the primary result.

## Robustness and retrieval

- Frozen-pair axis delta: mean 0.0155, median 0.0090, 16/24 wins.
- Against all six wrong axis texts: correct-minus-mean-wrong mean 0.0162; feature→axis Top-1 50.0%.
- Label-cluster means are positive for 7/7 axes; all leave-one-label-out means are positive.
- Train-reference 24-way Top-1: text→feature 54.2%, feature→text 70.8%.
- NLA-original 24-way Top-1: text→feature 62.5%, feature→text 58.3%.

## External-validity warnings

- Axis delta predicts heldout-valid status only descriptively: AUC 0.764; Spearman with test AUC 0.501.
- Context-label delta goes the wrong way for heldout generalization: valid-minus-invalid mean -0.0917.
- NLA paraphrases retain a median 60.1% of characters, versus 106.3% for authored references; private-code evidence is therefore suggestive, not identified.
- The base judge is not a valid quantitative oracle: generic and wrong-axis candidates frequently receive positive scores, and pooled q→judge AUC is below chance in the primary report.
