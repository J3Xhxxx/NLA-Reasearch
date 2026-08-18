#!/usr/bin/env python3
"""Deterministically build the final pre-text C1 v3r2 scenario-anchor freeze."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "server" / "c1_confirmatory_scenario_anchors_v3.json"
AUDIT_PATH = ROOT / "results" / "c1_confirmatory_scenario_anchor_audit_v3.json"
OUTPUT_PATH = ROOT / "server" / "c1_confirmatory_scenario_anchors_v3r2.json"

SOURCE_SHA256 = "061903133b748ccffe2f85f697c5e6a7d53fd631e63fb1c0302ae42fbd59e6d5"
AUDIT_SHA256 = "5dcc267ad9f6bf9765bbb7c6ab3965498c275c7df3165780f5f1d79301c2a396"

REPLACEMENTS = {
    "automatic_memory_reclamation_test_01": "Paired heap snapshots show entries disappearing from a weak-key cache after their keys become otherwise unreachable while equivalent entries in a strong-key cache persist; interpret weak reachability and eligibility for reclamation without discussing block placement.",
    "dynamic_memory_allocation_test_01": "A concurrency trace shows allocation stalls clustered when many worker threads request blocks simultaneously, and the stalls disappear with per-thread caches although the shape of free space barely changes; interpret lock contention, cache refills, and allocator scalability.",
    "cryptographic_authentication_test_01": "An incident timeline shows update manifests becoming forgeable only after their signing key is exposed, while manifests bound to an independently protected epoch key remain attributable; interpret key compromise, epoch binding, and the limits of cryptographic origin claims.",
    "error_detecting_codes_test_01": "A two-dimensional printed symbol reports one failed row check and one failed column check after a noisy scan; interpret the syndrome intersection, what accidental error can be localized, and which multi-bit patterns could remain undetected.",
    "protein_quality_control_test_00": "Translation proofreading becomes less accurate while temperature, oxidation, and total protein synthesis remain unchanged; predict effects on nascent-chain folding, chaperone load, selective disposal, and aggregation.",
    "protein_quality_control_test_01": "Ribosome profiling shows persistent translation stalls, truncated nascent chains bearing disposal tags, and restored protein synthesis after a rescue factor is supplied; interpret the cotranslational quality-control stage implicated by this evidence.",
    "membrane_vesicle_trafficking_test_00": "A soluble hydrolase loses its compartment-targeting modification while its folding and catalytic activity remain intact; predict its missorting, extracellular release, and depletion from the intended intracellular destination.",
    "membrane_vesicle_trafficking_test_01": "Electron micrographs show deeply invaginated coated pits remaining attached to the plasma membrane and very few newly internalized carriers; interpret whether cargo capture, membrane bending, scission, or downstream fusion is limiting.",
    "microbial_quorum_sensing_test_00": "A community retains its normal extracellular signal production, but the signal receptor has lower affinity; predict how the cell-density threshold, fraction of responding cells, and positive-feedback transition change.",
    "microbial_cross_feeding_test_00": "A recipient strain acquires the ability to synthesize the vitamin formerly supplied by its partner while release by the producer remains unchanged; predict changes in dependency, relative abundance, exchange stability, and producer burden.",
    "fault_rupture_mechanics_train_00": "A water reservoir is filled above a pre-stressed fault and shallow slip events begin only after a delay; request how pressure diffusion, effective normal stress, friction, and stored elastic strain can trigger rupture.",
    "fault_rupture_mechanics_train_01": "Two mapped fault strands experience similar regional loading, but a clay-rich strand creeps while a rough crystalline strand remains locked and ruptures intermittently; contrast frictional stability, stress accumulation, and event style.",
    "fault_rupture_mechanics_train_02": "A transport tunnel must cross a fault zone with uncertain surface displacement; ask about setbacks, flexible joints, rupture monitoring, design displacement, and residual-risk tradeoffs.",
    "fault_rupture_mechanics_train_03": "A subsurface rupture crosses several rock units but its surface offset terminates abruptly beneath thick unconsolidated sediment; diagnose rupture propagation, material contrasts, distributed deformation, and the missing surface break.",
    "fault_rupture_mechanics_test_00": "A brief train of distant seismic waves reaches a near-critical fault while its long-term loading and fluid pressure remain unchanged; predict when dynamic stress perturbations trigger slip and when they decay without rupture.",
    "fault_rupture_mechanics_test_01": "Geodetic measurements show slow preslip migrating along a fault before high-frequency radiation begins in one compact zone; interpret nucleation, locked and creeping sections, and the onset of dynamic rupture.",
    "slope_failure_mechanics_train_00": "A hillside above a transport corridor accelerates after snowmelt saturates fractured soil; request how infiltration, pore pressure, effective strength, gravity, and a developing slip surface cause failure.",
    "slope_failure_mechanics_train_01": "Two engineered cut slopes have the same geometry and rainfall, but one contains a continuous clay seam while the other is uniformly granular; contrast drainage, shear strength, slip-surface continuity, and failure tendency.",
    "slope_failure_mechanics_train_02": "A transport route must pass below a hillside with uncertain slide depth; ask about setbacks, drainage galleries, retaining structures, deformation monitoring, maintenance, and residual-risk tradeoffs.",
    "slope_failure_mechanics_train_03": "A shallow slide overtops its runout barrier after saturated debris entrains loose material downslope; diagnose progressive strength loss, entrainment, acceleration, and why the assumed runout was too short.",
    "slope_failure_mechanics_test_00": "A distant seismic wave train shakes a marginally stable dry slope while its geometry and groundwater level remain unchanged; predict when transient loading initiates movement and when deformation remains limited.",
    "slope_failure_mechanics_test_01": "Successive surface scans show a head scarp widening while toe bulging migrates upslope, with little change in groundwater level; interpret the slide geometry, progressive strength loss, and likely mode of movement.",
    "groundwater_contaminant_transport_test_01": "Samples along a dissolved plume show the parent compound declining while a transformation product and electron-acceptor depletion increase despite stable hydraulic heads; interpret evidence for in-situ degradation rather than dilution or source movement.",
    "coastal_saltwater_intrusion_test_01": "Water samples become more saline as their conservative ion ratios approach a seawater-mixing pattern while evaporation indicators remain unchanged; interpret whether marine intrusion rather than local concentration explains the salinity.",
    "census_classification_test_01": "Reported ages cluster at multiples of five even though linked event records show smooth birth-cohort totals; interpret digit preference, proxy reporting, and enumeration quality rather than a sudden demographic wave.",
    "cadastral_land_taxation_test_01": "Tax receipts from equal-area farms diverge sharply where cadastral soil classes differ even though recent yields converge; interpret how classification-based assessment, rather than parcel subdivision, produces the burden pattern.",
    "quarantine_regimes_test_01": "Among individually detained exposed travelers, symptom onset is concentrated before the release cutoff with a sparse late tail; interpret what the onset-time distribution implies for detention duration, residual release risk, and humane follow-up.",
    "lexical_semantic_ambiguity_test_00": "Readers receive repeated exposure to a formerly rare sense before encountering the ambiguous word in a neutral sentence; predict how learned sense frequency changes initial interpretation and later contextual reanalysis.",
    "phonological_assimilation_test_00": "The same target and neighboring sound occur once inside a prosodic phrase and once across a strong intonational boundary; predict how boundary strength changes the timing and magnitude of assimilation.",
    "morphological_agreement_test_01": "Corpus examples pair a collective controller with a singular verb but a plural pronoun, with the pattern varying by whether the group is construed as one unit; interpret syntactic versus semantic agreement and target-specific feature resolution.",
    "feedback_control_stability_test_01": "A swept-frequency test shows a resonant response peak and increasing phase lag near the loop crossover; interpret damping, gain and phase margins, and robustness to small delays from frequency-domain evidence.",
    "dynamical_state_estimation_test_01": "Posterior state samples split into two persistent location modes after a symmetric range measurement and collapse to one mode when a directional measurement arrives; interpret measurement ambiguity and the probabilistic state update without proposing a control action.",
    "fatigue_crack_growth_test_01": "A fracture surface has a small origin beside a machining notch, repeated bands that widen across the section, and a final rough region; interpret the fractographic evidence for cyclic crack advance and terminal overload.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchors(document: dict):
    for concept in document["concepts"]:
        for split in ("train", "test"):
            for anchor in concept[split]:
                yield concept, split, anchor


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {OUTPUT_PATH}")
    if sha256(SOURCE_PATH) != SOURCE_SHA256:
        raise RuntimeError("Source-anchor hash mismatch")
    if sha256(AUDIT_PATH) != AUDIT_SHA256:
        raise RuntimeError("Anchor-audit hash mismatch")

    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit.get("status") != "FAIL":
        raise RuntimeError("Expected conservative v3 anchor audit status FAIL")
    revised = copy.deepcopy(source)

    old_by_id = {anchor["anchor_id"]: anchor for _, _, anchor in anchors(source)}
    if len(REPLACEMENTS) != 33 or not set(REPLACEMENTS).issubset(old_by_id):
        raise RuntimeError("Replacement map must contain exactly 33 existing anchor IDs")

    audit_named_ids: set[str] = set()
    for decision_type in ("concept_decisions", "pair_decisions"):
        for decision in audit[decision_type]:
            for failure in decision.get("failures", []):
                audit_named_ids.update(failure.get("anchor_ids", []))
    if not set(REPLACEMENTS).issubset(audit_named_ids):
        raise RuntimeError("Replacement map contains an anchor not named by the failed audit")

    changed_ids: list[str] = []
    for _, _, anchor in anchors(revised):
        anchor_id = anchor["anchor_id"]
        if anchor_id in REPLACEMENTS:
            anchor["scenario_anchor"] = REPLACEMENTS[anchor_id]
            changed_ids.append(anchor_id)

    revised["experiment"] = "C1 confirmatory synthetic concept cohort v3r2 scenario anchors"
    revised["status"] = "frozen_before_v3r2_generation"
    revised["purpose"] = (
        "Freeze the final pre-text design iteration after the conservative v3 "
        "scenario-anchor audit failed, while leaving every final prose request ungenerated."
    )
    revised["sources"]["scenario_anchors_v3"] = {
        "path": "server/c1_confirmatory_scenario_anchors_v3.json",
        "sha256": SOURCE_SHA256,
    }
    revised["sources"]["scenario_anchor_audit_v3"] = {
        "path": "results/c1_confirmatory_scenario_anchor_audit_v3.json",
        "sha256": AUDIT_SHA256,
        "status": "FAIL",
    }
    revised["generation_contract"]["pre_text_design_iteration"] = True
    revised["generation_contract"]["iteration_label"] = "v3r2_final_pre_text_design_iteration"
    revised["generation_contract"]["stop_if_v3r2_anchor_audit_fails"] = True
    revised["generation_contract"]["pair_style_rule"] = (
        "For every reciprocal pair, both concepts use the same ordered discourse slots, "
        "split assignments, and matched framing specificity; content remains concept-specific."
    )
    revised["revision"] = {
        "basis": "Conservative v3 scenario-anchor audit FAIL",
        "pre_text_design_iteration": True,
        "scope": (
            "Only audit-implicated heldout anchors were replaced for within-concept overlap, "
            "except that all twelve fault-versus-slope anchors named by the pair-level style "
            "failure were reframed symmetrically."
        ),
        "changed_anchor_count": 33,
        "unchanged_anchor_count": 111,
        "changed_anchor_ids": changed_ids,
        "stop_rule": (
            "If an independent conservative audit of v3r2 fails, stop scenario-anchor "
            "iteration instead of generating another content revision."
        ),
    }
    revised["validation_attestation"].update(
        {
            "v3_anchor_audit_status": "FAIL",
            "v3r2_changed_anchor_count": 33,
            "v3r2_unchanged_anchor_count": 111,
            "v3r2_mechanical_self_check": "PASS",
            "v3r2_per_concept_semantic_self_check": "PASS",
            "final_pre_text_design_iteration": True,
        }
    )

    new_rows = list(anchors(revised))
    old_rows = list(anchors(source))
    if len(revised["concepts"]) != 24 or len(new_rows) != 144:
        raise RuntimeError("Expected 24 concepts and 144 anchors")
    if sum(split == "train" for _, split, _ in new_rows) != 96:
        raise RuntimeError("Expected 96 train anchors")
    if sum(split == "test" for _, split, _ in new_rows) != 48:
        raise RuntimeError("Expected 48 test anchors")
    if any(len(c["train"]) != 4 or len(c["test"]) != 2 for c in revised["concepts"]):
        raise RuntimeError("Every concept must retain four train and two test anchors")

    new_ids = [anchor["anchor_id"] for _, _, anchor in new_rows]
    new_texts = [anchor["scenario_anchor"] for _, _, anchor in new_rows]
    if len(new_ids) != len(set(new_ids)) or len(new_texts) != len(set(new_texts)):
        raise RuntimeError("Anchor IDs and scenario texts must remain globally unique")
    if re.search(r"https?://|\b\d{4}\b", "\n".join(new_texts)):
        raise RuntimeError("Forbidden URL or four-digit year in scenario text")

    actual_changes = {
        new_anchor["anchor_id"]
        for (_, _, old_anchor), (_, _, new_anchor) in zip(old_rows, new_rows, strict=True)
        if old_anchor != new_anchor
    }
    if actual_changes != set(REPLACEMENTS) or len(actual_changes) != 33:
        raise RuntimeError("Unexpected anchor-level differences")
    for (_, old_split, old_anchor), (_, new_split, new_anchor) in zip(
        old_rows, new_rows, strict=True
    ):
        if (
            old_split != new_split
            or old_anchor["anchor_id"] != new_anchor["anchor_id"]
            or old_anchor["slot_id"] != new_anchor["slot_id"]
            or old_anchor["role"] != new_anchor["role"]
        ):
            raise RuntimeError("An anchor identity, split, slot, or role changed")

    OUTPUT_PATH.write_text(
        json.dumps(revised, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote={OUTPUT_PATH}")
    print(f"changed_anchors={len(actual_changes)}")
    print(f"sha256={sha256(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
