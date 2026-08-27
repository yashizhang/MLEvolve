from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.coldstart import build_guidance_description, is_scientific_guidance
from engine.coldstart.naturebench import (
    collect_visible_path_inventory,
    flatten_task_description,
    render_recipe_guidance,
    resolve_asset,
    resolve_recipe,
)

COLDSTART_DIR = Path(__file__).resolve().parents[1] / "engine" / "coldstart"
RECIPE_PATH = COLDSTART_DIR / "naturebench_recipes.json"
ASSET_PATH = COLDSTART_DIR / "naturebench_assets.json"


def load_recipes():
    return json.loads(RECIPE_PATH.read_text(encoding="utf-8"))


def load_assets():
    return json.loads(ASSET_PATH.read_text(encoding="utf-8"))


EXAMPLES = {
    "spatial_multiomics_domain_identification": (
        "Identify spatial domains from paired RNA gene expression, ATAC chromatin "
        "accessibility, and spatial coordinates. Evaluate with adjusted Rand index."
    ),
    "inverse_protein_folding": (
        "Inverse protein folding: design amino-acid sequences from protein backbone "
        "PDB coordinates and evaluate sequence recovery."
    ),
    "smlm_point_cloud_clustering": (
        "Spatial clustering of single-molecule localization microscopy point clouds "
        "with background localizations."
    ),
    "xray_diffraction_denoising": (
        "Denoise low-count X-ray diffraction intensity images using paired clean "
        "targets while preserving diffraction peaks."
    ),
    "molecular_linker_design": (
        "3D molecular linker design: connect two molecular fragments at attachment "
        "points using coordinates and geometry, then score validity."
    ),
    "irregular_temporal_sequence_modeling": (
        "Temporal sequence modeling with irregular sampling, observation times, "
        "missingness masks, and time gaps."
    ),
}


@pytest.mark.parametrize("expected,description", EXAMPLES.items())
def test_routes_all_six_scientific_signatures(expected, description):
    match = resolve_recipe(description, [], load_recipes())
    assert match is not None
    assert match.recipe_id == expected
    assert match.margin >= match.minimum_margin


def test_description_dict_is_flattened_stably():
    text = flatten_task_description(
        {"Task goal": "inverse protein folding", "Evaluation": ["sequence recovery"]}
    )
    assert "Task goal" in text
    assert "inverse protein folding" in text
    assert "sequence recovery" in text


def test_case_identifier_or_doi_alone_never_routes():
    db = load_recipes()
    assert resolve_recipe("s41467-025-63418-x", [], db) is None
    assert resolve_recipe("10.1038/s41467-025-63418-x", [], db) is None
    assert resolve_recipe("NatureBench case 4", [], db) is None


def test_unknown_task_returns_no_guidance(tmp_path):
    desc = tmp_path / "description.md"
    desc.write_text("Predict a scalar from ordinary tabular rows.", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = make_cfg(tmp_path, desc, data_dir, resolver="semantic")
    assert build_guidance_description(cfg) == "None model"


def test_semantic_guidance_is_marked_and_actionable(tmp_path):
    desc = tmp_path / "description.md"
    desc.write_text(EXAMPLES["xray_diffraction_denoising"], encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train_low_count.npy").write_bytes(b"")
    (data_dir / "train_high_count.npy").write_bytes(b"")
    cfg = make_cfg(tmp_path, desc, data_dir, resolver="semantic")
    guidance = build_guidance_description(cfg)
    assert is_scientific_guidance(guidance)
    assert cfg.coldstart.selected_recipe_id == "xray_diffraction_denoising"
    assert cfg.coldstart.selected_recipe_score > 0
    assert cfg.coldstart.selected_recipe_margin > 0
    assert cfg.coldstart.selected_recipe_confidence in {"low", "medium", "high"}
    assert cfg.coldstart.selected_recipe_evidence
    assert "Matched recipe" in guidance
    assert "Cheap baselines" in guidance
    assert "NAFNet" in guidance
    assert "unavailable" in guidance
    assert "do not download" in guidance.casefold()


def test_legacy_mapping_wins_in_auto_mode(tmp_path):
    desc = tmp_path / "description.md"
    # Deliberately looks like a scientific task; exact legacy mapping must still win.
    desc.write_text(EXAMPLES["inverse_protein_folding"], encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = make_cfg(tmp_path, desc, data_dir, resolver="auto", exp_id="legacy-task")
    guidance = build_guidance_description(cfg)
    assert "LegacyBERT" in guidance
    assert not is_scientific_guidance(guidance)


def test_legacy_mode_does_not_fall_back_to_semantic(tmp_path):
    desc = tmp_path / "description.md"
    desc.write_text(EXAMPLES["inverse_protein_folding"], encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = make_cfg(tmp_path, desc, data_dir, resolver="legacy", exp_id="unlisted")
    assert build_guidance_description(cfg) == "None model"


def test_general_tier_excludes_specialist_model(tmp_path):
    desc = tmp_path / "description.md"
    desc.write_text(EXAMPLES["molecular_linker_design"], encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    general = build_guidance_description(
        make_cfg(tmp_path, desc, data_dir, resolver="semantic", prior_tier="general")
    )
    specialist = build_guidance_description(
        make_cfg(tmp_path, desc, data_dir, resolver="semantic", prior_tier="specialist")
    )
    assert "DeLinker" not in general
    assert "DeLinker" in specialist
    assert "[specialist]" in specialist


def test_missing_assets_skip_by_default_but_strict_mode_fails(tmp_path):
    desc = tmp_path / "description.md"
    desc.write_text(EXAMPLES["inverse_protein_folding"], encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    guidance = build_guidance_description(
        make_cfg(tmp_path, desc, data_dir, resolver="semantic", strict_assets=False)
    )
    assert "ProteinMPNN" in guidance
    assert "unavailable" in guidance

    with pytest.raises(FileNotFoundError):
        build_guidance_description(
            make_cfg(tmp_path, desc, data_dir, resolver="semantic", strict_assets=True)
        )


def test_asset_resolution_from_environment(tmp_path, monkeypatch):
    root = tmp_path / "ProteinMPNN"
    (root / "vanilla_model_weights").mkdir(parents=True)
    (root / "protein_mpnn_run.py").write_text("# local", encoding="utf-8")
    (root / "vanilla_model_weights" / "v_48_020.pt").write_bytes(b"weights")
    monkeypatch.setenv("PROTEINMPNN_DIR", str(root))

    status = resolve_asset("proteinmpnn", load_assets(), pretrain_model_dir="")
    assert status.available
    assert status.path == root.resolve()
    assert "PROTEINMPNN_DIR" in status.configured_by


def test_recipe_override_is_explicit_not_identifier_based():
    match = resolve_recipe(
        "ordinary text",
        [],
        load_recipes(),
        recipe_override="inverse_protein_folding",
    )
    assert match is not None
    assert match.overridden
    assert match.confidence == "explicit override"


def test_inventory_does_not_follow_symlinks_and_obeys_limit(tmp_path):
    root = tmp_path / "visible"
    root.mkdir()
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "b.txt").write_text("b", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.pdb").write_text("secret", encoding="utf-8")
    try:
        (root / "linked").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported")

    inventory = collect_visible_path_inventory(root, max_files=1)
    assert len(inventory) == 1
    assert all("secret" not in item for item in inventory)


def test_databases_contain_no_source_answer_keys_or_network_commands():
    combined = (RECIPE_PATH.read_text(encoding="utf-8") + "\n" + ASSET_PATH.read_text(encoding="utf-8")).casefold()
    # Exact source-task methods are intentionally excluded. DeLinker is a
    # separately labeled non-source specialist prior and is therefore allowed.
    for forbidden in ["multigate", "mapdiff", "miro", "difflinker"]:
        assert forbidden not in combined
    assert re.search(r"s\d{5}-\d{3}-\d{5}-\d", combined) is None
    for forbidden in ["http://", "https://", "wget ", "curl ", "git clone", "pip install"]:
        assert forbidden not in combined


def test_renderer_has_no_remote_fallback():
    db = load_recipes()
    match = resolve_recipe(EXAMPLES["inverse_protein_folding"], [], db)
    assert match is not None
    guidance = render_recipe_guidance(match, db, load_assets())
    lower = guidance.casefold()
    assert "do not download" in lower
    assert "skip this arm" in lower
    assert "http://" not in lower and "https://" not in lower


def make_cfg(
    tmp_path: Path,
    desc_file: Path,
    data_dir: Path,
    *,
    resolver: str,
    exp_id: str = "",
    prior_tier: str = "general",
    strict_assets: bool = False,
):
    legacy_tasks = tmp_path / "legacy_tasks.json"
    legacy_models = tmp_path / "legacy_models.json"
    if not legacy_tasks.exists():
        legacy_tasks.write_text(
            json.dumps({"legacy-task": "NLP"}), encoding="utf-8"
        )
    if not legacy_models.exists():
        legacy_models.write_text(
            json.dumps(
                {
                    "NLP": {
                        "LegacyBERT": {
                            "Description": "legacy description",
                            "Code_template": "model = 'legacy'",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    coldstart = SimpleNamespace(
        use_coldstart=True,
        task_json_path=str(legacy_tasks),
        model_json_path=str(legacy_models),
        description="",
        resolver=resolver,
        recipe_json_path=str(RECIPE_PATH),
        asset_manifest_path=str(ASSET_PATH),
        prior_tier=prior_tier,
        recipe_override="",
        strict_assets=strict_assets,
        max_inventory_files=256,
    )
    return SimpleNamespace(
        exp_id=exp_id,
        desc_file=desc_file,
        goal=None,
        eval=None,
        data_dir=data_dir,
        torch_hub_dir="",
        pretrain_model_dir=str(tmp_path / "missing-assets"),
        coldstart=coldstart,
    )
