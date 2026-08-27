"""Build cold-start guidance from legacy model mappings or scientific recipes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .naturebench import (
    NO_GUIDANCE,
    SCIENTIFIC_GUIDANCE_MARKER,
    collect_visible_path_inventory,
    is_scientific_guidance,
    render_recipe_guidance,
    resolve_recipe,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
INIT_SOLUTION_JSON = PACKAGE_ROOT / "init_solution_paths.json"
DEFAULT_RECIPE_JSON = PACKAGE_ROOT / "naturebench_recipes.json"
DEFAULT_ASSET_MANIFEST = PACKAGE_ROOT / "naturebench_assets.json"


def _load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    try:
        value = getattr(obj, name)
    except Exception:
        return default
    return default if value is None else value


def _safe_set(obj: Any, name: str, value: Any) -> None:
    try:
        setattr(obj, name, value)
    except Exception:
        # Old or externally supplied config schemas may not expose provenance fields.
        pass


def _resolve_config_path(raw_path: Any, default: Path | None = None) -> Path | None:
    value = raw_path if raw_path not in (None, "") else default
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _replace_common_placeholders(text: str, cfg: Any) -> str:
    replacements = {
        "{TORCH_HUB_DIR}": str(_safe_get(cfg, "torch_hub_dir", "") or "").rstrip("/"),
        "{PRETRAIN_MODEL_DIR}": str(
            _safe_get(cfg, "pretrain_model_dir", "") or ""
        ).rstrip("/"),
    }
    for placeholder, replacement in replacements.items():
        if replacement:
            text = text.replace(placeholder, replacement)
    return text


def collect_models_for_task(
    task_name: str, tasks: Dict, models: Dict
) -> List[Dict[str, str]]:
    """Match the legacy Kaggle model list by exact task name."""

    if task_name not in tasks:
        return []
    category = tasks[task_name]
    if category not in models:
        return []
    matched = []
    for model_name, model_info in models[category].items():
        matched.append(
            {
                "model_name": model_name,
                "description": model_info.get("Description", ""),
                "code_template": model_info.get("Code_template", ""),
            }
        )
    return matched


def _build_guidance_text(task_name: str, tasks: Dict, models: Dict) -> str:
    """Build the original MLEvolve pretrained-model guidance verbatim in spirit."""

    model_list = collect_models_for_task(task_name, tasks, models)
    if not model_list:
        return NO_GUIDANCE
    lines = []
    for index, model in enumerate(model_list):
        lines.append(f"\nModel{index + 1}: {model['model_name']}\n")
        lines.append(f"Description:{model['description']}\n")
        lines.append(
            "Code template (MUST copy exactly — do NOT change model variant names or file paths):\n"
            "```python\n"
            + model["code_template"]
            + "\n```"
        )
    return "\n".join(lines)


def get_init_solution_paths(exp_id: str) -> List[str]:
    """Load init solution paths for exp_id from the optional JSON file."""

    if not INIT_SOLUTION_JSON.exists():
        return []
    try:
        data = _load_json(INIT_SOLUTION_JSON)
        paths = data.get(exp_id)
        if isinstance(paths, list):
            return [str(path) for path in paths if path]
        return []
    except Exception:
        return []


def _task_description_from_cfg(cfg: Any) -> Any:
    desc_file = _safe_get(cfg, "desc_file", None)
    if desc_file not in (None, ""):
        try:
            path = Path(str(desc_file)).expanduser()
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8")
        except OSError:
            pass

    description: Dict[str, Any] = {}
    goal = _safe_get(cfg, "goal", None)
    evaluation = _safe_get(cfg, "eval", None)
    if goal not in (None, ""):
        description["Task goal"] = goal
    if evaluation not in (None, ""):
        description["Task evaluation"] = evaluation
    return description


def _build_legacy_guidance(cfg: Any, resolver: str) -> str | None:
    coldstart = cfg.coldstart
    task_path = _resolve_config_path(_safe_get(coldstart, "task_json_path", None))
    model_path = _resolve_config_path(_safe_get(coldstart, "model_json_path", None))
    if task_path is None or model_path is None:
        return None
    try:
        tasks = _load_json(task_path)
        exp_id = str(_safe_get(cfg, "exp_id", "") or "")
        if exp_id not in tasks:
            return None
        models = _load_json(model_path)
        return _replace_common_placeholders(
            _build_guidance_text(exp_id, tasks, models), cfg
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        if resolver == "legacy":
            raise
        return None


def build_guidance_description(cfg: Any, task_desc: Any = None) -> str:
    """Resolve cold-start guidance while preserving legacy Kaggle behavior.

    Resolver modes:
      - ``auto``: exact legacy task mapping first, semantic scientific recipe second.
      - ``legacy``: original task-ID/category/model mapping only.
      - ``semantic`` or ``naturebench``: scientific signature routing only.

    Scientific routing reads only the public task description and visible path
    names. It does not inspect file contents, evaluator metadata, task IDs, or
    source papers.
    """

    coldstart = cfg.coldstart
    _safe_set(coldstart, "selected_recipe_id", "")
    _safe_set(coldstart, "selected_recipe_score", 0.0)
    _safe_set(coldstart, "selected_recipe_margin", 0.0)
    _safe_set(coldstart, "selected_recipe_confidence", "")
    _safe_set(coldstart, "selected_recipe_evidence", [])
    resolver = str(_safe_get(coldstart, "resolver", "auto") or "auto").casefold()
    if resolver not in {"auto", "legacy", "semantic", "naturebench"}:
        raise ValueError(
            "coldstart.resolver must be one of: auto, legacy, semantic, naturebench"
        )

    if resolver in {"auto", "legacy"}:
        legacy = _build_legacy_guidance(cfg, resolver)
        if legacy is not None:
            return legacy
        if resolver == "legacy":
            return NO_GUIDANCE

    recipe_path = _resolve_config_path(
        _safe_get(coldstart, "recipe_json_path", None), DEFAULT_RECIPE_JSON
    )
    asset_path = _resolve_config_path(
        _safe_get(coldstart, "asset_manifest_path", None), DEFAULT_ASSET_MANIFEST
    )
    if recipe_path is None or asset_path is None:
        return NO_GUIDANCE

    try:
        recipes = _load_json(recipe_path)
        assets = _load_json(asset_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        if bool(_safe_get(coldstart, "strict_assets", False)):
            raise
        return NO_GUIDANCE

    if task_desc is None:
        task_desc = _task_description_from_cfg(cfg)
    max_inventory_files = int(
        _safe_get(coldstart, "max_inventory_files", 256) or 256
    )
    inventory = collect_visible_path_inventory(
        _safe_get(cfg, "data_dir", None), max_files=max_inventory_files
    )
    match = resolve_recipe(
        task_description=task_desc,
        inventory=inventory,
        recipe_database=recipes,
        recipe_override=str(
            _safe_get(coldstart, "recipe_override", "") or ""
        ),
    )
    if match is None:
        return NO_GUIDANCE

    _safe_set(coldstart, "selected_recipe_id", match.recipe_id)
    _safe_set(coldstart, "selected_recipe_score", float(match.score))
    _safe_set(coldstart, "selected_recipe_margin", float(match.margin))
    _safe_set(coldstart, "selected_recipe_confidence", match.confidence)
    _safe_set(coldstart, "selected_recipe_evidence", list(match.evidence))

    guidance = render_recipe_guidance(
        match=match,
        recipe_database=recipes,
        asset_manifest=assets,
        pretrain_model_dir=(
            _safe_get(cfg, "pretrain_model_dir", "")
            or _safe_get(cfg, "torch_hub_dir", "")
        ),
        prior_tier=str(_safe_get(coldstart, "prior_tier", "general") or "general"),
        strict_assets=bool(_safe_get(coldstart, "strict_assets", False)),
    )
    return _replace_common_placeholders(guidance, cfg)


__all__ = [
    "SCIENTIFIC_GUIDANCE_MARKER",
    "build_guidance_description",
    "collect_models_for_task",
    "get_init_solution_paths",
    "is_scientific_guidance",
]
