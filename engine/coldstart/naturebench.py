"""Semantic cold-start routing for heterogeneous scientific ML tasks.

The router deliberately uses only the visible task description and a shallow
inventory of visible file names.  It never keys on NatureBench case IDs, paper
DOIs, hidden evaluator metadata, or source-method names.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCIENTIFIC_GUIDANCE_MARKER = "## Scientific cold-start prior"
NO_GUIDANCE = "None model"
TIER_ORDER = {"general": 0, "specialist": 1, "oracle": 2}
_MAX_DESCRIPTION_CHARS = 100_000


@dataclass(frozen=True)
class RecipeMatch:
    """A deterministic semantic match between a task and a cold-start recipe."""

    recipe_id: str
    title: str
    score: float
    runner_up_score: float
    evidence: Tuple[str, ...]
    minimum_score: float
    minimum_margin: float
    overridden: bool = False

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score

    @property
    def confidence(self) -> str:
        if self.overridden:
            return "explicit override"
        score_surplus = self.score - self.minimum_score
        margin_surplus = self.margin - self.minimum_margin
        if score_surplus >= 8 and margin_surplus >= 3:
            return "high"
        if score_surplus >= 3 and margin_surplus >= 1:
            return "medium"
        return "low"


@dataclass(frozen=True)
class AssetStatus:
    """Resolved status of one locally prestaged cold-start asset."""

    asset_id: str
    available: bool
    path: Optional[Path]
    configured_by: str
    expected_locations: Tuple[str, ...]
    reason: str


def is_scientific_guidance(text: Any) -> bool:
    """Return whether guidance was emitted by the scientific recipe router."""

    return isinstance(text, str) and text.lstrip().startswith(SCIENTIFIC_GUIDANCE_MARKER)


def flatten_task_description(value: Any) -> str:
    """Convert string/dict/list task descriptions into stable searchable text."""

    parts: List[str] = []

    def visit(item: Any, prefix: str = "") -> None:
        if item is None:
            return
        if isinstance(item, Mapping):
            for key in sorted(item, key=lambda x: str(x)):
                key_text = str(key)
                parts.append(f"{prefix}{key_text}:")
                visit(item[key], prefix=f"{prefix}{key_text}. ")
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child, prefix=prefix)
            return
        parts.append(f"{prefix}{item}")

    visit(value)
    return "\n".join(parts)[:_MAX_DESCRIPTION_CHARS]


def collect_visible_path_inventory(root: Any, max_files: int = 256) -> List[str]:
    """Collect relative visible path names without reading files or following symlinks.

    The inventory is intentionally shallow in information content: it gives the
    resolver modality hints such as ``rna_matrix.npz`` or ``backbone.pdb`` while
    preserving the task's data boundary.
    """

    if root in (None, "") or max_files <= 0:
        return []
    try:
        root_path = Path(root).expanduser()
    except TypeError:
        return []
    if not root_path.exists() or root_path.is_symlink():
        return []
    if root_path.is_file():
        return [root_path.name]

    results: List[str] = []
    try:
        for current, dirnames, filenames in os.walk(root_path, followlinks=False):
            current_path = Path(current)
            # Never recurse through a symlink, even when it points back inside root.
            dirnames[:] = [
                name
                for name in sorted(dirnames)
                if not (current_path / name).is_symlink()
            ]
            for filename in sorted(filenames):
                candidate = current_path / filename
                if candidate.is_symlink():
                    continue
                try:
                    rel = candidate.relative_to(root_path).as_posix()
                except ValueError:
                    continue
                results.append(rel)
                if len(results) >= max_files:
                    return results
    except OSError:
        return results
    return results


def _safe_search(pattern: str, corpus: str) -> bool:
    try:
        return re.search(pattern, corpus, flags=re.IGNORECASE | re.MULTILINE) is not None
    except re.error:
        return pattern.casefold() in corpus.casefold()


def _corpus_for_source(source: str, text: str, paths: str) -> str:
    normalized = (source or "both").casefold()
    if normalized == "text":
        return text
    if normalized == "paths":
        return paths
    return f"{text}\n{paths}"


def _group_satisfied(group: Any, text: str, paths: str) -> bool:
    if isinstance(group, Mapping):
        patterns = group.get("any", [])
        source = str(group.get("source", "both"))
    else:
        patterns = group
        source = "both"
    if isinstance(patterns, str):
        patterns = [patterns]
    corpus = _corpus_for_source(source, text, paths)
    return any(_safe_search(str(pattern), corpus) for pattern in patterns)


def _score_recipe(
    recipe: Mapping[str, Any], task_text: str, path_text: str
) -> Optional[Tuple[float, Tuple[str, ...]]]:
    match_cfg = recipe.get("match", {})
    for group in match_cfg.get("required_groups", []):
        if not _group_satisfied(group, task_text, path_text):
            return None

    score = 0.0
    evidence: List[str] = []
    for rule in match_cfg.get("patterns", []):
        if isinstance(rule, str):
            rule = {"pattern": rule, "weight": 1.0}
        pattern = str(rule.get("pattern", ""))
        if not pattern:
            continue
        source = str(rule.get("source", "both"))
        corpus = _corpus_for_source(source, task_text, path_text)
        if _safe_search(pattern, corpus):
            weight = float(rule.get("weight", 1.0))
            score += weight
            evidence.append(str(rule.get("label", pattern)))

    for rule in match_cfg.get("negative_patterns", []):
        if isinstance(rule, str):
            rule = {"pattern": rule, "weight": 1.0}
        pattern = str(rule.get("pattern", ""))
        if not pattern:
            continue
        source = str(rule.get("source", "both"))
        corpus = _corpus_for_source(source, task_text, path_text)
        if _safe_search(pattern, corpus):
            score -= abs(float(rule.get("weight", 1.0)))

    # Stable deduplication keeps prompt evidence compact and deterministic.
    unique_evidence = tuple(dict.fromkeys(evidence))[:8]
    return score, unique_evidence


def _recipe_mapping(database: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    recipes = database.get("recipes", database)
    if not isinstance(recipes, Mapping):
        raise ValueError("Recipe database must contain an object named 'recipes'.")
    return recipes  # type: ignore[return-value]


def resolve_recipe(
    task_description: Any,
    inventory: Iterable[str],
    recipe_database: Mapping[str, Any],
    recipe_override: str = "",
) -> Optional[RecipeMatch]:
    """Resolve one recipe using deterministic weighted semantic signatures."""

    recipes = _recipe_mapping(recipe_database)
    override = (recipe_override or "").strip()
    if override:
        if override not in recipes:
            raise KeyError(f"Unknown cold-start recipe override: {override}")
        recipe = recipes[override]
        title = str(recipe.get("title", override))
        return RecipeMatch(
            recipe_id=override,
            title=title,
            score=1_000_000.0,
            runner_up_score=0.0,
            evidence=("explicit recipe override",),
            minimum_score=0.0,
            minimum_margin=0.0,
            overridden=True,
        )

    task_text = flatten_task_description(task_description)
    path_text = "\n".join(str(path) for path in inventory)
    scored: List[Tuple[float, str, Mapping[str, Any], Tuple[str, ...]]] = []
    for recipe_id, recipe in recipes.items():
        scored_result = _score_recipe(recipe, task_text, path_text)
        if scored_result is None:
            continue
        score, evidence = scored_result
        scored.append((score, str(recipe_id), recipe, evidence))

    if not scored:
        return None
    scored.sort(key=lambda row: (-row[0], row[1]))
    score, recipe_id, recipe, evidence = scored[0]
    runner_up_score = scored[1][0] if len(scored) > 1 else 0.0
    match_cfg = recipe.get("match", {})
    minimum_score = float(match_cfg.get("minimum_score", 1.0))
    minimum_margin = float(match_cfg.get("minimum_margin", 0.0))
    if score < minimum_score or score - runner_up_score < minimum_margin:
        return None
    return RecipeMatch(
        recipe_id=recipe_id,
        title=str(recipe.get("title", recipe_id)),
        score=score,
        runner_up_score=runner_up_score,
        evidence=evidence,
        minimum_score=minimum_score,
        minimum_margin=minimum_margin,
    )


def _iter_candidate_asset_paths(
    asset: Mapping[str, Any], pretrain_model_dir: Any
) -> Iterable[Tuple[Path, str]]:
    env_var = str(asset.get("env_var", "")).strip()
    if env_var:
        env_value = os.environ.get(env_var, "").strip()
        if env_value:
            yield Path(os.path.expandvars(env_value)).expanduser(), f"environment variable {env_var}"

    if pretrain_model_dir not in (None, ""):
        root = Path(os.path.expandvars(str(pretrain_model_dir))).expanduser()
        for relative in asset.get("relative_roots", []):
            yield root / str(relative), "pretrain_model_dir"

    for raw_path in asset.get("candidate_paths", []):
        expanded = os.path.expandvars(str(raw_path))
        if "{PRETRAIN_MODEL_DIR}" in expanded:
            expanded = expanded.replace(
                "{PRETRAIN_MODEL_DIR}", str(pretrain_model_dir or "")
            )
        if expanded.strip():
            yield Path(expanded).expanduser(), "asset manifest"


def _asset_probes_pass(path: Path, asset: Mapping[str, Any]) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True

    for required in asset.get("probe_all", []):
        if not (path / str(required)).exists():
            return False

    any_paths = list(asset.get("probe_any", []))
    if any_paths and not any((path / str(probe)).exists() for probe in any_paths):
        return False

    any_globs = list(asset.get("probe_any_glob", []))
    if any_globs and not any(any(path.glob(str(pattern))) for pattern in any_globs):
        return False
    return True


def resolve_asset(
    asset_id: str,
    asset_manifest: Mapping[str, Any],
    pretrain_model_dir: Any = "",
) -> AssetStatus:
    """Resolve one local asset without downloading or touching the network."""

    assets = asset_manifest.get("assets", {})
    if asset_id not in assets:
        return AssetStatus(
            asset_id=asset_id,
            available=False,
            path=None,
            configured_by="",
            expected_locations=(),
            reason="asset is not declared in the local manifest",
        )

    asset = assets[asset_id]
    candidates = list(_iter_candidate_asset_paths(asset, pretrain_model_dir))
    for candidate, configured_by in candidates:
        if _asset_probes_pass(candidate, asset):
            return AssetStatus(
                asset_id=asset_id,
                available=True,
                path=candidate.resolve(),
                configured_by=configured_by,
                expected_locations=tuple(str(path) for path, _ in candidates),
                reason="local asset passed manifest probes",
            )

    expected = tuple(str(path) for path, _ in candidates)
    env_var = str(asset.get("env_var", "")).strip()
    if not expected and env_var:
        expected = (f"set {env_var} to a local file or directory",)
    reason = "local asset was not found; skip this arm rather than downloading it"
    return AssetStatus(
        asset_id=asset_id,
        available=False,
        path=None,
        configured_by="",
        expected_locations=expected,
        reason=reason,
    )


def _as_lines(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def _add_numbered(lines: List[str], heading: str, items: Any) -> None:
    values = _as_lines(items)
    if not values:
        return
    lines.extend(["", f"### {heading}"])
    lines.extend(f"{index}. {item}" for index, item in enumerate(values, start=1))


def _add_bullets(lines: List[str], heading: str, items: Any) -> None:
    values = _as_lines(items)
    if not values:
        return
    lines.extend(["", f"### {heading}"])
    lines.extend(f"- {item}" for item in values)


def render_recipe_guidance(
    match: RecipeMatch,
    recipe_database: Mapping[str, Any],
    asset_manifest: Mapping[str, Any],
    pretrain_model_dir: Any = "",
    prior_tier: str = "general",
    strict_assets: bool = False,
) -> str:
    """Render an actionable recipe for MLEvolve's initial-draft prompt."""

    requested_tier = (prior_tier or "general").casefold()
    if requested_tier not in TIER_ORDER:
        raise ValueError(
            f"Unknown cold-start prior tier '{prior_tier}'. Expected one of {sorted(TIER_ORDER)}."
        )
    recipes = _recipe_mapping(recipe_database)
    recipe = recipes[match.recipe_id]

    lines: List[str] = [
        SCIENTIFIC_GUIDANCE_MARKER,
        "",
        f"**Matched recipe:** {match.title} (`{match.recipe_id}`)",
        f"**Router confidence:** {match.confidence}; score={match.score:g}; margin={match.margin:g}",
    ]
    if match.evidence:
        lines.append("**Visible evidence:** " + "; ".join(match.evidence))
    lines.extend(
        [
            "",
            "This is a reusable methodological prior inferred from visible task semantics, not an answer key. "
            "Inspect the actual arrays, schemas, and output contract before choosing an arm.",
        ]
    )

    _add_bullets(lines, "Problem signature", recipe.get("problem_signature"))
    _add_numbered(lines, "Recommended launch sequence", recipe.get("launch_sequence"))
    _add_bullets(lines, "Cheap baselines", recipe.get("cheap_baselines"))

    models = []
    missing_required_assets: List[str] = []
    for model in recipe.get("models", []):
        model_tier = str(model.get("tier", "general")).casefold()
        if model_tier not in TIER_ORDER:
            raise ValueError(
                f"Recipe {match.recipe_id} uses unknown tier '{model_tier}'."
            )
        if TIER_ORDER[model_tier] > TIER_ORDER[requested_tier]:
            continue
        models.append(model)

    if models:
        lines.extend(["", f"### Candidate methods (enabled tier: {requested_tier})"])
        for index, model in enumerate(models, start=1):
            name = str(model.get("name", f"Candidate {index}"))
            tier = str(model.get("tier", "general"))
            role = str(model.get("role", "")).strip()
            lines.append(f"{index}. **{name}** [{tier}]" + (f" — {role}" if role else ""))
            for note in _as_lines(model.get("instructions")):
                lines.append(f"   - {note}")
            asset_id = str(model.get("asset", "")).strip()
            if asset_id:
                status = resolve_asset(asset_id, asset_manifest, pretrain_model_dir)
                if status.available:
                    lines.append(
                        f"   - Local asset: **available** at `{status.path}` ({status.configured_by})."
                    )
                else:
                    lines.append(f"   - Local asset: **unavailable** — {status.reason}.")
                    if status.expected_locations:
                        lines.append(
                            "   - Expected local locations: "
                            + ", ".join(f"`{item}`" for item in status.expected_locations)
                        )
                    if bool(model.get("asset_required", False)):
                        missing_required_assets.append(f"{name} ({asset_id})")

    if strict_assets and missing_required_assets:
        raise FileNotFoundError(
            "Required cold-start assets are unavailable: " + ", ".join(missing_required_assets)
        )

    _add_bullets(lines, "Small initial sweep", recipe.get("initial_sweep"))
    _add_bullets(lines, "Validation and sanity checks", recipe.get("sanity_checks"))
    _add_bullets(lines, "Resource notes", recipe.get("resource_notes"))
    _add_bullets(lines, "Failure pivots", recipe.get("failure_pivots"))

    lines.extend(
        [
            "",
            "### Non-negotiable guardrails",
            "- Run a cheap, auditable baseline before committing most of the budget to a deep model.",
            "- Use only visible task files and locally prestaged assets. Do not download models, clone repositories, or call external services from candidate code.",
            "- An unavailable checkpoint is not a reason to fail the run: skip that arm and use the listed train-from-scratch or classical fallback.",
            "- Preserve the task's exact output schema and metric semantics; do not optimize an easier proxy as the primary score.",
            "- Treat model names and sweep values as starting hypotheses. Adapt them only when visible data evidence justifies the change.",
        ]
    )
    return "\n".join(lines).strip()


__all__ = [
    "AssetStatus",
    "NO_GUIDANCE",
    "RecipeMatch",
    "SCIENTIFIC_GUIDANCE_MARKER",
    "collect_visible_path_inventory",
    "flatten_task_description",
    "is_scientific_guidance",
    "render_recipe_guidance",
    "resolve_asset",
    "resolve_recipe",
]
