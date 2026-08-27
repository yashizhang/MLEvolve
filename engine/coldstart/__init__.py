"""Cold-start guidance for legacy competitions and scientific ML tasks."""

from .knowledge import (
    SCIENTIFIC_GUIDANCE_MARKER,
    build_guidance_description,
    get_init_solution_paths,
    is_scientific_guidance,
)
from .naturebench import (
    AssetStatus,
    RecipeMatch,
    collect_visible_path_inventory,
    flatten_task_description,
    render_recipe_guidance,
    resolve_asset,
    resolve_recipe,
)

__all__ = [
    "AssetStatus",
    "RecipeMatch",
    "SCIENTIFIC_GUIDANCE_MARKER",
    "build_guidance_description",
    "collect_visible_path_inventory",
    "flatten_task_description",
    "get_init_solution_paths",
    "is_scientific_guidance",
    "render_recipe_guidance",
    "resolve_asset",
    "resolve_recipe",
]
