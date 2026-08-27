#!/usr/bin/env python3
"""Preview MLEvolve cold-start guidance without launching a search run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.coldstart import build_guidance_description  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--exp-id", default="")
    parser.add_argument("--resolver", choices=["auto", "legacy", "semantic", "naturebench"], default="semantic")
    parser.add_argument("--prior-tier", choices=["general", "specialist", "oracle"], default="general")
    parser.add_argument("--pretrain-model-dir", default="")
    parser.add_argument("--recipe-override", default="")
    parser.add_argument("--strict-assets", action="store_true")
    parser.add_argument("--max-inventory-files", type=int, default=256)
    parser.add_argument(
        "--legacy-task-json",
        default="engine/coldstart/competition_tag_classified.json",
    )
    parser.add_argument(
        "--legacy-model-json",
        default="engine/coldstart/models_guidance_classified.json",
    )
    parser.add_argument(
        "--recipe-json",
        default="engine/coldstart/naturebench_recipes.json",
    )
    parser.add_argument(
        "--asset-manifest",
        default="engine/coldstart/naturebench_assets.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.description.is_file():
        raise SystemExit(f"Description file does not exist: {args.description}")
    if not args.data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {args.data_dir}")

    coldstart = SimpleNamespace(
        use_coldstart=True,
        task_json_path=args.legacy_task_json,
        model_json_path=args.legacy_model_json,
        description="",
        resolver=args.resolver,
        recipe_json_path=args.recipe_json,
        asset_manifest_path=args.asset_manifest,
        prior_tier=args.prior_tier,
        recipe_override=args.recipe_override,
        strict_assets=args.strict_assets,
        max_inventory_files=args.max_inventory_files,
    )
    cfg = SimpleNamespace(
        exp_id=args.exp_id,
        desc_file=args.description,
        goal=None,
        eval=None,
        data_dir=args.data_dir,
        torch_hub_dir="",
        pretrain_model_dir=args.pretrain_model_dir,
        coldstart=coldstart,
    )
    print(build_guidance_description(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
