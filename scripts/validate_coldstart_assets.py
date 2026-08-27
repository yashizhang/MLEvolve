#!/usr/bin/env python3
"""Validate local scientific cold-start assets and print machine-readable JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.coldstart.naturebench import resolve_asset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-model-dir", default="")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "engine" / "coldstart" / "naturebench_assets.json",
    )
    parser.add_argument("--require", action="append", default=[])
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    asset_ids = sorted(manifest.get("assets", {}))
    results = {}
    for asset_id in asset_ids:
        status = resolve_asset(asset_id, manifest, args.pretrain_model_dir)
        results[asset_id] = {
            "available": status.available,
            "path": str(status.path) if status.path else None,
            "configured_by": status.configured_by,
            "expected_locations": list(status.expected_locations),
            "reason": status.reason,
        }

    missing_required = [
        asset_id
        for asset_id in args.require
        if not results.get(asset_id, {}).get("available", False)
    ]
    payload = {
        "schema": "mlevolve.scientific-coldstart-assets.v1",
        "pretrain_model_dir": args.pretrain_model_dir,
        "assets": results,
        "missing_required": missing_required,
        "ok": not missing_required,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
