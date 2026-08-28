#!/usr/bin/env bash
# One-shot NatureBench Kimi K3-256K smoke launcher (case s41467-025-65557-7).
# Modeled on common/slurm/run_framework_smoke_array.sbatch but runs the bridge
# worker directly on this 4xV100 node with --solver-llm kimi.
# No secrets are placed in this file or on the command line; the Kimi relay
# reads OAuth credentials from KIMI_CREDENTIALS_FILE on the host side only.
set -euo pipefail

comparison_root=${NBFC_ROOT:-/gpfs/projects/AI4D/core-132/yashi/naturebench-lite-framework-comparison-20260825}
comparison_root=$(realpath -e "$comparison_root")
output_root="$comparison_root/outputs/naturebench_lite_framework_comparison"
case_id=s41467-025-65557-7
framework_source="$comparison_root/sources/MLEvolve"
framework_config="$output_root/configs/smoke/mlevolve_kimi_k3_256k_smoke.yaml"
run_root="$output_root/preflight/framework-smoke/mlevolve-kimi-k3-smlm"

common_python="$comparison_root/common/.venv/bin/python"
common_source="$comparison_root/common/src"
common_repo="$comparison_root/common"
catalog="$output_root/configs/task_catalog.json"
start_repos="$output_root/preflight/start-repos"
trial_config="$output_root/preflight/trial-config.json"
upstream_root=/gpfs/projects/AI4D/core-132/yashi/naturebench-lite-runtime-20260821/NatureBench-upstream
judge_python=/gpfs/projects/AI4D/core-132/yashi/EvolveFold/AutoScientists/.venv/bin/python
evolvefold_source=/gpfs/projects/AI4D/core-132/yashi/EvolveFold/AutoScientists/src
codex_home=${CODEX_HOME:-$HOME/.codex}
memory_model="$output_root/environment/models/bge-base-en-v1.5"
framework_env="$output_root/environment/overlays/mlevolve"
package_lock="$output_root/environment/mlevolve-software-lock.txt"

for required in \
  "$common_python" "$common_source" "$common_repo" "$catalog" \
  "$start_repos/$case_id" "$trial_config" "$upstream_root" \
  "$judge_python" "$evolvefold_source" "$codex_home/auth.json" \
  "$framework_source" "$framework_env" "$framework_config" "$package_lock" "$memory_model"; do
  [[ -e "$required" ]] || { echo "Missing smoke prerequisite: $required" >&2; exit 2; }
done

if [[ -e "$run_root" && -n "$(find "$run_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Smoke run root is not empty: $run_root" >&2
  exit 2
fi
mkdir -p "$run_root"

# Reuse the shared rootfs cache (the SMLM image is pre-extracted there).
cache_root="$comparison_root/cache"
mkdir -p "$cache_root"/{apptainer-cache,apptainer-tmp,tmp}
export APPTAINER_CACHEDIR="$cache_root/apptainer-cache"
export APPTAINER_TMPDIR="$cache_root/apptainer-tmp"
export TMPDIR="$cache_root/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

adapter_commit=$(git -C "$framework_source" rev-parse HEAD)
common_commit=$(git -C "$common_repo" rev-parse HEAD)

command=(
  "$common_python" -m naturebench_bridge.worker
  --framework mlevolve
  --case-id "$case_id"
  --mode smoke
  --solver-llm kimi
  --catalog "$catalog"
  --run-root "$run_root"
  --framework-source "$framework_source"
  --framework-env "$framework_env"
  --framework-config "$framework_config"
  --common-repo "$common_repo"
  --common-source "$common_source"
  --start-repos-root "$start_repos"
  --trial-config "$trial_config"
  --upstream-root "$upstream_root"
  --codex-home "$codex_home"
  --cache-root "$cache_root"
  --package-lock "$package_lock"
  --judge-python "$judge_python"
  --evolvefold-source "$evolvefold_source"
  --upstream-commit 7d8403c899c40f01941c0429f1c4ef51e82ae41c
  --adapter-commit "$adapter_commit"
  --common-commit "$common_commit"
  --experiment-version nbfc-kimi-k3-smoke
  --budget-seconds 3600
  --candidate-timeout-seconds 1200
  --memory-model "$memory_model"
  --verify-image-hash
  --require-four-v100
)

printf 'Launching:'
printf ' %q' "${command[@]}"
printf '\n'
PYTHONPATH="$common_source" "${command[@]}"
