# NatureBench Scientific Cold-Start Patch Report

## Repository and provenance

- Repository: `/gpfs/projects/AI4D/core-132/yashi/naturebench-lite-framework-comparison-20260825/sources/MLEvolve`
- This is the actual user-modified checkout used for `task-naturebench-lite` (branch
  `naturebench-lite-luna-xhigh`, two local commits `abfb803` + `e38df17` on top of the
  pinned upstream base). The pristine `EvolveFold/MLEvolve` clone in the workspace root
  was deliberately **not** patched.
- Original branch: `naturebench-lite-luna-xhigh`; patch branch: `feature/naturebench-scientific-coldstart`
- Original HEAD: `e38df170f1c6e670aa2d7749206a508464be0414` (pinned base `7d8403c` is an ancestor)
- Patch commit: `44b0ae83792492ecbb511ee6df8a081b62895da9` (`feat(coldstart): add semantic NatureBench scientific priors`); final HEAD is that commit plus this report-hash record commit
- Working tree before patch: clean (`diff_before.patch` is empty)
- Overlay: `mlevolve-naturebench-coldstart-v1.1.zip`, bundle version `scientific-coldstart-v1.1`,
  SHA-256 `7da182d091a34cf136369b1474eec8ac9059afd2ad3d3112ef060c785fa78589` (verified, matches GOAL.md)

## Installation mode

Normal guarded installation — **no manual merge was required** and `--force-replace` was not used.

- `engine/coldstart/knowledge.py` and `engine/coldstart/__init__.py` blob hashes matched the
  pinned upstream originals, so the installer replaced them under its guard.
- All text-patch anchors (`config/config.yaml`, `config/__init__.py`, `run.py`,
  `agents/draft_agent.py`, `agents/coder/stepwise_coder.py`) were present in the local versions.

### Conflict encountered and resolution (installer side effect, not a merge conflict)

The user's `run.py` and `config/config.yaml` use **mixed CRLF/LF line endings**. The
installer's newline handling (`read_normalized`/`write_preserving_newline`) rewrote both
files as uniform CRLF, turning ~370 untouched lines into spurious diff churn and failing
`git diff --check`. Resolution: rebuilt both files byte-identically from the pre-patch
copies with only the installer's intended hunks applied (one line in `run.py`, twelve
added lines in `config/config.yaml`, CRLF preserved to match surrounding style), and set
repo-local `git config core.whitespace cr-at-eol` (the repo already contains CRLF files).
Final diff is minimal: 7 modified files, 361 insertions(+), 67 deletions(-).

No other conflicts. Local NatureBench adapter changes (naturebench task mode, Luna/DeepSeek
backend, relay routing, executor/node changes) were not touched by the overlay and are intact.

## Test summary

| Command | Result | Duration |
|---|---|---|
| `python -m compileall -q engine/coldstart scripts tests` | PASS | ~2 s |
| `python -m json.tool` on recipes/assets/install manifest | PASS (3/3) | <1 s |
| `git diff --check` | PASS (after EOL fix + `core.whitespace cr-at-eol`) | <1 s |
| `pytest -q tests/test_naturebench_coldstart.py` (installed) | 19 passed | ~6 s |
| `pytest -q tests/` (full, incl. pre-existing `test_naturebench_adapter.py`) | 25 passed | ~5 s |
| overlay `pytest -q tests/test_naturebench_coldstart.py tests/test_installer.py` | 22 passed | <1 s |
| installer second dry-run (idempotency) | 15/15 `unchanged` | <1 s |
| installer second real pass | `git diff` byte-identical before/after | <1 s |
| legacy Kaggle regression (5 real exp IDs + unknown + placeholder) | PASS | <1 s |

Test environment note: the repo venv (`.venv`, Python 3.12) lacked pytest and the LLM-backend
deps; pytest 9.1.1 plus `backoff jsonschema funcy google-genai openai black genson humanize
numpy pandas` were installed into it with `uv` (venv is gitignored). Tests were run with
`PYTHONPATH=<comparison-root>/common/src` so `naturebench_bridge` resolves, matching the
existing harness launch scripts (`run_scored_worker.sh`). No pre-existing test failures were
observed; nothing was skipped or rewritten.

## Six-domain routing (Phase 5), condition: resolver=semantic, prior_tier=general, recipe_override="", strict_assets=false

Real public `README.md` + real agent-visible `problem/` directory per case. Gate: **6/6**.

| Case ID | Expected recipe | Selected recipe | Score | Margin | Confidence | Evidence |
|---|---|---|---|---|---|---|
| s41467-025-63418-x | spatial_multiomics_domain_identification | spatial_multiomics_domain_identification | 38.0 | 38.0 | high | spatial multi-omics wording; paired RNA and ATAC; spatial coordinates; spatial-domain objective; clustering metric; RNA-like files; ATAC-like files |
| s42256-025-01042-6 | inverse_protein_folding | inverse_protein_folding | 24.0 | 24.0 | high | see router_matrix.json |
| s41467-025-65557-7 | smlm_point_cloud_clustering | smlm_point_cloud_clustering | 15.0 | 15.0 | low | see router_matrix.json |
| s42256-024-00790-1 | xray_diffraction_denoising | xray_diffraction_denoising | 25.0 | 25.0 | high | see router_matrix.json |
| s42256-024-00815-9 | molecular_linker_design | molecular_linker_design | 31.0 | 31.0 | high | see router_matrix.json |
| s42256-022-00556-7 | irregular_temporal_sequence_modeling | irregular_temporal_sequence_modeling | 33.0 | 33.0 | high | see router_matrix.json |

Every preview output begins with `## Scientific cold-start prior`. Path inventories contain
only agent-visible relative paths, no hidden entries, no parent/absolute refs, and no
symlink traversal (`inventory_clean=true` in router_matrix.json). The SMLM case routes
correctly but at `low` confidence (margin 15) — worth watching, but no tuning was applied.

## Local asset availability (Phase 7)

No pretrained-asset root is configured anywhere in this checkout or the comparison harness
(`pretrain_model_dir: ""`). Validator run against the configured (empty) root and against an
intentionally empty directory:

| Asset | Available | Path |
|---|---|---|
| ProteinMPNN | no | — |
| ESM-IF1 | no | — |
| NAFNet | no | — |
| Restormer | no | — |
| DeLinker (specialist tier) | no | — |

With `strict_assets=false`, guidance for the asset-using recipes (inverse_protein_folding,
xray_diffraction_denoising) still renders, marks the checkpoint arms unavailable, forbids
downloading, and steers to the listed classical/train-from-scratch fallbacks. Resolution is
local-only (env var → conventional subdirectory → manifest paths). Details: `asset_report.json`.

## Legacy regression (Phase 4.5)

`legacy_regression.json`: `overall_pass=true`. For five sampled real experiment IDs from
`competition_tag_classified.json`, `resolver=legacy` output is byte-identical to the pre-patch
`build_guidance_description`; `resolver=auto` returns the exact legacy mapping; an unknown
experiment returns exactly `None model` under prepatch/legacy/auto; `{TORCH_HUB_DIR}`
placeholder replacement still works and matches the pre-patch text.

## Contamination and security (Phase 6)

`contamination_and_security.json`: `overall_pass=true` (11 checks): case-ID-only and
DOI-only descriptions do not route; unrelated and deliberately ambiguous descriptions return
exactly `None model`; recipe/asset databases contain no URLs, download/install commands,
case IDs, or DOIs; no source-method names (word-boundary scan for SPACell, CellCharter,
ProstT5, STORM, DECODE, DiffLinker, PROTAC, mTAN, Raindrop — the naive substring hit on
"Restormer" was verified to be the allowed general-tier denoising family); general-tier
linker guidance does not mention DeLinker; scientific prompt branch has no "copy exactly"
instruction; legacy prompt retains the original exact-template wording; cold-start code
(comment/docstring-stripped scan) contains no hidden-evaluator paths, metadata reads, case
IDs, or DOIs.

## Runtime smoke (Phase 8)

`runtime_smoke.json`: `overall_pass=true`, no LLM calls.

- Real `_load_cfg` path: `config/config.yaml` + `config/coldstart_naturebench.yaml` merged via
  the real `MLEVOLVE_CONFIG` overlay mechanism; profile values verified.
- `OmegaConf.structured(Config)` accepts the merged profile (all new `ColdstartConfig` fields
  type-check).
- Real public description loaded via `load_task_desc`; the exact `run.py` call
  `build_guidance_description(cfg, task_desc=task_desc)` returns marked, non-empty guidance
  and populates all five `selected_recipe_*` fields.
- Provenance survives `OmegaConf.to_container` and a save/reload round trip.
- Real `agents.draft_agent.run` (LLM boundary monkeypatched to capture the prompt) takes the
  `**Scientific Cold-Start Strategy**` branch for scientific guidance and the legacy
  `**Pretrained Model Strategy**` branch (with exact-template wording) for legacy guidance.
- `StepAgent._build_prompt` (model_design, draft stage) takes the scientific emphasis branch
  for scientific guidance and the pretrained emphasis branch for legacy guidance.

A live one-draft LLM smoke was **not** run: this goal used no API credentials. The comparison
harness's relay (`gpt-5.6-luna` via `llm/luna_responses.py`) is available in the Slurm
workers; recommend one short live draft smoke there before launching the full suite.

## Final diff hygiene

- 7 modified files (all intended), 8 new overlay files, the install manifest
  `.mlevolve_scientific_coldstart.json`, and this report directory.
- No task packages, evaluator files, checkpoints, archives, or temporary extraction
  directories added; no large binaries; `.venv` is gitignored.
- `config/config.yaml` default remains `resolver: "auto"` — ordinary Kaggle runs are
  unaffected; the NatureBench condition is activated only via `config/coldstart_naturebench.yaml`
  (or the equivalent CLI overrides).

## Remaining limitations / follow-up

1. No live LLM smoke in this goal — run one minimal draft/step against the Luna relay before
   the full six-task cold-start-enabled evaluation.
2. SMLM routing confidence is `low` (score/margin 15). Correct, but the weakest margin of the six.
3. All five optional checkpoint assets are absent on this cluster; ProteinMPNN/ESM-IF1 and
   NAFNet/Restormer arms will run as fallbacks unless a local asset cache is provisioned.
4. The installer's CRLF normalization bug (fixed here by hand) may be worth reporting/flagging
   if the overlay is ever re-applied to this checkout — a re-run is safe because the files are
   now uniform CRLF and the installer is idempotent, but a fresh overlay version should be
   dry-run first.
5. The cold-start-enabled scores are a **new evaluation condition**; do not mix them with the
   earlier cold-start-disabled MLEvolve results.
