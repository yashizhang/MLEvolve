# Scientific Cold-Start Recipes for MLEvolve

## Purpose

MLEvolve's original cold-start database maps an exact experiment ID to a broad Kaggle modality and then injects one or more pretrained-model snippets. That mechanism remains intact. This extension adds a second, deterministic resolver for heterogeneous scientific ML tasks:

```text
visible task description + visible path names
        -> semantic problem signature
        -> preprocessing + baseline + model + small sweep + sanity checks
```

The router does **not** key on a NatureBench case ID, DOI, source-paper title, hidden evaluator metadata, or source-method implementation. It reads no task-file contents during routing; the path inventory contains relative file names only and does not follow symlinks.

## Included recipes

| Recipe ID | Default starting prior | Local checkpoint families |
|---|---|---|
| `spatial_multiomics_domain_identification` | RNA PCA/SVD + ATAC LSI/SVD + spatial graph clustering | None required |
| `inverse_protein_folding` | ProteinMPNN/ESM-IF1 when available; compact geometry-aware network from scratch otherwise | ProteinMPNN, ESM-IF1 |
| `smlm_point_cloud_clustering` | DBSCAN/HDBSCAN/OPTICS, then a learned graph displacement field | None required |
| `xray_diffraction_denoising` | Residual U-Net control, NAFNet warm start, Restormer alternative | NAFNet, Restormer |
| `molecular_linker_design` | RDKit-constrained baseline + compact E(3)-equivariant conditional generator | Optional DeLinker in specialist tier |
| `irregular_temporal_sequence_modeling` | GRU-D/CfC/Neural CDE with subtype routing | None required |

Each recipe contains a required semantic signature, weighted evidence patterns, negative patterns, a launch sequence, cheap baselines, candidate methods, a deliberately small initial sweep, validity checks, resource notes, and failure pivots.

## Resolver modes

```yaml
coldstart:
  resolver: auto
```

- `auto`: preserve the original exact task-ID mapping when it exists; otherwise try semantic scientific routing.
- `legacy`: use only the original task-ID/category/model database.
- `semantic` or `naturebench`: use only the scientific signature router.

For the NatureBench-lite condition, use `semantic` so the result cannot depend on an accidental legacy competition mapping.

## Prior tiers

```yaml
coldstart:
  prior_tier: general
```

- `general`: reusable methods and specialist checkpoint families that are defensible from the visible problem signature.
- `specialist`: includes explicitly labeled task-specific priors such as DeLinker.
- `oracle`: structurally supported, but the shipped database intentionally contains no exact source-task implementations or checkpoints.

Use `general` as the primary reported condition. Report `specialist` separately when enabled.

## Local assets

Candidate code must not download models, clone repositories, or call external services. Asset resolution is local-only and checks, in order:

1. an asset-specific environment variable;
2. a conventional subdirectory under `pretrain_model_dir`;
3. any additional local path declared in the manifest.

Supported variables:

```text
PROTEINMPNN_DIR
ESM_IF1_DIR
NAFNET_DIR
RESTORMER_DIR
DELINKER_DIR
```

Validate the cache before launching a run:

```bash
python scripts/validate_coldstart_assets.py \
  --pretrain-model-dir /path/to/pretrained-assets
```

Require selected assets in an infrastructure preflight when appropriate:

```bash
python scripts/validate_coldstart_assets.py \
  --pretrain-model-dir /path/to/pretrained-assets \
  --require proteinmpnn \
  --require esm_if1
```

The default runtime setting is `strict_assets: false`: a missing checkpoint marks that arm unavailable and tells the agent to use the listed classical or train-from-scratch fallback. It does not terminate the scientific task.

## Configuration

The installer adds these fields to `ColdstartConfig` and `config/config.yaml`:

```yaml
coldstart:
  use_coldstart: true
  resolver: semantic
  task_json_path: engine/coldstart/competition_tag_classified.json
  model_json_path: engine/coldstart/models_guidance_classified.json
  recipe_json_path: engine/coldstart/naturebench_recipes.json
  asset_manifest_path: engine/coldstart/naturebench_assets.json
  prior_tier: general
  recipe_override: ""
  strict_assets: false
  max_inventory_files: 256
  selected_recipe_id: ""
  selected_recipe_score: 0.0
  selected_recipe_margin: 0.0
  selected_recipe_confidence: ""
  selected_recipe_evidence: []
```

The `selected_recipe_*` fields are populated before agent construction and are preserved in MLEvolve's saved run configuration for provenance.

`recipe_override` is a debugging and ablation tool. Ordinary evaluation should leave it blank so the router must infer the recipe from visible semantics.

A mergeable profile is included at `config/coldstart_naturebench.yaml`.

## Preview before a full run

```bash
python scripts/preview_coldstart.py \
  --description /path/to/public/problem/description.md \
  --data-dir /path/to/public/problem \
  --resolver semantic \
  --prior-tier general \
  --pretrain-model-dir /path/to/pretrained-assets
```

A successful preview begins with:

```text
## Scientific cold-start prior
```

An unknown or ambiguous task returns exactly:

```text
None model
```

That preserves MLEvolve's existing no-guidance behavior.

## Prompt integration

The original draft and stepwise prompts assume every cold-start entry is a pretrained model whose code must be copied exactly. The installer adds marker-aware prompt branches:

- legacy guidance retains the original pretrained-model wording;
- scientific guidance is presented as ranked hypotheses;
- a first draft is asked to implement one coherent serious arm plus a cheap baseline;
- classical and train-from-scratch recipes are treated as first-class recommendations;
- unavailable assets must be skipped rather than fetched remotely.

## Fair evaluation

The earlier NatureBench-lite comparison disabled MLEvolve's cold-start knowledge base. Enabling these recipes therefore defines a **new agent condition**, not a continuation of the old scores. Rerun all six tasks with one frozen configuration before comparing aggregate performance.

Recommended ablation:

```text
A. no cold start
B. semantic cold start, general tier
C. semantic cold start, specialist tier (reported separately)
```

Keep the LLM, search budget, hardware broker, wall-clock budget, seeds, task packages, evaluator, and retry policy fixed across A/B/C. Save the emitted recipe ID, match score, match margin, evidence, asset availability, and prior tier with each run.

## Security and contamination boundaries

The shipped tests assert that:

- case IDs and DOI-only text do not route;
- the recipe database contains no exact source-task method names;
- the databases contain no URLs or download/install commands;
- path inventory does not follow symlinks;
- missing assets fail safely;
- legacy exact mappings still take precedence in `auto` mode;
- the specialist model does not appear in the general-tier candidate list.

## Tests

```bash
python -m compileall -q engine/coldstart
python -m pytest -q tests/test_naturebench_coldstart.py
```
