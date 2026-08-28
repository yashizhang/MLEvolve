# NatureBench Kimi K3-256K End-to-End Smoke Report

## Verdict

**PASS.** One live end-to-end NatureBench-lite run (case `s41467-025-65557-7`, SMLM
point-cloud clustering) completed through the canonical comparison bridge with the
solver model exactly `k3-256k` at `reasoning_effort: low`, the scientific cold start
enabled, semantic recipe routing without override, one Kimi-generated candidate
executed through `naturebench_bridge.solver_client`, and a finite official metric
(`aggregate_improvement = -0.03523989460483358`) parsed and saved. No fallback,
retry, or substitution of model/provider/effort occurred anywhere.

## Repositories and provenance

- MLEvolve checkout: `/gpfs/projects/AI4D/core-132/yashi/naturebench-lite-framework-comparison-20260825/sources/MLEvolve`
  - branch `feature/naturebench-scientific-coldstart`
  - starting HEAD `383737e19dfa41a813598b1150d7fef06c846e66` (working tree clean)
  - final HEAD: see `git_status_after.txt` / `final_diff.patch` (one new commit)
- Bridge (`naturebench_bridge`): `/gpfs/projects/AI4D/core-132/yashi/naturebench-lite-framework-comparison-20260825/common`
  - branch `main`, starting HEAD `ad0a073` (clean), one new additive commit
- Environment: MLEvolve repo venv (Python 3.12.12) for tests; harness common venv;
  node `trixie-cn113` with 4x Tesla V100-SXM2-32GB (idle at launch), apptainer 1.5.x, bwrap.
- Canonical harness commands recorded in `commands.log`; the launch mirrors
  `common/slurm/run_framework_smoke_array.sbatch` but invokes
  `naturebench_bridge.worker` directly with `--solver-llm kimi`,
  `--case-id s41467-025-65557-7`, `--mode smoke`, budget 3600 s, candidate timeout 1200 s.

## Were code changes required?

Yes. The pre-existing code routed *every* NatureBench request through the Luna
Responses relay and hard-asserted `gpt-5.6-luna`/`xhigh`; the sandbox also has no
network, so a Kimi call required both solver-side provider selection and a
host-held relay. All changes are additive; the Luna/xhigh path is byte-identical
in behavior and remains the default.

### MLEvolve changes

- `llm/model_profiles.py` — treat `k3`/`k3-256k` as Kimi-family (profile,
  thinking extra-body, `supports_tool_choice_required`); new `is_kimi_family`.
- `llm/openai.py` — explicit provider selection in NatureBench mode via
  `provider_for_model`: `gpt-5.6-luna` keeps the exact Luna relay path;
  `k3-256k` uses the normal OpenAI-compatible client (direct `base_url`/`api_key`
  on the host, or the mounted `NATUREBENCH_KIMI_SOCKET` Unix relay inside the
  sandbox via an httpx UDS transport). The NatureBench prompt adaptation and
  execution contract are applied before any Kimi request. The configured
  `StageConfig.reasoning_effort` is sent in the request body (`extra_body`) for
  every request type: streamed generation, tool/function calls, and
  structured-output requests. Kimi temperature behavior (`temperature=1.0`,
  `top_p=0.95`) comes from the existing model profile.
- `naturebench_adapter.py` — `assert_startup_config` now validates the selected
  profile: Luna (`gpt-5.6-luna`/`xhigh`, requires `NATUREBENCH_LUNA_SOCKET`) and
  Kimi (`k3-256k`/`low`, requires `NATUREBENCH_KIMI_SOCKET` *or* configured
  credentials, never the Luna socket); mixed models/efforts, unknown models, and
  missing credentials fail early. Provider audit records actual provider/model/
  effort with `automatic_fallback: false`, `web_search: false`, no secrets.
  `materialize_candidate_command` metadata records the configured model/effort.
- `config/naturebench_kimi_k3_256k_smoke.yaml` — new tracked smoke profile:
  NatureBench task mode + scientific cold start (`use_coldstart: true`,
  `resolver: semantic`, `prior_tier: general`, `recipe_override: ""`,
  `strict_assets: false`, `max_inventory_files: 256`) + Kimi roles
  (`k3-256k`, `temp: null`, `reasoning_effort: low`, env-interpolated
  base URL with documented default, env-interpolated key) + one-step budget
  (`steps: 1`, `initial_drafts: 1`, `time_limit: 2400`, `exec.timeout: 1200`,
  `parallel_search_num: 1`, `num_gpus: 1`, `num_drafts: 1`, memory/stepwise/
  evolution/fusion/aggregation disabled for the smoke only).
  `config/naturebench.yaml` is unchanged.
- `tests/test_naturebench_kimi.py` — 18 new mocked/live-routing regression tests;
  `tests/test_naturebench_adapter.py` — candidate-command fixture now carries the
  configured stage (metadata is read from config, not constants).

### Bridge (common) changes

- `src/naturebench_bridge/kimi_relay.py` (new) — host-held Unix-socket relay for
  `POST /v1/chat/completions` → `$KIMI_BASE_URL/chat/completions`
  (default `https://api.kimi.com/coding/v1`). Holds the Kimi Code OAuth
  credentials host-side only (locked, rotation-aware refresh against
  `https://auth.kimi.com/api/oauth/token`; tokens persisted atomically with
  0600 and never logged). Enforces `model == k3-256k` and
  `reasoning_effort == low`, function-only tools, and writes a redacted
  `naturebench.kimi-request-audit` JSONL (same field names as the Luna audit).
- `src/naturebench_bridge/worker.py` — new `--solver-llm {luna,kimi}` flag
  (default `luna`, scored-mode frozen checks untouched). Kimi mode starts the
  Kimi relay instead of the Luna relay, mounts `/run/naturebench/kimi.sock`,
  sets `NATUREBENCH_KIMI_SOCKET`, records `k3-256k`/`low` in the run summary, and
  `_usage_summary` merges both audit files with per-provider contract checks.
- `tests/test_kimi_relay.py` — 11 new tests (policy rejections, redacted audit,
  streaming passthrough, 401 force-refresh, usage-summary merge).

## Test results

| Suite | Result |
|---|---|
| pre-change baseline `pytest tests/test_naturebench_coldstart.py tests/test_naturebench_adapter.py` | 25 passed |
| post-change `pytest tests/` (MLEvolve) | 43 passed |
| post-change `pytest tests/` (common bridge) | 53 passed |
| `compileall` both repos, `git diff --check` both repos | clean |

## Live probes (Phase 4)

Through the Kimi-configured `llm.openai` backend (`provider_probe.json`):

- Probe A (streamed `generate`): request `model=k3-256k`, `stream=true`,
  `extra_body.reasoning_effort=low`, `temperature=1.0`; provider response
  `model=k3-256k`, 35 chunks, nonempty text ("SMOKE OK").
- Probe B (`query` with `func_spec`): non-streaming, `model=k3-256k`,
  `reasoning_effort=low`, function `submit_metric`; provider returned
  `finish_reason=tool_calls`, parsed `{"metric_value": 0.25}`, response
  `model=k3-256k`, usage 519 in / 64 out (12 reasoning tokens — reasoning on at
  low effort).
- A relay-path integration check (backend → httpx UDS → `KimiRelay` →
  api.kimi.com) also returned 200 with correct audit rows before the smoke.
- No credential appears in any probe artifact (asserted programmatically).

## Live smoke run (Phase 5)

- Run root: `outputs/naturebench_lite_framework_comparison/preflight/framework-smoke/mlevolve-kimi-k3-smlm`
- Redacted resolved config: `resolved_config_redacted.yaml` (matches the fixed
  condition: both roles `k3-256k`/`low`, cold start semantic/general, no override,
  steps=1, initial_drafts=1, parallel_search_num=1).
- Startup audit (in-sandbox): provider `kimi-openai-compatible`, allowlists
  `[k3-256k]` / `[low]`, `automatic_fallback: false`, `web_search: false`
  (`provider_audit.json`).
- Semantic routing from the unmodified public README + visible problem paths:
  `smlm_point_cloud_clustering`, score 15.0, margin 15.0, confidence `low`
  (weakest of the six cases, as expected), evidence: single-molecule
  localization / point-cloud input / spatial clustering (`route_provenance.json`).
  Rendered guidance begins with `## Scientific cold-start prior`.
- Solver LLM usage: 3 requests, 3× HTTP 200, 0 retries, 0 infrastructure errors,
  models `[k3-256k]`, efforts `[low]`, `model_contract_valid: true`,
  7,992 in / 2,029 out tokens (run summary `solver_llm_usage`).
- Candidate: one draft (node `55de1e29`), nonempty 211-line `run.py`, committed
  into a fresh worktree (commit `75d0f275`), executed through
  `naturebench_bridge.solver_client` in the networkless candidate sandbox
  (offline: no requests/urllib/socket/download references; DATA_DIR/OUTPUT_DIR
  contract used).
- Official result: bridge payload `naturebench.framework-candidate-result` with
  `metric_value = -0.03523989460483358` (`aggregate_improvement`, per-instance
  improvements present); the same finite value is saved in the MLEvolve journal
  (`candidate_result.json`, `run_summary.json`).
- Termination: `framework_stop`, returncode 0, worker status `COMPLETE`,
  `administrative_errors: []`. Effective solve time 412 s of 3600 s budget;
  literal solver wall-clock 969 s; total worker wall-clock ≈ 31 min
  (image/rootfs reuse, candidate execution, fresh final evaluation).
- The negative score is a valid official result (candidate slightly below the
  reference baseline); per the goal this is an integration smoke, not a
  performance result.

## Fallback / retry statement

No automatic provider, model, or effort fallback exists or occurred. Zero
transport retries. Every recorded request is `k3-256k` + `low`.

## Secret hygiene

Kimi OAuth tokens live only in `~/.kimi-code/credentials/kimi-code.json`
(0600, host-side). No key or token is written to any tracked file, config,
command line, log, audit, or report artifact; this was verified by scanning all
report artifacts for the live token material. The sandboxed solver only ever
sees the placeholder `naturebench-kimi-relay`.

## Remaining concerns before broader experiments

1. **Token lifetime vs. run length**: the Kimi Code OAuth access token lives
   900 s and the refresh token rotates on every refresh. The relay handles
   refresh under a lock, but the Kimi CLI and the relay can race a rotation;
   the provider re-reads the file after locking, which recovers in practice.
   For long scored campaigns, watch `kimi-relay-audit.jsonl` for 401s.
2. **SMLM routing confidence is `low`** (score/margin 15.0) — correct but weak;
   semantic routing should be monitored on any rerun.
3. Kimi smoke support is **smoke-mode only** by design: scored mode still
   requires the frozen Luna/xhigh manifest, so the production comparison is
   unaffected; running Kimi scored would need a deliberate new frozen manifest.
