"""NatureBench Kimi K3-256K provider-selection regression tests (mocked clients)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import llm.openai as openai_backend
from naturebench_adapter import (
    EFFORT,
    KIMI_EFFORT,
    KIMI_MODEL,
    MODEL,
    adapt_prompt,
    assert_startup_config,
    materialize_candidate_command,
    provider_for_model,
)


ROOT = Path(__file__).resolve().parents[1]
SMLM_PROBLEM = Path(
    "/gpfs/projects/AI4D/core-132/yashi/naturebench-lite-runtime-20260821/"
    "task-naturebench-lite-pro/public-problems/s41467-025-65557-7/problem"
)


def _stage(model: str, effort: str, api_key: str = "test-key") -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        reasoning_effort=effort,
        api_key=api_key,
        base_url="https://api.kimi.com/coding/v1",
    )


def _luna_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        task_mode="naturebench",
        agent=SimpleNamespace(code=_stage(MODEL, EFFORT), feedback=_stage(MODEL, EFFORT)),
    )


def _kimi_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        task_mode="naturebench",
        agent=SimpleNamespace(
            code=_stage(KIMI_MODEL, KIMI_EFFORT), feedback=_stage(KIMI_MODEL, KIMI_EFFORT)
        ),
    )


class _FakeStreamChunk:
    def __init__(self, text: str) -> None:
        self.choices = [SimpleNamespace(delta=SimpleNamespace(content=text))]


class _FakeCompletions:
    """Captures chat.completions.create params and returns canned responses."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **params):
        self.calls.append(params)
        if params.get("stream"):
            return iter([_FakeStreamChunk("hello "), _FakeStreamChunk("kimi")])
        tools = params.get("tools") or []
        if tools:
            name = tools[0]["function"]["name"]
            tool_call = SimpleNamespace(
                function=SimpleNamespace(name=name, arguments=json.dumps({"verdict": "ok"}))
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
        else:
            message = SimpleNamespace(content="plain text", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
            model=params["model"],
            created=1,
        )


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


class KimiProviderSelectionTests(unittest.TestCase):
    def test_provider_mapping_is_explicit(self) -> None:
        self.assertEqual(provider_for_model(MODEL), "host-unix-responses-relay")
        self.assertEqual(provider_for_model(KIMI_MODEL), "kimi-openai-compatible")
        with self.assertRaises(RuntimeError):
            provider_for_model("k3")  # only the exact 256K ID is supported
        with self.assertRaises(RuntimeError):
            provider_for_model("gpt-5.5")

    def test_kimi_query_uses_openai_client_and_never_luna(self) -> None:
        completions = _FakeCompletions()
        with (
            patch.object(openai_backend, "_openai_client", return_value=_FakeClient(completions)),
            patch.dict(sys.modules, {"llm.luna_responses": None}),
        ):
            output, _, in_tok, out_tok, info = openai_backend.query(
                system_message="review the candidate",
                user_message="candidate code",
                func_spec=None,
                cfg=_kimi_cfg(),
                model=KIMI_MODEL,
            )
        self.assertEqual(output, "plain text")
        self.assertEqual((in_tok, out_tok), (3, 4))
        self.assertEqual(info["model"], KIMI_MODEL)
        self.assertEqual(len(completions.calls), 1)
        params = completions.calls[0]
        self.assertEqual(params["model"], KIMI_MODEL)
        self.assertEqual(params["extra_body"]["reasoning_effort"], KIMI_EFFORT)
        self.assertEqual(params["temperature"], 1.0)

    def test_kimi_tool_query_sends_low_effort(self) -> None:
        completions = _FakeCompletions()
        func_spec = SimpleNamespace(
            name="grade",
            as_openai_tool_dict={
                "type": "function",
                "function": {"name": "grade", "description": "d", "parameters": {}},
            },
            openai_tool_choice_dict={"type": "function", "function": {"name": "grade"}},
        )
        with patch.object(openai_backend, "_openai_client", return_value=_FakeClient(completions)):
            output, *_ = openai_backend.query(
                system_message="judge",
                user_message="code",
                func_spec=func_spec,
                cfg=_kimi_cfg(),
                model=KIMI_MODEL,
            )
        self.assertEqual(output, {"verdict": "ok"})
        params = completions.calls[0]
        self.assertEqual(params["model"], KIMI_MODEL)
        self.assertEqual(params["extra_body"]["reasoning_effort"], KIMI_EFFORT)
        # Kimi models do not support tool_choice="required"
        self.assertNotIn("tool_choice", params)

    def test_kimi_streamed_generate_sends_low_effort(self) -> None:
        completions = _FakeCompletions()
        with patch.object(openai_backend, "_openai_client", return_value=_FakeClient(completions)):
            text = openai_backend.generate("write the solver", cfg=_kimi_cfg())
        self.assertEqual(text, "hello kimi")
        params = completions.calls[0]
        self.assertEqual(params["model"], KIMI_MODEL)
        self.assertTrue(params["stream"])
        self.assertEqual(params["extra_body"]["reasoning_effort"], KIMI_EFFORT)

    def test_kimi_structured_output_keeps_model_and_effort(self) -> None:
        completions = _FakeCompletions()
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        with patch.object(openai_backend, "_openai_client", return_value=_FakeClient(completions)):
            openai_backend.generate("return json", cfg=_kimi_cfg(), json_schema=schema)
        params = completions.calls[0]
        self.assertEqual(params["model"], KIMI_MODEL)
        self.assertEqual(params["extra_body"]["reasoning_effort"], KIMI_EFFORT)
        self.assertEqual(params["response_format"]["type"], "json_schema")
        self.assertEqual(params["response_format"]["json_schema"]["schema"], schema)

    def test_kimi_prompts_carry_execution_contract(self) -> None:
        completions = _FakeCompletions()
        with patch.object(openai_backend, "_openai_client", return_value=_FakeClient(completions)):
            openai_backend.query(
                system_message="You are writing code for a Kaggle competition.",
                user_message="draft",
                func_spec=None,
                cfg=_kimi_cfg(),
                model=KIMI_MODEL,
            )
            openai_backend.generate(
                {"system": "plan", "user": "produce run.py"}, cfg=_kimi_cfg()
            )
        query_messages = completions.calls[0]["messages"]
        generate_messages = completions.calls[1]["messages"]
        self.assertIn("NATUREBENCH EXECUTION CONTRACT", query_messages[0]["content"])
        self.assertNotIn("Kaggle", query_messages[0]["content"])
        self.assertIn("NATUREBENCH EXECUTION CONTRACT", generate_messages[0]["content"])

    def test_unknown_model_fails_early(self) -> None:
        cfg = _kimi_cfg()
        cfg.agent.code.model = "gpt-4o"
        with self.assertRaises(RuntimeError):
            openai_backend.generate("hi", cfg=cfg)


class StartupProfileTests(unittest.TestCase):
    def _env(self, extra: dict[str, str]) -> dict[str, str]:
        env = {
            "NATUREBENCH_START_REPO": "/start",
            "NATUREBENCH_PUBLIC_PROBLEM": "/problem",
            "NATUREBENCH_PROVIDER_AUDIT": "",
        }
        env.update(extra)
        return env

    def test_luna_profile_still_valid(self) -> None:
        env = self._env({"NATUREBENCH_LUNA_SOCKET": "/run/naturebench/luna.sock"})
        with patch.dict(os.environ, env, clear=False):
            audit = assert_startup_config(_luna_cfg())
        self.assertEqual(audit["provider"], "host-unix-responses-relay")
        self.assertEqual(audit["model_allowlist"], [MODEL])
        self.assertEqual(audit["reasoning_effort_allowlist"], [EFFORT])
        self.assertFalse(audit["automatic_fallback"])
        self.assertFalse(audit["web_search"])

    def test_luna_profile_requires_socket(self) -> None:
        with (
            patch.dict(os.environ, self._env({}), clear=False),
            patch.dict(os.environ, {"NATUREBENCH_LUNA_SOCKET": ""}),
        ):
            with self.assertRaisesRegex(RuntimeError, "NATUREBENCH_LUNA_SOCKET"):
                assert_startup_config(_luna_cfg())

    def test_kimi_profile_valid_with_credentials(self) -> None:
        with patch.dict(os.environ, self._env({}), clear=False):
            audit = assert_startup_config(_kimi_cfg())
        self.assertEqual(audit["provider"], "kimi-openai-compatible")
        self.assertEqual(audit["model_allowlist"], [KIMI_MODEL])
        self.assertEqual(audit["reasoning_effort_allowlist"], [KIMI_EFFORT])
        self.assertNotIn("api_key", json.dumps(audit))

    def test_kimi_profile_valid_with_relay_socket(self) -> None:
        cfg = _kimi_cfg()
        cfg.agent.code.api_key = ""
        cfg.agent.feedback.api_key = ""
        env = self._env({"NATUREBENCH_KIMI_SOCKET": "/run/naturebench/kimi.sock"})
        with patch.dict(os.environ, env, clear=False):
            audit = assert_startup_config(cfg)
        self.assertEqual(audit["provider"], "kimi-openai-compatible")

    def test_kimi_profile_rejects_wrong_or_missing_effort(self) -> None:
        for effort in ("high", "max", "none", ""):
            cfg = _kimi_cfg()
            cfg.agent.code.reasoning_effort = effort
            cfg.agent.feedback.reasoning_effort = effort
            with patch.dict(os.environ, self._env({}), clear=False):
                with self.assertRaises(RuntimeError, msg=f"effort={effort!r}"):
                    assert_startup_config(cfg)

    def test_mixed_roles_rejected(self) -> None:
        cfg = _kimi_cfg()
        cfg.agent.feedback.model = MODEL
        cfg.agent.feedback.reasoning_effort = EFFORT
        env = self._env({"NATUREBENCH_LUNA_SOCKET": "/run/luna.sock"})
        with patch.dict(os.environ, env, clear=False):
            with self.assertRaises(RuntimeError):
                assert_startup_config(cfg)
        cfg = _kimi_cfg()
        cfg.agent.feedback.reasoning_effort = "high"
        with patch.dict(os.environ, self._env({}), clear=False):
            with self.assertRaises(RuntimeError):
                assert_startup_config(cfg)

    def test_unknown_model_rejected(self) -> None:
        cfg = _kimi_cfg()
        cfg.agent.code.model = "kimi-k2.5"
        cfg.agent.feedback.model = "kimi-k2.5"
        with patch.dict(os.environ, self._env({}), clear=False):
            with self.assertRaisesRegex(RuntimeError, "unsupported NatureBench solver model"):
                assert_startup_config(cfg)

    def test_kimi_profile_requires_credentials_without_socket(self) -> None:
        cfg = _kimi_cfg()
        cfg.agent.code.api_key = ""
        with patch.dict(os.environ, self._env({}), clear=False):
            with self.assertRaisesRegex(RuntimeError, "api_key"):
                assert_startup_config(cfg)

    def test_common_environment_still_required(self) -> None:
        with patch.dict(os.environ, {"NATUREBENCH_START_REPO": "", "NATUREBENCH_PUBLIC_PROBLEM": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                assert_startup_config(_kimi_cfg())


class CandidateMetadataTests(unittest.TestCase):
    def test_candidate_metadata_records_configured_model(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            starter = tmp / "starter"
            starter.mkdir()
            (starter / "run.py").write_text("pass\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=starter, check=True)
            subprocess.run(["git", "add", "run.py"], cwd=starter, check=True)
            subprocess.run(
                [
                    "git", "-c", "user.name=NatureBench Test",
                    "-c", "user.email=test@example.invalid",
                    "commit", "-q", "-m", "starter",
                ],
                cwd=starter,
                check=True,
            )
            stage = _stage(KIMI_MODEL, KIMI_EFFORT)
            cfg = SimpleNamespace(
                task_mode="naturebench",
                workspace_dir=str(tmp / "workspace"),
                agent=SimpleNamespace(
                    code=stage,
                    feedback=stage,
                    search=SimpleNamespace(num_gpus=1),
                ),
                exec=SimpleNamespace(timeout=300),
            )
            node = SimpleNamespace(
                parent=None, stage="draft", branch_id="b", step=1, from_topk=False
            )
            with (
                patch.dict(os.environ, {"NATUREBENCH_START_REPO": str(starter)}, clear=False),
                patch("naturebench_adapter.expired", return_value=False),
            ):
                command, _ = materialize_candidate_command(
                    cfg=cfg, code="print('x')\n", candidate_id="cand-1", node=node, cpu_ids=set()
                )
            metadata = json.loads(command[command.index("--metadata") + 1])
            self.assertEqual(metadata["llm_model"], KIMI_MODEL)
            self.assertEqual(metadata["reasoning_effort"], KIMI_EFFORT)


class RealSmlmRoutingTests(unittest.TestCase):
    @unittest.skipUnless(SMLM_PROBLEM.is_dir(), "real SMLM public problem not available")
    def test_real_smlm_readme_routes_to_smlm_recipe(self) -> None:
        from engine.coldstart import build_guidance_description, is_scientific_guidance

        cfg = SimpleNamespace(
            exp_id="",
            desc_file=SMLM_PROBLEM / "README.md",
            goal=None,
            eval=None,
            data_dir=SMLM_PROBLEM,
            torch_hub_dir="",
            pretrain_model_dir="",
            coldstart=SimpleNamespace(
                use_coldstart=True,
                task_json_path=str(ROOT / "engine/coldstart/competition_tag_classified.json"),
                model_json_path=str(ROOT / "engine/coldstart/models_guidance_classified.json"),
                description="",
                resolver="semantic",
                recipe_json_path=str(ROOT / "engine/coldstart/naturebench_recipes.json"),
                asset_manifest_path=str(ROOT / "engine/coldstart/naturebench_assets.json"),
                prior_tier="general",
                recipe_override="",
                strict_assets=False,
                max_inventory_files=256,
            ),
        )
        guidance = build_guidance_description(cfg)
        self.assertTrue(is_scientific_guidance(guidance))
        self.assertEqual(cfg.coldstart.selected_recipe_id, "smlm_point_cloud_clustering")
        self.assertGreater(cfg.coldstart.selected_recipe_score, 0)
        self.assertTrue(cfg.coldstart.selected_recipe_evidence)


if __name__ == "__main__":
    unittest.main()
