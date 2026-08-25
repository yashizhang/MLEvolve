from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import _load_cfg
from naturebench_adapter import (
    EFFORT,
    MODEL,
    adapt_prompt,
    parse_candidate_result,
)


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "mlevolve_luna_responses", ROOT / "llm" / "luna_responses.py"
)
assert _SPEC and _SPEC.loader
luna_responses = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = luna_responses
_SPEC.loader.exec_module(luna_responses)


def _cfg() -> SimpleNamespace:
    stage = SimpleNamespace(model=MODEL, reasoning_effort=EFFORT)
    return SimpleNamespace(
        task_mode="naturebench",
        agent=SimpleNamespace(code=stage, feedback=stage),
    )


class NatureBenchAdapterTests(unittest.TestCase):
    def test_default_mode_remains_mlebench(self) -> None:
        with patch.dict(os.environ, {"MLEVOLVE_CONFIG": ""}, clear=False):
            config = _load_cfg(use_cli_args=False)
        self.assertEqual(config.task_mode, "mlebench")
        self.assertEqual(config.evaluation_backend, "mlebench")
        self.assertEqual(config.agent.time_limit, 43_200)
        self.assertEqual(config.exec.timeout, 32_400)
        self.assertTrue(config.coldstart.use_coldstart)

    def test_frozen_overlay_scales_only_absolute_schedule(self) -> None:
        environment = {
            "MLEVOLVE_CONFIG": str(ROOT / "config" / "naturebench.yaml"),
            "NATUREBENCH_PUBLIC_PROBLEM": "/public/problem",
            "NATUREBENCH_MLEVOLVE_RUN_ROOT": "/run",
            "NATUREBENCH_MEMORY_MODEL_PATH": "/cache/bge-base-en-v1.5",
        }
        with patch.dict(os.environ, environment, clear=False):
            config = _load_cfg(use_cli_args=False)
        self.assertEqual(config.task_mode, "naturebench")
        self.assertEqual(config.evaluation_backend, "naturebench_bridge")
        self.assertEqual(config.agent.time_limit, 14_400)
        self.assertEqual(config.exec.timeout, 10_800)
        self.assertEqual(config.agent.search.fusion_min_time_hours, 2.0)
        self.assertAlmostEqual(config.agent.search.fusion_max_time_hours, 10.0 / 3.0)
        self.assertEqual(config.agent.search.parallel_search_num, 3)
        self.assertEqual(config.agent.search.num_drafts, 5)
        self.assertEqual(config.agent.search.num_improves, 3)
        self.assertEqual(config.agent.search.top_candidates_size, 20)
        self.assertEqual(list(config.agent.decay.phase_ratios), [0.3, 0.7])
        self.assertEqual(config.agent.seed, 42)
        self.assertFalse(config.coldstart.use_coldstart)
        self.assertTrue(config.agent.use_global_memory)
        self.assertEqual(config.agent.memory_embedding_device, "cpu")
        self.assertEqual(config.agent.code.model, MODEL)
        self.assertEqual(config.agent.feedback.reasoning_effort, EFFORT)

        manifest = json.loads(
            (ROOT / "config" / "naturebench_scaled_constants.json").read_text()
        )
        self.assertEqual(manifest["scale_factor"], 1.0 / 3.0)
        self.assertFalse(manifest["dimensionless_and_count_settings_changed"])

    def test_query_and_generate_emit_exact_responses_profile(self) -> None:
        captured: list[dict] = []

        def fake_stream(payload, _config):
            captured.append(payload)
            return luna_responses.StreamResult(
                text="ok",
                function_calls=(),
                input_tokens=1,
                output_tokens=2,
                model=MODEL,
            )

        with patch.object(luna_responses, "_stream", side_effect=fake_stream):
            output, _, _, _, info = luna_responses.query(
                system_message="review",
                user_message="candidate",
                func_spec=None,
                cfg=_cfg(),
            )
            generated = luna_responses.generate(
                prompt_messages=[
                    {"role": "system", "content": "write code"},
                    {"role": "user", "content": "implement"},
                ],
                cfg=_cfg(),
                json_schema=None,
                max_tokens=4096,
            )
        self.assertEqual(output, "ok")
        self.assertEqual(generated, "ok")
        self.assertEqual(info["reasoning_effort"], EFFORT)
        self.assertEqual(len(captured), 2)
        for payload in captured:
            self.assertEqual(payload["model"], MODEL)
            self.assertEqual(payload["reasoning"], {"effort": EFFORT, "summary": "auto"})
            self.assertTrue(payload["stream"])
            self.assertNotIn("temperature", payload)
            self.assertNotIn("max_output_tokens", payload)

    def test_prompt_adapter_removes_inherited_kaggle_contracts(self) -> None:
        source = (
            "Kaggle MLE-bench: write submission.csv matching sample_submission.csv. "
            "Check the leaderboard. During code development, you can and should use online resources.\n"
            "The no internet access restriction applies ONLY to submission evaluation.\n"
            "📦 **Packages & Internet**: torch.hub.load(), HuggingFace, etc. available during development.\n"
            "**Do NOT question internet access concerns - all pretrained models are available."
        )
        adapted = adapt_prompt(source, _cfg()).casefold()
        self.assertIn("data_dir", adapted)
        self.assertIn("output_dir", adapted)
        self.assertNotIn("kaggle", adapted)
        self.assertNotIn("mle-bench", adapted)
        self.assertNotIn("submission.csv", adapted)
        self.assertNotIn("sample_submission", adapted)
        self.assertNotIn("leaderboard", adapted)
        self.assertNotIn("online resources", adapted)
        self.assertNotIn("available during development", adapted)
        self.assertNotIn("applies only", adapted)

    def test_direct_official_result_parsing(self) -> None:
        payload = {
            "schema": "naturebench.framework-candidate-result",
            "metric_value": -0.125,
            "metrics": {"aggregate_improvement": -0.125},
            "candidate_root": "/immutable/candidate",
        }
        parsed = parse_candidate_result("diagnostic\n" + json.dumps(payload) + "\n")
        self.assertEqual(parsed, payload)


if __name__ == "__main__":
    unittest.main()
