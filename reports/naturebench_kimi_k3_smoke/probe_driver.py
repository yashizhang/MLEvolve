"""Phase 4 live Kimi probes through the MLEvolve OpenAI backend.

Runs llm.openai.generate (streamed generation) and llm.openai.query
(function/tool call) against the configured Kimi endpoint using the smoke
profile, with a local stub clock endpoint standing in for the NatureBench
evaluation clock (the probes exercise the LLM path only; candidate evaluation
is covered by the real bridge in Phase 5).

Never prints or persists credential material. Writes provider_probe.json and
the startup provider_audit.json into the report directory.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMMON_SRC = Path(os.environ["NATUREBENCH_COMMON_SRC"])
REPORT_DIR = Path(__file__).resolve().parent
SMLM_PROBLEM = Path(
    "/gpfs/projects/AI4D/core-132/yashi/naturebench-lite-runtime-20260821/"
    "task-naturebench-lite-pro/public-problems/s41467-025-65557-7/problem"
)
START_REPO = Path(
    "/gpfs/projects/AI4D/core-132/yashi/naturebench-lite-framework-comparison-20260825/"
    "outputs/naturebench_lite_framework_comparison/preflight/start-repos/s41467-025-65557-7"
)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(COMMON_SRC))


class _ClockStub(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/time_remaining"):
            payload = {
                "timeout_seconds": 2400.0,
                "elapsed_seconds": 1.0,
                "remaining_seconds": 2399.0,
                "total_paused_seconds": 0.0,
                "is_paused": False,
            }
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args) -> None:
        return


def _redact_messages(params: dict) -> dict:
    redacted = {}
    for key, value in params.items():
        if key == "messages":
            redacted[key] = [
                {"role": m.get("role"), "content_chars": len(str(m.get("content", "")))}
                for m in value
            ]
        elif key == "tools":
            redacted[key] = [
                {"type": t.get("type"), "name": t.get("function", {}).get("name")}
                for t in value
            ]
        else:
            redacted[key] = value
    return redacted


def main() -> int:
    from naturebench_bridge.kimi_relay import KimiCodeOAuthCredentialProvider

    provider = KimiCodeOAuthCredentialProvider(
        Path(os.environ.get("KIMI_CREDENTIALS_FILE", "~/.kimi-code/credentials/kimi-code.json")).expanduser()
    )
    os.environ["KIMI_API_KEY"] = provider.get()  # fresh access token, never printed
    os.environ.pop("KIMI_BASE_URL", None)  # exercise the documented default

    clock_server = HTTPServer(("127.0.0.1", 0), _ClockStub)
    threading.Thread(target=clock_server.serve_forever, daemon=True).start()
    os.environ["EVOLVEFOLD_NATUREBENCH_EVAL_URL"] = f"http://127.0.0.1:{clock_server.server_port}"
    os.environ["EVOLVEFOLD_NATUREBENCH_EVAL_TOKEN"] = "probe-clock-stub"
    os.environ["NATUREBENCH_PUBLIC_PROBLEM"] = str(SMLM_PROBLEM)
    os.environ["NATUREBENCH_MLEVOLVE_RUN_ROOT"] = str(REPORT_DIR / "probe-run-root")
    os.environ["NATUREBENCH_START_REPO"] = str(START_REPO)
    os.environ["NATUREBENCH_PROVIDER_AUDIT"] = str(REPORT_DIR / "provider_audit.json")
    os.environ["MLEVOLVE_CONFIG"] = str(REPO / "config" / "naturebench_kimi_k3_256k_smoke.yaml")

    from config import _load_cfg
    from naturebench_adapter import assert_startup_config
    import llm.openai as openai_backend
    from llm import FunctionSpec

    cfg = _load_cfg(use_cli_args=False)
    audit = assert_startup_config(cfg)

    # Observability wrapper: the real client still performs every HTTP call;
    # we only record the redacted request shape and safe response metadata.
    captured: list[dict] = []
    real_client_factory = openai_backend._openai_client

    def recording_factory(stage, cfg_arg):
        client = real_client_factory(stage, cfg_arg)
        real_create = client.chat.completions.create

        def recording_create(**params):
            started = time.monotonic()
            response = real_create(**params)
            record = {"request": _redact_messages(params), "latency_seconds": time.monotonic() - started}
            if params.get("stream"):
                chunks = []

                def wrap():
                    for chunk in chunks:
                        yield chunk

                for chunk in response:
                    chunks.append(chunk)
                final = chunks[-1] if chunks else None
                usage = getattr(final, "usage", None) if final else None
                record["response"] = {
                    "model": getattr(final, "model", None),
                    "chunk_count": len(chunks),
                    "usage": usage.model_dump() if usage is not None else None,
                }
                captured.append(record)
                return wrap()
            usage = getattr(response, "usage", None)
            record["response"] = {
                "model": getattr(response, "model", None),
                "usage": usage.model_dump() if usage is not None else None,
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
            }
            captured.append(record)
            return response

        client.chat.completions.create = recording_create
        return client

    openai_backend._openai_client = recording_factory

    report: dict = {
        "schema": "naturebench.kimi-k3-provider-probe",
        "startup_audit": audit,
        "probe_a": {},
        "probe_b": {},
    }

    # Probe A: ordinary streamed generation
    try:
        text = openai_backend.generate(
            "Reply with exactly the two words: SMOKE OK", cfg=cfg, max_tokens=2048
        )
        report["probe_a"] = {
            "ok": bool(text.strip()),
            "response_nonempty": bool(text.strip()),
            "response_excerpt": text.strip()[:120],
            **captured[-1],
        }
    except Exception as exc:
        report["probe_a"] = {"ok": False, "error": traceback.format_exc()[-1500:]}

    # Probe B: function/tool call
    func_spec = FunctionSpec(
        name="submit_metric",
        description="Submit a numeric metric value",
        json_schema={
            "type": "object",
            "properties": {"metric_value": {"type": "number"}},
            "required": ["metric_value"],
        },
    )
    try:
        output, req_time, in_tok, out_tok, info = openai_backend.query(
            system_message="You report numeric results by calling the submit_metric function.",
            user_message="The experiment scored 0.25. Report it.",
            func_spec=func_spec,
            cfg=cfg,
            model="k3-256k",
        )
        report["probe_b"] = {
            "ok": isinstance(output, dict) and "metric_value" in output,
            "parsed_output": output,
            "tokens": {"input": in_tok, "output": out_tok},
            "info": {k: v for k, v in info.items() if k in {"model", "created"}},
            **captured[-1],
        }
    except Exception as exc:
        report["probe_b"] = {"ok": False, "error": traceback.format_exc()[-1500:]}

    clock_server.shutdown()
    out = REPORT_DIR / "provider_probe.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    # Secret hygiene: the probe artifact must not contain the token.
    token = os.environ["KIMI_API_KEY"]
    assert token not in out.read_text(encoding="utf-8"), "token leaked into probe artifact"
    print(json.dumps({"probe_a_ok": report["probe_a"].get("ok"), "probe_b_ok": report["probe_b"].get("ok")}))
    return 0 if report["probe_a"].get("ok") and report["probe_b"].get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
