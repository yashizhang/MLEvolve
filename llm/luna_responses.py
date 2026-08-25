"""Strict Luna/xhigh Responses client over the host-only NatureBench Unix relay."""

from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Any

from naturebench_adapter import EFFORT, MODEL, llm_timeout_seconds


logger = logging.getLogger("MLEvolve")


class LunaResponsesError(RuntimeError):
    pass


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, *, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        connection.connect(self.socket_path)
        self.sock = connection


@dataclass(frozen=True)
class StreamResult:
    text: str
    function_calls: tuple[dict[str, Any], ...]
    input_tokens: int
    output_tokens: int
    model: str


def _tool(func_spec: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "name": func_spec.name,
        "description": func_spec.description,
        "parameters": func_spec.json_schema,
        "strict": False,
    }


def build_request(
    *,
    system_message: str | None,
    user_message: str | None,
    reasoning_effort: str,
    func_spec: Any = None,
    json_schema: dict[str, Any] | None = None,
    max_tokens: int = 16384,
) -> dict[str, Any]:
    if reasoning_effort != EFFORT:
        raise LunaResponsesError("MLEvolve NatureBench requests must use reasoning_effort=xhigh")
    input_items: list[dict[str, str]] = []
    if user_message:
        input_items.append({"role": "user", "content": user_message})
    elif system_message:
        input_items.append(
            {"role": "user", "content": "Complete the task specified in the instructions."}
        )
    else:
        raise ValueError("Either system_message or user_message must be provided")
    payload: dict[str, Any] = {
        "model": MODEL,
        "instructions": system_message or "",
        "input": input_items,
        "reasoning": {"effort": EFFORT, "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "store": False,
        "stream": True,
    }
    if func_spec is not None:
        payload["tools"] = [_tool(func_spec)]
        payload["tool_choice"] = {"type": "function", "name": func_spec.name}
        payload["parallel_tool_calls"] = False
    if json_schema is not None:
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": "structured_output",
                "schema": json_schema,
                "strict": False,
            }
        }
    else:
        payload["text"] = {"verbosity": "medium"}
    return payload


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _message_text(item: dict[str, Any]) -> str:
    content = item.get("content", [])
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "".join(parts)


def _stream(payload: dict[str, Any], cfg: Any) -> StreamResult:
    socket_path = os.environ.get("NATUREBENCH_LUNA_SOCKET", "").strip()
    if not socket_path or not os.path.isabs(socket_path):
        raise LunaResponsesError("NATUREBENCH_LUNA_SOCKET must name an absolute Unix socket")
    timeout = llm_timeout_seconds(cfg)
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    connection = _UnixConnection(socket_path, timeout=timeout)
    started = time.monotonic()
    try:
        connection.request(
            "POST",
            "/v1/responses",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            error = response.read(1024 * 1024).decode("utf-8", errors="replace")
            raise LunaResponsesError(f"Luna relay returned HTTP {response.status}: {error[:2000]}")
        text_deltas: list[str] = []
        message_items: list[str] = []
        function_calls: list[dict[str, Any]] = []
        input_tokens = 0
        output_tokens = 0
        response_model = MODEL
        completed = False
        while True:
            line = response.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == b"[DONE]":
                continue
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "response.output_text.delta" and isinstance(event.get("delta"), str):
                text_deltas.append(event["delta"])
            elif event_type == "response.output_item.done":
                item = _as_dict(event.get("item"))
                if item.get("type") == "function_call":
                    function_calls.append(item)
                elif item.get("type") == "message" and not text_deltas:
                    item_text = _message_text(item)
                    if item_text:
                        message_items.append(item_text)
            elif event_type == "response.completed":
                completed = True
                final = _as_dict(event.get("response"))
                usage = _as_dict(final.get("usage"))
                input_tokens = int(usage.get("input_tokens") or 0)
                output_tokens = int(usage.get("output_tokens") or 0)
                response_model = str(final.get("model") or MODEL)
        if not completed:
            raise LunaResponsesError("Luna stream ended without response.completed")
        logger.info("Luna Responses request completed in %.3fs", time.monotonic() - started)
        return StreamResult(
            text="".join(text_deltas) or "\n\n".join(message_items),
            function_calls=tuple(function_calls),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=response_model,
        )
    finally:
        connection.close()


def query(
    *,
    system_message: str | None,
    user_message: str | None,
    func_spec: Any,
    cfg: Any,
    max_tokens: int = 16384,
) -> tuple[str | dict[str, Any], float, int, int, dict[str, Any]]:
    stage = cfg.agent.feedback
    if str(stage.model) != MODEL:
        raise LunaResponsesError(f"unexpected NatureBench feedback model: {stage.model}")
    payload = build_request(
        system_message=system_message,
        user_message=user_message,
        reasoning_effort=str(stage.reasoning_effort),
        func_spec=func_spec,
        max_tokens=max_tokens,
    )
    started = time.monotonic()
    result = _stream(payload, cfg)
    if func_spec is None:
        output: str | dict[str, Any] = result.text
    else:
        matching = [call for call in result.function_calls if call.get("name") == func_spec.name]
        if not matching:
            raise LunaResponsesError(f"expected function call {func_spec.name!r}")
        arguments = matching[0].get("arguments") or "{}"
        output = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        if not isinstance(output, dict):
            raise LunaResponsesError("function-call arguments are not a JSON object")
    return (
        output,
        time.monotonic() - started,
        result.input_tokens,
        result.output_tokens,
        {"model": result.model, "created": int(time.time()), "reasoning_effort": EFFORT},
    )


def generate(
    *, prompt_messages: list[dict[str, str]], cfg: Any, json_schema: dict | None,
    max_tokens: int,
) -> str:
    stage = cfg.agent.code
    if str(stage.model) != MODEL:
        raise LunaResponsesError(f"unexpected NatureBench code model: {stage.model}")
    system_parts = [row["content"] for row in prompt_messages if row.get("role") == "system"]
    user_parts = [
        row["content"]
        for row in prompt_messages
        if row.get("role") in {"user", "assistant"}
    ]
    payload = build_request(
        system_message="\n\n".join(system_parts) or None,
        user_message="\n\n".join(user_parts) or None,
        reasoning_effort=str(stage.reasoning_effort),
        json_schema=json_schema,
        max_tokens=max_tokens,
    )
    return _stream(payload, cfg).text
