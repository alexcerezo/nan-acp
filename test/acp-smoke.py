#!/usr/bin/env python3
"""Minimal ACP client smoke test for nan-acp.

Usage: python3 test/acp-smoke.py [path-to-nan-acp]
It starts the driver, creates a session, asks it to read/write/bash, and
auto-approves permission requests. Exits non-zero on failure.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time

DRIVER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "nan-acp.mjs")
CWD = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
READ_FILE = "/tmp/nan-test.txt"
WRITE_FILE = "/tmp/nan-acp-written.txt"


def main() -> int:
    with open(READ_FILE, "w", encoding="utf-8") as handle:
        handle.write("hello from nan test\n")

    env = dict(os.environ)
    env["NAN_ACP_LOG"] = "1"
    process = subprocess.Popen(
        [DRIVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=CWD,
        text=True,
        bufsize=1,
        env=env,
    )
    messages = queue.Queue()
    stderr_lines = []

    threading.Thread(target=lambda: [messages.put(line) for line in process.stdout], daemon=True).start()
    threading.Thread(target=lambda: [stderr_lines.append(line) for line in process.stderr], daemon=True).start()

    next_id = [0]

    def send(message):
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    def recv(timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = messages.get(timeout=0.5)
            except queue.Empty:
                continue
            if not line.strip():
                continue
            return json.loads(line)
        raise TimeoutError("timed out waiting for ACP message")

    def request(method, params, timeout=120):
        next_id[0] += 1
        request_id = next_id[0]
        send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            message = recv(timeout)
            if message.get("method") == "session/request_permission":
                options = message.get("params", {}).get("options", [])
                choice = next((o for o in options if o.get("optionId") == "allow-once"), options[0] if options else None)
                send({
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"outcome": {"outcome": "selected", "optionId": choice["optionId"]}},
                })
                continue
            if message.get("method") == "session/update":
                update = message.get("params", {}).get("update", {})
                kind = update.get("sessionUpdate")
                if kind == "tool_call":
                    print(f"  tool: {update.get('title')}")
                elif kind == "tool_call_update" and update.get("status") in ("completed", "failed"):
                    print(f"  tool {update.get('status')}: {update.get('toolCallId')}")
                continue
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    try:
        init = request("initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {"fs": {"readTextFile": True, "writeTextFile": True}, "terminal": True},
            "clientInfo": {"name": "nan-acp-smoke", "version": "0"},
        })
        assert init["protocolVersion"] == 1, init
        assert init["agentInfo"]["name"] == "nan-acp", init
        print("initialize: ok")

        session = request("session/new", {"cwd": CWD, "mcpServers": []})
        models = session.get("configOptions", [{}])[0].get("options", [])
        assert len(models) == 4, models
        assert session["models"]["currentModelId"] == "deepseek-v4-flash", session
        print(f"session: {session['sessionId']} models={len(models)}")

        result = request("session/prompt", {
            "sessionId": session["sessionId"],
            "prompt": [{"type": "text", "text": "Read /tmp/nan-test.txt and tell me its exact contents. Use the read_file tool."}],
        })
        assert result.get("stopReason") == "end_turn", result
        print("read prompt: ok")

        result = request("session/prompt", {
            "sessionId": session["sessionId"],
            "prompt": [{"type": "text", "text": "Write the exact text 'written by nan-acp test' to /tmp/nan-acp-written.txt using write_file."}],
        })
        assert result.get("stopReason") == "end_turn", result
        with open(WRITE_FILE, encoding="utf-8") as handle:
            written = handle.read()
        assert written.strip() == "written by nan-acp test", repr(written)
        print("write prompt: ok")

        result = request("session/prompt", {
            "sessionId": session["sessionId"],
            "prompt": [{"type": "text", "text": "Run the bash command `printf nan-bash-ok` and report the output."}],
        })
        assert result.get("stopReason") == "end_turn", result
        print("bash prompt: ok")

        result = request("session/set_config_option", {
            "sessionId": session["sessionId"],
            "configId": "model",
            "value": "qwen3.8-flash",
        })
        assert result == {}, result
        print("model switch: ok")

        print("ALL OK")
        return 0
    finally:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        if stderr_lines and os.environ.get("NAN_ACP_LOG"):
            print("--- driver stderr ---")
            print("".join(stderr_lines[-80:]))


if __name__ == "__main__":
    sys.exit(main())
