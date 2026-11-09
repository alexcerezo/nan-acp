# nan-acp

Standalone ACP (Agent Client Protocol) driver for the NaN OpenAI-compatible API.
T3 Code can launch it as a **Local ACP command**, so NaN models no longer need omp.

## What it does

- Speaks ACP v1 over stdio (newline-delimited JSON-RPC).
- Exposes the four NaN chat models:
  - `deepseek-v4-flash`
  - `glm5.3-flash`
  - `qwen3.8-flash`
  - `mimo-v2.6-flash`
- Streams assistant text and reasoning chunks to T3.
- Runs an OpenAI tool loop with:
  - `read_file`, `write_file`, `edit_file`
  - `bash`
  - `grep`, `glob`, `list_dir`
- Asks T3 for permission before `write_file`, `edit_file`, and `bash`.
- Supports model switching via ACP config options or `session/set_model`.
- Supports `session/cancel`.

## Requirements

- Node.js 22 or newer (`fetch`, `AbortController`, `node:crypto`).
- A NaN API key.

The API key is read from:

1. `NAN_API_KEY` (configured directly in T3 Code Provider environment settings or shell environment)
2. `NAN_API_KEY_FILE`

## Install

```sh
mkdir -p ~/.local/bin
ln -sfn "$PWD/nan-acp.mjs" ~/.local/bin/nan-acp
chmod +x nan-acp.mjs
```

Verify it starts and exits cleanly:

```sh
NAN_ACP_LOG=1 ~/.local/bin/nan-acp </dev/null
```

## T3 Code provider instance

Settings → Providers → Add provider → Show more → Local ACP command:

- Display name: `NaN`
- Executable: `/home/ubuntu/.local/bin/nan-acp`
- Arguments: none
- Instance ID: `nan` (T3 stores it as `acpRegistry_nan`)

Or merge this into `~/.t3/userdata/settings.json`:

```json
"acpRegistry_nan": {
  "driver": "acpRegistry",
  "displayName": "NaN",
  "enabled": true,
  "config": {
    "source": "local",
    "commandPath": "/home/ubuntu/.local/bin/nan-acp",
    "commandArgs": []
  }
}
```

Start a test thread:

```sh
t3 thread start . "Reply with exactly: nan-ok" \
  --provider acpRegistry_nan --model deepseek-v4-flash --wait
```

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `NAN_API_KEY` | — | API key |
| `NAN_API_KEY_FILE` | — | Path to the key file |
| `NAN_BASE_URL` | `https://api.nan.builders/v1` | API base URL |
| `NAN_MAX_TOKENS` | `8192` | Output token cap per request |
| `NAN_REQUEST_TIMEOUT_MS` | `180000` | Per-request timeout |
| `NAN_ACP_LOG` | unset | `1` logs to stderr |

## Smoke test

```sh
python3 test/acp-smoke.py
```

The test launches the driver directly, creates a session, reads a file, writes a
file, runs a bash command, and switches models. It auto-approves permission
requests.

## Limitations

- No `session/load`, `session/list`, or native session import.
- No automatic compaction; long sessions stay in memory.
- No MCP servers, skills, subagents, or checkpoints.
- Tools run directly on the host. Permission requests cover mutations and bash,
  but reads and searches are not sandboxed.
