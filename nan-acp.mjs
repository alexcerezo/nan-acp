#!/usr/bin/env node
/**
 * nan-acp — a small ACP (Agent Client Protocol) server for the NaN
 * OpenAI-compatible API.
 *
 * It exposes the chat models configured in ~/.omp/agent/models.yml for the
 * `nan` provider, streams assistant output to the ACP client (T3 Code), and
 * runs a compact coding tool loop (read/write/edit/bash/grep/glob/list).
 *
 * Transport: newline-delimited JSON-RPC 2.0 over stdio.
 * Logging: stderr only, enabled with NAN_ACP_LOG=1.
 */
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";
import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";

const VERSION = "0.1.0";
const DEFAULT_MODEL = "deepseek-v4-flash";
const DEFAULT_BASE_URL = "https://api.nan.builders/v1";
const MAX_OUTPUT_CHARS = 120_000;
const MAX_TOOL_RESULT_CHARS = 60_000;
const MAX_TOOL_STEPS = 50;
const PERMISSION_TIMEOUT_MS = 5 * 60_000;

const MODELS = [
  {
    id: "deepseek-v4-flash",
    name: "NaN DeepSeek V4 Flash",
    reasoning: true,
    contextWindow: 1_000_000,
    maxTokens: 32_000,
  },
  {
    id: "glm5.3-flash",
    name: "NaN GLM 5.3 Flash",
    reasoning: true,
    contextWindow: 1_000_000,
    maxTokens: 32_000,
  },
  {
    id: "qwen3.8-flash",
    name: "NaN Qwen 3.8 Flash",
    reasoning: false,
    contextWindow: 262_000,
    maxTokens: 32_000,
  },
  {
    id: "mimo-v2.6-flash",
    name: "NaN MiMo V2.6 Flash",
    reasoning: true,
    contextWindow: 1_000_000,
    maxTokens: 32_000,
  },
];
const MODEL_BY_ID = new Map(MODELS.map((model) => [model.id, model]));

const BASE_URL = (process.env.NAN_BASE_URL || DEFAULT_BASE_URL).replace(/\/+$/, "");
const MAX_TOKENS = Number.parseInt(process.env.NAN_MAX_TOKENS || "8192", 10);
const REQUEST_TIMEOUT_MS = Number.parseInt(process.env.NAN_REQUEST_TIMEOUT_MS || "180000", 10);

function log(...args) {
  if (process.env.NAN_ACP_LOG) {
    console.error("[nan-acp]", ...args);
  }
}

function truncate(text, max = MAX_OUTPUT_CHARS) {
  if (typeof text !== "string") return "";
  if (text.length <= max) return text;
  return `${text.slice(0, max)}\n... [truncated ${text.length - max} chars]`;
}

function resolveUserPath(cwd, value) {
  let input = String(value ?? "");
  if (input === "~") input = os.homedir();
  else if (input.startsWith("~/")) input = path.join(os.homedir(), input.slice(2));
  return path.isAbsolute(input) ? input : path.resolve(cwd, input);
}

function displayPath(cwd, file) {
  const relative = path.relative(cwd, file);
  return relative && !relative.startsWith("..") && !path.isAbsolute(relative)
    ? relative
    : file;
}

function apiKey() {
  const direct = process.env.NAN_API_KEY;
  if (direct && direct.trim()) return direct.trim();
  if (process.env.NAN_API_KEY_FILE) {
    try {
      const value = fs.readFileSync(process.env.NAN_API_KEY_FILE, "utf8").trim();
      if (value) return value;
    } catch {
      // ignore
    }
  }
  return null;
}

function normalizeModelId(value) {
  let id = String(value ?? "").trim();
  if (id.startsWith("nan/")) id = id.slice(4);
  if (id.includes("/")) id = id.slice(id.lastIndexOf("/") + 1);
  if (id.endsWith(":max")) id = id.slice(0, -4);
  return id;
}

function modelConfigOptions(current) {
  return [
    {
      id: "model",
      name: "Model",
      category: "model",
      type: "select",
      currentValue: current,
      options: MODELS.map((model) => ({
        value: model.id,
        name: model.name,
        description: `${model.id} · ${Math.round(model.contextWindow / 1000)}k ctx`,
      })),
    },
  ];
}

function modelsPayload(current) {
  return {
    availableModels: MODELS.map((model) => ({
      modelId: model.id,
      name: model.name,
      description: `${model.id} · ${Math.round(model.contextWindow / 1000)}k ctx`,
    })),
    currentModelId: current,
  };
}

function systemPrompt(cwd) {
  return `You are nan-acp, a pragmatic coding agent running inside T3 Code.

Working directory: ${cwd}

Rules:
- Use the provided tools for file access and command execution. Do not pretend to have read or changed a file without calling a tool.
- Inspect before modifying. Prefer read_file, grep, glob, and list_dir over guessing.
- Use edit_file for exact replacements; use write_file only for new files or full rewrites.
- Run commands with bash when needed, and report the important output.
- Keep going until the task is genuinely complete. Verify changes with a real check when possible.
- Be concise. Do not invent results, files, or command output.`;
}

const TOOLS = [
  {
    type: "function",
    function: {
      name: "read_file",
      description: "Read a UTF-8 text file. Returns its contents, optionally sliced by line.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "File path, absolute or relative to the working directory." },
          offset: { type: "integer", description: "1-based first line to return." },
          limit: { type: "integer", description: "Maximum number of lines to return." },
        },
        required: ["path"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "write_file",
      description: "Create or overwrite a file with the given text. Creates parent directories.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "File path, absolute or relative to the working directory." },
          content: { type: "string", description: "Full file contents." },
        },
        required: ["path", "content"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "edit_file",
      description: "Replace an exact string in a file. Fails if the old string is absent or, unless replace_all is true, ambiguous.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "File path, absolute or relative to the working directory." },
          old_string: { type: "string", description: "Exact text to replace." },
          new_string: { type: "string", description: "Replacement text." },
          replace_all: { type: "boolean", description: "Replace every occurrence instead of requiring exactly one." },
        },
        required: ["path", "old_string", "new_string"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "bash",
      description: "Run a shell command in the working directory. Returns exit code, stdout and stderr.",
      parameters: {
        type: "object",
        properties: {
          command: { type: "string", description: "Command line to run with /bin/bash -lc." },
          timeout_seconds: { type: "integer", description: "Timeout in seconds (default 120, max 600)." },
        },
        required: ["command"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "grep",
      description: "Search file contents with a regular expression. Returns matching lines with file and line number.",
      parameters: {
        type: "object",
        properties: {
          pattern: { type: "string", description: "Regular expression." },
          path: { type: "string", description: "Directory or file to search. Defaults to the working directory." },
          glob: { type: "string", description: "Optional glob filter, for example '*.ts'." },
        },
        required: ["pattern"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "glob",
      description: "List files matching a glob pattern.",
      parameters: {
        type: "object",
        properties: {
          pattern: { type: "string", description: "Glob pattern, for example 'src/**/*.ts'." },
          path: { type: "string", description: "Directory to search. Defaults to the working directory." },
        },
        required: ["pattern"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list_dir",
      description: "List a directory's entries.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Directory path. Defaults to the working directory." },
        },
        additionalProperties: false,
      },
    },
  },
];

class CancelledError extends Error {
  constructor(message = "cancelled") {
    super(message);
    this.name = "CancelledError";
  }
}

class ToolError extends Error {
  constructor(message) {
    super(message);
    this.name = "ToolError";
  }
}

// ---------------------------------------------------------------------------
// ACP transport
// ---------------------------------------------------------------------------

const sessions = new Map();
let nextRequestId = 1;
const pendingClientRequests = new Map();

function send(message) {
  try {
    process.stdout.write(`${JSON.stringify(message)}\n`);
  } catch (error) {
    log("stdout write failed", error?.message || error);
  }
}

function sendResult(id, result) {
  send({ jsonrpc: "2.0", id, result });
}

function sendError(id, code, message, data) {
  send({
    jsonrpc: "2.0",
    id,
    error: { code, message, ...(data === undefined ? {} : { data }) },
  });
}

function notify(method, params) {
  send({ jsonrpc: "2.0", method, params });
}

function requestClient(method, params, timeoutMs = PERMISSION_TIMEOUT_MS) {
  const id = nextRequestId++;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pendingClientRequests.delete(id);
      reject(new Error(`Client request timed out: ${method}`));
    }, timeoutMs);
    pendingClientRequests.set(id, { resolve, reject, timer, method, params });
    send({ jsonrpc: "2.0", id, method, params });
  });
}

function sessionUpdate(session, update) {
  notify("session/update", { sessionId: session.id, update });
}

function emitMessageChunk(session, messageId, text) {
  if (!text) return;
  sessionUpdate(session, {
    sessionUpdate: "agent_message_chunk",
    content: { type: "text", text },
    messageId,
  });
}

function emitThoughtChunk(session, messageId, text) {
  if (!text) return;
  sessionUpdate(session, {
    sessionUpdate: "agent_thought_chunk",
    content: { type: "text", text },
    messageId,
  });
}

async function handleMessage(message) {
  if (
    message &&
    typeof message === "object" &&
    message.method === undefined &&
    message.id !== undefined &&
    ("result" in message || "error" in message)
  ) {
    const pending = pendingClientRequests.get(message.id);
    if (!pending) return;
    pendingClientRequests.delete(message.id);
    clearTimeout(pending.timer);
    if (message.error) {
      pending.reject(new Error(message.error.message || "client request failed"));
    } else {
      pending.resolve(message.result || {});
    }
    return;
  }

  if (!message || typeof message.method !== "string") return;

  if (message.id === undefined) {
    handleNotification(message).catch((error) => log("notification error", error));
    return;
  }

  handleRequest(message).catch((error) => {
    log("request error", message.method, error);
    sendError(message.id, -32603, error?.message || String(error));
  });
}

async function handleNotification(message) {
  if (message.method !== "session/cancel") return;
  const session = sessions.get(message.params?.sessionId);
  if (!session) return;
  session.cancelled = true;
  session.abort?.abort();
  for (const [id, pending] of pendingClientRequests) {
    if (
      pending.method === "session/request_permission" &&
      pending.params?.sessionId === session.id
    ) {
      pendingClientRequests.delete(id);
      clearTimeout(pending.timer);
      pending.resolve({ outcome: { outcome: "cancelled" } });
    }
  }
}

async function handleRequest(message) {
  const { id, method, params } = message;
  switch (method) {
    case "initialize":
      return sendResult(id, initializeResult());
    case "authenticate":
      return sendResult(id, {});
    case "session/new":
      return sendResult(id, newSessionResult(params));
    case "session/prompt":
      return sendResult(id, await promptResult(params));
    case "session/set_model":
      return sendResult(id, setSessionModel(params));
    case "session/set_config_option":
      return sendResult(id, setConfigOption(params));
    case "session/close": {
      const sessionId = params?.sessionId;
      if (sessionId) sessions.delete(sessionId);
      return sendResult(id, {});
    }
    case "session/load":
    case "session/resume":
      return sendError(id, -32601, `${method} is not supported by nan-acp`);
    default:
      return sendError(id, -32601, `Method not found: ${method}`);
  }
}

function initializeResult() {
  return {
    protocolVersion: 1,
    agentInfo: { name: "nan-acp", title: "NaN", version: VERSION },
    authMethods: [
      {
        id: "nan",
        name: "Use configured NaN API key",
        description: "Uses NAN_API_KEY configured in T3 Code Provider settings.",
      },
    ],
    agentCapabilities: {
      loadSession: false,
      promptCapabilities: { image: true, embeddedContext: true },
      mcpCapabilities: { http: false, sse: false },
    },
  };
}

function newSessionResult(params) {
  const cwd = params?.cwd || process.cwd();
  const session = {
    id: randomUUID(),
    cwd,
    model: DEFAULT_MODEL,
    messages: [{ role: "system", content: systemPrompt(cwd) }],
    abort: null,
    cancelled: false,
    running: false,
    alwaysAllow: new Set(),
  };
  sessions.set(session.id, session);
  log("session created", session.id, cwd);
  return {
    sessionId: session.id,
    configOptions: modelConfigOptions(session.model),
    models: modelsPayload(session.model),
  };
}

function setSessionModel(params) {
  const session = sessions.get(params?.sessionId);
  if (!session) throw new Error("Unknown session");
  const requested = normalizeModelId(params?.modelId ?? params?.model ?? params?.value);
  if (!MODEL_BY_ID.has(requested)) {
    throw new Error(`Unknown model: ${params?.modelId ?? params?.model ?? params?.value}`);
  }
  session.model = requested;
  log("model set", session.id, requested);
  return {};
}

function setConfigOption(params) {
  const session = sessions.get(params?.sessionId);
  if (!session) throw new Error("Unknown session");
  const configId = params?.configId ?? params?.config_id ?? params?.id;
  if (configId === "model") {
    const requested = normalizeModelId(params?.value);
    if (!MODEL_BY_ID.has(requested)) throw new Error(`Unknown model: ${params?.value}`);
    session.model = requested;
    log("model set via config", session.id, requested);
  }
  return {};
}

async function promptResult(params) {
  const session = sessions.get(params?.sessionId);
  if (!session) throw new Error("Unknown session");
  if (session.running) throw new Error("A prompt is already running for this session");
  session.running = true;
  session.cancelled = false;
  try {
    const content = promptToApiContent(params?.prompt, session);
    session.messages.push({ role: "user", content });
    return await runTurn(session);
  } catch (error) {
    if (error instanceof CancelledError) return { stopReason: "cancelled" };
    throw error;
  } finally {
    session.running = false;
  }
}

function promptToApiContent(prompt, session) {
  const blocks = Array.isArray(prompt) ? prompt : [{ type: "text", text: String(prompt ?? "") }];
  const parts = [];
  for (const block of blocks) {
    if (!block || typeof block !== "object") continue;
    if (block.type === "text") {
      parts.push({ type: "text", text: String(block.text ?? "") });
    } else if (block.type === "image" && block.data) {
      parts.push({
        type: "image_url",
        image_url: { url: `data:${block.mimeType || "image/png"};base64,${block.data}` },
      });
    } else if (block.type === "resource_link") {
      const uri = String(block.uri ?? "");
      const name = block.name || uri;
      let text = "";
      if (uri.startsWith("file://")) {
        try {
          text = fs.readFileSync(new URL(uri), "utf8");
        } catch (error) {
          text = `[could not read ${uri}: ${error.message}]`;
        }
      }
      parts.push({
        type: "text",
        text: `Context file: ${name}\n${truncate(text, 80_000)}`,
      });
    } else if (block.type === "resource" && block.resource) {
      const resource = block.resource;
      const text =
        typeof resource.text === "string"
          ? resource.text
          : resource.uri?.startsWith("file://")
            ? fs.readFileSync(new URL(resource.uri), "utf8")
            : "";
      if (text) {
        parts.push({ type: "text", text: `Context: ${resource.uri || "resource"}\n${truncate(text, 80_000)}` });
      }
    }
  }
  if (parts.length === 0) return "";
  if (parts.length === 1 && parts[0].type === "text") return parts[0].text;
  return parts;
}

// ---------------------------------------------------------------------------
// Agent loop
// ---------------------------------------------------------------------------

async function runTurn(session) {
  const model = MODEL_BY_ID.get(session.model) || MODELS[0];
  let totalInput = 0;
  let totalOutput = 0;
  let lastFinish = "stop";
  const turnController = new AbortController();
  session.abort = turnController;

  try {
    for (let step = 0; step < MAX_TOOL_STEPS; step += 1) {
      if (session.cancelled) return { stopReason: "cancelled" };
      let response;
      try {
        response = await callModelStreaming(session, model, turnController.signal);
      } catch (error) {
        if (error instanceof CancelledError || session.cancelled) {
          return { stopReason: "cancelled" };
        }
        throw error;
      }

    totalInput += response.usage.inputTokens;
    totalOutput += response.usage.outputTokens;
    lastFinish = response.finishReason || "stop";

    if (response.toolCalls.length > 0) {
      session.messages.push({
        role: "assistant",
        content: response.text || null,
        tool_calls: response.toolCalls.map((toolCall) => ({
          id: toolCall.id,
          type: "function",
          function: {
            name: toolCall.name,
            arguments: toolCall.arguments || "{}",
          },
        })),
      });

      for (const toolCall of response.toolCalls) {
        if (session.cancelled) return { stopReason: "cancelled" };
        const resultText = await executeToolCall(session, toolCall);
        session.messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: resultText,
        });
      }
      continue;
    }

    if (response.text) {
      session.messages.push({ role: "assistant", content: response.text });
    }

    sessionUpdate(session, {
      sessionUpdate: "usage_update",
      size: model.contextWindow,
      used: Math.max(totalInput + totalOutput, 1),
    });

    return {
      stopReason: lastFinish === "length" ? "max_tokens" : "end_turn",
      usage: {
        inputTokens: totalInput,
        outputTokens: totalOutput,
        totalTokens: totalInput + totalOutput,
      },
    };
  }

    return {
      stopReason: "max_turn_requests",
      usage: {
        inputTokens: totalInput,
        outputTokens: totalOutput,
        totalTokens: totalInput + totalOutput,
      },
    };
  } finally {
    if (session.abort === turnController) session.abort = null;
  }
}

async function callModelStreaming(session, model, signal) {
  const key = apiKey();
  if (!key) {
    throw new Error(
      "No NaN API key found. Please configure the NAN_API_KEY environment variable in T3 Code Provider settings.",
    );
  }

  const messageId = randomUUID();
  const body = {
    model: model.id,
    messages: session.messages,
    tools: TOOLS,
    tool_choice: "auto",
    stream: true,
    stream_options: { include_usage: true },
    max_tokens: Number.isFinite(MAX_TOKENS) && MAX_TOKENS > 0 ? MAX_TOKENS : 8192,
  };

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const onAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener("abort", onAbort, { once: true });
  }

  let response;
  try {
    response = await fetch(`${BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${key}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timeout);
    signal?.removeEventListener?.("abort", onAbort);
    if (signal?.aborted) throw new CancelledError();
    throw new Error(`NaN request failed: ${error.message}`);
  }

  if (!response.ok || !response.body) {
    clearTimeout(timeout);
    const text = await response.text().catch(() => "");
    throw new Error(`NaN API ${response.status}: ${truncate(text, 2000)}`);
  }

  let text = "";
  let reasoning = "";
  let finishReason = "stop";
  let usage = { inputTokens: 0, outputTokens: 0 };
  const toolCalls = [];
  let streamDone = false;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (!streamDone) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const raw = buffer.slice(0, newline).replace(/\r$/, "");
        buffer = buffer.slice(newline + 1);
        if (!raw.startsWith("data:")) continue;
        const data = raw.slice(5).trim();
        if (!data) continue;
        if (data === "[DONE]") {
          streamDone = true;
          break;
        }
        let chunk;
        try {
          chunk = JSON.parse(data);
        } catch {
          log("bad SSE chunk", data.slice(0, 300));
          continue;
        }
        if (chunk.usage) {
          usage = {
            inputTokens: Number(chunk.usage.prompt_tokens) || usage.inputTokens,
            outputTokens: Number(chunk.usage.completion_tokens) || usage.outputTokens,
          };
        }
        const choice = chunk.choices?.[0];
        if (!choice) continue;
        if (choice.finish_reason) finishReason = choice.finish_reason;
        const delta = choice.delta || choice.message || {};
        if (typeof delta.reasoning_content === "string" && delta.reasoning_content) {
          reasoning += delta.reasoning_content;
          emitThoughtChunk(session, messageId, delta.reasoning_content);
        }
        if (typeof delta.content === "string" && delta.content) {
          text += delta.content;
          emitMessageChunk(session, messageId, delta.content);
        }
        if (Array.isArray(delta.tool_calls)) {
          for (const part of delta.tool_calls) {
            const index = Number.isInteger(part.index) ? part.index : 0;
            if (!toolCalls[index]) {
              toolCalls[index] = { id: "", name: "", arguments: "" };
            }
            const toolCall = toolCalls[index];
            if (part.id) toolCall.id = part.id;
            if (part.function?.name) toolCall.name += part.function.name;
            if (part.function?.arguments) toolCall.arguments += part.function.arguments;
          }
        }
      }
    }
  } catch (error) {
    if (signal?.aborted || controller.signal.aborted) throw new CancelledError();
    throw error;
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener?.("abort", onAbort);
    try {
      await reader.cancel();
    } catch {
      // ignore
    }
  }

  const normalizedToolCalls = toolCalls
    .filter((toolCall) => toolCall && (toolCall.name || toolCall.id))
    .map((toolCall, index) => ({
      id: toolCall.id || `call_${messageId}_${index}`,
      name: toolCall.name,
      arguments: toolCall.arguments,
    }));

  if (usage.inputTokens === 0 && usage.outputTokens === 0) {
    const inputChars = JSON.stringify(session.messages).length;
    usage = {
      inputTokens: Math.ceil(inputChars / 4),
      outputTokens: Math.ceil((text.length + reasoning.length) / 4),
    };
  }

  return {
    text,
    reasoning,
    finishReason,
    usage,
    toolCalls: normalizedToolCalls,
  };
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

function toolDescriptor(name, args, session) {
  const cwd = session.cwd;
  switch (name) {
    case "read_file": {
      const file = resolveUserPath(cwd, args.path);
      return { title: `Reading ${displayPath(cwd, file)}`, kind: "read", locations: [{ path: file }] };
    }
    case "write_file": {
      const file = resolveUserPath(cwd, args.path);
      return { title: `Writing ${displayPath(cwd, file)}`, kind: "edit", locations: [{ path: file }] };
    }
    case "edit_file": {
      const file = resolveUserPath(cwd, args.path);
      return { title: `Editing ${displayPath(cwd, file)}`, kind: "edit", locations: [{ path: file }] };
    }
    case "bash": {
      const command = String(args.command ?? "");
      return {
        title: `Running ${command.length > 100 ? `${command.slice(0, 97)}...` : command}`,
        kind: "execute",
        locations: [],
      };
    }
    case "grep": {
      return { title: `Searching ${String(args.pattern ?? "")}`, kind: "search", locations: [] };
    }
    case "glob": {
      return { title: `Finding ${String(args.pattern ?? "")}`, kind: "search", locations: [] };
    }
    case "list_dir": {
      const dir = resolveUserPath(cwd, args.path || ".");
      return { title: `Listing ${displayPath(cwd, dir)}`, kind: "read", locations: [{ path: dir }] };
    }
    default:
      return { title: name, kind: "other", locations: [] };
  }
}

async function executeToolCall(session, toolCall) {
  const toolCallId = toolCall.id || randomUUID();
  let args = {};
  let parseError = null;
  try {
    args = safeParseArgs(toolCall.arguments);
  } catch (error) {
    parseError = error;
  }
  const descriptor = toolDescriptor(toolCall.name, args, session);
  sessionUpdate(session, {
    sessionUpdate: "tool_call",
    toolCallId,
    title: descriptor.title,
    name: toolCall.name,
    kind: descriptor.kind,
    status: "pending",
    rawInput: args,
    locations: descriptor.locations,
  });

  if (parseError) {
    const message = `Invalid tool arguments: ${parseError.message}`;
    sessionUpdate(session, {
      sessionUpdate: "tool_call_update",
      toolCallId,
      status: "failed",
      content: [{ type: "content", content: { type: "text", text: message } }],
    });
    return message;
  }

  const needsPermission = ["write_file", "edit_file", "bash"].includes(toolCall.name);
  if (needsPermission && !session.alwaysAllow.has(toolCall.name)) {
    const allowed = await requestPermission(session, {
      toolCallId,
      title: descriptor.title,
      name: toolCall.name,
      kind: descriptor.kind,
      rawInput: args,
      locations: descriptor.locations,
    });
    if (!allowed) {
      const message = "Permission denied by the user.";
      sessionUpdate(session, {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "failed",
        content: [{ type: "content", content: { type: "text", text: message } }],
      });
      return message;
    }
  }

  sessionUpdate(session, {
    sessionUpdate: "tool_call_update",
    toolCallId,
    status: "in_progress",
  });

  let result;
  try {
    result = await runTool(session, toolCall.name, args);
  } catch (error) {
    if (error instanceof CancelledError) throw error;
    const message = error instanceof ToolError ? error.message : `Tool failed: ${error.message}`;
    sessionUpdate(session, {
      sessionUpdate: "tool_call_update",
      toolCallId,
      status: "failed",
      content: [{ type: "content", content: { type: "text", text: truncate(message, 20_000) } }],
    });
    return message;
  }

  const text = truncate(result.text || "", MAX_TOOL_RESULT_CHARS);
  sessionUpdate(session, {
    sessionUpdate: "tool_call_update",
    toolCallId,
    status: "completed",
    rawOutput: result.rawOutput ?? { output: text },
    content: [{ type: "content", content: { type: "text", text } }],
    locations: descriptor.locations,
  });
  return text;
}

function safeParseArgs(value) {
  if (value && typeof value === "object") return value;
  const text = String(value ?? "").trim();
  if (!text) return {};
  return JSON.parse(text);
}

async function requestPermission(session, toolCall) {
  const options = [
    { optionId: "allow-once", name: "Allow once", kind: "allow_once" },
    {
      optionId: "allow-always",
      name: `Always allow ${toolCall.name} in this session`,
      kind: "allow_always",
    },
    { optionId: "reject-once", name: "Reject", kind: "reject_once" },
  ];
  let response;
  try {
    response = await requestClient("session/request_permission", {
      sessionId: session.id,
      toolCall: {
        toolCallId: toolCall.toolCallId,
        title: toolCall.title,
        name: toolCall.name,
        kind: toolCall.kind,
        status: "pending",
        rawInput: toolCall.rawInput,
        locations: toolCall.locations,
      },
      options,
    });
  } catch (error) {
    log("permission request failed", error.message);
    return false;
  }
  const outcome = response?.outcome;
  if (!outcome || outcome.outcome === "cancelled") {
    throw new CancelledError();
  }
  if (outcome.outcome === "selected") {
    if (outcome.optionId === "allow-always") {
      session.alwaysAllow.add(toolCall.name);
      return true;
    }
    return outcome.optionId === "allow-once";
  }
  return false;
}

async function runTool(session, name, args) {
  switch (name) {
    case "read_file":
      return readFileTool(session, args);
    case "write_file":
      return writeFileTool(session, args);
    case "edit_file":
      return editFileTool(session, args);
    case "bash":
      return bashTool(session, args);
    case "grep":
      return grepTool(session, args);
    case "glob":
      return globTool(session, args);
    case "list_dir":
      return listDirTool(session, args);
    default:
      throw new ToolError(`Unknown tool: ${name}`);
  }
}

async function readFileTool(session, args) {
  if (!args.path) throw new ToolError("path is required");
  const file = resolveUserPath(session.cwd, args.path);
  const stat = await fsp.stat(file).catch(() => null);
  if (!stat) throw new ToolError(`File not found: ${file}`);
  if (!stat.isFile()) throw new ToolError(`Not a file: ${file}`);
  if (stat.size > 4_000_000) throw new ToolError(`File is too large (${stat.size} bytes): ${file}`);
  let text = await fsp.readFile(file, "utf8");
  const offset = Number(args.offset);
  const limit = Number(args.limit);
  if (Number.isFinite(offset) || Number.isFinite(limit)) {
    const lines = text.split("\n");
    const start = Math.max(0, (Number.isFinite(offset) && offset > 0 ? offset : 1) - 1);
    const end = Number.isFinite(limit) ? start + Math.max(0, limit) : lines.length;
    text = lines.slice(start, end).join("\n");
  }
  return { text: truncate(text), rawOutput: { path: file, bytes: stat.size } };
}

async function writeFileTool(session, args) {
  if (!args.path) throw new ToolError("path is required");
  const file = resolveUserPath(session.cwd, args.path);
  const content = String(args.content ?? "");
  await fsp.mkdir(path.dirname(file), { recursive: true });
  await fsp.writeFile(file, content, "utf8");
  return {
    text: `Wrote ${Buffer.byteLength(content, "utf8")} bytes to ${file}`,
    rawOutput: { path: file, bytes: Buffer.byteLength(content, "utf8") },
  };
}

async function editFileTool(session, args) {
  if (!args.path) throw new ToolError("path is required");
  const file = resolveUserPath(session.cwd, args.path);
  const oldString = String(args.old_string ?? "");
  const newString = String(args.new_string ?? "");
  if (!oldString) throw new ToolError("old_string must not be empty");
  const original = await fsp.readFile(file, "utf8");
  const first = original.indexOf(oldString);
  if (first < 0) throw new ToolError(`old_string was not found in ${file}`);
  const second = original.indexOf(oldString, first + oldString.length);
  if (second >= 0 && !args.replace_all) {
    throw new ToolError(
      `old_string appears more than once in ${file}; make it unique or set replace_all=true`,
    );
  }
  const updated = args.replace_all
    ? original.split(oldString).join(newString)
    : `${original.slice(0, first)}${newString}${original.slice(first + oldString.length)}`;
  await fsp.writeFile(file, updated, "utf8");
  const count = args.replace_all ? original.split(oldString).length - 1 : 1;
  return {
    text: `Replaced ${count} occurrence${count === 1 ? "" : "s"} in ${file}`,
    rawOutput: { path: file, replacements: count },
  };
}

async function bashTool(session, args) {
  const command = String(args.command ?? "").trim();
  if (!command) throw new ToolError("command must not be empty");
  const requested = Number(args.timeout_seconds);
  const timeoutMs = Math.min(Math.max(Number.isFinite(requested) ? requested : 120, 1), 600) * 1000;
  return await new Promise((resolve, reject) => {
    let settled = false;
    let stdout = "";
    let stderr = "";
    const child = spawn("/bin/bash", ["-lc", command], {
      cwd: session.cwd,
      env: process.env,
      detached: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const kill = () => {
      try {
        process.kill(-child.pid, "SIGKILL");
      } catch {
        try {
          child.kill("SIGKILL");
        } catch {
          // ignore
        }
      }
    };
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      session.abort?.signal?.removeEventListener?.("abort", onAbort);
      resolve(result);
    };
    const fail = (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      session.abort?.signal?.removeEventListener?.("abort", onAbort);
      reject(error);
    };
    const onAbort = () => {
      kill();
      fail(new CancelledError());
    };
    const timer = setTimeout(() => {
      kill();
      finish({
        text: `Command timed out after ${timeoutMs / 1000}s.\nstdout:\n${truncate(stdout, 20_000)}\nstderr:\n${truncate(stderr, 20_000)}`,
        rawOutput: { timedOut: true },
      });
    }, timeoutMs);
    if (session.abort?.signal) {
      if (session.abort.signal.aborted) onAbort();
      else session.abort.signal.addEventListener("abort", onAbort, { once: true });
    }
    child.stdout.on("data", (chunk) => {
      if (stdout.length < MAX_OUTPUT_CHARS) stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      if (stderr.length < MAX_OUTPUT_CHARS) stderr += chunk.toString();
    });
    child.on("error", (error) => fail(new ToolError(`Failed to start command: ${error.message}`)));
    child.on("close", (code, signal) => {
      finish({
        text: `exit code: ${code ?? "null"}${signal ? ` (signal ${signal})` : ""}\nstdout:\n${truncate(stdout, 40_000)}\nstderr:\n${truncate(stderr, 40_000)}`,
        rawOutput: { exitCode: code, signal },
      });
    });
  });
}

async function grepTool(session, args) {
  const pattern = String(args.pattern ?? "");
  if (!pattern) throw new ToolError("pattern must not be empty");
  const searchPath = resolveUserPath(session.cwd, args.path || ".");
  const rgArgs = ["--line-number", "--no-heading", "--color", "never", "--max-count", "200"];
  if (args.glob) rgArgs.push("-g", String(args.glob));
  rgArgs.push(pattern, searchPath);
  const result = await runSimpleProcess("rg", rgArgs, session.cwd, session);
  if (result.code === 127 || /ENOENT/.test(result.error || "")) {
    return fallbackGrep(session, pattern, searchPath, args.glob);
  }
  if (result.code !== 0 && result.code !== 1) {
    throw new ToolError(result.stderr || `rg exited with ${result.code}`);
  }
  return {
    text: truncate(result.stdout || "(no matches)"),
    rawOutput: { matches: result.stdout ? result.stdout.split("\n").filter(Boolean).length : 0 },
  };
}

async function globTool(session, args) {
  const pattern = String(args.pattern ?? "");
  if (!pattern) throw new ToolError("pattern must not be empty");
  const searchPath = resolveUserPath(session.cwd, args.path || ".");
  const result = await runSimpleProcess("rg", ["--files", "-g", pattern, searchPath], session.cwd, session);
  if (result.code === 127 || /ENOENT/.test(result.error || "")) {
    return fallbackGlob(session, pattern, searchPath);
  }
  if (result.code !== 0 && result.code !== 1) {
    throw new ToolError(result.stderr || `rg exited with ${result.code}`);
  }
  const lines = (result.stdout || "").split("\n").filter(Boolean).slice(0, 1000);
  return { text: truncate(lines.join("\n") || "(no files)"), rawOutput: { count: lines.length } };
}

async function listDirTool(session, args) {
  const dir = resolveUserPath(session.cwd, args.path || ".");
  const entries = await fsp.readdir(dir, { withFileTypes: true });
  const lines = entries
    .sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name))
    .map((entry) => `${entry.isDirectory() ? "d" : entry.isSymbolicLink() ? "l" : "-"} ${entry.name}`);
  return { text: truncate(lines.join("\n") || "(empty)"), rawOutput: { path: dir, count: entries.length } };
}

function runSimpleProcess(command, args, cwd, session) {
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let error = "";
    const timer = setTimeout(() => {
      try {
        child.kill("SIGKILL");
      } catch {
        // ignore
      }
    }, 60_000);
    const onAbort = () => {
      try {
        child.kill("SIGKILL");
      } catch {
        // ignore
      }
    };
    session.abort?.signal?.addEventListener("abort", onAbort, { once: true });
    child.stdout.on("data", (chunk) => {
      if (stdout.length < MAX_OUTPUT_CHARS) stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      if (stderr.length < MAX_OUTPUT_CHARS) stderr += chunk.toString();
    });
    child.on("error", (err) => {
      error = err.message;
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      session.abort?.signal?.removeEventListener?.("abort", onAbort);
      resolve({ code, stdout, stderr, error });
    });
  });
}

async function walkFiles(root, limit = 2000) {
  const files = [];
  const stack = [root];
  while (stack.length > 0 && files.length < limit) {
    const current = stack.pop();
    let entries;
    try {
      entries = await fsp.readdir(current, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      if (entry.name === ".git" || entry.name === "node_modules") continue;
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) stack.push(full);
      else if (entry.isFile()) files.push(full);
      if (files.length >= limit) break;
    }
  }
  return files;
}

function globToRegExp(pattern) {
  let output = "^";
  for (let index = 0; index < pattern.length; index += 1) {
    const char = pattern[index];
    if (char === "*") {
      if (pattern[index + 1] === "*") {
        output += ".*";
        index += 1;
      } else {
        output += "[^/]*";
      }
    } else if (char === "?") {
      output += ".";
    } else if ("\\^$+?.()|{}[]".includes(char)) {
      output += `\\${char}`;
    } else {
      output += char;
    }
  }
  return new RegExp(`${output}$`);
}

async function fallbackGlob(session, pattern, searchPath) {
  const stat = await fsp.stat(searchPath).catch(() => null);
  const files = stat?.isFile() ? [searchPath] : await walkFiles(searchPath);
  const regex = globToRegExp(pattern);
  const matches = files.filter((file) => regex.test(path.relative(searchPath, file) || path.basename(file)));
  return { text: truncate(matches.join("\n") || "(no files)"), rawOutput: { count: matches.length } };
}

async function fallbackGrep(session, pattern, searchPath, glob) {
  const stat = await fsp.stat(searchPath).catch(() => null);
  const files = stat?.isFile() ? [searchPath] : await walkFiles(searchPath);
  const regex = new RegExp(pattern);
  const globRegex = glob ? globToRegExp(glob) : null;
  const lines = [];
  for (const file of files) {
    if (globRegex && !globRegex.test(path.relative(searchPath, file) || path.basename(file))) continue;
    let content;
    try {
      const fileStat = await fsp.stat(file);
      if (fileStat.size > 2_000_000) continue;
      content = await fsp.readFile(file, "utf8");
    } catch {
      continue;
    }
    const rows = content.split("\n");
    for (let index = 0; index < rows.length; index += 1) {
      if (regex.test(rows[index])) {
        lines.push(`${file}:${index + 1}:${rows[index]}`);
        if (lines.length >= 200) break;
      }
    }
    if (lines.length >= 200) break;
  }
  return { text: truncate(lines.join("\n") || "(no matches)"), rawOutput: { count: lines.length } };
}

// ---------------------------------------------------------------------------
// stdio
// ---------------------------------------------------------------------------

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let message;
  try {
    message = JSON.parse(trimmed);
  } catch (error) {
    log("invalid JSON", trimmed.slice(0, 200), error.message);
    return;
  }
  handleMessage(message).catch((error) => log("message error", error));
});
rl.on("close", () => {
  log("stdin closed");
  process.exit(0);
});

process.on("uncaughtException", (error) => log("uncaught", error));
process.on("unhandledRejection", (error) => log("unhandled", error));

log("nan-acp started", VERSION, BASE_URL);
