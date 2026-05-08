import argparse
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer


DEFAULT_HOST = "http://127.0.0.1"
DEFAULT_BIND_HOST = "localhost"
DEFAULT_PORT = 9379

DEFAULT_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
DEFAULT_MODEL_FILE = "gemma-4-E2B-it.litertlm"
DEFAULT_MODEL_ID = "gemma4e2b"
DEFAULT_CONTEXT_TOKENS = 32768
DEFAULT_RESERVE_OUTPUT_TOKENS = 1024
DEFAULT_LOG_FILE = "litert_lm_server.log"
DEFAULT_PID_FILE = "litert_lm_server.pid"
DEFAULT_SERVER_START_TIMEOUT = 180
DEFAULT_SESSION_DIR = "litert_lm_sessions"
DEFAULT_SESSION = "default"


SERVER_ARGS = None
SERVER_ENGINE = None
SERVER_LITERT = None
SERVER_STARTED_AT = time.time()


GEMINI_RE = re.compile(
    r"^/v1beta/models/([^/\\:]+):(generateContent|streamGenerateContent)$"
)


def estimate_output_tokens(text):
    if not text:
        return 0

    # This is a client-side estimate. The Gemini-style HTTP wrapper does not
    # receive exact token counts from LiteRT-LM, so use a conservative
    # no-dependency approximation that is stable enough for speed comparisons.
    normalized = text.replace("\r\n", "\n")
    visible_chars = len(normalized)

    if visible_chars <= 0:
        return 0

    return max(1, int(round(visible_chars / 4.0)))


class ClientMeter:
    def __init__(self, enabled):
        self.enabled = enabled
        self.start_time = time.perf_counter()
        self.first_text_time = None
        self.end_time = None
        self.output_parts = []
        self.chunk_count = 0

    def add_text(self, text):
        if not text:
            return

        now = time.perf_counter()

        if self.first_text_time is None:
            self.first_text_time = now

        self.output_parts.append(text)
        self.chunk_count += 1

    def finish(self):
        self.end_time = time.perf_counter()

    def output_text(self):
        return "".join(self.output_parts)

    def output_tokens(self):
        return estimate_output_tokens(self.output_text())

    def total_seconds(self):
        end_time = self.end_time or time.perf_counter()
        return max(0.0, end_time - self.start_time)

    def first_token_seconds(self):
        if self.first_text_time is None:
            return None

        return max(0.0, self.first_text_time - self.start_time)

    def generation_seconds(self):
        if self.first_text_time is None:
            return None

        end_time = self.end_time or time.perf_counter()
        return max(0.001, end_time - self.first_text_time)

    def tokens_per_second(self):
        seconds = self.generation_seconds()

        if seconds is None:
            return None

        return self.output_tokens() / seconds


def print_meter(args, meter, streaming):
    if not args.meter:
        return

    meter.finish()

    tokens = meter.output_tokens()
    total_seconds = meter.total_seconds()

    if streaming:
        first_seconds = meter.first_token_seconds()
        generation_seconds = meter.generation_seconds()
        token_rate = meter.tokens_per_second()

        if first_seconds is None or generation_seconds is None or token_rate is None:
            print(
                "[meter] no output | total "
                + format(total_seconds, ".2f")
                + "s",
                file=sys.stderr,
            )
            return

        print(
            "[meter] first token "
            + format(first_seconds, ".2f")
            + "s | output ~"
            + str(tokens)
            + " tok | generation "
            + format(generation_seconds, ".2f")
            + "s | speed ~"
            + format(token_rate, ".2f")
            + " tok/s | chunks "
            + str(meter.chunk_count)
            + " | total "
            + format(total_seconds, ".2f")
            + "s",
            file=sys.stderr,
        )
        return

    token_rate = tokens / max(0.001, total_seconds)

    print(
        "[meter] output ~"
        + str(tokens)
        + " tok | total "
        + format(total_seconds, ".2f")
        + "s | average ~"
        + format(token_rate, ".2f")
        + " tok/s",
        file=sys.stderr,
    )


def cmd_text(cmd):
    try:
        return subprocess.list2cmdline(cmd)
    except Exception:
        return " ".join(cmd)


def run_capture(cmd):
    print()
    print("Running:")
    print(cmd_text(cmd))
    print()

    try:
        return subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        print("ERROR: Could not find:", cmd[0])
        print("Make sure litert-lm is installed and available in PATH.")
        return None


def run_live(cmd):
    print()
    print("Running:")
    print(cmd_text(cmd))
    print()

    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("ERROR: Could not find:", cmd[0])
        print("Make sure litert-lm is installed and available in PATH.")
        return 1


def python_can_import_litert():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import litert_lm; print(litert_lm)",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    output = result.stdout or ""
    print(output)

    return result.returncode == 0


def install_python_litert():
    print("Installing LiteRT-LM Python packages into this Python:")
    print(sys.executable)

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "litert-lm-api",
        "litert-lm",
    ]

    return run_live(cmd) == 0


def ensure_python_litert(args):
    print("Checking Python LiteRT-LM import...")

    if python_can_import_litert():
        print("Python can import litert_lm.")
        return True

    print("Python cannot import litert_lm.")

    if not args.auto_python_install:
        print()
        print("Run this manually:")
        print("  " + sys.executable + " -m pip install --upgrade litert-lm-api litert-lm")
        return False

    if not install_python_litert():
        print("ERROR: Failed to install LiteRT-LM Python packages.")
        return False

    print("Rechecking Python LiteRT-LM import...")

    if python_can_import_litert():
        print("Python can now import litert_lm.")
        return True

    print("ERROR: Package install finished, but import still failed.")
    return False


def model_path_from_id(model_id):
    return os.path.join(
        os.path.expanduser("~"),
        ".litert-lm",
        "models",
        model_id.replace("/", "--"),
        "model.litertlm",
    )


def model_is_imported(args):
    path = model_path_from_id(args.model_id)

    if os.path.isfile(path):
        print("Model file exists:")
        print(path)
        return True

    result = run_capture([args.litert, "list"])

    if result is None:
        return False

    output = result.stdout or ""
    print(output)

    for line in output.splitlines():
        stripped = line.strip()

        if stripped == args.model_id:
            return True

        if stripped.startswith(args.model_id + " "):
            return True

        if stripped.startswith(args.model_id + "\t"):
            return True

    return False


def import_model(args):
    cmd = [
        args.litert,
        "import",
        "--from-huggingface-repo=" + args.repo,
        args.model_file,
        args.model_id,
    ]

    return run_live(cmd) == 0


def ensure_model(args):
    if model_is_imported(args):
        print("Model already imported:", args.model_id)
        return True

    print("Model is not imported yet.")
    print("Importing default Gemma 4 E2B model...")

    if not import_model(args):
        print("ERROR: Import failed.")
        return False

    if not model_is_imported(args):
        print("ERROR: Import finished, but model id was not found.")
        print("Expected model id:", args.model_id)
        print("Expected path:")
        print(model_path_from_id(args.model_id))
        return False

    print("Model imported successfully:", args.model_id)
    return True


def setup_all(args):
    if not ensure_python_litert(args):
        return False

    if not ensure_model(args):
        return False

    print()
    print("Setup complete.")
    print("Python import: OK")
    print("Model import:  OK")
    print("Model id:      " + args.model_id)
    print("Model path:    " + model_path_from_id(args.model_id))
    print()
    print("Now start server:")
    print("  py " + os.path.basename(sys.argv[0]) + " --serve")

    return True


def speculative_value(text):
    value = text.lower().strip()

    if value == "true":
        return True

    if value == "false":
        return False

    return None


def import_litert_lm():
    try:
        import litert_lm
        return litert_lm
    except ModuleNotFoundError:
        print("ERROR: Python could not import litert_lm.")
        print()
        print("The custom GPU server must run in a Python environment where LiteRT-LM is importable.")
        print("Your litert-lm command may exist, but this Python may not have the package.")
        print()
        print("Try:")
        print("  py " + os.path.basename(sys.argv[0]) + " --setup")
        print()
        print("Or manually:")
        print("  " + sys.executable + " -m pip install --upgrade litert-lm-api litert-lm")
        raise


def backend_value(litert_lm, backend_name):
    if backend_name.lower() == "gpu":
        return litert_lm.Backend.GPU

    return litert_lm.Backend.CPU


def load_engine_once(args):
    global SERVER_ENGINE
    global SERVER_LITERT

    if SERVER_ENGINE is not None:
        return SERVER_ENGINE

    model_path = model_path_from_id(args.model_id)

    if not os.path.isfile(model_path):
        raise FileNotFoundError("Model file not found: " + model_path)

    litert_lm = import_litert_lm()
    SERVER_LITERT = litert_lm

    try:
        if args.verbose:
            litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)
        else:
            # Keep startup readable unless the user explicitly asks for logs.
            litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
    except Exception:
        pass

    backend = backend_value(litert_lm, args.backend)
    spec = speculative_value(args.speculative)

    print()
    print("Loading LiteRT-LM engine once and keeping it alive:")
    print("Model id:     ", args.model_id)
    print("Model path:   ", model_path)
    print("Backend:      ", args.backend)
    print("Speculative:  ", args.speculative)
    print()

    SERVER_ENGINE = litert_lm.Engine(
        model_path,
        backend=backend,
        enable_speculative_decoding=spec,
    )

    SERVER_ENGINE.__enter__()

    print("Engine loaded and ready.")
    print()

    return SERVER_ENGINE


def close_engine():
    global SERVER_ENGINE

    if SERVER_ENGINE is not None:
        try:
            SERVER_ENGINE.__exit__(None, None, None)
        except Exception:
            traceback.print_exc()

        SERVER_ENGINE = None


def gemini_to_litertlm_message(gemini_content):
    role = gemini_content.get("role")

    if role == "model":
        role = "assistant"
    elif not role:
        role = "user"

    parts = gemini_content.get("parts", [])
    litertlm_parts = []
    tool_calls = []

    for part in parts:
        if "text" in part:
            litertlm_parts.append({
                "type": "text",
                "text": part["text"],
            })

        if "functionCall" in part:
            function_call = part["functionCall"]
            tool_calls.append({
                "function": {
                    "name": function_call.get("name"),
                    "arguments": function_call.get("args"),
                }
            })

        if "functionResponse" in part:
            function_response = part["functionResponse"]
            litertlm_parts.append({
                "type": "tool_response",
                "name": function_response.get("name"),
                "response": function_response.get("response"),
            })
            role = "tool"

    message = {
        "role": role,
    }

    if litertlm_parts:
        message["content"] = litertlm_parts

    if tool_calls:
        message["tool_calls"] = tool_calls

    return message


def litertlm_to_gemini_response(litertlm_response, finish_reason="STOP"):
    parts = []

    for item in litertlm_response.get("content", []):
        if item.get("type") == "text":
            parts.append({
                "text": item.get("text", "")
            })

    for tool_call in litertlm_response.get("tool_calls", []):
        function_data = tool_call.get("function", {})
        parts.append({
            "functionCall": {
                "name": function_data.get("name"),
                "args": function_data.get("arguments"),
            }
        })

    candidate = {
        "content": {
            "role": "model",
            "parts": parts,
        },
        "index": 0,
    }

    if finish_reason:
        candidate["finishReason"] = finish_reason

    return {
        "candidates": [
            candidate
        ]
    }


def extract_system_instruction(body):
    system_instruction = None
    data = body.get("systemInstruction") or body.get("system_instruction")

    if data:
        parts = data.get("parts", [])
        system_instruction = "".join(part.get("text", "") for part in parts)

    return system_instruction


def build_context_messages(body, messages):
    context_messages = []

    system_instruction = extract_system_instruction(body)

    if system_instruction:
        context_messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_instruction,
                }
            ],
        })

    context_messages.extend(messages)

    return context_messages



def now_stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    chunks.append(item.get("text"))
                elif isinstance(item.get("text"), str):
                    chunks.append(item.get("text"))
        return "".join(chunks)

    return str(value)


def litert_message(role, text):
    mapped_role = role

    if mapped_role == "model":
        mapped_role = "assistant"

    if mapped_role not in ("system", "user", "assistant", "tool"):
        mapped_role = "user"

    return {
        "role": mapped_role,
        "content": [
            {
                "type": "text",
                "text": safe_text(text),
            }
        ],
    }


def litert_text(response):
    parts = []

    if not isinstance(response, dict):
        return ""

    for item in response.get("content", []):
        if item.get("type") == "text":
            parts.append(item.get("text", ""))

    return "".join(parts)


def server_status_data():
    args = SERVER_ARGS
    data = {
        "ready": SERVER_ENGINE is not None,
        "engine_loaded": SERVER_ENGINE is not None,
        "uptime_seconds": max(0, int(time.time() - SERVER_STARTED_AT)),
    }

    if args is not None:
        data.update({
            "model_id": args.model_id,
            "model_path": model_path_from_id(args.model_id),
            "backend": args.backend,
            "speculative": args.speculative,
            "context_tokens": args.context_tokens,
            "reserve_output_tokens": args.reserve_output_tokens,
            "host": args.bind_host,
            "port": args.port,
        })

    return data


def openai_messages_to_litert(messages):
    converted = []

    for item in messages:
        if not isinstance(item, dict):
            continue

        role = item.get("role", "user")
        text = safe_text(item.get("content", ""))

        if role == "assistant":
            role = "assistant"
        elif role == "system":
            role = "system"
        elif role == "tool":
            role = "tool"
        else:
            role = "user"

        converted.append(litert_message(role, text))

    return converted


def openai_plain_response(model_id, text):
    return {
        "id": "chatcmpl-litert-local-" + str(int(time.time() * 1000)),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
    }


def openai_stream_chunk(model_id, text, finish_reason=None):
    choice = {
        "index": 0,
        "delta": {},
        "finish_reason": finish_reason,
    }

    if text:
        choice["delta"]["content"] = text

    return {
        "id": "chatcmpl-litert-local-" + str(int(time.time() * 1000)),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [choice],
    }

class GeminiLiteRTHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (
            self.address_string(),
            self.log_date_time_string(),
            fmt % args,
        ))

    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)

        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            return None

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/health", "/ready"):
            data = server_status_data()
            if data.get("ready"):
                self.send_json(200, data)
            else:
                self.send_json(503, data)
            return

        if path == "/status":
            self.send_json(200, server_status_data())
            return

        if path == "/v1/models":
            model_id = SERVER_ARGS.model_id if SERVER_ARGS is not None else DEFAULT_MODEL_ID
            self.send_json(200, {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "owned_by": "local",
                    }
                ],
            })
            return

        self.send_error(404, "Not Found")

    def handle_openai_chat(self, body):
        if not isinstance(body, dict):
            self.send_error(400, "Invalid JSON")
            return

        messages = body.get("messages", [])

        if not messages:
            self.send_error(400, "No messages provided")
            return

        converted = openai_messages_to_litert(messages)

        if not converted:
            self.send_error(400, "No usable messages provided")
            return

        converted, _removed = trim_litert_messages_to_context(
            converted,
            SERVER_ARGS.context_tokens,
            SERVER_ARGS.reserve_output_tokens,
            verbose=SERVER_ARGS.verbose,
        )

        last_message = converted.pop()
        context_messages = converted

        try:
            engine = load_engine_once(SERVER_ARGS)

            with engine.create_conversation(
                messages=context_messages,
                tools=None,
                automatic_tool_calling=False,
            ) as conversation:
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()

                    for chunk in conversation.send_message_async(last_message):
                        text = litert_text(chunk)
                        if text:
                            data = json.dumps(
                                openai_stream_chunk(SERVER_ARGS.model_id, text),
                                ensure_ascii=False,
                            )
                            self.wfile.write(("data: " + data + "\n\n").encode("utf-8"))
                            self.wfile.flush()

                    final = json.dumps(
                        openai_stream_chunk(SERVER_ARGS.model_id, "", finish_reason="stop"),
                        ensure_ascii=False,
                    )
                    self.wfile.write(("data: " + final + "\n\n").encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return

                response = conversation.send_message(last_message)
                self.send_json(200, openai_plain_response(SERVER_ARGS.model_id, litert_text(response)))

        except Exception as error:
            traceback.print_exc()
            try:
                self.send_error(500, str(error))
            except Exception:
                pass

    def do_POST(self):
        global SERVER_ARGS

        path = self.path.split("?")[0]

        if path == "/v1/chat/completions":
            body = self.read_json_body()
            self.handle_openai_chat(body)
            return

        match = GEMINI_RE.match(path)

        if not match:
            self.send_error(404, "Not Found")
            return

        requested_model = match.group(1).split(",")[0]
        operation = match.group(2)

        if requested_model != SERVER_ARGS.model_id:
            self.send_error(
                404,
                "Model "
                + requested_model
                + " not found. This server is serving "
                + SERVER_ARGS.model_id,
            )
            return

        body = self.read_json_body()

        if body is None:
            self.send_error(400, "Invalid JSON")
            return

        contents = body.get("contents", [])

        if not contents:
            self.send_error(400, "No contents provided")
            return

        system_tokens = estimate_output_tokens(extract_system_instruction(body) or "")
        contents, _removed = trim_contents_to_context(
            contents,
            max(1, SERVER_ARGS.context_tokens - system_tokens),
            SERVER_ARGS.reserve_output_tokens,
            verbose=SERVER_ARGS.verbose,
        )

        messages = [gemini_to_litertlm_message(item) for item in contents]
        last_message = messages.pop()
        context_messages = build_context_messages(body, messages)

        try:
            engine = load_engine_once(SERVER_ARGS)

            with engine.create_conversation(
                messages=context_messages,
                tools=None,
                automatic_tool_calling=False,
            ) as conversation:
                if operation == "streamGenerateContent":
                    self.handle_stream(conversation, last_message)
                else:
                    self.handle_plain(conversation, last_message)

        except Exception as error:
            traceback.print_exc()
            try:
                self.send_error(500, str(error))
            except Exception:
                pass

    def handle_plain(self, conversation, last_message):
        response = conversation.send_message(last_message)
        gemini_response = litertlm_to_gemini_response(response)
        self.send_json(200, gemini_response)

    def handle_stream(self, conversation, last_message):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            for chunk in conversation.send_message_async(last_message):
                response = litertlm_to_gemini_response(chunk, finish_reason="")
                data = json.dumps(response, ensure_ascii=False)
                self.wfile.write(("data: " + data + "\n\n").encode("utf-8"))
                self.wfile.flush()

            final_response = litertlm_to_gemini_response(
                {
                    "content": []
                },
                finish_reason="STOP",
            )

            final_data = json.dumps(final_response, ensure_ascii=False)
            self.wfile.write(("data: " + final_data + "\n\n").encode("utf-8"))
            self.wfile.flush()

        except Exception as error:
            traceback.print_exc()
            error_data = {
                "error": str(error)
            }

            try:
                self.wfile.write(
                    (
                        "event: error\ndata: "
                        + json.dumps(error_data, ensure_ascii=False)
                        + "\n\n"
                    ).encode("utf-8")
                )
                self.wfile.flush()
            except Exception:
                pass


def serve(args):
    global SERVER_ARGS

    SERVER_ARGS = args

    if args.auto_setup:
        if not setup_all(args):
            return 1

    try:
        load_engine_once(args)
    except Exception:
        print("ERROR: Failed to load engine.")
        traceback.print_exc()
        return 1

    address = (args.bind_host, args.port)

    print("Starting custom LiteRT-LM GPU API server:")
    print("Address:      ", args.bind_host + ":" + str(args.port))
    print("Model id:     ", args.model_id)
    print("Backend:      ", args.backend)
    print("Speculative:  ", args.speculative)
    print("Model context:", str(args.context_tokens) + " tokens client budget")
    print("Streaming URL:", "/v1beta/models/" + args.model_id + ":streamGenerateContent")
    print()

    try:
        with HTTPServer(address, GeminiLiteRTHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print()
        print("Shutting down server...")
    finally:
        close_engine()

    return 0


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_gemini_text(data):
    if not isinstance(data, dict):
        return ""

    candidates = data.get("candidates", [])

    if not candidates:
        return ""

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])

    output = []

    for part in parts:
        text = part.get("text")
        if text:
            output.append(text)

    return "".join(output)


def estimate_part_tokens(part):
    if "text" in part:
        return estimate_output_tokens(part.get("text", ""))

    # Tool payloads are uncommon in this simple client, but count them if a
    # future agent path stores them in history. Use compact JSON so the
    # estimate stays deterministic and dependency-free.
    try:
        return estimate_output_tokens(json.dumps(part, ensure_ascii=False, separators=(",", ":")))
    except Exception:
        return estimate_output_tokens(str(part))


def estimate_content_tokens(content):
    tokens = 4  # small role/message overhead estimate

    for part in content.get("parts", []):
        tokens += estimate_part_tokens(part)

    return tokens


def estimate_contents_tokens(contents):
    return sum(estimate_content_tokens(item) for item in contents)


def trim_contents_to_context(contents, context_tokens, reserve_output_tokens, verbose=False):
    if context_tokens <= 0:
        return contents, 0

    budget = max(1, context_tokens - max(0, reserve_output_tokens))
    trimmed = list(contents)
    removed = 0

    # Preserve the newest user prompt. Drop oldest chat turns first.
    while len(trimmed) > 1 and estimate_contents_tokens(trimmed) > budget:
        trimmed.pop(0)
        removed += 1

    if verbose and removed:
        print(
            "[context] trimmed "
            + str(removed)
            + " old message(s); sending ~"
            + str(estimate_contents_tokens(trimmed))
            + "/"
            + str(budget)
            + " input tokens, reserving "
            + str(max(0, reserve_output_tokens))
            + " output tokens",
            file=sys.stderr,
        )

    if verbose and estimate_contents_tokens(trimmed) > budget:
        print(
            "[context] current prompt alone is over the requested context budget; sending it anyway",
            file=sys.stderr,
        )

    return trimmed, removed



def estimate_litert_message_tokens(message):
    tokens = 4

    for part in message.get("content", []):
        if isinstance(part, dict):
            if part.get("type") == "text":
                tokens += estimate_output_tokens(part.get("text", ""))
            else:
                try:
                    tokens += estimate_output_tokens(json.dumps(part, ensure_ascii=False, separators=(",", ":")))
                except Exception:
                    tokens += estimate_output_tokens(str(part))

    return tokens


def estimate_litert_messages_tokens(messages):
    return sum(estimate_litert_message_tokens(item) for item in messages)


def trim_litert_messages_to_context(messages, context_tokens, reserve_output_tokens, verbose=False):
    if context_tokens <= 0:
        return messages, 0

    budget = max(1, context_tokens - max(0, reserve_output_tokens))
    trimmed = list(messages)
    removed = 0

    while len(trimmed) > 1 and estimate_litert_messages_tokens(trimmed) > budget:
        # Preserve a leading system message as long as possible.
        if len(trimmed) > 2 and trimmed[0].get("role") == "system":
            trimmed.pop(1)
        else:
            trimmed.pop(0)
        removed += 1

    if verbose and removed:
        print(
            "[context] trimmed "
            + str(removed)
            + " old message(s); sending ~"
            + str(estimate_litert_messages_tokens(trimmed))
            + "/"
            + str(budget)
            + " input tokens",
            file=sys.stderr,
        )

    return trimmed, removed


def read_text_file(path):
    if not path:
        return ""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except Exception as error:
        print("Could not read file " + str(path) + ": " + str(error), file=sys.stderr)
        return ""


def system_text_from_args(args):
    pieces = []

    file_text = read_text_file(getattr(args, "system_file", ""))
    inline_text = getattr(args, "system", "") or ""

    if file_text.strip():
        pieces.append(file_text.strip())

    if inline_text.strip():
        pieces.append(inline_text.strip())

    return "\n\n".join(pieces).strip()

def build_contents(prompt, history, args):
    contents = []

    for item in history:
        contents.append(item)

    contents.append({
        "role": "user",
        "parts": [
            {
                "text": prompt
            }
        ]
    })

    contents, _removed = trim_contents_to_context(
        contents,
        args.context_tokens,
        args.reserve_output_tokens,
        verbose=args.verbose,
    )

    return contents


def build_payload(prompt, history, args):
    payload = {
        "contents": build_contents(prompt, history, args)
    }

    system_text = system_text_from_args(args)

    if system_text:
        payload["systemInstruction"] = {
            "parts": [
                {
                    "text": system_text
                }
            ]
        }

    return payload


def generate_url(args):
    return (
        args.host.rstrip("/")
        + ":"
        + str(args.port)
        + "/v1beta/models/"
        + args.model_id
        + ":generateContent"
    )


def stream_url(args):
    return (
        args.host.rstrip("/")
        + ":"
        + str(args.port)
        + "/v1beta/models/"
        + args.model_id
        + ":streamGenerateContent"
    )


def post_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            return response.status, text

    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", errors="replace")
        return error.code, text

    except urllib.error.URLError as error:
        return 0, str(error)


def request_model(args, prompt, history):
    if args.stream:
        return request_stream(args, prompt, history)

    return request_plain(args, prompt, history)


def request_plain(args, prompt, history):
    url = generate_url(args)
    payload = build_payload(prompt, history, args)
    meter = ClientMeter(args.meter)

    status, text = post_json(url, payload, args.timeout)

    if status == 0:
        print("CONNECTION ERROR:")
        print(text)
        print()
        print("Start the custom GPU server first:")
        print("  py " + os.path.basename(sys.argv[0]) + " --serve")
        return None

    data = parse_json(text)

    if args.raw:
        if data is not None:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(text)

    if status < 200 or status >= 300:
        print("HTTP ERROR:", status)
        print(text)
        return None

    answer = extract_gemini_text(data).strip()

    if not answer:
        print("No text found.")
        if not args.raw:
            print(text)
        return None

    if not args.raw:
        print(answer)

    meter.add_text(answer)
    print_meter(args, meter, streaming=False)

    return answer


def request_stream(args, prompt, history):
    url = stream_url(args)
    payload = build_payload(prompt, history, args)
    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )

    try:
        return read_stream(args, request)

    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", errors="replace")
        print("HTTP ERROR:", error.code)
        print(text)
        return None

    except urllib.error.URLError as error:
        print("CONNECTION ERROR:")
        print(error)
        print()
        print("Start the custom GPU server first:")
        print("  py " + os.path.basename(sys.argv[0]) + " --serve")
        return None


def read_stream(args, request):
    full_answer = []
    event_lines = []
    meter = ClientMeter(args.meter)

    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")

            if line == "":
                if event_lines:
                    text = handle_sse_event(args, event_lines)

                    if text:
                        full_answer.append(text)
                        meter.add_text(text)

                    event_lines = []

                continue

            event_lines.append(line)

        if event_lines:
            text = handle_sse_event(args, event_lines)

            if text:
                full_answer.append(text)
                meter.add_text(text)

    answer = "".join(full_answer).strip()

    if not args.raw:
        print()

    print_meter(args, meter, streaming=True)

    if not answer:
        return None

    return answer


def handle_sse_event(args, event_lines):
    data_parts = []

    for line in event_lines:
        if line.startswith("data:"):
            data_parts.append(line[5:].strip())

    if not data_parts:
        return ""

    data_text = "\n".join(data_parts)

    if data_text == "[DONE]":
        return ""

    data = parse_json(data_text)

    if args.raw:
        if data is not None:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(data_text)

    text = extract_gemini_text(data)

    if text and not args.raw:
        print(text, end="", flush=True)

    return text


def add_history(history, prompt, answer):
    history.append({
        "role": "user",
        "parts": [
            {
                "text": prompt
            }
        ]
    })

    history.append({
        "role": "model",
        "parts": [
            {
                "text": answer
            }
        ]
    })


def safe_session_name(name):
    name = (name or DEFAULT_SESSION).strip()

    if not name:
        name = DEFAULT_SESSION

    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.strip("._-")

    if not name:
        name = DEFAULT_SESSION

    return name[:80]


def session_path(args, name=None):
    session_name = safe_session_name(name or args.session)
    return os.path.abspath(os.path.join(args.session_dir, session_name + ".json"))


def load_session_history(args):
    path = session_path(args)

    if getattr(args, "new_session", False):
        return []

    if getattr(args, "clear_session", False):
        try:
            os.remove(path)
            print("Cleared session:", safe_session_name(args.session))
        except FileNotFoundError:
            pass
        except Exception as error:
            print("Could not clear session:", error)
        return []

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        history = data.get("history", [])

        if isinstance(history, list):
            return history

    except FileNotFoundError:
        return []
    except Exception as error:
        print("Could not load session " + safe_session_name(args.session) + ": " + str(error))

    return []


def save_session_history(args, history):
    if not getattr(args, "save_session", True):
        return

    path = session_path(args)
    folder = os.path.dirname(path)

    try:
        if folder:
            os.makedirs(folder, exist_ok=True)

        data = {
            "version": 1,
            "updated": now_stamp(),
            "session": safe_session_name(args.session),
            "model_id": args.model_id,
            "system_hash_hint": estimate_output_tokens(system_text_from_args(args)),
            "history": history,
        }

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
    except Exception as error:
        print("Could not save session " + safe_session_name(args.session) + ": " + str(error))


def list_sessions(args):
    folder = os.path.abspath(args.session_dir)

    if not os.path.isdir(folder):
        print("No session folder found:", folder)
        return

    names = []

    for item in os.listdir(folder):
        if item.endswith(".json"):
            names.append(item[:-5])

    if not names:
        print("No sessions found in:", folder)
        return

    print("Sessions:")
    for name in sorted(names):
        marker = "*" if safe_session_name(args.session) == name else " "
        print(" " + marker + " " + name)


def print_history(history, limit=20):
    if not history:
        print("History is empty.")
        return

    start_index = max(0, len(history) - limit)

    for index, item in enumerate(history[start_index:], start=start_index + 1):
        role = item.get("role", "?")
        text = ""

        for part in item.get("parts", []):
            if isinstance(part, dict) and "text" in part:
                text += part.get("text", "")

        text = text.replace("\r", " ").replace("\n", " ").strip()

        if len(text) > 160:
            text = text[:157] + "..."

        print(str(index) + ". " + role + ": " + text)


def print_context_stats(args, history):
    contents = list(history)
    input_tokens = estimate_contents_tokens(contents)
    system_tokens = estimate_output_tokens(system_text_from_args(args))
    budget = max(1, args.context_tokens - args.reserve_output_tokens)

    print("Session:       ", safe_session_name(args.session))
    print("Messages:      ", len(history))
    print("History tokens:~" + str(input_tokens))
    print("System tokens: ~" + str(system_tokens))
    print("Input budget:  " + str(budget))
    print("Context total: " + str(args.context_tokens))
    print("Reserve output:" , args.reserve_output_tokens)


def http_get_json(url, timeout=3):
    request = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            data = parse_json(text)
            return response.status, data, text
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", errors="replace")
        return error.code, parse_json(text), text
    except Exception as error:
        return 0, None, str(error)


def base_url(args):
    return args.host.rstrip("/") + ":" + str(args.port)


def health_url(args):
    return base_url(args) + "/health"


def status_url(args):
    return base_url(args) + "/status"


def server_health_ready(args):
    status, data, _text = http_get_json(health_url(args), timeout=1)

    if status == 200 and isinstance(data, dict) and data.get("ready"):
        return True

    return False


def print_server_status(args):
    status, data, text = http_get_json(status_url(args), timeout=3)

    if status == 0:
        print("Server is not reachable:", text)
        return 1

    if data is not None:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(text)

    return 0 if status >= 200 and status < 300 else 1


def print_info(args):
    print("LiteRT-LM wrapper info")
    print("Script:       ", os.path.abspath(sys.argv[0]))
    print("Python:       ", sys.executable)
    print("LiteRT-LM cmd:", args.litert)
    print("litert_lm py: ", end="")

    spec = importlib.util.find_spec("litert_lm")
    if spec is None:
        print("not importable")
    else:
        print(spec.origin or "importable")

    print("Host:         ", args.host + ":" + str(args.port))
    print("Bind host:    ", args.bind_host)
    print("Model id:     ", args.model_id)
    print("Model path:   ", model_path_from_id(args.model_id))
    print("Repo:         ", args.repo)
    print("Model file:   ", args.model_file)
    print("Backend:      ", args.backend)
    print("Speculative:  ", args.speculative)
    print("Context:      ", str(args.context_tokens) + " tokens")
    print("Reserve:      ", str(args.reserve_output_tokens) + " output tokens")
    print("System file:  ", args.system_file or "")
    print("System inline:", "yes" if (args.system or "").strip() else "no")
    print("Session:      ", safe_session_name(args.session))
    print("Session file: ", session_path(args))
    print("Log file:     ", os.path.abspath(args.log_file))
    print("PID file:     ", os.path.abspath(args.pid_file))
    print("Server ready: ", "yes" if server_health_ready(args) else "no")


def print_chat_help():
    print("Commands:")
    print("  /help                         show commands")
    print("  /system                       show current system prompt")
    print("  /system TEXT                  set inline system prompt")
    print("  /systemfile PATH              set system prompt file")
    print("  /session NAME                 save current history and switch session")
    print("  /sessions                     list saved sessions")
    print("  /new                          clear current in-memory history")
    print("  /clear                        clear current session history file and memory")
    print("  /history                      show recent history")
    print("  /context                      show context/session token estimates")
    print("  /save                         save current session")
    print("  /load                         reload current session from disk")
    print("  /status                       fetch server /status")
    print("  /info                         print local wrapper info")
    print("  /log                          print server log tail")
    print("  /stream on|off                toggle streaming")
    print("  /raw on|off                   toggle raw JSON printing")
    print("  /meter on|off                 toggle meter")
    print("  /exit                         quit")


def set_bool_command(args, name, value):
    value = value.strip().lower()

    if value in ("on", "true", "yes", "1"):
        setattr(args, name, True)
        print(name + ": on")
        return True

    if value in ("off", "false", "no", "0"):
        setattr(args, name, False)
        print(name + ": off")
        return True

    print("Use: /" + name + " on   or   /" + name + " off")
    return False


def handle_chat_command(args, history, prompt):
    lower = prompt.lower()

    if lower in ("/help", "help"):
        print_chat_help()
        return history, True, False

    if lower in ("/exit", "/quit", "exit", "quit"):
        return history, True, True

    if lower == "/system":
        text = system_text_from_args(args)
        if text:
            print(text)
        else:
            print("No system prompt set.")
        return history, True, False

    if lower.startswith("/system "):
        args.system = prompt[len("/system "):].strip()
        print("System prompt updated.")
        return history, True, False

    if lower.startswith("/systemfile "):
        args.system_file = prompt[len("/systemfile "):].strip().strip('"')
        print("System file:", args.system_file)
        return history, True, False

    if lower.startswith("/session "):
        save_session_history(args, history)
        args.session = safe_session_name(prompt[len("/session "):].strip())
        history = load_session_history(args)
        print("Switched session:", safe_session_name(args.session), "| messages:", len(history))
        return history, True, False

    if lower == "/sessions":
        list_sessions(args)
        return history, True, False

    if lower == "/new":
        history = []
        print("Started empty in-memory history. Use /save to write it to this session.")
        return history, True, False

    if lower == "/clear":
        history = []
        try:
            os.remove(session_path(args))
        except FileNotFoundError:
            pass
        except Exception as error:
            print("Could not remove session file:", error)
        print("Current session cleared.")
        return history, True, False

    if lower == "/history":
        print_history(history)
        return history, True, False

    if lower == "/context":
        print_context_stats(args, history)
        return history, True, False

    if lower == "/save":
        save_session_history(args, history)
        print("Saved session:", safe_session_name(args.session))
        return history, True, False

    if lower == "/load":
        history = load_session_history(args)
        print("Loaded session:", safe_session_name(args.session), "| messages:", len(history))
        return history, True, False

    if lower == "/status":
        print_server_status(args)
        return history, True, False

    if lower == "/info":
        print_info(args)
        return history, True, False

    if lower == "/log":
        tail = read_log_tail(args.log_file, max_chars=8000)
        if tail:
            print(tail.strip())
        else:
            print("No log text found:", os.path.abspath(args.log_file))
        return history, True, False

    if lower.startswith("/stream "):
        set_bool_command(args, "stream", prompt[len("/stream "):])
        return history, True, False

    if lower.startswith("/raw "):
        set_bool_command(args, "raw", prompt[len("/raw "):])
        return history, True, False

    if lower.startswith("/meter "):
        set_bool_command(args, "meter", prompt[len("/meter "):])
        return history, True, False

    if prompt.startswith("/"):
        print("Unknown command. Type /help.")
        return history, True, False

    return history, False, False


def chat(args):
    history = load_session_history(args)

    print("LiteRT-LM GPU API chat")
    print("Host:       ", args.host + ":" + str(args.port))
    print("Model id:   ", args.model_id)
    print("Streaming:  ", "yes" if args.stream else "no")
    print("Meter:      ", "yes" if args.meter else "no")
    print("Max context:", str(args.context_tokens) + " tokens")
    print("Reserve:    ", str(args.reserve_output_tokens) + " output tokens")
    print("Session:    ", safe_session_name(args.session), "(" + str(len(history)) + " messages)")

    if system_text_from_args(args):
        print("System:     ", "set")
    else:
        print("System:     ", "none")

    print()
    print("Type /help for commands. Type /exit to quit.")
    print()

    while True:
        try:
            prompt = input("You> ").strip()
        except KeyboardInterrupt:
            print()
            break

        if not prompt:
            continue

        history, handled, should_exit = handle_chat_command(args, history, prompt)

        if should_exit:
            break

        if handled:
            print()
            continue

        print()
        print("Model> ", end="", flush=True)

        answer = request_model(args, prompt, history)

        if answer:
            add_history(history, prompt, answer)
            save_session_history(args, history)

        print()

    save_session_history(args, history)


def run_prompt_with_session(args, prompt):
    history = load_session_history(args)
    answer = request_model(args, prompt, history)

    if answer:
        add_history(history, prompt, answer)
        save_session_history(args, history)
        return 0

    return 1


def socket_host_from_args(args):
    host = args.host.strip()

    if host.startswith("http://"):
        host = host[len("http://"):]

    if host.startswith("https://"):
        host = host[len("https://"):]

    host = host.split("/")[0]
    host = host.split(":")[0]

    if not host:
        host = "127.0.0.1"

    return host


def server_port_open(args):
    host = socket_host_from_args(args)

    try:
        with socket.create_connection((host, args.port), timeout=0.5):
            return True
    except OSError:
        return False


def read_log_tail(log_file, max_chars=5000):
    try:
        with open(log_file, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_chars), os.SEEK_SET)
            data = handle.read()

        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def bool_server_flag(cmd, args, attribute, flag):
    if getattr(args, attribute):
        cmd.append(flag)


def false_server_flag(cmd, args, attribute, flag):
    if not getattr(args, attribute):
        cmd.append(flag)


def build_server_command(args):
    script = os.path.abspath(sys.argv[0])

    cmd = [
        sys.executable,
        "-u",
        script,
        "--serve",
        "--litert", args.litert,
        "--bind-host", args.bind_host,
        "--port", str(args.port),
        "--repo", args.repo,
        "--model-file", args.model_file,
        "--model-id", args.model_id,
        "--backend", args.backend,
        "--speculative", args.speculative,
        "--context-tokens", str(args.context_tokens),
        "--reserve-output-tokens", str(args.reserve_output_tokens),
    ]

    false_server_flag(cmd, args, "auto_setup", "--no-auto-setup")
    false_server_flag(cmd, args, "auto_python_install", "--no-python-install")
    bool_server_flag(cmd, args, "verbose", "--verbose")

    return cmd


def write_pid_file(args, pid):
    try:
        pid_file = os.path.abspath(args.pid_file)
        pid_dir = os.path.dirname(pid_file)

        if pid_dir:
            os.makedirs(pid_dir, exist_ok=True)

        with open(pid_file, "w", encoding="utf-8") as handle:
            handle.write(str(pid) + "\n")
    except Exception:
        pass


def read_pid_file(args):
    try:
        with open(args.pid_file, "r", encoding="utf-8") as handle:
            text = handle.read().strip()

        return int(text)
    except Exception:
        return None


def stop_server_from_pid_file(args):
    pid = read_pid_file(args)

    if pid is None:
        print("No server pid file found:", os.path.abspath(args.pid_file))
        return 1

    try:
        os.kill(pid, signal.SIGTERM)
        print("Stopped server process:", pid)
    except ProcessLookupError:
        print("Server process is not running anymore:", pid)
    except Exception as error:
        print("Could not stop server process", str(pid) + ":", error)
        return 1

    try:
        os.remove(args.pid_file)
    except Exception:
        pass

    return 0


def start_server_process(args):
    log_file = os.path.abspath(args.log_file)
    log_dir = os.path.dirname(log_file)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    command = build_server_command(args)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    creationflags = 0

    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    with open(log_file, "a", encoding="utf-8", errors="replace", buffering=1) as log:
        log.write("\n")
        log.write("=" * 72 + "\n")
        log.write("Launcher started server at " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        log.write("Command: " + cmd_text(command) + "\n")
        log.write("=" * 72 + "\n")
        log.flush()

        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
        )

        write_pid_file(args, process.pid)
        log.write("PID: " + str(process.pid) + "\n")
        log.flush()

    return process, log_file


def wait_for_server(args, process, log_file):
    deadline = time.time() + max(1, args.server_start_timeout)

    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            print("Server process exited before it became reachable.")
            print("Server log:", log_file)
            tail = read_log_tail(log_file)

            if tail:
                print()
                print("Last server log lines:")
                print(tail.strip())

            return False

        if server_health_ready(args):
            return True

        # Compatibility fallback for an older copy of this script that has no /health endpoint.
        if server_port_open(args):
            status, _data, _text = http_get_json(health_url(args), timeout=0.5)
            if status == 404:
                return True

        time.sleep(0.25)

    print("Timed out waiting for the server to become reachable.")
    print("Server log:", log_file)
    tail = read_log_tail(log_file)

    if tail:
        print()
        print("Last server log lines:")
        print(tail.strip())

    return False


def stop_server_process(process):
    if process is None:
        return

    if process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=8)
        return
    except Exception:
        pass

    try:
        process.kill()
        process.wait(timeout=5)
    except Exception:
        pass


def run_server_and_client(args, prompt):
    started_process = None
    log_file = os.path.abspath(args.log_file)

    if server_health_ready(args):
        print("Using already running ready server on " + socket_host_from_args(args) + ":" + str(args.port))
        print()
    elif server_port_open(args):
        print("Using already running server on " + socket_host_from_args(args) + ":" + str(args.port))
        print("Warning: /health did not report ready; this may be an older wrapper copy.")
        print()
    else:
        print("Starting server in background...")
        started_process, log_file = start_server_process(args)
        print("Server log:", log_file)
        print("Waiting until the server is reachable...")

        if not wait_for_server(args, started_process, log_file):
            stop_server_process(started_process)
            return 1

        print("Server is ready.")
        print()

    try:
        if args.chat or not prompt:
            chat(args)
            return 0

        return run_prompt_with_session(args, prompt)

    except KeyboardInterrupt:
        print()
        return 1

    finally:
        if started_process is not None and not args.keep_server:
            print()
            print("Stopping launcher-started server. Use --keep-server to leave it running.")
            stop_server_process(started_process)

            try:
                os.remove(args.pid_file)
            except Exception:
                pass


def load_config_file(path):
    if not path:
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, dict):
            return data

        print("Config file is not a JSON object:", path)
    except FileNotFoundError:
        print("Config file not found:", path)
    except Exception as error:
        print("Could not read config " + str(path) + ": " + str(error))

    return {}


def config_value(config, key, default):
    if key in config:
        return config[key]

    alt = key.replace("_", "-")
    if alt in config:
        return config[alt]

    return default


def write_default_config(path):
    data = {
        "host": DEFAULT_HOST,
        "bind_host": DEFAULT_BIND_HOST,
        "port": DEFAULT_PORT,
        "repo": DEFAULT_REPO,
        "model_file": DEFAULT_MODEL_FILE,
        "model_id": DEFAULT_MODEL_ID,
        "backend": "gpu",
        "speculative": "false",
        "context_tokens": DEFAULT_CONTEXT_TOKENS,
        "reserve_output_tokens": DEFAULT_RESERVE_OUTPUT_TOKENS,
        "system_file": "system.txt",
        "session": DEFAULT_SESSION,
        "session_dir": DEFAULT_SESSION_DIR,
        "stream": True,
        "meter": True,
        "log_file": DEFAULT_LOG_FILE,
        "pid_file": DEFAULT_PID_FILE,
        "keep_server": True,
    }

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)

    print("Wrote config:", os.path.abspath(path))


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    pre_args, _unknown = pre_parser.parse_known_args()
    config = load_config_file(pre_args.config)

    def cfg(key, default):
        return config_value(config, key, default)

    parser = argparse.ArgumentParser(
        description="Custom LiteRT-LM GPU + speculative decoding API server/client."
    )

    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text. If omitted, chat mode starts.",
    )

    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--write-default-config")

    parser.add_argument("--litert", default=cfg("litert", "litert-lm"))

    parser.add_argument("--host", default=cfg("host", DEFAULT_HOST))
    parser.add_argument("--bind-host", default=cfg("bind_host", DEFAULT_BIND_HOST))
    parser.add_argument("--port", type=int, default=int(cfg("port", DEFAULT_PORT)))

    parser.add_argument("--repo", default=cfg("repo", DEFAULT_REPO))
    parser.add_argument("--model-file", default=cfg("model_file", DEFAULT_MODEL_FILE))
    parser.add_argument("--model-id", default=cfg("model_id", DEFAULT_MODEL_ID))

    parser.add_argument("--backend", default=cfg("backend", "gpu"), choices=["cpu", "gpu"])
    parser.add_argument("--speculative", default=cfg("speculative", "true"), choices=["auto", "true", "false"])

    parser.add_argument(
        "--context-tokens",
        type=int,
        default=int(cfg("context_tokens", DEFAULT_CONTEXT_TOKENS)),
        help="Client-side max context budget. Default is 32768 for the LiteRT-LM Gemma 4 E2B package.",
    )
    parser.add_argument(
        "--reserve-output-tokens",
        type=int,
        default=int(cfg("reserve_output_tokens", DEFAULT_RESERVE_OUTPUT_TOKENS)),
        help="Tokens kept free inside the context budget for the next response.",
    )

    parser.add_argument("--system", default=cfg("system", ""))
    parser.add_argument("--system-file", default=cfg("system_file", ""))

    parser.add_argument("--session", default=cfg("session", DEFAULT_SESSION))
    parser.add_argument("--session-dir", default=cfg("session_dir", DEFAULT_SESSION_DIR))
    parser.add_argument("--new-session", action="store_true")
    parser.add_argument("--clear-session", action="store_true")
    parser.add_argument("--list-sessions", action="store_true")
    parser.add_argument("--no-save-session", dest="save_session", action="store_false")
    parser.set_defaults(save_session=bool(cfg("save_session", True)))

    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--no-auto-setup", dest="auto_setup", action="store_false")
    parser.set_defaults(auto_setup=bool(cfg("auto_setup", True)))

    parser.add_argument("--no-python-install", dest="auto_python_install", action="store_false")
    parser.set_defaults(auto_python_install=bool(cfg("auto_python_install", True)))

    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--no-meter", dest="meter", action="store_false")
    parser.set_defaults(meter=bool(cfg("meter", True)))

    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=bool(cfg("stream", True)))

    parser.add_argument("--timeout", type=int, default=int(cfg("timeout", 300)))

    parser.add_argument(
        "--client-only",
        action="store_true",
        help="Do not auto-start the server. Use the old behavior and connect to an already running server.",
    )
    parser.add_argument(
        "--log-file",
        default=cfg("log_file", DEFAULT_LOG_FILE),
        help="Where the launcher-started server writes stdout and stderr.",
    )
    parser.add_argument(
        "--pid-file",
        default=cfg("pid_file", DEFAULT_PID_FILE),
        help="Where the launcher stores the background server process id.",
    )
    parser.add_argument(
        "--server-start-timeout",
        type=int,
        default=int(cfg("server_start_timeout", DEFAULT_SERVER_START_TIMEOUT)),
        help="Seconds to wait for a launcher-started server to become reachable.",
    )
    parser.add_argument(
        "--keep-server",
        dest="keep_server",
        action="store_true",
        help="Leave the launcher-started server running after the client exits. This is the default.",
    )
    parser.add_argument(
        "--stop-server-on-exit",
        dest="keep_server",
        action="store_false",
        help="Stop a launcher-started server when the foreground client exits.",
    )
    parser.set_defaults(keep_server=bool(cfg("keep_server", True)))
    parser.add_argument(
        "--stop-server",
        action="store_true",
        help="Stop the background server recorded in the pid file, then exit.",
    )
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--status", action="store_true")

    args = parser.parse_args()

    if args.write_default_config:
        write_default_config(args.write_default_config)
        return

    args.session = safe_session_name(args.session)

    if args.list_sessions:
        list_sessions(args)
        return

    if args.info:
        print_info(args)
        return

    if args.status:
        sys.exit(print_server_status(args))

    if args.setup:
        if setup_all(args):
            return

        sys.exit(1)

    if args.stop_server:
        sys.exit(stop_server_from_pid_file(args))

    if args.serve:
        sys.exit(serve(args))

    prompt = " ".join(args.prompt).strip()

    if args.client_only:
        if args.chat or not prompt:
            chat(args)
            return

        sys.exit(run_prompt_with_session(args, prompt))

    sys.exit(run_server_and_client(args, prompt))


if __name__ == "__main__":
    main()
