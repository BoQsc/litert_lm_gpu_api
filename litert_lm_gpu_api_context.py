import argparse
import json
import os
import re
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

SERVER_ARGS = None
SERVER_ENGINE = None
SERVER_LITERT = None

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
    print("  py litert_lm_gpu_api.py --serve")

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
        print("  py litert_lm_gpu_api.py --setup")
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

    if args.verbose:
        try:
            litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)
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

    def do_POST(self):
        global SERVER_ARGS

        path = self.path.split("?")[0]
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
    return {
        "contents": build_contents(prompt, history, args)
    }


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
        print("  py litert_lm_gpu_api.py --serve")
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
        print("  py litert_lm_gpu_api.py --serve")
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


def chat(args):
    history = []

    print("LiteRT-LM GPU API chat")
    print("Host:       ", args.host + ":" + str(args.port))
    print("Model id:   ", args.model_id)
    print("Streaming:  ", "yes" if args.stream else "no")
    print("Meter:      ", "yes" if args.meter else "no")
    print("Max context:", str(args.context_tokens) + " tokens")
    print("Reserve:    ", str(args.reserve_output_tokens) + " output tokens")
    print()
    print("Type /exit to quit.")
    print()

    while True:
        try:
            prompt = input("You> ").strip()
        except KeyboardInterrupt:
            print()
            break

        if not prompt:
            continue

        if prompt.lower() in ("/exit", "/quit", "exit", "quit"):
            break

        print()
        print("Model> ", end="", flush=True)

        answer = request_model(args, prompt, history)

        if answer:
            add_history(history, prompt, answer)

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Custom LiteRT-LM GPU + speculative decoding API server/client."
    )

    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text. If omitted, chat mode starts.",
    )

    parser.add_argument("--litert", default="litert-lm")

    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--bind-host", default=DEFAULT_BIND_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)

    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)

    parser.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--speculative", default="true", choices=["auto", "true", "false"])

    parser.add_argument(
        "--context-tokens",
        type=int,
        default=DEFAULT_CONTEXT_TOKENS,
        help="Client-side max context budget. Default is 32768 for the LiteRT-LM Gemma 4 E2B package.",
    )
    parser.add_argument(
        "--reserve-output-tokens",
        type=int,
        default=DEFAULT_RESERVE_OUTPUT_TOKENS,
        help="Tokens kept free inside the context budget for the next response.",
    )

    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--no-auto-setup", dest="auto_setup", action="store_false")
    parser.set_defaults(auto_setup=True)

    parser.add_argument("--no-python-install", dest="auto_python_install", action="store_false")
    parser.set_defaults(auto_python_install=True)

    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--no-meter", dest="meter", action="store_false")
    parser.set_defaults(meter=True)

    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=True)

    parser.add_argument("--timeout", type=int, default=300)

    args = parser.parse_args()

    if args.setup:
        if setup_all(args):
            return

        sys.exit(1)

    if args.serve:
        sys.exit(serve(args))

    prompt = " ".join(args.prompt).strip()

    if args.chat or not prompt:
        chat(args)
        return

    request_model(args, prompt, [])


if __name__ == "__main__":
    main()