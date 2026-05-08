"""Agent-first coding variant built on top of litert_lm_gpu_api_full.

This keeps the LiteRT setup and model-loading logic from the full wrapper,
but swaps the user experience to a focused coding agent with native tool use.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import html
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from typing import Any

from litert_lm_gpu_api_full import (
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_ID,
    DEFAULT_REPO,
    DEFAULT_RESERVE_OUTPUT_TOKENS,
    DEFAULT_SESSION,
    DEFAULT_SESSION_DIR,
    close_engine,
    estimate_litert_messages_tokens,
    estimate_output_tokens,
    ensure_model,
    ensure_python_litert,
    litert_message,
    load_engine_once,
    model_path_from_id,
    now_stamp,
    safe_session_name,
    system_text_from_args,
    trim_litert_messages_to_context,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
START_CWD = os.getcwd()
DEFAULT_WORKSPACE = SCRIPT_DIR
DEFAULT_AGENT_SESSION = "coding-agent"
SESSION_SUFFIX = ".agent.json"
DEFAULT_FILE_LIMIT = 300
DEFAULT_SEARCH_LIMIT = 50
DEFAULT_READ_CHARS = 12000
DEFAULT_COMMAND_CHARS = 12000
DEFAULT_DIFF_CHARS = 20000
DEFAULT_FETCH_CHARS = 12000


AGENT_SYSTEM_PROMPT = """You are a local coding agent working inside a single workspace.

Your job is to help the user inspect code, plan changes, make edits, run tests,
and explain the result accurately.

Rules:
- Never guess about file contents, command output, or test results. Verify with tools.
- Start by exploring the workspace and relevant files before editing.
- Prefer the smallest change that solves the request.
- Use `calc` for exact math instead of mental arithmetic when precision matters.
- Use `list_files`, `search_text`, and `read_file` for context.
- Use `create_folder` for directories. Never use `run_command` for folder creation.
- If the user asks for a folder but does not give its name or path, ask one short clarifying question before acting.
- Use `write_file` for new files or full rewrites.
- Use `replace_text` for targeted edits after reading the file.
- Use `run_command` for tests and scripts, staying within the workspace.
- Use `git_status` and `git_diff` before concluding a change.
- Use `fetch_url` when you need online docs or references.
- Stay inside the workspace root unless the user explicitly asks otherwise.
- If the task is underspecified or risky, ask one short clarifying question.
- Keep user-facing responses concise. Do not reveal hidden reasoning.

Workflow:
1. Inspect the workspace and relevant files.
2. Make the minimal safe edit.
3. Run the most relevant checks you can.
4. Summarize what changed and what you verified.
"""


def workspace_root(args):
    return os.path.abspath(args.workspace)


def is_within_workspace(root, path):
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


def resolve_workspace_path(root, path):
    if not path:
        return root

    candidate = path if os.path.isabs(path) else os.path.join(root, path)
    candidate = os.path.abspath(candidate)

    if not is_within_workspace(root, candidate):
        raise ValueError("Path is outside the workspace: " + path)

    return candidate


def to_workspace_relpath(root, path):
    try:
        rel = os.path.relpath(path, root)
        if rel == ".":
            return "."
        return rel.replace("\\", "/")
    except Exception:
        return path


def preview_text(text, limit=240):
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= limit:
        return text

    return text[: max(0, limit - 3)] + "..."


def preview_json(value, limit=240):
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    return preview_text(text, limit=limit)


def message_text(message):
    if not isinstance(message, dict):
        return str(message)

    parts = message.get("content", [])
    chunks = []

    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    chunks.append(part.get("text"))
                elif isinstance(part.get("text"), str):
                    chunks.append(part.get("text"))
            elif isinstance(part, str):
                chunks.append(part)
    elif isinstance(parts, str):
        chunks.append(parts)

    return "".join(chunks)


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def write_text(path, content):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def numbered_text(text, start_line=1, end_line=0, max_chars=DEFAULT_READ_CHARS):
    lines = text.splitlines()
    if end_line <= 0 or end_line > len(lines):
        end_line = len(lines)

    if start_line < 1:
        start_line = 1

    if start_line > end_line:
        selected = []
    else:
        selected = lines[start_line - 1 : end_line]

    numbered = []
    for offset, line in enumerate(selected, start=start_line):
        numbered.append(f"{offset:>5} | {line}")

    output = "\n".join(numbered)
    truncated = False

    if len(output) > max_chars:
        output = output[:max_chars]
        truncated = True

    return output, len(lines), truncated


def run_process(args_list, cwd, timeout_seconds=120):
    try:
        completed = subprocess.run(
            args_list,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout or "",
            "timed_out": False,
            "command": list(args_list),
            "cwd": cwd,
        }
    except subprocess.TimeoutExpired as error:
        output = ""
        if error.stdout:
            output = error.stdout if isinstance(error.stdout, str) else error.stdout.decode("utf-8", errors="replace")
        return {
            "ok": False,
            "returncode": None,
            "stdout": output,
            "timed_out": True,
            "command": list(args_list),
            "cwd": cwd,
            "error": "Command timed out after " + str(timeout_seconds) + " seconds",
        }
    except FileNotFoundError:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "timed_out": False,
            "command": list(args_list),
            "cwd": cwd,
            "error": "Could not find executable: " + str(args_list[0]) if args_list else "Could not find executable",
        }
    except Exception as error:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "timed_out": False,
            "command": list(args_list),
            "cwd": cwd,
            "error": str(error),
        }


def run_git(root, git_args, timeout_seconds=30):
    return run_process(["git"] + list(git_args), cwd=root, timeout_seconds=timeout_seconds)


def parse_git_status(root):
    result = run_git(root, ["status", "--short", "--branch"])

    if not result["ok"]:
        return {
            "is_git_repo": False,
            "error": result.get("error") or "Git status failed",
            "stdout": result.get("stdout", ""),
        }

    stdout = result.get("stdout", "").strip()
    lines = stdout.splitlines() if stdout else []
    branch_line = lines[0] if lines else ""
    branch = ""
    if branch_line.startswith("##"):
        branch = branch_line[2:].strip()
    else:
        branch_result = run_git(root, ["branch", "--show-current"])
        if branch_result["ok"]:
            branch = (branch_result.get("stdout") or "").strip()

    body_lines = lines[1:] if len(lines) > 1 else []
    dirty = bool(body_lines)
    untracked = sum(1 for line in body_lines if line.startswith("??"))
    changed = sum(1 for line in body_lines if line and not line.startswith("??"))

    root_result = run_git(root, ["rev-parse", "--show-toplevel"])
    git_root = root if not root_result["ok"] else (root_result.get("stdout") or "").strip() or root

    return {
        "is_git_repo": True,
        "git_root": git_root,
        "branch": branch,
        "dirty": dirty,
        "changed_files": changed,
        "untracked_files": untracked,
        "status_text": stdout,
        "status_lines": body_lines,
    }


def git_diff_data(root, path="", staged=False, max_chars=DEFAULT_DIFF_CHARS):
    cmd = ["diff"]
    if staged:
        cmd.append("--staged")
    if path:
        cmd.extend(["--", path])

    result = run_git(root, cmd, timeout_seconds=60)

    if not result["ok"]:
        return {
            "ok": False,
            "error": result.get("error") or "Git diff failed",
            "stdout": result.get("stdout", ""),
        }

    text = result.get("stdout", "")
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    return {
        "ok": True,
        "path": path or "",
        "staged": bool(staged),
        "truncated": truncated,
        "diff": text,
    }


def safe_eval_calc(expression):
    allowed_binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }
    allowed_unary = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }
    allowed_funcs = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "floor": math.floor,
        "ceil": math.ceil,
        "trunc": math.trunc,
        "log": math.log,
        "log10": math.log10,
    }
    allowed_names = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant: " + repr(node.value))
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_binops:
                raise ValueError("Unsupported operator: " + op_type.__name__)
            return allowed_binops[op_type](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_unary:
                raise ValueError("Unsupported unary operator: " + op_type.__name__)
            return allowed_unary[op_type](eval_node(node.operand))
        if isinstance(node, ast.Name):
            if node.id in allowed_names:
                return allowed_names[node.id]
            raise ValueError("Unknown name: " + node.id)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed")
            func_name = node.func.id
            if func_name not in allowed_funcs:
                raise ValueError("Unsupported function: " + func_name)
            func = allowed_funcs[func_name]
            args = [eval_node(arg) for arg in node.args]
            kwargs = {}
            for kw in node.keywords:
                if kw.arg is None:
                    raise ValueError("Keyword unpacking is not allowed")
                kwargs[kw.arg] = eval_node(kw.value)
            return func(*args, **kwargs)
        if isinstance(node, ast.List):
            return [eval_node(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(eval_node(item) for item in node.elts)
        raise ValueError("Unsupported expression element: " + node.__class__.__name__)

    tree = ast.parse(expression, mode="eval")
    return eval_node(tree)


def summarize_tool_response(value):
    if isinstance(value, dict):
        if "error" in value and value["error"]:
            return "ERROR: " + preview_text(value["error"], limit=180)

        if "workspace_root" in value and "git_root" in value:
            branch = value.get("branch") or "unknown"
            git_root = value.get("git_root") or "none"
            dirty = "dirty" if value.get("dirty") else "clean"
            return "workspace " + preview_text(value.get("workspace_root", ""), 120) + " | git " + branch + " | " + dirty + " | " + preview_text(git_root, 120)

        if "paths" in value:
            count = value.get("count")
            if count is None:
                count = len(value.get("paths", []))
            suffix = " (truncated)" if value.get("truncated") else ""
            return str(count) + " file path(s)" + suffix

        if "matches" in value:
            count = value.get("count")
            if count is None:
                count = len(value.get("matches", []))
            suffix = " (truncated)" if value.get("truncated") else ""
            return str(count) + " match(es)" + suffix

        if "content" in value and "line_count" in value:
            suffix = " (truncated)" if value.get("truncated") else ""
            return (
                preview_text(value.get("path", ""), 120)
                + ": "
                + str(value.get("line_count", 0))
                + " line(s)"
                + suffix
            )

        if "returncode" in value and "stdout" in value:
            suffix = " (timed out)" if value.get("timed_out") else ""
            return "returncode " + str(value.get("returncode")) + suffix + " | " + str(len(value.get("stdout", ""))) + " char(s)"

        if "diff" in value:
            suffix = " (truncated)" if value.get("truncated") else ""
            return "diff ready" + suffix

        if "text" in value:
            return preview_text(value.get("text", ""), 180)

        return preview_json(value, limit=220)

    if isinstance(value, list):
        return str(len(value)) + " item(s)"

    return preview_text(value, limit=220)


def list_files_data(root, path="", glob_pattern="*", max_results=DEFAULT_FILE_LIMIT, include_hidden=False):
    search_root = resolve_workspace_path(root, path)

    results = []

    if os.path.isfile(search_root):
        rel = to_workspace_relpath(root, search_root)
        return {
            "root": to_workspace_relpath(root, os.path.dirname(search_root)),
            "count": 1,
            "paths": [rel],
            "truncated": False,
        }

    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--files", search_root]
        if include_hidden:
            cmd.extend(["--hidden", "--no-ignore-vcs"])
        if glob_pattern and glob_pattern != "*":
            cmd.extend(["-g", glob_pattern])
        result = run_process(cmd, cwd=root, timeout_seconds=30)
        if result["ok"]:
            for line in (result.get("stdout") or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                abs_path = os.path.abspath(os.path.join(search_root, line)) if not os.path.isabs(line) else os.path.abspath(line)
                if is_within_workspace(root, abs_path) and os.path.isfile(abs_path):
                    results.append(to_workspace_relpath(root, abs_path))
        else:
            results = []

    if not results:
        for dirpath, dirnames, filenames in os.walk(search_root):
            if not include_hidden:
                dirnames[:] = [name for name in dirnames if not name.startswith(".")]
                filenames = [name for name in filenames if not name.startswith(".")]
            for name in filenames:
                if glob_pattern and glob_pattern != "*" and not fnmatch.fnmatch(name, glob_pattern):
                    continue
                abs_path = os.path.abspath(os.path.join(dirpath, name))
                if is_within_workspace(root, abs_path):
                    results.append(to_workspace_relpath(root, abs_path))
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

    results = sorted(dict.fromkeys(results))
    truncated = len(results) > max_results
    if truncated:
        results = results[:max_results]

    return {
        "root": to_workspace_relpath(root, search_root),
        "count": len(results),
        "paths": results,
        "truncated": truncated,
    }


def search_text_data(root, query, path="", glob_pattern="*", max_results=DEFAULT_SEARCH_LIMIT, include_hidden=False, regex=False):
    search_root = resolve_workspace_path(root, path)
    matches = []

    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--line-number", "--color", "never"]
        if include_hidden:
            cmd.extend(["--hidden", "--no-ignore-vcs"])
        if not regex:
            cmd.append("--fixed-strings")
        if glob_pattern and glob_pattern != "*":
            cmd.extend(["-g", glob_pattern])
        cmd.append(query)
        cmd.append(search_root)

        result = run_process(cmd, cwd=root, timeout_seconds=60)
        if result["ok"]:
            for line in (result.get("stdout") or "").splitlines():
                if not line.strip():
                    continue
                parts = line.rsplit(":", 2)
                if len(parts) >= 3:
                    file_path, line_no, text = parts[0], parts[1], parts[2]
                    abs_path = os.path.abspath(file_path)
                    if not os.path.isabs(file_path):
                        abs_path = os.path.abspath(os.path.join(search_root, file_path))
                    if is_within_workspace(root, abs_path):
                        matches.append(
                            {
                                "path": to_workspace_relpath(root, abs_path),
                                "line": int(line_no) if line_no.isdigit() else line_no,
                                "text": text,
                            }
                        )
        else:
            matches = []

    if not matches:
        if os.path.isfile(search_root):
            file_paths = [search_root]
        else:
            file_paths = []
            for dirpath, dirnames, filenames in os.walk(search_root):
                if not include_hidden:
                    dirnames[:] = [name for name in dirnames if not name.startswith(".")]
                    filenames = [name for name in filenames if not name.startswith(".")]
                for name in filenames:
                    if glob_pattern and glob_pattern != "*" and not fnmatch.fnmatch(name, glob_pattern):
                        continue
                    file_paths.append(os.path.join(dirpath, name))

        needle = query if regex else None
        pattern = re.compile(query) if regex else None

        for file_path in file_paths:
            try:
                text = read_text(file_path)
            except Exception:
                continue

            for line_no, line in enumerate(text.splitlines(), start=1):
                ok = False
                if regex and pattern and pattern.search(line):
                    ok = True
                elif not regex and query in line:
                    ok = True

                if ok:
                    matches.append(
                        {
                            "path": to_workspace_relpath(root, file_path),
                            "line": line_no,
                            "text": line,
                        }
                    )
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break

    truncated = len(matches) > max_results
    if truncated:
        matches = matches[:max_results]

    return {
        "query": query,
        "root": to_workspace_relpath(root, search_root),
        "count": len(matches),
        "matches": matches,
        "truncated": truncated,
    }


def read_file_data(root, path, start_line=1, end_line=0, max_chars=DEFAULT_READ_CHARS):
    file_path = resolve_workspace_path(root, path)
    if not os.path.isfile(file_path):
        return {
            "ok": False,
            "path": to_workspace_relpath(root, file_path),
            "error": "File not found",
        }

    try:
        text = read_text(file_path)
    except Exception as error:
        return {
            "ok": False,
            "path": to_workspace_relpath(root, file_path),
            "error": str(error),
        }

    content, line_count, truncated = numbered_text(text, start_line=start_line, end_line=end_line, max_chars=max_chars)
    return {
        "ok": True,
        "path": to_workspace_relpath(root, file_path),
        "start_line": max(1, start_line),
        "end_line": 0 if end_line <= 0 else min(end_line, line_count),
        "line_count": line_count,
        "content": content,
        "truncated": truncated,
    }


def write_file_data(root, path, content, make_dirs=True):
    file_path = resolve_workspace_path(root, path)
    created = not os.path.exists(file_path)

    if make_dirs:
        folder = os.path.dirname(file_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    write_text(file_path, content)

    return {
        "ok": True,
        "path": to_workspace_relpath(root, file_path),
        "bytes_written": len(content.encode("utf-8")),
        "created": created,
    }


def create_folder_data(root, path, parents=True):
    folder_path = resolve_workspace_path(root, path)

    if not path:
        return {
            "ok": False,
            "path": to_workspace_relpath(root, folder_path),
            "error": "Folder path is required",
        }

    existed = os.path.isdir(folder_path)

    try:
        if parents:
            os.makedirs(folder_path, exist_ok=True)
        else:
            os.mkdir(folder_path)
    except FileExistsError:
        existed = True
    except Exception as error:
        return {
            "ok": False,
            "path": to_workspace_relpath(root, folder_path),
            "error": str(error),
        }

    return {
        "ok": True,
        "path": to_workspace_relpath(root, folder_path),
        "created": not existed,
        "existed": existed,
    }


def replace_text_data(root, path, old_text, new_text, replace_all=False):
    file_path = resolve_workspace_path(root, path)

    if not os.path.isfile(file_path):
        return {
            "ok": False,
            "path": to_workspace_relpath(root, file_path),
            "error": "File not found",
        }

    if not old_text:
        return {
            "ok": False,
            "path": to_workspace_relpath(root, file_path),
            "error": "Old text is empty",
        }

    try:
        original = read_text(file_path)
    except Exception as error:
        return {
            "ok": False,
            "path": to_workspace_relpath(root, file_path),
            "error": str(error),
        }

    count = original.count(old_text)
    if count == 0:
        return {
            "ok": False,
            "path": to_workspace_relpath(root, file_path),
            "error": "Old text not found",
        }

    if not replace_all and count != 1:
        return {
            "ok": False,
            "path": to_workspace_relpath(root, file_path),
            "error": "Old text appears " + str(count) + " times; use a narrower snippet or replace_all=true",
        }

    updated = original.replace(old_text, new_text, -1 if replace_all else 1)
    write_text(file_path, updated)

    return {
        "ok": True,
        "path": to_workspace_relpath(root, file_path),
        "replacements": count if replace_all else 1,
        "bytes_before": len(original.encode("utf-8")),
        "bytes_after": len(updated.encode("utf-8")),
    }


def delete_file_data(root, path):
    file_path = resolve_workspace_path(root, path)

    if not os.path.exists(file_path):
        return {
            "ok": True,
            "path": to_workspace_relpath(root, file_path),
            "deleted": False,
        }

    if os.path.isdir(file_path):
        return {
            "ok": False,
            "path": to_workspace_relpath(root, file_path),
            "error": "Refusing to delete a directory with delete_file",
        }

    os.remove(file_path)
    return {
        "ok": True,
        "path": to_workspace_relpath(root, file_path),
        "deleted": True,
    }


def command_is_unsafe(args_list):
    if not args_list:
        return True, "Empty command"

    first = str(args_list[0]).lower()
    command_text = " ".join(str(item) for item in args_list).lower()
    danger_patterns = [
        "rm -rf",
        "rm -fr",
        "rmdir /s",
        "del /s",
        "del /q",
        "remove-item -recurse",
        "remove-item -r",
        "git reset --hard",
        "git clean -fdx",
        "git clean -xdf",
    ]
    blocked_first = {
        "rm",
        "rmdir",
        "del",
        "erase",
        "format",
        "shutdown",
        "reboot",
        "poweroff",
        "halt",
        "mkfs",
        "fdisk",
    }

    if first in blocked_first:
        return True, "Refusing potentially destructive command: " + str(args_list[0])

    for pattern in danger_patterns:
        if pattern in command_text:
            return True, "Refusing destructive command pattern: " + pattern

    if first == "git":
        if " reset --hard" in command_text or " clean -fdx" in command_text or " clean -xdf" in command_text:
            return True, "Refusing destructive git command"

    return False, ""


def run_command_data(root, args_list, cwd="", timeout_seconds=120, max_chars=DEFAULT_COMMAND_CHARS):
    if not isinstance(args_list, list) or not args_list:
        return {
            "ok": False,
            "error": "args must be a non-empty list of command arguments",
            "returncode": None,
            "stdout": "",
        }

    for item in args_list:
        if not isinstance(item, str):
            return {
                "ok": False,
                "error": "all command arguments must be strings",
                "returncode": None,
                "stdout": "",
            }

    unsafe, reason = command_is_unsafe(args_list)
    if unsafe:
        return {
            "ok": False,
            "error": reason,
            "returncode": None,
            "stdout": "",
        }

    first = str(args_list[0]).lower()
    if first in ("mkdir", "md"):
        folder_args = [item for item in args_list[1:] if item not in ("-p", "/p", "--parents")]
        if not folder_args:
            return {
                "ok": False,
                "error": "mkdir needs a folder path. Use create_folder for directory creation.",
                "returncode": None,
                "stdout": "",
            }
        if len(folder_args) > 1:
            return {
                "ok": False,
                "error": "mkdir expects a single folder path in this wrapper.",
                "returncode": None,
                "stdout": "",
            }
        return create_folder_data(root, folder_args[0], parents=True)

    workdir = resolve_workspace_path(root, cwd) if cwd else root
    if not os.path.isdir(workdir):
        return {
            "ok": False,
            "error": "Working directory does not exist: " + cwd,
            "returncode": None,
            "stdout": "",
        }

    result = run_process(args_list, cwd=workdir, timeout_seconds=timeout_seconds)
    stdout = result.get("stdout", "")
    truncated = False
    if len(stdout) > max_chars:
        stdout = stdout[:max_chars]
        truncated = True

    return {
        "ok": result.get("ok", False),
        "command": result.get("command", list(args_list)),
        "cwd": workdir,
        "returncode": result.get("returncode"),
        "stdout": stdout,
        "truncated": truncated,
        "timed_out": result.get("timed_out", False),
        "error": result.get("error", ""),
    }


def fetch_url_data(url, max_chars=DEFAULT_FETCH_CHARS, timeout_seconds=15):
    if not url.startswith(("http://", "https://")):
        return {
            "ok": False,
            "error": "Only http and https URLs are supported",
            "url": url,
        }

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 LiteRT-Coding-Agent",
            "Accept": "*/*",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
            final_url = response.geturl()
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            text = body.decode(charset, errors="replace")

        title = ""
        if content_type == "text/html" or "<html" in text[:1000].lower():
            stripped = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
            title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", stripped)
            if title_match:
                title = html.unescape(title_match.group(1))
            stripped = re.sub(r"(?s)<[^>]+>", " ", stripped)
            text = html.unescape(stripped)
            text = re.sub(r"\s+", " ", text).strip()
        else:
            text = text.strip()

        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated = True

        return {
            "ok": True,
            "url": url,
            "final_url": final_url,
            "content_type": content_type,
            "title": preview_text(title, 180),
            "text": text,
            "truncated": truncated,
        }

    except urllib.error.HTTPError as error:
        body_text = error.read().decode("utf-8", errors="replace") if error.fp else ""
        truncated = False
        if len(body_text) > max_chars:
            body_text = body_text[:max_chars]
            truncated = True
        return {
            "ok": False,
            "url": url,
            "status": error.code,
            "error": str(error),
            "text": body_text,
            "truncated": truncated,
        }
    except urllib.error.URLError as error:
        return {
            "ok": False,
            "url": url,
            "error": str(error),
        }
    except Exception as error:
        return {
            "ok": False,
            "url": url,
            "error": str(error),
        }


def workspace_info_data(args):
    root = workspace_root(args)
    git = parse_git_status(root)
    return {
        "workspace_root": root,
        "launch_cwd": START_CWD,
        "script_dir": SCRIPT_DIR,
        "session": safe_session_name(args.session),
        "session_file": agent_session_path(args),
        "model_id": args.model_id,
        "model_path": model_path_from_id(args.model_id),
        "git": git,
    }


def print_dict(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def print_human_status(args):
    info = workspace_info_data(args)
    git = info.get("git", {})

    print("LiteRT coding agent")
    print("Workspace:    ", info.get("workspace_root"))
    print("Launch cwd:   ", info.get("launch_cwd"))
    print("Script dir:   ", info.get("script_dir"))
    print("Model id:     ", info.get("model_id"))
    print("Model path:   ", info.get("model_path"))
    print("Session:      ", info.get("session"))
    print("Session file: ", info.get("session_file"))
    print("Git repo:     ", "yes" if git.get("is_git_repo") else "no")
    if git.get("is_git_repo"):
        print("Git root:     ", git.get("git_root"))
        print("Branch:       ", git.get("branch") or "unknown")
        print("Dirty:        ", "yes" if git.get("dirty") else "no")
        print("Changed files:", git.get("changed_files", 0))
        print("Untracked:    ", git.get("untracked_files", 0))
    print("Context:      ", str(args.context_tokens) + " tokens")
    print("Reserve:      ", str(args.reserve_output_tokens) + " output tokens")
    print("Streaming:    ", "yes" if args.stream else "no")


def print_agent_help():
    print("Commands:")
    print("  /help                         show commands")
    print("  /status                       print workspace and git status")
    print("  /info                         print local wrapper info")
    print("  /files [PATH]                 list files under PATH")
    print("  /diff                         show git diff")
    print("  /context                      show context/session token estimates")
    print("  /history                      show recent turns")
    print("  /session NAME                 switch sessions")
    print("  /sessions                     list saved sessions")
    print("  /new                          clear in-memory session")
    print("  /clear                        clear current session file and memory")
    print("  /save                         save current session")
    print("  /load                         reload current session from disk")
    print("  /stream on|off                toggle streaming")
    print("  /exit                         quit")


def agent_session_path(args, name=None):
    session_name = safe_session_name(name or args.session)
    return os.path.abspath(os.path.join(args.session_dir, session_name + SESSION_SUFFIX))


def load_agent_history(args):
    path = agent_session_path(args)

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

        messages = data.get("messages")
        if isinstance(messages, list):
            return messages

        history = data.get("history")
        if isinstance(history, list):
            return history
    except FileNotFoundError:
        return []
    except Exception as error:
        print("Could not load session " + safe_session_name(args.session) + ": " + str(error))

    return []


def save_agent_history(args, history):
    if not getattr(args, "save_session", True):
        return

    path = agent_session_path(args)
    folder = os.path.dirname(path)

    try:
        if folder:
            os.makedirs(folder, exist_ok=True)

        data = {
            "version": 1,
            "updated": now_stamp(),
            "session": safe_session_name(args.session),
            "model_id": args.model_id,
            "workspace": workspace_root(args),
            "messages": history,
        }

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
    except Exception as error:
        print("Could not save session " + safe_session_name(args.session) + ": " + str(error))


def list_agent_sessions(args):
    folder = os.path.abspath(args.session_dir)

    if not os.path.isdir(folder):
        print("No session folder found:", folder)
        return

    names = []
    for item in os.listdir(folder):
        if item.endswith(SESSION_SUFFIX):
            names.append(item[: -len(SESSION_SUFFIX)])

    if not names:
        print("No sessions found in:", folder)
        return

    print("Sessions:")
    for name in sorted(names):
        marker = "*" if safe_session_name(args.session) == name else " "
        print(" " + marker + " " + name)


def print_agent_history(history, limit=20):
    if not history:
        print("History is empty.")
        return

    start_index = max(0, len(history) - limit)

    for index, item in enumerate(history[start_index:], start=start_index + 1):
        role = item.get("role", "?")
        text = preview_text(message_text(item), limit=180)
        print(str(index) + ". " + role + ": " + text)


def print_context_stats(args, history):
    input_tokens = estimate_litert_messages_tokens(history)
    system_tokens = estimate_output_tokens(agent_system_text(args))
    budget = max(1, args.context_tokens - args.reserve_output_tokens)

    print("Session:       ", safe_session_name(args.session))
    print("Messages:      ", len(history))
    print("History tokens:~" + str(input_tokens))
    print("System tokens: ~" + str(system_tokens))
    print("Input budget:  " + str(budget))
    print("Context total: " + str(args.context_tokens))
    print("Reserve output:", args.reserve_output_tokens)


def agent_system_text(args):
    pieces = [AGENT_SYSTEM_PROMPT.strip()]
    extra = system_text_from_args(args)
    if extra.strip():
        pieces.append(extra.strip())
    return "\n\n".join(pieces).strip()


def build_agent_tools(args):
    root = workspace_root(args)

    def workspace_info():
        """Show a compact snapshot of the current workspace and git state."""
        return workspace_info_data(args)

    def list_files(path: str = "", glob_pattern: str = "*", max_results: int = DEFAULT_FILE_LIMIT, include_hidden: bool = False):
        """List files inside the workspace.

        Args:
            path: Workspace-relative directory to scan, or a file path.
            glob_pattern: Optional file name glob like ``*.py``.
            max_results: Maximum number of file paths to return.
            include_hidden: Include dotfiles and hidden directories when true.
        """
        try:
            return list_files_data(root, path, glob_pattern, max_results, include_hidden)
        except Exception as error:
            return {"ok": False, "error": str(error)}

    def search_text(query: str, path: str = "", glob_pattern: str = "*", max_results: int = DEFAULT_SEARCH_LIMIT, include_hidden: bool = False, regex: bool = False):
        """Search for text inside files.

        Args:
            query: Text or regular expression to search for.
            path: Workspace-relative directory or file to search within.
            glob_pattern: Optional file name glob like ``*.py``.
            max_results: Maximum number of matches to return.
            include_hidden: Include dotfiles and hidden directories when true.
            regex: Treat the query as a regular expression when true.
        """
        try:
            return search_text_data(root, query, path, glob_pattern, max_results, include_hidden, regex)
        except Exception as error:
            return {"ok": False, "error": str(error)}

    def read_file(path: str, start_line: int = 1, end_line: int = 0, max_chars: int = DEFAULT_READ_CHARS):
        """Read a text file from the workspace.

        Args:
            path: Workspace-relative file path.
            start_line: 1-based starting line.
            end_line: 1-based ending line. Use 0 to read to EOF.
            max_chars: Maximum number of characters to return.
        """
        try:
            return read_file_data(root, path, start_line, end_line, max_chars)
        except Exception as error:
            return {"ok": False, "error": str(error)}

    def write_file(path: str, content: str, make_dirs: bool = True):
        """Create or replace a file with full content.

        Args:
            path: Workspace-relative file path.
            content: Entire file contents to write.
            make_dirs: Create missing parent directories when true.
        """
        try:
            return write_file_data(root, path, content, make_dirs)
        except Exception as error:
            return {"ok": False, "error": str(error)}

    def create_folder(path: str, parents: bool = True):
        """Create a folder inside the workspace.

        This is the preferred tool for directory creation.

        Args:
            path: Workspace-relative folder path to create.
            parents: Create missing parent directories when true.
        """
        try:
            return create_folder_data(root, path, parents)
        except Exception as error:
            return {"ok": False, "error": str(error)}

    def replace_text(path: str, old_text: str, new_text: str, replace_all: bool = False):
        """Replace a unique text block in a file.

        Args:
            path: Workspace-relative file path.
            old_text: Exact text to replace.
            new_text: Replacement text.
            replace_all: Replace every occurrence when true.
        """
        try:
            return replace_text_data(root, path, old_text, new_text, replace_all)
        except Exception as error:
            return {"ok": False, "error": str(error)}

    def delete_file(path: str):
        """Delete a file inside the workspace."""
        try:
            return delete_file_data(root, path)
        except Exception as error:
            return {"ok": False, "error": str(error)}

    def run_command(args: list[str], cwd: str = "", timeout_seconds: int = 120, max_chars: int = DEFAULT_COMMAND_CHARS):
        """Run a local command in the workspace.

        Use `create_folder` for folder creation instead of shell commands.

        Args:
            args: Command arguments. Pass one argument per list item.
            cwd: Workspace-relative working directory. Leave empty for the workspace root.
            timeout_seconds: Command timeout in seconds.
            max_chars: Maximum stdout characters to return.
        """
        try:
            return run_command_data(root, args, cwd, timeout_seconds, max_chars)
        except Exception as error:
            return {"ok": False, "error": str(error)}

    def calc(expression: str):
        """Evaluate a math expression safely.

        Args:
            expression: Arithmetic or math expression to evaluate.
        """
        try:
            value = safe_eval_calc(expression)
            return {"ok": True, "expression": expression, "result": value}
        except Exception as error:
            return {"ok": False, "expression": expression, "error": str(error)}

    def git_status():
        """Return the current git branch and working tree state."""
        try:
            return parse_git_status(root)
        except Exception as error:
            return {"is_git_repo": False, "error": str(error)}

    def git_diff(path: str = "", staged: bool = False, max_chars: int = DEFAULT_DIFF_CHARS):
        """Return a git diff for the workspace.

        Args:
            path: Optional workspace-relative file path.
            staged: Return the staged diff when true.
            max_chars: Maximum characters of diff text to return.
        """
        try:
            return git_diff_data(root, path, staged, max_chars)
        except Exception as error:
            return {"ok": False, "error": str(error)}

    def fetch_url(url: str, max_chars: int = DEFAULT_FETCH_CHARS, timeout_seconds: int = 15):
        """Fetch a web page or raw document.

        Args:
            url: HTTP or HTTPS URL to fetch.
            max_chars: Maximum characters of cleaned text to return.
            timeout_seconds: Request timeout in seconds.
        """
        try:
            return fetch_url_data(url, max_chars, timeout_seconds)
        except Exception as error:
            return {"ok": False, "url": url, "error": str(error)}

    return [
        workspace_info,
        list_files,
        search_text,
        read_file,
        write_file,
        create_folder,
        replace_text,
        delete_file,
        run_command,
        calc,
        git_status,
        git_diff,
        fetch_url,
    ]


class AgentToolEvents:
    def approve_tool_call(self, tool_call):
        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        name = function.get("name", "?")
        args = function.get("arguments", {})
        print()
        print("[tool] " + name + "(" + preview_json(args, limit=260) + ")")
        return True

    def process_tool_response(self, tool_response):
        print("[tool result] " + summarize_tool_response(tool_response))
        return tool_response


def build_conversation(args, history, automatic_tool_calling=True, tool_event_handler=None):
    engine = load_engine_once(args)
    system_message = agent_system_text(args)
    tools = build_agent_tools(args)
    if tool_event_handler is None and automatic_tool_calling:
        tool_event_handler = AgentToolEvents()
    extra_context = {
        "workspace_root": workspace_root(args),
        "session": safe_session_name(args.session),
        "script": os.path.basename(sys.argv[0]),
        "mode": "coding-agent",
    }

    trimmed_history, _removed = trim_litert_messages_to_context(
        history,
        args.context_tokens,
        args.reserve_output_tokens,
        verbose=args.verbose,
    )

    return engine.create_conversation(
        messages=trimmed_history,
        tools=tools,
        tool_event_handler=tool_event_handler,
        automatic_tool_calling=automatic_tool_calling,
        extra_context=extra_context,
        system_message=system_message,
    )


def run_agent_turn(args, history, prompt):
    assistant_text = []
    conversation = None

    try:
        with build_conversation(args, history) as conversation:
            if args.stream:
                print()
                print("Agent> ", end="", flush=True)
                for chunk in conversation.send_message_async(prompt):
                    text = message_text(chunk)
                    if text:
                        assistant_text.append(text)
                        print(text, end="", flush=True)
                print()
            else:
                print()
                response = conversation.send_message(prompt)
                text = message_text(response)
                if text:
                    assistant_text.append(text)
                    print(text)

    except KeyboardInterrupt:
        try:
            if conversation is not None:
                conversation.cancel_process()
        except Exception:
            pass
        print()
        print("Interrupted.")
        return None
    except Exception as error:
        print()
        print("Agent error:", error)
        traceback.print_exc()
        return None

    answer = "".join(assistant_text).strip()
    if not answer:
        print("No final text returned.")
        return None

    return answer


def handle_agent_command(args, history, prompt):
    lower = prompt.lower()

    if lower in ("/help", "help"):
        print_agent_help()
        return history, True, False

    if lower in ("/exit", "/quit", "exit", "quit"):
        return history, True, True

    if lower == "/status":
        print_human_status(args)
        return history, True, False

    if lower == "/info":
        print_agent_info(args)
        return history, True, False

    if lower.startswith("/files"):
        path = prompt[len("/files"):].strip()
        if path.startswith(" "):
            path = path.strip()
        try:
            data = list_files_data(workspace_root(args), path=path)
            print_dict(data)
        except Exception as error:
            print("Could not list files:", error)
        return history, True, False

    if lower == "/diff":
        data = git_diff_data(workspace_root(args))
        print_dict(data)
        return history, True, False

    if lower == "/history":
        print_agent_history(history)
        return history, True, False

    if lower == "/context":
        print_context_stats(args, history)
        return history, True, False

    if lower == "/session":
        print("Use /session NAME.")
        return history, True, False

    if lower.startswith("/session "):
        save_agent_history(args, history)
        args.session = safe_session_name(prompt[len("/session "):].strip())
        history = load_agent_history(args)
        print("Switched session:", safe_session_name(args.session), "| messages:", len(history))
        return history, True, False

    if lower == "/sessions":
        list_agent_sessions(args)
        return history, True, False

    if lower == "/new":
        history = []
        print("Started empty in-memory history. Use /save to write it to this session.")
        return history, True, False

    if lower == "/clear":
        history = []
        try:
            os.remove(agent_session_path(args))
        except FileNotFoundError:
            pass
        except Exception as error:
            print("Could not remove session file:", error)
        print("Current session cleared.")
        return history, True, False

    if lower == "/save":
        save_agent_history(args, history)
        print("Saved session:", safe_session_name(args.session))
        return history, True, False

    if lower == "/load":
        history = load_agent_history(args)
        print("Loaded session:", safe_session_name(args.session), "| messages:", len(history))
        return history, True, False

    if lower.startswith("/stream "):
        value = prompt[len("/stream "):].strip().lower()
        if value in ("on", "true", "yes", "1"):
            args.stream = True
            print("stream: on")
        elif value in ("off", "false", "no", "0"):
            args.stream = False
            print("stream: off")
        else:
            print("Use /stream on or /stream off")
        return history, True, False

    if prompt.startswith("/"):
        print("Unknown command. Type /help.")
        return history, True, False

    return history, False, False


def print_agent_info(args):
    print("LiteRT coding agent info")
    print("Script:       ", os.path.abspath(sys.argv[0]))
    print("Launch cwd:   ", START_CWD)
    print("Workspace:    ", workspace_root(args))
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
    print("Session file: ", agent_session_path(args))
    print("Session dir:  ", os.path.abspath(args.session_dir))
    print("Python:       ", sys.executable)
    print("litert_lm:    ", end="")
    spec = importlib.util.find_spec("litert_lm")
    if spec is None:
        print("not importable")
    else:
        print(spec.origin or "importable")


def print_agent_banner(args, history):
    info = workspace_info_data(args)
    git = info.get("git", {})

    print("LiteRT coding agent")
    print("Workspace:  ", info.get("workspace_root"))
    print("Model id:   ", args.model_id)
    print("Session:    ", safe_session_name(args.session), "(" + str(len(history)) + " messages)")
    print("Streaming:  ", "yes" if args.stream else "no")
    print("Git:        ", "yes" if git.get("is_git_repo") else "no")
    if git.get("is_git_repo"):
        print("Branch:     ", git.get("branch") or "unknown")
        print("Dirty:      ", "yes" if git.get("dirty") else "no")
    print()
    print("Type /help for commands. Type /exit to quit.")
    print()


def agent_chat(args):
    history = load_agent_history(args)
    print_agent_banner(args, history)

    while True:
        try:
            prompt = input("Task> ").strip()
        except KeyboardInterrupt:
            print()
            break
        except EOFError:
            print()
            break

        if not prompt:
            continue

        history, handled, should_exit = handle_agent_command(args, history, prompt)
        if should_exit:
            break

        if handled:
            print()
            continue

        print()
        answer = run_agent_turn(args, history, prompt)
        if answer:
            history.append(litert_message("user", prompt))
            history.append(litert_message("assistant", answer))
            save_agent_history(args, history)
        print()

    save_agent_history(args, history)


def run_agent_prompt(args, prompt):
    history = load_agent_history(args)
    answer = run_agent_turn(args, history, prompt)
    if answer:
        history.append(litert_message("user", prompt))
        history.append(litert_message("assistant", answer))
        save_agent_history(args, history)
        return 0
    return 1


def preload_agent_runtime(args):
    print()
    print("Preloading LiteRT-LM engine before opening the prompt...")
    load_engine_once(args)
    print("Priming agent runtime...")

    try:
        with build_conversation(
            args,
            [],
            automatic_tool_calling=False,
            tool_event_handler=None,
        ) as conversation:
            conversation.send_message(
                "Warm up the runtime. Do not use tools. Reply with a single short confirmation."
            )
    except Exception as error:
        print("Warm-up skipped:", error)

    print("Ready.")
    print()


def setup_agent_all(args):
    if not ensure_python_litert(args):
        return False

    if not ensure_model(args):
        return False

    print()
    print("Setup complete.")
    print("Python import: OK")
    print("Model import:  OK")
    print("Agent is ready to start.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Agent-first LiteRT-LM coding wrapper."
    )

    parser.add_argument(
        "prompt",
        nargs="*",
        help="Task text. If omitted, interactive agent mode starts.",
    )

    parser.add_argument("--litert", default="litert-lm")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--speculative", default="true", choices=["auto", "true", "false"])
    parser.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    parser.add_argument("--reserve-output-tokens", type=int, default=DEFAULT_RESERVE_OUTPUT_TOKENS)
    parser.add_argument("--system", default="")
    parser.add_argument("--system-file", default="")
    parser.add_argument("--session", default=DEFAULT_AGENT_SESSION)
    parser.add_argument("--session-dir", default=DEFAULT_SESSION_DIR)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--new-session", action="store_true")
    parser.add_argument("--clear-session", action="store_true")
    parser.add_argument("--list-sessions", action="store_true")
    parser.add_argument("--no-save-session", dest="save_session", action="store_false")
    parser.set_defaults(save_session=True)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--no-auto-setup", dest="auto_setup", action="store_false")
    parser.set_defaults(auto_setup=True)
    parser.add_argument("--no-python-install", dest="auto_python_install", action="store_false")
    parser.set_defaults(auto_python_install=True)
    parser.add_argument("--stream", dest="stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--status", action="store_true")

    args = parser.parse_args()

    if not os.path.isabs(args.workspace):
        args.workspace = os.path.abspath(os.path.join(START_CWD, args.workspace))
    else:
        args.workspace = os.path.abspath(args.workspace)

    if not os.path.isdir(args.workspace):
        print("Workspace does not exist:", args.workspace)
        sys.exit(1)

    try:
        os.chdir(args.workspace)
    except Exception as error:
        print("Could not switch to workspace:", error)
        sys.exit(1)

    args.session = safe_session_name(args.session)

    if args.list_sessions:
        list_agent_sessions(args)
        return

    if args.info:
        print_agent_info(args)
        return

    if args.status:
        print_human_status(args)
        return

    if args.setup:
        if setup_agent_all(args):
            return
        sys.exit(1)

    prompt = " ".join(args.prompt).strip()

    if args.auto_setup:
        if not setup_agent_all(args):
            sys.exit(1)

    try:
        preload_agent_runtime(args)
        if not prompt:
            agent_chat(args)
            return

        sys.exit(run_agent_prompt(args, prompt))
    finally:
        close_engine()


if __name__ == "__main__":
    main()
