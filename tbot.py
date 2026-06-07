#!/usr/bin/env python3
"""tbot - Terminal chatbot for OpenRouter with PC tool support."""

import os, sys, json, time, subprocess, platform, re, html, socket, urllib.parse
import argparse, textwrap, atexit, tempfile, shutil, shlex
from pathlib import Path
import requests
try:
    import readline
except ImportError:
    readline = None

CONFIG_DIR = Path.home() / ".config" / "tbot"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history.txt"
SKILLS_DIR = CONFIG_DIR / "skills"
SYSTEM_PROMPT_FILE = CONFIG_DIR / "system_prompt.txt"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    GRAY = "\033[90m"

# ── Tool definitions (opencode-inspired) ──────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "invalid",
            "description": "Reports an invalid tool call. Do not use this tool directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "description": "The tool name that was called with invalid arguments"},
                    "error": {"type": "string", "description": "Description of the validation error"},
                },
                "required": ["tool", "error"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "question",
            "description": "Ask the user one or more questions and get their answers. Use this when you need clarification or additional information from the user to proceed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string", "description": "The complete question to ask"},
                                "header": {"type": "string", "description": "Very short label (max 30 chars)"},
                                "options": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {"type": "string", "description": "Display text (1-5 words)"},
                                            "description": {"type": "string", "description": "Explanation of choice"},
                                        },
                                        "required": ["label", "description"],
                                    },
                                    "description": "Available choices (omit for free-text input)",
                                },
                                "multiple": {"type": "boolean", "description": "Allow selecting more than one option"},
                            },
                            "required": ["question", "header", "options"],
                        },
                        "description": "Questions to ask the user",
                    },
                },
                "required": ["questions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command on the local machine with timeout and working directory support. Runs in the project directory by default. 'cd' commands update the persistent working directory for subsequent tool calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute"},
                    "description": {"type": "string", "description": "Clear concise description of what this command does in 5-10 words"},
                    "timeout": {"type": "integer", "description": "Timeout in milliseconds (default: 120000)", "default": 120000},
                    "workdir": {"type": "string", "description": "Working directory (relative paths resolve against current directory). Use this instead of 'cd' commands for one-off directory changes."},
                },
                "required": ["command", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file or directory. Read each file ONCE only — re-reading will be warned and blocked. Read all files you need in parallel in a single round.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Path to the file or directory (relative paths resolve against current directory — use them)"},
                    "offset": {"type": "integer", "description": "The line number to start reading from (1-indexed)"},
                    "limit": {"type": "integer", "description": "The maximum number of lines to read (defaults to 2000)"},
                },
                "required": ["filePath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Fast file pattern matching tool that works with any codebase size. Supports glob patterns like '**/*.js' or 'src/**/*.ts'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The glob pattern to match files against"},
                    "path": {"type": "string", "description": "The directory to search in. Defaults to current working directory."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Fast content search tool that searches file contents using regular expressions. Returns file paths and line numbers with matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The regex pattern to search for in file contents"},
                    "path": {"type": "string", "description": "The directory to search in. Defaults to current working directory."},
                    "include": {"type": "string", "description": "File pattern to include (e.g. '*.js', '*.{ts,tsx}')"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Performs exact string replacements in files. Replaces oldString with newString. Supports replaceAll and multiple fallback strategies for fuzzy matching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Path to the file (relative paths resolve against current directory)"},
                    "oldString": {"type": "string", "description": "The text to replace"},
                    "newString": {"type": "string", "description": "The text to replace it with (must be different from oldString)"},
                    "replaceAll": {"type": "boolean", "description": "Replace all occurrences of oldString (default false)"},
                },
                "required": ["filePath", "oldString", "newString"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Writes a file to the local filesystem. Overwrites existing file if one exists. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Path to the file (relative paths resolve against current directory)"},
                    "content": {"type": "string", "description": "The content to write to the file"},
                },
                "required": ["filePath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Launch a new agent to handle complex multistep tasks autonomously. Use this for tasks that need independent research or processing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "A short (3-5 words) description of the task"},
                    "prompt": {"type": "string", "description": "The task for the agent to perform"},
                    "subagent_type": {
                        "type": "string",
                        "enum": ["general", "explore"],
                        "description": "The type of agent to use: 'general' for research/execution, 'explore' for codebase exploration",
                    },
                    "command": {"type": "string", "description": "The command that triggered this task"},
                },
                "required": ["description", "prompt", "subagent_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "webfetch",
            "description": "Fetches content from a specified URL. Returns the content in text, markdown, or HTML format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch content from"},
                    "format": {
                        "type": "string",
                        "enum": ["text", "markdown", "html"],
                        "description": "The format to return the content in (text, markdown, or html). Defaults to markdown.",
                    },
                    "timeout": {"type": "integer", "description": "Optional timeout in seconds (max 120)"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todowrite",
            "description": "Write the task list to TASK.md. Call EXACTLY ONCE to plan (all tasks pending except first in_progress). Then call again ONLY when a task is actually completed (update its status to 'completed' and advance the next). Do NOT call for any other reason. If the output warns about looping, you violated this rule — stop calling and use edit/write/bash.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "Brief description of the task"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "cancelled"],
                                    "description": "Current status of the task",
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                    "description": "Priority level of the task",
                                },
                            },
                            "required": ["content", "status", "priority"],
                        },
                        "description": "The updated todo list",
                    },
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "websearch",
            "description": "Search the web using DuckDuckGo and fetch page content. Returns page titles, snippets, URLs, and the actual text content of each page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "numResults": {"type": "integer", "description": "Number of search results to return (default: 8)"},
                    "livecrawl": {
                        "type": "string",
                        "enum": ["fallback", "preferred"],
                        "description": "Live crawl mode (default: fallback)",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["auto", "fast", "deep"],
                        "description": "Search type (default: auto)",
                    },
                    "contextMaxCharacters": {
                        "type": "integer",
                        "description": "Maximum characters for context string (default: 10000)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill",
            "description": "[ONE-TIME] Load a specialized skill. Call ONCE per skill — the instructions stay in context. Do NOT reload the same skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name of the skill to load from installed skills"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified format patch to one or more files. The patch text must contain standard unified diff hunks with file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patchText": {"type": "string", "description": "The full unified diff patch text describing all changes to be made"},
                },
                "required": ["patchText"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get OS, architecture, hostname, and other system information.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "install_skill",
            "description": "Download and install a skill from a URL. The URL must point directly to a raw SKILL.md file or a git repository containing SKILL.md files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to raw SKILL.md, GitHub repo URL, or gh:user/repo"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": "Create a new skill with instructions. Skills are instruction sets that guide the model on how to perform specific tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill name — lowercase, alphanumeric, underscores and hyphens"},
                    "description": {"type": "string", "description": "Short description of what the skill does"},
                    "content": {"type": "string", "description": "Markdown instructions for the model"},
                    "schema": {"type": "object", "description": "Optional JSON Schema for skill parameters"},
                },
                "required": ["name", "description", "content"],
            },
        },
    },
]

# ── Alias: read_file → read ─────────────────────────────────
_read_fn = next(t["function"] for t in TOOLS if t["function"]["name"] == "read")
TOOLS.append({
    "type": "function",
    "function": {
        **_read_fn,
        "name": "read_file",
        "description": "Alias for read. Reads a file. Relative paths work fine — do NOT retry with an absolute path.",
    },
})
del _read_fn

# ── Project directory context ────────────────────────────────

CURRENT_DIR = Path(os.getcwd()).resolve()


def _chdir(path):
    global CURRENT_DIR
    CURRENT_DIR = Path(path).expanduser().resolve()


def _resolve_path(p, *, also_try_cwd=True):
    p = Path(p)
    if not p.is_absolute():
        resolved = (CURRENT_DIR / p).expanduser().resolve()
        if also_try_cwd and not resolved.exists():
            alt = (Path(os.getcwd()) / p).expanduser().resolve()
            if alt.exists():
                return alt
        return resolved
    return p.expanduser().resolve()


# ── Handler helpers ──────────────────────────────────────────

MAX_TOOL_OUTPUT = 8000

_READ_ONLY_TOOLS = frozenset({
    "read", "read_file", "glob", "grep", "todowrite",
    "question", "webfetch", "websearch", "get_system_info",
    "skill", "skills", "invalid",
})

def _is_read_only_tool(name):
    return name in _READ_ONLY_TOOLS


def _truncate_output(text):
    if len(text) > MAX_TOOL_OUTPUT:
        return text[:MAX_TOOL_OUTPUT] + f"\n... (truncated, {len(text)} total chars)"
    return text


def _clean_html_text(html_text):
    """Strip HTML tags and extract readable paragraphs from HTML."""
    text = re.sub(r'<script[^>]*>.*?</script>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '\n', text)
    text = html.unescape(text)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    filtered = [l for l in lines if len(l) > 40]
    return '\n'.join(filtered) if filtered else '\n'.join(lines)


def _fetch_page_text(url, max_chars=4000):
    """Fetch a URL and extract readable text content. Returns None on failure."""
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; tbot/1.0; +https://github.com/user/tbot)",
            "Accept": "text/html,text/plain,*/*",
        })
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "text/html" not in ct and "text/plain" not in ct:
            return None
        text = _clean_html_text(resp.text)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return text if len(text) > 100 else None
    except Exception:
        return None


# ── New / upgraded handlers (opencode-inspired) ──────────────

def handle_invalid(args):
    return f"The arguments provided to the tool '{args.get('tool', '?')}' are invalid: {args.get('error', 'unknown error')}"


def handle_question(args):
    questions = args.get("questions", [])
    answers = []
    for q in questions:
        header = q.get("header", "")
        question = q.get("question", "")
        options = q.get("options", [])
        multiple = q.get("multiple", False)
        print(f"\n{C.CYAN}── {header} ──{C.RESET}")
        print(f"{C.BOLD}{question}{C.RESET}")
        if options:
            for i, opt in enumerate(options, 1):
                desc = opt.get("description", "")
                print(f"  {C.YELLOW}{i}.{C.RESET} {opt['label']}  {C.GRAY}{desc}{C.RESET}")
            print(f"  {C.YELLOW}0.{C.RESET} Type your own answer")
            while True:
                try:
                    raw = input(f"{C.GREEN}choice{C.RESET} {'(comma-separated)' if multiple else ''}: ").strip()
                    if not raw:
                        continue
                    parts = [p.strip() for p in raw.split(",") if p.strip()]
                    selected = []
                    for p in parts:
                        if p == "0":
                            custom = input(f"{C.GREEN}your answer:{C.RESET} ").strip()
                            if custom:
                                selected.append(custom)
                        else:
                            try:
                                idx = int(p) - 1
                                if 0 <= idx < len(options):
                                    selected.append(options[idx]["label"])
                            except ValueError:
                                selected.append(p)
                    if selected:
                        answers.append(selected)
                        break
                except (EOFError, KeyboardInterrupt):
                    answers.append([])
                    break
        else:
            try:
                ans = input(f"{C.GREEN}answer:{C.RESET} ").strip()
                answers.append([ans] if ans else [])
            except (EOFError, KeyboardInterrupt):
                answers.append([])
    formatted = ", ".join(
        f'"{q.get("question", "")}"="{", ".join(a) if a else "Unanswered"}"'
        for q, a in zip(questions, answers)
    )
    return f"User has answered your questions: {formatted}. You can now continue with the user's answers in mind."


_CD_RE = re.compile(r'^\s*cd\s+(.+?)(?:\s*[;&|#]|$)')


def _update_cwd(cmd, last_cwd):
    """Detect `cd <dir>` in command and update CURRENT_DIR."""
    m = _CD_RE.match(cmd)
    if not m:
        return
    target = m.group(1).strip().strip("'\"")
    resolved = _resolve_path(target)
    if resolved.is_dir():
        _chdir(resolved)


_output_re = re.compile(r'(?:-(?:o|O|output)\s+|\>\s*|\>\>\s*)(\S+)')

def _list_outputs(cmd, cwd):
    files = _output_re.findall(cmd)
    if not files:
        return ""
    lines = []
    for f in files:
        f = f.strip("\"'")
        p = Path(f) if f.startswith("/") else Path(cwd) / f
        if p.exists():
            lines.append(f"→ {p.name} ({p.stat().st_size} bytes)")
    return "\n" + "\n".join(lines) if lines else ""

def handle_bash(args):
    cmd = _pick(args, "command", "cmd")
    if not cmd:
        return "Error: command is required"
    desc = args.get("description", "")
    timeout_ms = args.get("timeout", 120000)
    workdir = args.get("workdir")
    timeout_s = timeout_ms / 1000
    cwd = str(_resolve_path(workdir)) if workdir else str(CURRENT_DIR)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_s, cwd=cwd)
        out = r.stdout
        if r.stderr:
            out += "\n--- stderr ---\n" + r.stderr
        out += f"\n--- exit code: {r.returncode} ---"
        if r.returncode == 0:
            out += _list_outputs(cmd, cwd)
        _update_cwd(cmd, cwd)
        result = out.strip() or f"(no output)  [exit {r.returncode}]"
        return _truncate_output(result + f"\n\n<cwd>{CURRENT_DIR}</cwd>")
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout_s}s (exit: -1)\n\n<cwd>{CURRENT_DIR}</cwd>"
    except Exception as e:
        return f"Error: {e} (exit: -1)\n\n<cwd>{CURRENT_DIR}</cwd>"


# --- doom loop detection ---
_doom_trail = []

def _check_doom_loop(tool_calls):
    global _doom_trail
    for tc in tool_calls:
        name = tc["function"]["name"]
        args_raw = tc["function"].get("arguments", "{}")
        try:
            normalized = json.dumps(json.loads(args_raw), sort_keys=True)
        except json.JSONDecodeError:
            normalized = args_raw
        _doom_trail.append((name, normalized))
    if len(_doom_trail) > 9:
        _doom_trail = _doom_trail[-9:]
    if len(_doom_trail) >= 3:
        last_3 = _doom_trail[-3:]
        if all(t == last_3[0] for t in last_3):
            _doom_trail.clear()
            return (
                f"DOOM LOOP: You called {last_3[0][0]} 3 times with identical arguments. "
                "This is a loop. The tool was NOT executed. "
                "Use different arguments, a different tool, or respond with text."
            )
    return None

# --- read tool ---
def handle_read(args):
    raw = args.get("filePath", "")
    if not raw:
        return "Error: filePath is required"
    filepath = str(_resolve_path(raw))
    offset = args.get("offset", 1)
    limit = args.get("limit", 2000)
    p = Path(filepath)
    if not p.exists():
        return (
            f"Error: file not found: {filepath}\n"
            f"(raw input: {raw!r}, CURRENT_DIR: {CURRENT_DIR}, "
            f"os.getcwd(): {os.getcwd()})\n"
            f"Use an absolute path like {Path(os.getcwd()) / raw}"
        )
    if p.is_dir():
        entries = sorted(
            f"{e.name}/" if e.is_dir() else e.name
            for e in p.iterdir()
        )
        total = len(entries)
        start = max(0, offset - 1)
        sliced = entries[start:start + limit]
        result = f"<path>{filepath}</path>\n<type>directory</type>\n<entries>\n"
        result += "\n".join(sliced)
        if start + len(sliced) < total:
            result += f"\n(Showing {len(sliced)} of {total} entries. Use 'offset' parameter to read beyond entry {offset + len(sliced)})"
        else:
            result += f"\n({total} entries)"
        result += "\n</entries>"
        return result
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = text.split("\n")
    total = len(lines)
    start = max(0, offset - 1)
    end = min(start + limit, total)
    sliced = lines[start:end]
    result = f"<path>{filepath}</path>\n<type>file</type>\n<content>\n"
    for i, line in enumerate(sliced, start + 1):
        result += f"{i}: {line}\n"
    last = end
    if last < total:
        result += f"\n(Showing lines {offset}-{last} of {total}. Use offset={last + 1} to continue.)"
    else:
        result += f"\n(End of file - total {total} lines)"
    result += "\n</content>"
    return result


def handle_glob(args):
    pattern = _pick(args, "pattern")
    if not pattern:
        return "Error: pattern is required"
    search_path = args.get("path", ".")
    import glob as glob_mod
    p = _resolve_path(search_path)
    matches = sorted(glob_mod.glob(pattern, root_dir=p, recursive=True))
    if not matches:
        matches = sorted(glob_mod.glob(str(p / pattern), recursive=True))
        if matches:
            matches = [str(Path(m).relative_to(p)) for m in matches]
    if not matches:
        return "No files found matching pattern."
    limit = 200
    if len(matches) > limit:
        return "\n".join(matches[:limit]) + f"\n... ({len(matches) - limit} more matches)"
    return "\n".join(matches)


def handle_grep(args):
    pattern = _pick(args, "pattern")
    if not pattern:
        return "Error: pattern is required"
    search_path = args.get("path", ".")
    include = args.get("include")
    root = _resolve_path(search_path)
    matches = []
    try:
        import subprocess
        cmd = ["rg", "-n", pattern, str(root)]
        if include:
            cmd.extend(["-g", include])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode not in (0, 1):
            pass
        if r.stdout:
            matches = r.stdout.rstrip().split("\n")
    except FileNotFoundError:
        pass
    except Exception:
        pass
    if not matches:
        for fpath in root.rglob("*"):
            if fpath.is_file():
                if include:
                    import fnmatch
                    if not fnmatch.fnmatch(fpath.name, include):
                        continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(text.split("\n"), 1):
                        if re.search(pattern, line):
                            rel = fpath.relative_to(root)
                            matches.append(f"{rel}:{i}:{line[:200]}")
                except Exception:
                    continue
    limit = 500
    if len(matches) > limit:
        matches = matches[:limit] + [f"... ({len(matches) - limit} more matches)"]
    return "\n".join(matches) if matches else "No files found"


def _pick(args, *keys):
    """Return the first matching key's value from args, or None."""
    for k in keys:
        if k in args:
            return args[k]
    return None

def handle_edit(args):
    fp = _pick(args, "filePath", "file_path")
    if not fp:
        return "Error: filePath is required"
    filepath = str(_resolve_path(fp))
    old = _pick(args, "oldString", "old_string")
    new = _pick(args, "newString", "new_string")
    if old is None:
        return "Error: oldString (or old_string) is required"
    if new is None:
        return "Error: newString (or new_string) is required"
    replace_all = args.get("replaceAll", args.get("replace_all", False))
    if old == new:
        return "No changes to apply: oldString and newString are identical."
    p = Path(filepath)
    if not p.exists():
        return f"Error: file not found: {filepath}"
    if p.is_dir():
        return f"Error: path is a directory, not a file: {filepath}"
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"
    if not old:
        return f"Error: oldString is empty. Use 'write' tool to create new files or provide content to replace."
    if replace_all:
        if old not in text:
            return f"Error: could not find:\n{old[:500]}"
        count = text.count(old)
        new_text = text.replace(old, new)
        p.write_text(new_text, encoding="utf-8")
        return f"Replaced {count} occurrence(s) in {filepath}"
    idx = text.find(old)
    if idx == -1:
        lines = text.split('\n')
        clue_lines = []
        for old_line in old.strip().split('\n')[:3]:
            stripped = old_line.strip()
            if stripped:
                for i, fline in enumerate(lines):
                    if stripped in fline:
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        ctx = '\n'.join(f'{j+1}: {lines[j]}' for j in range(start, end))
                        clue_lines.append(f"  Near line {i+1}:\n{ctx}")
                        break
        hint = ""
        if clue_lines:
            hint = "\nClosest matches in file:\n" + "\n".join(clue_lines[:2])
        return f"Error: could not find:\n{old[:500]}{hint}"
    last_idx = text.rfind(old)
    if idx != last_idx:
        return "Found multiple matches for oldString. Provide more surrounding context to make the match unique."
    new_text = text[:idx] + new + text[idx + len(old):]
    p.write_text(new_text, encoding="utf-8")
    return f"Replaced 1 occurrence in {filepath}"


def handle_write(args):
    fp = _pick(args, "filePath", "file_path")
    if not fp:
        return "Error: filePath is required"
    filepath = str(_resolve_path(fp))
    content = _pick(args, "content")
    if content is None:
        return "Error: content is required"
    p = Path(filepath)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"


def handle_task(args):
    desc = args.get("description", "task")
    prompt = args.get("prompt", "")
    subagent_type = args.get("subagent_type", "general")
    return (
        f"Task '{desc}' would be dispatched to a {subagent_type} subagent.\n"
        f"Subagent support requires recursive tbot execution.\n"
        f"Prompt: {prompt[:200]}"
    )


def handle_webfetch(args):
    url = _pick(args, "url")
    if not url:
        return "Error: url is required"
    fmt = args.get("format", "markdown")
    timeout = min(args.get("timeout", 30), 120)
    if not url.startswith(("http://", "https://")):
        return "URL must start with http:// or https://"
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; tbot/1.0; +https://github.com/user/tbot)",
        })
        resp.raise_for_status()
    except Exception as e:
        return f"Error fetching URL: {e}"
    if fmt == "html":
        return _truncate_output(resp.text)
    if fmt == "text":
        text = re.sub(r'<script[^>]*>.*?</script>', '', resp.text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '\n', text)
        text = html.unescape(text)
        text = re.sub(r'[ \t]+', ' ', text)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return _truncate_output('\n'.join(lines))
    text = _clean_html_text(resp.text)
    return text or "(no readable content found)"


# --- todowrite state (loop prevention) ---
_todo_prev_fingerprint = None
_todo_noop_count = 0
_todo_last_call_time = 0
_todo_has_write_since_last = False  # set by execute_tool_calls when edit/write/bash runs

def _todo_fingerprint(todos):
    return tuple(sorted(
        (t.get("content", ""), t.get("status", ""), t.get("priority", ""))
        for t in todos
    ))

def handle_todowrite(args):
    global _todo_prev_fingerprint, _todo_noop_count, _todo_last_call_time, _todo_has_write_since_last

    now = time.time()

    # --- rate limiter: <15s since last call = rapid-fire loop ---
    if 0 < now - _todo_last_call_time < 15:
        _todo_last_call_time = now
        return (
            "TOOL LOOP BLOCKED: todowrite called <15s after previous call. "
            "TASK.md was NOT updated. Use edit/write/bash/apply_patch to make progress."
        )
    _todo_last_call_time = now

    todos = args.get("todos", [])
    fingerprint = _todo_fingerprint(todos)

    # --- loop detection: same content back-to-back ---
    if fingerprint == _todo_prev_fingerprint:
        _todo_noop_count += 1
    else:
        _todo_noop_count = 0
        _todo_prev_fingerprint = fingerprint

    if _todo_noop_count >= 2:
        return (
            "TOOL LOOP BLOCKED: todowrite called 2+ times with identical task list. "
            "TASK.md was NOT updated. If you made code changes, update the task statuses "
            "(set completed tasks to 'completed', advance next to 'in_progress'). "
            "Otherwise stop planning and use edit/write/bash."
        )

    # --- write TASK.md ---
    task_file = CURRENT_DIR / "TASK.md"
    lines = ["# Task List", ""]
    in_progress = None
    completed_count = 0
    for t in todos:
        status_map = {"pending": " ", "in_progress": "~", "completed": "x", "cancelled": "-"}
        m = status_map.get(t.get("status", "pending"), " ")
        priority = t.get("priority", "medium")
        prio_tag = f" [{priority}]" if priority != "medium" else ""
        lines.append(f"- [{m}]{prio_tag} {t['content']}")
        if t.get("status") == "in_progress":
            in_progress = t["content"]
        if t.get("status") == "completed":
            completed_count += 1
    lines.append("")
    lines.append(f"<!-- Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')} -->")

    try:
        task_file.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        return f"Error writing task list: {e}"

    # --- build rich response ---
    response = f"✓ TASK.md updated ({len(todos)} tasks, {completed_count} done)"
    if _todo_noop_count >= 1:
        if _todo_has_write_since_last:
            response += " | ⚠ You made code changes but no task status changed. Mark done tasks as 'completed' and advance the next to 'in_progress'."
        else:
            response += " | ⚠ same list as before — take action instead"
    if in_progress:
        response += f" | → working on: {in_progress[:80]}"
    if _todo_has_write_since_last and _todo_noop_count == 0:
        response += " | progress tracked — continue working"
    _todo_has_write_since_last = False

    return response


def _resolve_ddg_url(url):
    """Resolve DuckDuckGo redirect URLs to the actual target."""
    url = url.strip()
    if url.startswith("//"):
        url = "https:" + url
    m = re.search(r'[?&]uddg=([^&]+)', url)
    if m:
        return urllib.parse.unquote(m.group(1))
    return url


def handle_websearch(args):
    query = _pick(args, "query")
    if not query:
        return "Error: query is required"
    num_results = min(args.get("numResults", 8), 10)
    try:
        sess = requests.Session()
        sess.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://duckduckgo.com/",
            "DNT": "1",
        })
        sess.get("https://duckduckgo.com/", timeout=8)
        resp = sess.post("https://html.duckduckgo.com/html/", data={"q": query}, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"Search failed: {e}"
    results = []
    blocks = re.split(r'<div[^>]*class="[^"]*result__body[^"]*"', resp.text)[1:]
    for block in blocks[:num_results]:
        href = None
        for pat in [
            r'href="(https?://[^"]+)"[^>]*>[^<]*<[^>]+class="result__a"',
            r'class="result__a"[^>]+href="(https?://[^"]*)"',
            r'class="result__a"[^>]+href="(//[^"]*)"',
        ]:
            m = re.search(pat, block)
            if m:
                href = html.unescape(m.group(1))
                break
        if not href:
            continue
        href = _resolve_ddg_url(href)
        tm = re.search(r'class="result__a"[^>]*>(.*?)</a>', block, re.DOTALL)
        title = re.sub(r'<[^>]+>', '', tm.group(1)).strip() if tm else ""
        sm = re.search(r'class="result__snippet"[^>]*>(.*?)</(?:span|div)>', block, re.DOTALL)
        snippet = re.sub(r'<[^>]+>', '', sm.group(1)).strip() if sm else ""
        title = html.unescape(title)
        snippet = html.unescape(snippet)
        if title:
            results.append((title, snippet, href))
    if not results:
        return "No results found."
    out = []
    for i, (title, snippet, href) in enumerate(results):
        out.append(f"• {title}")
        if snippet:
            out.append(f"  {snippet[:300]}")
        out.append(f"  {href}")
        if i < 2:
            page_text = _fetch_page_text(href, max_chars=3000)
            if page_text:
                out.append(f"  ── page content ({len(page_text)} chars) ──")
                for line in page_text.split('\n')[:10]:
                    out.append(f"  {line.strip()}")
    return "\n".join(out)


def handle_skill(args):
    name = args.get("name", "")
    if not name:
        return "Error: name is required"
    skills = load_skills()
    for n, desc, schema, doc in skills:
        if n == name:
            return f"Skill '{name}' is already loaded — do NOT call this tool again.\n\n{doc}"
    return f"Skill '{name}' not found. Use /skills to list available skills."


def handle_apply_patch(args):
    patch_text = args.get("patchText", "")
    if not patch_text:
        return "patchText is required"
    lines = patch_text.split("\n")
    files = {}
    current_file = None
    current_hunk = []
    in_hunk = False
    for line in lines:
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            current_file = line[4:].strip()
            files.setdefault(current_file, [])
            in_hunk = False
            continue
        if line.startswith("@@"):
            if current_hunk and current_file:
                files[current_file].append(current_hunk)
            current_hunk = []
            in_hunk = True
            continue
        if in_hunk and current_file:
            current_hunk.append(line)
    if current_hunk and current_file:
        files[current_file].append(current_hunk)
    applied = 0
    errors = []
    for filepath, hunks in files.items():
        if filepath == "/dev/null":
            continue
        fp = _resolve_path(filepath)
        if not fp.exists():
            errors.append(f"file not found: {filepath}")
            continue
        try:
            text = fp.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"cannot read {filepath}: {e}")
            continue
        for hunk in hunks:
            added_lines = [l[1:] for l in hunk if l.startswith("+")]
            removed_lines = [l[1:] for l in hunk if l.startswith("-")]
            context_lines = [l[1:] for l in hunk if l.startswith(" ")]
            if not removed_lines and not context_lines:
                text += "\n" + "\n".join(added_lines) + "\n"
                applied += 1
                continue
            old_block = "\n".join(
                l[1:] for l in hunk if l.startswith("-") or l.startswith(" ")
            )
            new_block = "\n".join(
                l[1:] for l in hunk if l.startswith("+") or l.startswith(" ")
            )
            idx = text.find(old_block)
            if idx == -1:
                errors.append(f"hunk not found in {filepath}")
                continue
            text = text[:idx] + new_block + text[idx + len(old_block):]
            applied += 1
        fp.write_text(text, encoding="utf-8")
    result = f"Patch applied: {applied} hunk(s)"
    if errors:
        result += "\nErrors:\n" + "\n".join(errors)
    return result


# ── Legacy handlers (kept) ──────────────────────────────────

def handle_install_skill(args):
    url = args.get("url", "")
    if not url:
        return "Error: url is required"
    return _install_skill_from_url(url)


def handle_create_skill(args):
    name = args.get("name", "")
    if not name:
        return "Error: name is required"
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', name):
        return f"{C.RED}invalid skill name — use letters, numbers, underscores{C.RESET}"
    desc = args.get("description", name)
    content = args.get("content", "")
    schema = args.get("schema", {
        "type": "object",
        "properties": {"input": {"type": "string", "description": "Input"}},
        "required": ["input"],
    })
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skill_dir = SKILLS_DIR / name
    if skill_dir.exists():
        return f"{C.RED}skill '{name}' already exists{C.RESET}"
    skill_dir.mkdir(parents=True)
    esc_desc = desc.replace("\\", "\\\\").replace('"', '\\"')
    esc_schema = json.dumps(schema, indent=2)
    md = f'''---
name: "{name}"
description: "{esc_desc}"
schema: {esc_schema}
---

# {name}

{content}
'''
    (skill_dir / "SKILL.md").write_text(md, encoding="utf-8")
    clear_skill_cache()
    return f"{C.GREEN}skill '{name}' created{C.RESET}"


def handle_get_system_info(_args):
    return json.dumps({
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME", ""),
        "home": str(Path.home()),
    }, indent=2)


TOOL_HANDLERS = {
    "invalid": handle_invalid,
    "question": handle_question,
    "bash": handle_bash,
    "read": handle_read,
    "read_file": handle_read,
    "glob": handle_glob,
    "grep": handle_grep,
    "edit": handle_edit,
    "write": handle_write,
    "task": handle_task,
    "webfetch": handle_webfetch,
    "todowrite": handle_todowrite,
    "websearch": handle_websearch,
    "skill": handle_skill,
    "apply_patch": handle_apply_patch,
    "get_system_info": handle_get_system_info,
    "install_skill": handle_install_skill,
    "create_skill": handle_create_skill,
}

# ── Skills (SKILL.md v1 — directory format) ───────────────────

_skill_cache = None


try:
    import yaml
    _has_yaml = True
except ImportError:
    _has_yaml = False


def _parse_skill_text(text):
    """Parse SKILL.md text with YAML frontmatter (between --- delimiters)."""
    m = re.match(r'^---\s*\n(.*?)\n(?:---|\.\.\.)\s*\n(.*)', text, re.DOTALL)
    if not m:
        return None
    raw_yaml = m.group(1)
    if _has_yaml:
        meta = yaml.safe_load(raw_yaml) or {}
    else:
        meta = {}
        for line in raw_yaml.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                key, _, val = line.partition(':')
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if val.lower() in ('true', 'yes'):
                    val = True
                elif val.lower() in ('false', 'no'):
                    val = False
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                meta[key] = val
    if not isinstance(meta, dict):
        meta = {}
    meta["_doc"] = m.group(2).strip()
    return meta


def _parse_skill_md(path):
    """Parse SKILL.md file with YAML frontmatter."""
    return _parse_skill_text(path.read_text(encoding="utf-8"))


def load_skills():
    global _skill_cache
    if _skill_cache is not None:
        return _skill_cache
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    default_schema = {
        "type": "object",
        "properties": {"input": {"type": "string", "description": "Input"}},
        "required": ["input"],
    }
    skills = []
    for entry in sorted(SKILLS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        md_path = entry / "SKILL.md"
        if not md_path.exists():
            continue
        meta = _parse_skill_md(md_path)
        if not meta:
            continue
        name = meta.get("name", entry.name)
        desc = meta.get("description", "") or (meta["_doc"][:80] if meta.get("_doc") else name)
        schema = meta.get("schema", default_schema)
        doc = meta.get("_doc")
        if doc:
            skills.append((name, desc, schema, doc))
    _skill_cache = skills
    return skills


def clear_skill_cache():
    global _skill_cache
    _skill_cache = None


def _install_from_skill_url(skill_url):
    """Install a skill from a direct URL to SKILL.md."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(skill_url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"{C.RED}download failed: {e}{C.RESET}"
    text = resp.text
    if text.strip().startswith("<!DOCTYPE") or text.strip().startswith("<html"):
        return (f"{C.RED}URL returned HTML (web page), not a SKILL.md file{C.RESET}\n"
                f"{C.GRAY}Use a raw URL (e.g. raw.githubusercontent.com/...) or a git repo URL{C.RESET}")
    meta = _parse_skill_text(text)
    if not meta:
        return f"{C.RED}invalid SKILL.md — no frontmatter{C.RESET}"
    name = meta.get("name", "")
    if not name or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', name):
        return f"{C.RED}invalid or missing skill name in SKILL.md frontmatter{C.RESET}"
    skill_dir = SKILLS_DIR / name
    if skill_dir.exists():
        return f"{C.RED}skill '{name}' already exists{C.RESET}"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    siblings = _download_github_siblings(skill_url, skill_dir)
    clear_skill_cache()
    deps = _install_skill_dependencies(skill_dir)
    msg = f"{C.GREEN}skill '{name}' installed{C.RESET}"
    if deps:
        msg += "\n" + "\n".join(deps)
    return msg


def _install_from_git(repo_url):
    """Install skill(s) from a Git repository (shallow clone)."""
    if not shutil.which("git"):
        return f"{C.RED}git is not installed — install git or use a direct SKILL.md URL{C.RESET}"
    tmpdir = tempfile.mkdtemp()
    try:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        ret = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, tmpdir],
            capture_output=True, text=True, timeout=120, env=env
        )
        if ret.returncode != 0:
            return f"{C.RED}git clone failed: {ret.stderr.strip() or ret.stdout.strip()}{C.RESET}"
        skill_files = list(Path(tmpdir).rglob("SKILL.md"))
        if not skill_files:
            return f"{C.RED}no SKILL.md found in repository{C.RESET}"
        installed = []
        for sf in skill_files:
            meta = _parse_skill_md(sf)
            if not meta:
                continue
            name = meta.get("name", sf.parent.name)
            if not name or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', name):
                continue
            target = SKILLS_DIR / name
            if target.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(sf.parent, target, dirs_exist_ok=True)
            installed.append(name)
        if not installed:
            return f"{C.YELLOW}no new skills to install (already exist or invalid){C.RESET}"
        clear_skill_cache()
        names = ", ".join(f"'{n}'" for n in installed)
        msg = f"{C.GREEN}skills installed: {names}{C.RESET}"

        # Dependencies are optional — install in background, don't block return
        import threading as _thr
        def _install_deps_bg():
            for name in installed:
                try:
                    deps = _install_skill_dependencies(SKILLS_DIR / name)
                except Exception:
                    pass
        _thr.Thread(target=_install_deps_bg, daemon=True).start()

        return msg
    except subprocess.TimeoutExpired:
        return f"{C.RED}git clone timed out{C.RESET}"
    except Exception as e:
        return f"{C.RED}error: {e}{C.RESET}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _install_skill_dependencies(skill_dir):
    """Read SKILL.md, parse ## Dependencies section, and install missing deps."""
    md_path = skill_dir / "SKILL.md"
    if not md_path.exists():
        return []
    text = md_path.read_text(encoding="utf-8")
    m = re.search(
        r'^##\s+Dependenc(?:ies|ias)\s*\n(.*?)(?=\n##\s|\Z)',
        text, re.MULTILINE | re.DOTALL
    )
    if not m:
        return []
    results = [f"  {C.CYAN}Dependencies:{C.RESET}"]
    for line in m.group(1).split('\n'):
        line = line.strip()
        if not line.startswith('- '):
            continue
        content = line[2:].strip()
        bt = re.match(r'^`([^`]+)`', content)
        cmd_str = bt.group(1) if bt else content.split(' - ')[0].split(' — ')[0].split(' – ')[0].strip()
        if not cmd_str:
            continue
        if cmd_str.startswith(('pip install', 'pip3 install')):
            exe = shutil.which('pip') or shutil.which('pip3')
            if not exe:
                results.append(f"    {C.YELLOW}⚠ pip not found{C.RESET}")
                continue
            args = shlex.split(cmd_str)
            pip_args = [exe, *args[1:]]
            if '--user' not in pip_args:
                pip_args.append('--user')
            if '--break-system-packages' not in pip_args:
                pip_args.append('--break-system-packages')
            try:
                r = subprocess.run(pip_args, capture_output=True, text=True, timeout=60)
                status = f"{C.GREEN}✓{C.RESET}" if r.returncode == 0 else f"{C.RED}✗{C.RESET}"
                if r.returncode != 0:
                    detail = r.stderr.strip()[-200:]
                    results.append(f"    {status} {cmd_str} — {detail}")
                else:
                    results.append(f"    {status} {cmd_str}")
            except Exception as e:
                results.append(f"    {C.RED}✗{C.RESET} {cmd_str}: {e}")
        elif cmd_str.startswith(('npm install', 'npm i')):
            exe = shutil.which('npm')
            if not exe:
                results.append(f"    {C.YELLOW}⚠ npm not found{C.RESET}")
                continue
            args = shlex.split(cmd_str)
            try:
                r = subprocess.run([exe, *args[1:]], capture_output=True, text=True, timeout=120)
                status = f"{C.GREEN}✓{C.RESET}" if r.returncode == 0 else f"{C.RED}✗{C.RESET}"
                results.append(f"    {status} {cmd_str}")
            except Exception as e:
                results.append(f"    {C.RED}✗{C.RESET} {cmd_str}: {e}")
        elif cmd_str.startswith(('brew install',)):
            args = shlex.split(cmd_str)
            try:
                r = subprocess.run(args, capture_output=True, text=True, timeout=300)
                status = f"{C.GREEN}✓{C.RESET}" if r.returncode == 0 else f"{C.RED}✗{C.RESET}"
                results.append(f"    {status} {cmd_str}")
            except Exception as e:
                results.append(f"    {C.RED}✗{C.RESET} {cmd_str}: {e}")
        else:
            par = re.search(r'\((`[^`]+`|[^)]+)\)', cmd_str)
            if par:
                binary = par.group(1).strip('`')
                if shutil.which(binary):
                    results.append(f"    {C.GREEN}✓{C.RESET} {binary} found")
                else:
                    results.append(f"    {C.YELLOW}⚠{C.RESET} {binary} not found — install manually")
            else:
                results.append(f"    {C.YELLOW}⚠{C.RESET} {cmd_str} — skipped")
    return results


_GITHUB_TREE_RE = re.compile(r'https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)')
_GITHUB_RAW_RE = re.compile(r'https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)')


def _github_tree_to_raw(url):
    """Convert GitHub tree URL to raw.githubusercontent.com URL for SKILL.md."""
    m = _GITHUB_TREE_RE.match(url)
    if m:
        owner, repo, branch, path = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/SKILL.md"
    return None


def _download_github_siblings(skill_url, skill_dir):
    """Download all files from the same GitHub directory as SKILL.md via the GitHub API."""
    m = _GITHUB_RAW_RE.match(skill_url)
    if not m:
        return []
    owner, repo, branch, path = m.group(1), m.group(2), m.group(3), m.group(4)
    dir_path = path.rsplit("/", 1)[0] if "/" in path else "."
    if not dir_path or dir_path == ".":
        return []
    downloaded = []

    def _fetch_dir(dir_path, local_dir):
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{dir_path}?ref={branch}"
        try:
            resp = requests.get(api_url, timeout=15, headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "tbot/1.0",
            })
            resp.raise_for_status()
            items = resp.json()
            if not isinstance(items, list):
                return
            for item in items:
                name = item.get("name", "")
                if name == "SKILL.md":
                    continue
                if item.get("type") == "dir":
                    subdir = local_dir / name
                    subdir.mkdir(parents=True, exist_ok=True)
                    _fetch_dir(f"{dir_path}/{name}", subdir)
                elif item.get("type") == "file":
                    dl_url = item.get("download_url")
                    if not dl_url:
                        continue
                    try:
                        fresp = requests.get(dl_url, timeout=15)
                        if fresp.status_code == 200:
                            (local_dir / name).write_bytes(fresp.content)
                            downloaded.append(name)
                    except Exception:
                        pass
        except Exception:
            pass

    _fetch_dir(dir_path, skill_dir)
    return downloaded


def _install_skill_from_url(url):
    """Install a skill from a URL or remote Git repository.

    Supports:
      - Direct URL to SKILL.md (raw content)
      - GitHub tree URL (e.g. github.com/owner/repo/tree/branch/path)
      - GitHub repository URL (auto-discovers SKILL.md files)
      - gh:user/repo shorthand
    """
    url = url.strip()
    if url.startswith("gh:"):
        url = f"https://github.com/{url[3:]}.git"

    raw_url = _github_tree_to_raw(url)
    if raw_url:
        return _install_from_skill_url(raw_url)

    has_skill_md = "SKILL.md" in url
    is_git = url.endswith(".git") or (not has_skill_md and "github.com" in url) or url.startswith("git@")
    if is_git:
        return _install_from_git(url)
    if has_skill_md:
        return _install_from_skill_url(url)
    return _install_from_skill_url(url.rstrip("/") + "/SKILL.md")


def skills_to_tools(skills):
    tools = []
    for n, d, s, *_ in skills:
        desc = f"[ONE-TIME] Load instructions for '{n}' skill. Call ONCE, then follow the instructions — do NOT call again. {d}"
        tools.append({"type": "function", "function": {"name": f"skill_{n}", "description": desc[:500], "parameters": s}})
    return tools


def skill_tool_handler(name, args, messages=None):
    for n, desc, schema, doc in load_skills():
        if n == name:
            if messages is not None:
                already = any(
                    m.get("role") == "system"
                    and f"## Skill: {name}" in m.get("content", "")
                    for m in messages[-5:]
                )
                if not already:
                    messages.append({"role": "system", "content": f"## Skill: {name}\n\n{doc}"})
            return f"Skill '{name}' instructions are already in context. Do NOT call this tool again — just follow the instructions."
    return f"Skill '{name}' not found"

def default_cfg():
    return {
        "api_key": "",
        "model": "deepseek/deepseek-v4-flash",
        "temperature": 0.7,
        "max_tokens": 524288,
        "system_prompt": "",  # empty = load from system_prompt.txt
        "max_history_chars": 200000,
        "tools_enabled": True,
        "trust_mode": False,
    }


def load_cfg():
    if not CONFIG_FILE.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return default_cfg()
    try:
        data = json.loads(CONFIG_FILE.read_text())
        base = default_cfg()
        base.update(data)
        return base
    except Exception:
        return default_cfg()


def save_cfg(cfg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

# ── API ─────────────────────────────────────────────────────────

def resolve_key(cfg):
    key = cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    print(f"{C.YELLOW}No API key configured.{C.RESET}")
    print(f"Get one at: {C.CYAN}https://openrouter.ai/keys{C.RESET}")
    key = input(f"{C.GREEN}Enter API key (sk-or-v1-...): {C.RESET}").strip()
    if not key:
        print(f"{C.RED}No key provided.{C.RESET}")
        sys.exit(1)
    cfg["api_key"] = key
    save_cfg(cfg)
    return key


def show_error(title, detail, hint=""):
    width = min(72, os.get_terminal_size().columns if hasattr(os, 'get_terminal_size') else 72)
    print(f"\n{C.RED}╭─{'─' * (width-4)}─╮{C.RESET}")
    print(f"{C.RED}│{C.RESET} {C.BOLD}{C.RED}✗ {title}{C.RESET}{' ' * (width - len(title) - 7)}{C.RED}│{C.RESET}")
    print(f"{C.RED}│{C.RESET} {C.RED}{'─' * (width - 6)}{C.RESET} {C.RED}│{C.RESET}")
    for line in detail.split("\n"):
        wrapped = textwrap.wrap(line, width - 6)
        for w in wrapped or [""]:
            print(f"{C.RED}│{C.RESET} {C.GRAY}{w}{C.RESET}{' ' * (width - len(w) - 5)}{C.RED}│{C.RESET}")
    if hint:
        print(f"{C.RED}│{C.RESET} {' ' * (width - 5)}{C.RED}│{C.RESET}")
        for line in hint.split("\n"):
            wrapped = textwrap.wrap(line, width - 6)
            for w in wrapped or [""]:
                print(f"{C.RED}│{C.RESET} {C.YELLOW}{w}{C.RESET}{' ' * (width - len(w) - 5)}{C.RED}│{C.RESET}")
    print(f"{C.RED}╰─{'─' * (width-4)}─╯{C.RESET}\n")


def chat_completion(messages, cfg, stream=True, tools=None):
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/user/tbot",
        "X-Title": "tbot",
    }
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, stream=stream, timeout=120)
    except requests.exceptions.Timeout:
        return {"error": "timeout", "title": "Connection timed out",
                "detail": "The request to OpenRouter took too long to respond.",
                "hint": "Check your internet connection or try again. If the problem persists, the service may be slow."}
    except requests.exceptions.SSLError as e:
        return {"error": "ssl", "title": "SSL certificate error",
                "detail": f"Could not verify the SSL certificate: {e}",
                "hint": "Check your system date/time. If on Termux, try: pkg install ca-certificates"}
    except requests.exceptions.ConnectionError:
        return {"error": "connection", "title": "Could not connect to OpenRouter",
                "detail": "No route to host. Your device may be offline or OpenRouter is blocked.",
                "hint": "Check your internet connection with: ping openrouter.ai\nIf on Termux, try: pkg install openssl && pkg reinstall python"}
    except requests.exceptions.ProxyError:
        return {"error": "proxy", "title": "Proxy connection failed",
                "detail": "Could not connect through the configured proxy.",
                "hint": "Check your proxy settings or disable the proxy and try again."}
    except Exception as e:
        return {"error": "unknown", "title": "Unexpected error",
                "detail": str(e),
                "hint": "This is an unexpected error. Check your setup and try again."}

    if resp.status_code != 200:
        try:
            data = resp.json()
            err = data.get("error", {})
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        except Exception:
            msg = resp.text[:300]
        return {"error": f"http_{resp.status_code}", "title": f"HTTP {resp.status_code}",
                "detail": msg,
                "hint": "The API returned an error. Check your API key, model name, and OpenRouter status at https://status.openrouter.ai"}

    return {"stream": resp}

# ── Stream parsing ──────────────────────────────────────────────

def parse_stream(resp):
    content_parts = []
    tool_calls = {}

    sock = None
    try:
        conn = getattr(resp.raw, "connection", None)
        sock = getattr(conn, "sock", None) if conn else None
    except Exception:
        pass

    try:
        iterator = resp.iter_lines()
    except Exception:
        return None, None

    received = False
    try:
        for raw in iterator:
            if not raw:
                continue
            if not received:
                received = True
                if sock:
                    try:
                        sock.settimeout(3.0)
                    except Exception:
                        sock = None
            try:
                raw = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            if not raw.startswith("data: "):
                continue
            chunk = raw[6:].strip()
            if chunk == "[DONE]":
                try:
                    resp.raw.drain_conn()
                except Exception:
                    pass
                break
            try:
                data = json.loads(chunk)
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                c = delta.get("content")
                if c:
                    content_parts.append(c)
                    print(c, end="", flush=True)
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                    entry = tool_calls[idx]
                    if "id" in tc:
                        entry["id"] = tc["id"]
                    if "function" in tc:
                        fn = tc["function"]
                        if "name" in fn:
                            entry["function"]["name"] += fn["name"]
                        if "arguments" in fn:
                            entry["function"]["arguments"] += fn["arguments"]
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
    except (socket.timeout, OSError):
        pass

    content = "".join(content_parts)
    calls = list(tool_calls.values()) if tool_calls else None
    return content, calls

# ── Tool execution ──────────────────────────────────────────────

def execute_tool_calls(tool_calls, messages, cfg):
    global _todo_has_write_since_last
    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
        except json.JSONDecodeError:
            args = {}

        print(f"\n{C.GRAY}── {C.CYAN}{name}{C.RESET} {C.GRAY}{json.dumps(args)[:200]}{C.RESET}")

        handler = TOOL_HANDLERS.get(name)
        if not handler and name.startswith("skill_"):
            handler = lambda a, _n=name[6:], _msgs=messages: skill_tool_handler(_n, a, _msgs)
        if not handler:
            result = f"Error: unknown tool '{name}'"
        else:
            if cfg.get("trust_mode"):
                ok = True
            else:
                try:
                    ans = input(f"  {C.YELLOW}run?{C.RESET} [Y/n/q] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return False
                if ans == "q":
                    return False
                ok = ans in ("", "y", "yes")
            if ok:
                try:
                    result = handler(args)
                except KeyboardInterrupt:
                    print(f"\n  {C.YELLOW}cancelled{C.RESET}")
                    return False
                if len(result) > 8000:
                    result = result[:8000] + f"\n... (truncated, {len(result)} total chars)"
            else:
                result = "TOOL_CALL_DECLINED"

        preview = result[:500].replace("\n", "\\n")
        print(f"  {C.GRAY}→ {preview}{'...' if len(result) > 500 else ''}{C.RESET}")
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        if name not in _READ_ONLY_TOOLS and ok:
            _todo_has_write_since_last = True
    return True

# ── UI ──────────────────────────────────────────────────────────

def setup_history():
    if readline is None:
        return
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        readline.read_history_file(str(HISTORY_FILE))
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)
    try:
        readline.parse_and_bind('"\\C-e": "/edit\\C-j"')
    except Exception:
        pass


def save_history():
    if readline is None:
        return
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(HISTORY_FILE))
    except OSError:
        pass


def show_banner(cfg):
    tools_flag = f"{C.CYAN}tools{C.RESET}" if cfg["tools_enabled"] else f"{C.GRAY}tools off{C.RESET}"
    trust_flag = f"{C.GREEN}trust{C.RESET}" if cfg["trust_mode"] else f"{C.YELLOW}confirm{C.RESET}"
    print(f"\n{C.BOLD}{C.CYAN}  tbot{C.RESET}  {C.GRAY}— OpenRouter CLI{C.RESET}")
    print(f"  {C.GRAY}model:{C.RESET} {C.YELLOW}{cfg['model']}{C.RESET}")
    print(f"  {C.GRAY}temp:{C.RESET}  {C.YELLOW}{cfg['temperature']}{C.RESET}  "
          f"{tools_flag}  {trust_flag}")
    print(f"  {C.GRAY}type{C.RESET} {C.BOLD}/help{C.RESET} {C.GRAY}for commands{C.RESET}\n")


def print_help():
    print(f"{C.CYAN}Commands:{C.RESET}")
    print(f"  /help              This help")
    print(f"  /new               Reset conversation")
    print(f"  /model [name]      Show or switch model")
    print(f"  /temp [n]          Show or set temperature")
    print(f"  /sys [prompt]      Show or set system prompt")
    print(f"  /edit  [text]      Multi-line editor (or Ctrl+E / /edit)")
    print(f"  /tools             Toggle tool calling on/off")
    print(f"  /trust             Toggle auto-approve tools")
    print(f"  /skills            List installed skills")
    print(f"  /skill add|rm|show  Manage skills")
    print(f"  /exit              Quit")
    print()
    print(f"{C.CYAN}Tools ({len(TOOLS)}):{C.RESET}")
    for t in TOOLS:
        fn = t["function"]
        print(f"  {C.YELLOW}{fn['name']}{C.RESET}  {C.GRAY}{fn['description'].split('.')[0]}.{C.RESET}")
    print(f"  {C.GRAY}--- skills are injected dynamically via the skill tool ---{C.RESET}")


def open_editor(initial_text=""):
    editor_cmd = shlex.split(os.environ.get('VISUAL') or os.environ.get('EDITOR') or 'vi')
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+', delete=False) as f:
        f.write(initial_text)
        f.flush()
        tmp_path = f.name
    try:
        subprocess.run(editor_cmd + [tmp_path], check=True)
        result = Path(tmp_path).read_text(encoding='utf-8')
        return result.rstrip('\n')
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"{C.RED}editor error: {e}{C.RESET}")
        return initial_text
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── System prompt loading ───────────────────────────────────────

SYSTEM_PROMPT_DEFAULT = """You are tbot, an interactive CLI that helps with software engineering tasks.

# Tone
Be concise and direct. No introductions, conclusions, or summaries after editing. Use tools to act, text to communicate. Never use tool calls or code comments to communicate.

# Tool use
- Prefer read, edit, write, glob, grep over bash for file ops. Reserve bash for system commands (git, pip, builds, tests).
- Call independent tools in parallel. Sequential only when dependencies exist.
- After editing, the change is applied — no need to re-read unless you need to verify a specific detail.
- If a tool errors, check the message and adjust — don't retry the same call verbatim.

# Conventions
Match the surrounding code's style, comment density, and idioms. Do NOT add comments unless the code is non-obvious.
NEVER commit changes unless the user explicitly asks.

# Environment
Today's date: {date}
Working directory: {cwd}
Platform: {platform}"""


def load_system_prompt(cfg):
    """Load system prompt from file (or config override). Injects dynamic vars."""
    prompt = cfg.get("system_prompt", "")
    if prompt:
        for k, v in _env_vars().items():
            if "{" + k + "}" in prompt:
                prompt = prompt.replace("{" + k + "}", v)
        return prompt

    # Load from file (create on first run)
    try:
        data = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        SYSTEM_PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = SYSTEM_PROMPT_DEFAULT
        try:
            SYSTEM_PROMPT_FILE.write_text(data, encoding="utf-8")
        except OSError:
            pass
    except OSError:
        data = SYSTEM_PROMPT_DEFAULT

    for k, v in _env_vars().items():
        if "{" + k + "}" in data:
            data = data.replace("{" + k + "}", v)
    return data


def _env_vars():
    return {
        "date": time.strftime("%Y-%m-%d"),
        "cwd": str(Path.cwd()),
        "platform": platform.system().lower(),
    }


# ── Main ────────────────────────────────────────────────────────

def _init_messages(cfg):
    prompt = load_system_prompt(cfg)
    return [{"role": "system", "content": prompt}] if prompt else []


def main():
    cfg = load_cfg()
    cfg["api_key"] = resolve_key(cfg)

    parser = argparse.ArgumentParser(description="tbot - Terminal chatbot for OpenRouter")
    parser.add_argument("-m", "--model", help="Model slug (e.g. deepseek/deepseek-chat)")
    parser.add_argument("-t", "--temperature", type=float, help="Temperature 0.0-2.0")
    parser.add_argument("-s", "--system", help="System prompt")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool calling")
    parser.add_argument("--trust", action="store_true", help="Auto-approve tool execution")
    args = parser.parse_args()

    if args.model:
        cfg["model"] = args.model
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.system is not None:
        cfg["system_prompt"] = args.system
    if args.no_tools:
        cfg["tools_enabled"] = False
    if args.trust:
        cfg["trust_mode"] = True

    setup_history()
    atexit.register(save_history)
    if readline is not None:
        readline.set_startup_hook(lambda: sys.stdout.write(f"{C.BOLD}{C.BLUE}"))
    messages = _init_messages(cfg)
    show_banner(cfg)

    while True:
        try:
            if readline is not None:
                line = input(">>> ")
                save_history()
            else:
                line = input(f"{C.BOLD}{C.BLUE}>>>{C.RESET} ")
            sys.stdout.write(C.RESET)
        except (EOFError, KeyboardInterrupt):
            save_history()
            print(f"\n{C.YELLOW}bye{C.RESET}")
            break

        line = line.strip()
        if not line:
            continue

        # ── commands ──
        if line.startswith("/"):
            parts = line[1:].strip().split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("exit", "quit"):
                save_history()
                print(f"{C.YELLOW}bye{C.RESET}")
                break
            elif cmd == "help":
                print_help()
            elif cmd == "new":
                messages = _init_messages(cfg)
                _doom_trail.clear()
                print(f"{C.GREEN}reset{C.RESET}")
            elif cmd == "model":
                if arg:
                    cfg["model"] = arg
                    save_cfg(cfg)
                    print(f"{C.GREEN}model → {cfg['model']}{C.RESET}")
                else:
                    print(f"{C.YELLOW}{cfg['model']}{C.RESET}")
            elif cmd == "temp":
                if arg:
                    try:
                        cfg["temperature"] = max(0.0, min(2.0, float(arg)))
                        save_cfg(cfg)
                        print(f"{C.GREEN}temp → {cfg['temperature']}{C.RESET}")
                    except ValueError:
                        print(f"{C.RED}invalid number{C.RESET}")
                else:
                    print(f"{C.YELLOW}{cfg['temperature']}{C.RESET}")
            elif cmd == "sys":
                if arg:
                    cfg["system_prompt"] = arg
                    save_cfg(cfg)
                    if messages and messages[0]["role"] == "system":
                        messages[0]["content"] = load_system_prompt(cfg)
                    print(f"{C.GREEN}system prompt updated{C.RESET}")
                else:
                    effective = cfg["system_prompt"] or f"(from {SYSTEM_PROMPT_FILE})"
                    print(f"{C.YELLOW}{effective}{C.RESET}")
            elif cmd == "tools":
                cfg["tools_enabled"] = not cfg["tools_enabled"]
                save_cfg(cfg)
                print(f"{C.GREEN}tools {'on' if cfg['tools_enabled'] else 'off'}{C.RESET}")
            elif cmd == "trust":
                cfg["trust_mode"] = not cfg["trust_mode"]
                save_cfg(cfg)
                print(f"{C.GREEN}trust {'on' if cfg['trust_mode'] else 'off'}{C.RESET}")
            elif cmd == "edit":
                if arg:
                    last_user = -1
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i]["role"] == "user":
                            last_user = i
                            break
                    if last_user == -1:
                        print(f"{C.RED}no user message to edit{C.RESET}")
                    else:
                        messages[last_user]["content"] = arg
                        print(f"{C.GREEN}last message updated{C.RESET}")
                else:
                    last_user = -1
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i]["role"] == "user":
                            last_user = i
                            break
                    initial = messages[last_user]["content"] if last_user != -1 else ""
                    print(f"{C.YELLOW}opening editor...{C.RESET}")
                    content = open_editor(initial)
                    if content:
                        appended = last_user == -1
                        if last_user != -1:
                            messages[last_user]["content"] = content
                        else:
                            messages.append({"role": "user", "content": content})
                        lines = content.split("\n")
                        print(f"{C.GREEN}message ({len(lines)} lines, {len(content)} chars){C.RESET}")
                        if len(lines) <= 3:
                            print(content)
                        send_conversation(messages, cfg, pop_on_first_error=appended)
                    else:
                        print(f"{C.YELLOW}cancelled{C.RESET}")
            elif cmd == "skills":
                skills = load_skills()
                if not skills:
                    print(f"{C.YELLOW}no skills installed{C.RESET}")
                    print(f"{C.GRAY}create one: /skill add <name>{C.RESET}")
                else:
                    print(f"{C.CYAN}skills ({len(skills)}):{C.RESET}")
                    for name, desc, *_ in skills:
                        print(f"  {C.GREEN}{name}{C.RESET}  {C.GRAY}{desc}{C.RESET}")
            elif cmd == "skill":
                sub = arg.split(maxsplit=1) if arg else []
                sub_cmd = sub[0].lower() if sub else ""
                sub_arg = sub[1] if len(sub) > 1 else ""
                if sub_cmd == "add" and sub_arg:
                    name = sub_arg.strip()
                    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', name):
                        print(f"{C.RED}invalid skill name{C.RESET}")
                    else:
                        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
                        skill_dir = SKILLS_DIR / name
                        if skill_dir.exists():
                            print(f"{C.RED}skill '{name}' already exists{C.RESET}")
                        else:
                            skill_dir.mkdir(parents=True)
                            md = f'''---
name: "{name}"
description: "{name} skill"
schema:
  type: object
  properties:
    input:
      type: string
      description: Input
  required: [input]
---

# {name}

Replace this with instructions for the model.
'''
                            (skill_dir / "SKILL.md").write_text(md)
                            clear_skill_cache()
                            print(f"{C.GREEN}skill '{name}' created{C.RESET}")
                            print(f"{C.GRAY}  {skill_dir}/SKILL.md{C.RESET}")
                elif sub_cmd == "rm" and sub_arg:
                    skill_dir = SKILLS_DIR / sub_arg.strip()
                    if not skill_dir.exists() or not skill_dir.is_dir():
                        print(f"{C.RED}skill '{sub_arg}' not found{C.RESET}")
                    else:
                        shutil.rmtree(skill_dir)
                        clear_skill_cache()
                        print(f"{C.GREEN}skill '{sub_arg}' removed{C.RESET}")
                elif sub_cmd == "show" and sub_arg:
                    skill_dir = SKILLS_DIR / sub_arg.strip()
                    if not skill_dir.exists() or not skill_dir.is_dir():
                        print(f"{C.RED}skill '{sub_arg}' not found{C.RESET}")
                    else:
                        for f in sorted(skill_dir.iterdir()):
                            print(f"{C.YELLOW}── {f.name} ──{C.RESET}")
                            print(f.read_text().rstrip())
                            print()
                elif sub_cmd == "install" and sub_arg:
                    msg = _install_skill_from_url(sub_arg.strip())
                    print(msg)
                else:
                    print(f"{C.YELLOW}usage:{C.RESET}")
                    print(f"  /skill add <name>      create a new skill")
                    print(f"  /skill rm <name>       delete a skill")
                    print(f"  /skill show <name>     show skill files")
                    print(f"  /skill install <url>   install from URL or git repo")
                    print(f"  /skills                list all skills")
            else:
                print(f"{C.RED}unknown: /{cmd}{C.RESET}")
            continue

        # ── message ──
        messages.append({"role": "user", "content": line})
        send_conversation(messages, cfg, pop_on_first_error=True)


def send_conversation(messages, cfg, pop_on_first_error=False):
    total = sum(len(m.get("content", "")) for m in messages if isinstance(m.get("content"), str))
    while total > cfg["max_history_chars"] and sum(1 for m in messages if m["role"] not in ("system",)) > 1:
        for i, m in enumerate(messages):
            if m["role"] not in ("system", "tool"):
                total -= len(m.get("content", ""))
                messages.pop(i)
                break
    tools = None
    if cfg["tools_enabled"]:
        tools = list(TOOLS)
        skills = load_skills()
        if skills:
            tools += skills_to_tools(skills)
    max_rounds = 30
    round_n = 0
    while round_n < max_rounds:
        round_n += 1
        try:
            result = chat_completion(messages, cfg, stream=True, tools=tools)
            if "error" in result:
                show_error(result.get("title", "Error"),
                          result.get("detail", result["error"]),
                          result.get("hint", ""))
                if round_n == 1 and pop_on_first_error:
                    messages.pop()
                break
            content, tool_calls = parse_stream(result["stream"])
            result["stream"].close()
            if tool_calls:
                if content:
                    print()
                doom_warning = _check_doom_loop(tool_calls)
                if doom_warning:
                    print(f"\n  {C.YELLOW}{doom_warning[:100]}{C.RESET}")
                    messages.append({"role": "system", "content": doom_warning})
                    continue
                ok = execute_tool_calls(tool_calls, messages, cfg)
                if not ok:
                    break
                continue
            if content:
                print()
                messages.append({"role": "assistant", "content": content})
            break
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}cancelled{C.RESET}")
            if round_n == 1 and pop_on_first_error:
                messages.pop()
            break
    if round_n >= max_rounds:
        show_error("Max tool rounds reached",
                   f"The model used {max_rounds} consecutive tool calls without producing a final response.",
                   "This may indicate a bug in the model or an infinite loop. Try a different model.")


if __name__ == "__main__":
    main()
