#!/usr/bin/env python3
"""tbot - Terminal chatbot for OpenRouter with PC tool support."""

import os, sys, json, time, subprocess, platform, re, html
import argparse, textwrap, atexit, importlib.util
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

# ── Tool definitions ───────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Run a shell command on the local machine. Returns stdout + stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path where to write the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path", "default": "."},
                },
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
            "name": "search_web",
            "description": "Search the web using DuckDuckGo. Returns page titles, snippets, and URLs. Use this to get current information, documentation, or answers to questions you don't know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Number of results", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Apply a surgical find-and-replace edit to a file. Replaces only the FIRST occurrence of 'find' with 'replace'. Use this instead of write_file when you need to change specific lines while preserving the rest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "find": {"type": "string", "description": "The exact text to search for (first occurrence)"},
                    "replace": {"type": "string", "description": "The replacement text"},
                },
                "required": ["path", "find", "replace"],
            },
        },
    },
]


def handle_execute_command(args):
    cmd = args["command"]
    timeout = args.get("timeout", 30)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = r.stdout
        if r.stderr:
            out += "\n--- stderr ---\n" + r.stderr
        if r.returncode != 0:
            out += f"\n--- exit code: {r.returncode} ---"
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def handle_read_file(args):
    path = Path(args["path"]).expanduser().resolve()
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"


def handle_write_file(args):
    path = Path(args["path"]).expanduser().resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return f"Written {len(args['content'])} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def handle_list_directory(args):
    path = Path(args.get("path", ".")).expanduser().resolve()
    try:
        entries = []
        for p in path.iterdir():
            suffix = "/" if p.is_dir() else ""
            entries.append(f"{p.name}{suffix}")
        return "\n".join(sorted(entries)) if entries else "(empty directory)"
    except Exception as e:
        return f"Error listing directory: {e}"


def handle_search_web(args):
    query = args["query"]
    max_results = min(args.get("max_results", 5), 15)
    current_year = str(time.localtime().tm_year)
    if current_year not in query:
        query = f"{query} {current_year}"
    try:
        resp = requests.post("https://html.duckduckgo.com/html/", data={"q": query, "df": f"{current_year}-01-01..{current_year}-12-31"}, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        return f"Search failed: {e}"
    results = []
    blocks = re.split(r'<div[^>]*class="[^"]*result__body[^"]*"', resp.text)[1:]
    for block in blocks[:max_results]:
        m = re.search(r'href="(https?://[^"]+)"[^>]*>[^<]*<[^>]+class="result__a"', block)
        if not m:
            m = re.search(r'class="result__a"[^>]+href="(https?://[^"]*)"', block)
        if not m:
            continue
        href = html.unescape(m.group(1))
        tm = re.search(r'class="result__a"[^>]*>(.*?)</a>', block, re.DOTALL)
        title = re.sub(r'<[^>]+>', '', tm.group(1)).strip() if tm else ""
        sm = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL)
        snippet = re.sub(r'<[^>]+>', '', sm.group(1)).strip() if sm else ""
        title = html.unescape(title)
        snippet = html.unescape(snippet)
        if title:
            results.append((title, snippet, href))
    if not results:
        return "No results found."
    out = []
    for title, snippet, href in results:
        out.append(f"• {title}")
        if snippet:
            out.append(f"  {snippet[:200]}")
        out.append(f"  {href}")
    return "\n".join(out)


def handle_edit_file(args):
    path = Path(args["path"]).expanduser().resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"
    find = args["find"]
    replace = args["replace"]
    idx = text.find(find)
    if idx == -1:
        return f"Error: could not find:\n{find[:200]}"
    new_text = text[:idx] + replace + text[idx + len(find):]
    try:
        path.write_text(new_text, encoding="utf-8")
    except Exception as e:
        return f"Error writing file: {e}"
    return f"Replaced 1 occurrence in {path}"


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
    "execute_command": handle_execute_command,
    "read_file": handle_read_file,
    "write_file": handle_write_file,
    "list_directory": handle_list_directory,
    "get_system_info": handle_get_system_info,
    "edit_file": handle_edit_file,
    "search_web": handle_search_web,
}

# ── Skills (SKILL.md v1 — directory format) ───────────────────

_skill_cache = None
_skill_teardowns = []


try:
    import yaml
    _has_yaml = True
except ImportError:
    _has_yaml = False


def _parse_skill_md(path):
    """Parse SKILL.md with YAML frontmatter (between --- delimiters)."""
    text = path.read_text(encoding="utf-8")
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


def load_skills():
    global _skill_cache, _skill_teardowns
    if _skill_cache is not None:
        return _skill_cache
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    for fn in _skill_teardowns:
        try:
            fn()
        except Exception:
            pass
    _skill_teardowns.clear()
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
        desc = meta.get("description", "") or meta["_doc"][:80] if meta.get("_doc") else name
        schema = meta.get("schema", default_schema)
        doc = meta.get("_doc")
        handler_file = meta.get("handler", "run.py")
        handler_path = entry / handler_file

        if handler_path.exists():
            spec = importlib.util.spec_from_file_location(f"tbot_skill_{name}", handler_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                except Exception:
                    continue
                run_fn = getattr(mod, "run", None)
                if not callable(run_fn):
                    continue
                setup_fn = getattr(mod, "setup", None)
                if callable(setup_fn):
                    try:
                        setup_fn()
                    except Exception:
                        pass
                teardown_fn = getattr(mod, "teardown", None)
                if callable(teardown_fn):
                    _skill_teardowns.append(teardown_fn)
                skills.append((name, desc, schema, run_fn, None))
        elif doc:
            skills.append((name, desc, schema, None, doc))
    _skill_cache = skills
    return skills


def clear_skill_cache():
    global _skill_cache, _skill_teardowns
    _skill_cache = None
    _skill_teardowns.clear()


def skills_to_tools(skills):
    return [
        {"type": "function", "function": {"name": f"skill_{n}", "description": d, "parameters": s}}
        for n, d, s, *_ in skills
    ]


def skill_tool_handler(name, args, messages=None):
    for n, desc, schema, run_fn, doc in load_skills():
        if n == name:
            if doc and run_fn is None:
                if messages is not None:
                    messages.append({"role": "system", "content": f"## Skill: {name}\n\n{doc}"})
                return f"Skill '{name}' loaded. Follow the instructions above."
            try:
                return str(run_fn(args))
            except Exception as e:
                return f"Skill error: {e}"
    return f"Skill '{name}' not found"

def default_cfg():
    return {
        "api_key": "",
        "model": "deepseek/deepseek-v4-flash",
        "temperature": 0.7,
        "max_tokens": 524288,
        "system_prompt": "You are a helpful assistant with access to PC tools.",
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
    try:
        iterator = resp.iter_lines()
    except Exception:
        return None, None
    for raw in iterator:
        if not raw:
            continue
        try:
            raw = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        if not raw.startswith("data: "):
            continue
        chunk = raw[6:].strip()
        if chunk == "[DONE]":
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
    content = "".join(content_parts)
    calls = list(tool_calls.values()) if tool_calls else None
    return content, calls

# ── Tool execution ──────────────────────────────────────────────

def execute_tool_calls(tool_calls, messages, cfg):
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
                ans = input(f"  {C.YELLOW}run?{C.RESET} [Y/n/q] ").strip().lower()
                if ans == "q":
                    return False
                ok = ans in ("", "y", "yes")
            if ok:
                result = handler(args)
                if len(result) > 8000:
                    result = result[:8000] + f"\n... (truncated, {len(result)} total chars)"
            else:
                result = "TOOL_CALL_DECLINED"

        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
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
    print(f"  /edit  [text]      View or replace last user message")
    print(f"  /tools             Toggle tool calling on/off")
    print(f"  /trust             Toggle auto-approve tools")
    print(f"  /skills            List installed skills")
    print(f"  /skill add|rm|show  Manage skills")
    print(f"  /exit              Quit")
    print()
    print(f"{C.CYAN}Built-in tools:{C.RESET}")
    for t in TOOLS:
        fn = t["function"]
        print(f"  {C.YELLOW}{fn['name']}{C.RESET}  {C.GRAY}{fn['description']}{C.RESET}")
    skills = load_skills()
    if skills:
        print(f"\n{C.CYAN}Skills:{C.RESET}")
        for name, desc, *_ in skills:
            print(f"  {C.MAGENTA}skill_{name}{C.RESET}  {C.GRAY}{desc}{C.RESET}")

# ── Main ────────────────────────────────────────────────────────

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
    messages = [{"role": "system", "content": cfg["system_prompt"]}] if cfg["system_prompt"] else []
    show_banner(cfg)

    while True:
        try:
            if readline is not None:
                line = input(">>> ")
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
                messages = [{"role": "system", "content": cfg["system_prompt"]}] if cfg["system_prompt"] else []
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
                        messages[0]["content"] = arg
                    print(f"{C.GREEN}system prompt updated{C.RESET}")
                else:
                    print(f"{C.YELLOW}{cfg['system_prompt']}{C.RESET}")
            elif cmd == "tools":
                cfg["tools_enabled"] = not cfg["tools_enabled"]
                save_cfg(cfg)
                print(f"{C.GREEN}tools {'on' if cfg['tools_enabled'] else 'off'}{C.RESET}")
            elif cmd == "trust":
                cfg["trust_mode"] = not cfg["trust_mode"]
                save_cfg(cfg)
                print(f"{C.GREEN}trust {'on' if cfg['trust_mode'] else 'off'}{C.RESET}")
            elif cmd == "edit":
                last_user = -1
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i]["role"] == "user":
                        last_user = i
                        break
                if last_user == -1:
                    print(f"{C.RED}no user message to edit{C.RESET}")
                elif arg:
                    messages[last_user]["content"] = arg
                    print(f"{C.GREEN}last message updated{C.RESET}")
                else:
                    print(f"{C.YELLOW}last message:{C.RESET}")
                    print(messages[last_user]["content"])
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
                    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
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
handler: run.py
---

# {name}

Replace this with usage documentation.
'''
                            (skill_dir / "SKILL.md").write_text(md)
                            py = '''def run(args):
    return f"Hello from {args.get('input', '')}"
'''
                            (skill_dir / "run.py").write_text(py)
                            clear_skill_cache()
                            print(f"{C.GREEN}skill '{name}' created{C.RESET}")
                            print(f"{C.GRAY}  {skill_dir}/SKILL.md{C.RESET}")
                            print(f"{C.GRAY}  {skill_dir}/run.py{C.RESET}")
                elif sub_cmd == "rm" and sub_arg:
                    skill_dir = SKILLS_DIR / sub_arg.strip()
                    if not skill_dir.exists() or not skill_dir.is_dir():
                        print(f"{C.RED}skill '{sub_arg}' not found{C.RESET}")
                    else:
                        import shutil
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
                else:
                    print(f"{C.YELLOW}usage:{C.RESET}")
                    print(f"  /skill add <name>    create a new skill")
                    print(f"  /skill rm <name>     delete a skill")
                    print(f"  /skill show <name>   show skill files")
                    print(f"  /skills              list all skills")
            else:
                print(f"{C.RED}unknown: /{cmd}{C.RESET}")
            continue

        # ── message ──
        messages.append({"role": "user", "content": line})

        # trim history
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
        max_rounds = 12
        round_n = 0

        while round_n < max_rounds:
            round_n += 1
            result = chat_completion(messages, cfg, stream=True, tools=tools)

            if "error" in result:
                show_error(result.get("title", "Error"),
                          result.get("detail", result["error"]),
                          result.get("hint", ""))
                if round_n == 1:
                    messages.pop()
                break

            content, tool_calls = parse_stream(result["stream"])

            if tool_calls:
                if content:
                    print()
                ok = execute_tool_calls(tool_calls, messages, cfg)
                if not ok:
                    break
                continue

            if content:
                print()
                messages.append({"role": "assistant", "content": content})
            break

        if round_n >= max_rounds:
            show_error("Max tool rounds reached",
                       f"The model used {max_rounds} consecutive tool calls without producing a final response.",
                       "This may indicate a bug in the model or an infinite loop. Try a different model.")


if __name__ == "__main__":
    main()
