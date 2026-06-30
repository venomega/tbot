#!/usr/bin/env python3
"""tbot - Terminal chatbot for OpenRouter with PC tool support."""

import os, sys, json, time, subprocess, platform, re, html, socket, urllib.parse, base64, functools, signal, threading
import argparse, textwrap, atexit, tempfile, shutil, shlex
from pathlib import Path
import requests

try:
    import readline
except ImportError:
    readline = None

_COMMANDS = [
    "help",
    "new",
    "model",
    "preset",
    "session",
    "provider",
    "temp",
    "sys",
    "edit",
    "tools",
    "trust",
    "export",
    "skills",
    "skill",
    "rag",
    "memory",
    "episodic",
    "mcp",
    "commit",
    "exit",
]
_PRESET_SUBCMDS = ["save", "load", "rm", "ls", "show"]
_SKILL_SUBCMDS = ["add", "rm", "show", "install"]
_RAG_SUBCMDS = ["index", "search", "status"]
_MEMORY_SUBCMDS = ["show", "edit", "gate", "search"]
_EPISODIC_SUBCMDS = ["search", "stats", "prune"]
_MCP_SUBCMDS = ["status", "connect", "disconnect", "reconnect", "discover"]

_total_tokens = 0
_last_cost = 0
_acc_cost = 0


def _format_cost(cost):
    if not cost:
        return ""
    if cost < 0.001:
        s = f"{cost:.6f}"
    else:
        s = f"{cost:.4f}"
    s = s.rstrip("0").rstrip(".")
    return f" ${s}"


CONFIG_DIR = Path.home() / ".config" / "tbot"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history.txt"
SKILLS_DIR = CONFIG_DIR / "skills"
SYSTEM_PROMPT_FILE = CONFIG_DIR / "system_prompt.txt"
LOG_DIR = CONFIG_DIR / "log"
PRESETS_DIR = CONFIG_DIR / "presets"
PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "openai": {
        "name": "OpenAI",
        "url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
    },
    "deepseek": {
        "name": "DeepSeek",
        "url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "groq": {
        "name": "Groq",
        "url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
    },
    "together": {
        "name": "Together AI",
        "url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
    },
    "perplexity": {
        "name": "Perplexity",
        "url": "https://api.perplexity.ai",
        "env_key": "PERPLEXITY_API_KEY",
    },
    "xai": {
        "name": "xAI (Grok)",
        "url": "https://api.x.ai/v1",
        "env_key": "XAI_API_KEY",
    },
    "mistral": {
        "name": "Mistral AI",
        "url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
    },
    "fireworks": {
        "name": "Fireworks AI",
        "url": "https://api.fireworks.ai/inference/v1",
        "env_key": "FIREWORKS_API_KEY",
    },
    "custom": {
        "name": "Custom API",
    },
}

_DEFAULT_PROVIDER_URL = PROVIDERS["openrouter"]["url"]

_log_fh = None
_current_log_path = None

# ── Memory system constants ──────────────────────────────────

MEMORY_DIR = CONFIG_DIR / "memories"
FACTS_DIR = MEMORY_DIR / "facts"
MEMORY_FILE = FACTS_DIR / "MEMORY.md"
USER_FILE = FACTS_DIR / "USER.md"
EPISODIC_DB = MEMORY_DIR / "episodic.db"

_session_state = {"id": None}

_MEMORY_FACT_COUNTER = {}

# ── MCP (Model Context Protocol) ──────────────────────────────────
try:
    from mcp_client import MCPServerManager
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False
    MCPServerManager = None

_mcp_manager = None


def ensure_memory_dirs():
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    FACTS_DIR.mkdir(parents=True, exist_ok=True)


def _read_fact_file(path):
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _write_fact_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _init_episodic_db():
    """Create/open episodic.db, create tables + FTS5 if not exist."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import sqlite3

        conn = sqlite3.connect(str(EPISODIC_DB))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            model TEXT,
            provider TEXT
        )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            tool_name TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )"""
        )
        # Create FTS5 virtual table (IF NOT EXISTS handled by checking existence)
        try:
            conn.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content, role, tool_name,
                content=messages, content_rowid=id
            )"""
            )
        except Exception:
            # Table may already exist from previous run
            pass
        # Create triggers unconditionally (IF NOT EXISTS)
        conn.execute(
            """CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content, role, tool_name)
            VALUES (new.id, new.content, new.role, new.tool_name);
        END"""
        )
        conn.execute(
            """CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid) VALUES ('delete', old.id);
        END"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)"
        )
        conn.close()
    except ImportError:
        pass


def _episodic_store(role, content, session_id=None, tool_name=None):
    """Insert a message into episodic.db."""
    if session_id is None:
        session_id = _session_state.get("id")
    if not session_id or not content:
        return
    try:
        import sqlite3

        conn = sqlite3.connect(str(EPISODIC_DB))
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_name) VALUES (?, ?, ?, ?)",
            (session_id, role, content[:100000], tool_name),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _episodic_search(query, top_k=10):
    """Search FTS5 and return formatted results."""
    if not query:
        return ""
    try:
        import sqlite3

        conn = sqlite3.connect(str(EPISODIC_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """SELECT m.id, m.session_id, m.role, m.content, m.created_at, m.tool_name,
                      rank
               FROM messages_fts
               JOIN messages m ON messages_fts.rowid = m.id
               WHERE messages_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, top_k),
        )
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return ""
        result = f"Episodic memory results (FTS5) for: {query}\n\n"
        for r in rows:
            role = r["role"]
            content = r["content"][:500]
            ts = r["created_at"]
            sid = r["session_id"][:8]
            tool = f" [{r['tool_name']}]" if r["tool_name"] else ""
            result += f"[{ts}] session:{sid} {role}{tool}\n{content}\n---\n"
        return result
    except Exception as e:
        return f"Error searching episodic memory: {e}"


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


# ── File picker ───────────────────────────────────────────────

_FILE_RE = re.compile(r"@@(\S*)")


def _check_gitignore(paths):
    ignored = set()
    try:
        input_data = "\n".join(str(p) for p in paths)
        r = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(CURRENT_DIR),
        )
        if r.returncode == 0 and r.stdout.strip():
            for line in r.stdout.strip().split("\n"):
                ignored.add(line.strip())
    except Exception:
        pass
    return ignored


_FILE_SKIP_DIRS = frozenset({".git", ".gitignore"})

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"})

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


def _is_image(path):
    return Path(path).suffix.lower() in _IMAGE_EXTS


def _image_mime(path):
    return _IMAGE_MIME.get(Path(path).suffix.lower(), "image/png")


def _content_str_len(content):
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                total += len(part.get("text", ""))
                url = part.get("image_url", {})
                if isinstance(url, dict):
                    total += len(url.get("url", ""))
        return total
    return 0


def _fzf_file_selector(initial_query=""):
    """Use fzf to select files from the project. Returns comma-separated paths or None."""
    if not shutil.which("fzf"):
        return None
    all_files = _collect_files()
    if not all_files:
        return None
    fzf_input = "\n".join(f["path"] for f in all_files)
    cmd = ["fzf", "--multi", "--height=~80%", "--layout=reverse"]
    if initial_query:
        cmd.extend(["--query", initial_query])
    try:
        r = subprocess.run(
            cmd, input=fzf_input, capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0 and r.stdout.strip():
            selected = [s.strip() for s in r.stdout.strip().split("\n") if s.strip()]
            return ",".join(selected)
    except subprocess.TimeoutExpired:
        pass
    return None


def _collect_files():
    candidates = []
    for f in CURRENT_DIR.rglob("*"):
        if not f.is_file():
            continue
        if any(p.name in _FILE_SKIP_DIRS for p in f.parents):
            continue
        if f.stat().st_size > 3_000_000:
            continue
        try:
            rel = f.relative_to(CURRENT_DIR)
        except ValueError:
            continue
        candidates.append((f, rel))
    ignored = _check_gitignore([rel for _, rel in candidates])
    files = []
    for f, rel in candidates:
        if str(rel) in ignored:
            continue
        files.append({"path": str(rel), "size": f.stat().st_size})
    return sorted(files, key=lambda x: x["path"])


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
                    "tool": {
                        "type": "string",
                        "description": "The tool name that was called with invalid arguments",
                    },
                    "error": {
                        "type": "string",
                        "description": "Description of the validation error",
                    },
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
                                "question": {
                                    "type": "string",
                                    "description": "The complete question to ask",
                                },
                                "header": {
                                    "type": "string",
                                    "description": "Very short label (max 30 chars)",
                                },
                                "options": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {
                                                "type": "string",
                                                "description": "Display text (1-5 words)",
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "Explanation of choice",
                                            },
                                        },
                                        "required": ["label", "description"],
                                    },
                                    "description": "Available choices (omit for free-text input)",
                                },
                                "multiple": {
                                    "type": "boolean",
                                    "description": "Allow selecting more than one option",
                                },
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
            "description": "Execute a shell command on the local machine with timeout and working directory support. Runs in the project directory by default.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    },
                    "description": {
                        "type": "string",
                        "description": "Clear concise description of what this command does in 5-10 words",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in milliseconds (default: 120000)",
                        "default": 120000,
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory. Use this instead of 'cd' cd commands for directory changes.",
                    },
                },
                "required": ["command", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file or directory. Supports offset and limit for partial reads. Also reads images (PNG, JPG, GIF, WebP, BMP) and returns them as data URIs so vision models can see them. Re-reading a file you already read this round is allowed (content will be refreshed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "Path to the file or directory. For images, the content is returned as a data URI and made visible to vision models.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "The line number to start reading from (1-indexed)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "The maximum number of lines to read (defaults to 2000)",
                    },
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
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match files against",
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory to search in. Defaults to current working directory.",
                    },
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
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for in file contents",
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory to search in. Defaults to current working directory.",
                    },
                    "include": {
                        "type": "string",
                        "description": "File pattern to include (e.g. '*.js', '*.{ts,tsx}')",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Performs exact string replacements in files. Replaces oldString with newString. Supports replaceAll.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Path to the file"},
                    "oldString": {
                        "type": "string",
                        "description": "The text to replace",
                    },
                    "newString": {
                        "type": "string",
                        "description": "The text to replace it with (must be different from oldString)",
                    },
                    "replaceAll": {
                        "type": "boolean",
                        "description": "Replace all occurrences of oldString (default false)",
                    },
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
                    "filePath": {"type": "string", "description": "Path to the file"},
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
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
                    "description": {
                        "type": "string",
                        "description": "A short (3-5 words) description of the task",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task for the agent to perform",
                    },
                    "subagent_type": {
                        "type": "string",
                        "enum": ["general", "explore"],
                        "description": "The type of agent to use: 'general' for research/execution, 'explore' for codebase exploration",
                    },
                    "command": {
                        "type": "string",
                        "description": "The command that triggered this task",
                    },
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
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["text", "markdown", "html"],
                        "description": "The format to return the content in (text, markdown, or html). Defaults to markdown.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds (max 120)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todowrite",
            "description": "Write the task list to TASK.md. Call ONCE to plan, then ONLY when a task completes. If blocked, STOP calling and use edit/write instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Brief description of the task",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "in_progress",
                                        "completed",
                                        "cancelled",
                                    ],
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
                    "numResults": {
                        "type": "integer",
                        "description": "Number of search results to return (default: 8)",
                    },
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
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to load from installed skills",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_index",
            "description": "Build or rebuild the RAG index for a directory. Call this BEFORE rag_search if the index does not exist yet. The index is stored in ~/.config/tbot/rag_index/. Indexing is fast (<1s for most projects).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to index (default: current project directory, i.e. the directory you are working in)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the RAG index using BM25 (keyword-based, no LLM). Returns relevant chunks with snippet context (the best matching lines with actual text). Call rag_index first if the index doesn't exist. The snippet IS the relevant text — no need to read files separately. Ideal for knowledge base Q&A.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords, function names, concepts, or questions)",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 20)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_status",
            "description": "Show RAG index status (exists, chunk count, file count, term count). Use this to check if the index is ready before searching.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": "Crea una nueva skill en ~/.config/tbot/skills/<name>/ para que el modelo pueda reutilizar este procedimiento en el futuro. Úsalo DESPUÉS de completar un procedimiento complejo que involucró trial-and-error. Antes de llamarlo, carga skill-guide con skill_skill-guide() para conocer el formato exacto. NO llames esto si el usuario no ha pedido la tarea o si el procedimiento es trivial.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nombre de la skill. Debe coincidir con el del frontmatter. Minúsculas, guiones, alfanumérico (ej: 'deploy-railway', 'docker-cleanup')."
                    },
                    "description": {
                        "type": "string",
                        "description": "Descripción de 1-2 oraciones de cuándo el modelo debe usar esta skill."
                    },
                    "content": {
                        "type": "string",
                        "description": "El cuerpo completo de la skill en Markdown, incluyendo el frontmatter YAML. Debe seguir el formato documentado en skill-guide."
                    },
                    "schema": {
                        "type": "object",
                        "description": "JSON Schema opcional para los parámetros de la skill tool. Omitir para skills de solo lectura/lookup.",
                        "default": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "Input"
                                }
                            },
                            "required": ["input"]
                        }
                    },
                    "replace": {
                        "type": "boolean",
                        "description": "Si es true y la skill ya existe, la sobrescribe en lugar de dar error.",
                        "default": False
                    },
                    "files": {
                        "type": "object",
                        "description": "Opcional. Diccionario de nombre_archivo → contenido para archivos adicionales en la carpeta de la skill (ej: {'script.py': '...', 'config.template': '...'}).",
                        "default": {}
                    }
                },
                "required": ["name", "description", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fact_read",
            "description": "Lee MEMORY.md (hechos del entorno: proyectos, configuraciones, convenciones aprendidas) o USER.md (preferencias del usuario, estilo de comunicación). Se inyectan automáticamente al inicio de la sesión.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store": {
                        "type": "string",
                        "enum": ["memory", "user"],
                        "description": "Qué archivo leer: 'memory' para MEMORY.md, 'user' para USER.md"
                    }
                },
                "required": ["store"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fact_write",
            "description": "Agrega o reemplaza una entrada en MEMORY.md o USER.md. Si el texto ya existe (substring match), se reemplaza. Si no, se agrega. Respeta el límite de caracteres.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store": {
                        "type": "string",
                        "enum": ["memory", "user"],
                        "description": "'memory' para MEMORY.md, 'user' para USER.md"
                    },
                    "content": {
                        "type": "string",
                        "description": "El texto de la entrada a guardar. Debe ser una línea informativa y compacta."
                    }
                },
                "required": ["store", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fact_remove",
            "description": "Elimina una entrada de MEMORY.md o USER.md por substring matching. El substring debe identificar exactamente una sola entrada.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store": {
                        "type": "string",
                        "enum": ["memory", "user"],
                        "description": "'memory' para MEMORY.md, 'user' para USER.md"
                    },
                    "substring": {
                        "type": "string",
                        "description": "Substring único que identifica la entrada a eliminar"
                    }
                },
                "required": ["store", "substring"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "episodic_search",
            "description": "Busca en el historial completo de conversaciones usando FTS5. Útil para recordar discusiones pasadas, decisiones técnicas, o configuraciones mencionadas en sesiones anteriores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Términos de búsqueda (ej: 'despliegue coolify postgres', 'migración base de datos')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Número de resultados (default: 5, max: 20)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_manage",
            "description": "Gestiona la configuración de servidores MCP: listar, agregar, modificar o eliminar servidores. La config se guarda en ~/.config/tbot/config.json ('mcp_servers'). Para aplicar cambios, usa /mcp reconnect o reinicia tbot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "add", "remove", "modify"],
                        "description": "Acción a realizar: list (lista servidores configurados), add (agregar nuevo), remove (eliminar), modify (modificar existente)"
                    },
                    "name": {
                        "type": "string",
                        "description": "Nombre del servidor (requerido para add/remove/modify)"
                    },
                    "transport": {
                        "type": "string",
                        "enum": ["stdio", "http"],
                        "description": "Tipo de transporte (para add/modify, default: stdio)"
                    },
                    "command": {
                        "type": "string",
                        "description": "Comando a ejecutar (para transporte stdio)"
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Argumentos del comando (para transporte stdio)"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL del servidor (para transporte http)"
                    },
                    "env": {
                        "type": "object",
                        "description": "Variables de entorno del servidor (para transporte stdio)",
                        "additionalProperties": {"type": "string"}
                    },
                    "headers": {
                        "type": "object",
                        "description": "Headers HTTP personalizados (para transporte http, ej: API keys)",
                        "additionalProperties": {"type": "string"}
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "Si el servidor está habilitado (default: true)"
                    },
                    "auto_reconnect": {
                        "type": "boolean",
                        "description": "Reconexión automática al desconectarse (default: true)"
                    },
                    "request_timeout": {
                        "type": "integer",
                        "description": "Timeout de requests en segundos (default: 60)"
                    }
                },
                "required": ["action"]
            }
        }
    },
]

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

MAX_TOOL_OUTPUT = 32000

_READ_ONLY_TOOLS = frozenset(
    {
        "read",
        "glob",
        "grep",
        "question",
        "webfetch",
        "websearch",
        "skill",
        "skills",
        "invalid",
        "fact_read",
        "episodic_search",
    }
)


def _is_read_only_tool(name):
    return name in _READ_ONLY_TOOLS


def _truncate_output(text):
    if len(text) > MAX_TOOL_OUTPUT:
        return text[:MAX_TOOL_OUTPUT] + f"\n... (truncated, {len(text)} total chars)"
    return text


# ── Session log ──────────────────────────────────────────────


def _log_path():
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    return LOG_DIR / f"{ts}.log"


def _log_init():
    global _log_fh, _current_log_path
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _current_log_path = _log_path()
    _log_fh = open(_current_log_path, "a", encoding="utf-8")
    _log_write("── session started ──")


def _log_reopen():
    global _log_fh, _current_log_path
    if _log_fh is not None:
        _log_write("── session ended ──")
        try:
            _log_fh.close()
        except Exception:
            pass
    _current_log_path = _log_path()
    _log_fh = open(_current_log_path, "a", encoding="utf-8")
    _log_write("── session started ──")


def _log_close():
    global _log_fh
    if _log_fh is not None:
        _log_write("── session ended ──")
        try:
            _log_fh.close()
        except Exception:
            pass


def _log_write(text, role=None, tool_name=None):
    if _log_fh is not None:
        try:
            _log_fh.write(text.rstrip("\n") + "\n")
            _log_fh.flush()
        except Exception:
            pass
    # Also store in episodic database
    if role and text:
        _episodic_store(role, text, tool_name=tool_name)


def _llm_convert(text, target_format, cfg):
    base_url = _provider_url(cfg)
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    fmt_name = target_format.lstrip(".")
    system = (
        f"Convierte el siguiente log de conversación a formato {fmt_name.upper()}. "
        f"Preserva TODO el contenido, incluyendo preguntas y respuestas del usuario y asistente. "
        f"Usa el formato {fmt_name.upper()} apropiado con estructura clara. "
        f"Devuelve SOLO el resultado en el formato solicitado, sin explicaciones adicionales."
    )
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "temperature": 0.3,
        "max_tokens": 16384,
        "stream": False,
    }
    try:
        resp = requests.post(
            base_url + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return None


def _clean_html_text(html_text, min_line_length=0):
    """Strip HTML tags and extract readable paragraphs from HTML."""
    text = re.sub(
        r"<script[^>]*>.*?</script>", "", html_text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"<footer[^>]*>.*?</footer>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(
        r"<header[^>]*>.*?</header>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<[^>]+>", "\n", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if min_line_length:
        filtered = [l for l in lines if len(l) > min_line_length]
        return "\n".join(filtered) if filtered else "\n".join(lines)
    return "\n".join(lines)


def _fetch_page_text(url, max_chars=4000):
    """Fetch a URL and extract readable text content. Returns None on failure."""
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; tbot/1.0; +https://github.com/user/tbot)",
                "Accept": "text/html,text/plain,*/*",
            },
        )
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "text/html" not in ct and "text/plain" not in ct:
            return None
        text = _clean_html_text(resp.text, min_line_length=40)
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
                print(
                    f"  {C.YELLOW}{i}.{C.RESET} {opt['label']}  {C.GRAY}{desc}{C.RESET}"
                )
            print(f"  {C.YELLOW}0.{C.RESET} Type your own answer")
            while True:
                try:
                    raw = input(
                        f"{C.GREEN}choice{C.RESET} {'(comma-separated)' if multiple else ''}: "
                    ).strip()
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


_CD_RE = re.compile(r"^\s*cd\s+(.+?)(?:\s*[;&|#]|$)")


def _update_cwd(cmd, last_cwd):
    """Detect `cd <dir>` in command and update CURRENT_DIR."""
    m = _CD_RE.match(cmd)
    if not m:
        return
    target = m.group(1).strip().strip("'\"")
    resolved = _resolve_path(target)
    if resolved.is_dir():
        _chdir(resolved)


_output_re = re.compile(r"(?:-(?:o|O|output)\s+|\>\s*|\>\>\s*)(\S+)")


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
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout_s, cwd=cwd
        )
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
_read_trail = set()
_todo_blocked_count = 0


def _clear_trails():
    _doom_trail.clear()


def _detect_pattern(trail):
    if len(trail) < 20:
        return False
    names_only = [t[0] for t in trail[-6:]]
    # A,B,A,B,A,B
    if names_only[:2] == names_only[2:4] == names_only[4:6]:
        return True
    # A,B,C,A,B,C
    if len(trail) >= 6 and names_only[:3] == names_only[3:6]:
        return True
    return False


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
    if len(_doom_trail) > 12:
        _doom_trail = _doom_trail[-12:]
    if len(_doom_trail) >= 3:
        last_3 = _doom_trail[-3:]
        if all(t == last_3[0] for t in last_3):
            _doom_trail.clear()
            return (
                f"DOOM LOOP: You called {last_3[0][0]} 3 times with identical arguments. "
                "This is a loop. The tool was NOT executed. "
                "Use different arguments, a different tool, or respond with text."
            )
    if _detect_pattern(_doom_trail):
        pattern = " → ".join(t[0] for t in _doom_trail[-6:])
        _doom_trail.clear()
        return (
            f"PATTERN LOOP: Detected repeating tool pattern: {pattern}. "
            "This is a loop — repeating the same sequence of tools. "
            "Stop and respond with text to the user."
        )
    return None


# --- read tool ---
def handle_read(args):
    global _read_trail
    raw = args.get("filePath", "")
    if not raw:
        return "Error: filePath is required"
    filepath = str(_resolve_path(raw))
    offset = args.get("offset", 1)
    key = (filepath, offset)

    is_rerun = key in _read_trail
    if not is_rerun:
        _read_trail.add(key)

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
        entries = sorted(f"{e.name}/" if e.is_dir() else e.name for e in p.iterdir())
        total = len(entries)
        start = max(0, offset - 1)
        sliced = entries[start : start + limit]
        result = f"<path>{filepath}</path>\n<type>directory</type>\n<entries>\n"
        result += "\n".join(sliced)
        if start + len(sliced) < total:
            result += f"\n(Showing {len(sliced)} of {total} entries. Use 'offset' parameter to read beyond entry {offset + len(sliced)})"
        else:
            result += f"\n({total} entries)"
        result += "\n</entries>"
        return result
    if _is_image(p):
        sz = p.stat().st_size
        max_img = 20_000_000
        if sz > max_img:
            return f"Error: image too large ({sz} bytes, max {max_img})"
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        mime = _image_mime(p)
        return f"data:{mime};base64,{data}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = text.split("\n")
    total = len(lines)
    start = max(0, offset - 1)
    end = min(start + limit, total)
    sliced = lines[start:end]

    header = f"<path>{filepath}</path>\n<type>file</type>\n<content>\n"
    footer_prefix = "\n</content>"

    # Reserve ~200 chars for the boundary line so truncation is accurate
    max_body = MAX_TOOL_OUTPUT - len(header) - 200 - len(footer_prefix)
    if max_body < 200:
        max_body = 200

    re_read_prefix = f"(Re-read of '{raw}' at offset {offset})\n" if is_rerun else ""
    result = re_read_prefix + header
    shown_lines = 0
    for i, line in enumerate(sliced, start + 1):
        entry = f"{i}: {line}\n"
        if len(result) + len(entry) > max_body:
            # Can't fit this line — show accurate boundary
            remaining = total - i + 1
            result += f"\n(Showing lines {offset}-{i - 1} of {total}. Use offset={i} to continue.)"
            break
        result += entry
        shown_lines += 1
    else:
        # All requested lines fit
        if end < total:
            result += f"\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
        else:
            result += f"\n(End of file - total {total} lines)"

    result += footer_prefix
    return result


def handle_glob(args):
    pattern = _pick(args, "pattern")
    if not pattern:
        return "Error: pattern is required"
    search_path = args.get("path", ".")
    import glob as glob_mod

    p = _resolve_path(search_path)
    try:
        matches = sorted(glob_mod.glob(pattern, root_dir=p, recursive=True))
    except TypeError:
        cwd = os.getcwd()
        os.chdir(p)
        try:
            matches = sorted(glob_mod.glob(pattern, recursive=True))
        finally:
            os.chdir(cwd)
    if not matches:
        matches = sorted(glob_mod.glob(str(p / pattern), recursive=True))
        if matches:
            matches = [str(Path(m).relative_to(p)) for m in matches]
    if not matches:
        return "No files found matching pattern."
    limit = 200
    if len(matches) > limit:
        return (
            "\n".join(matches[:limit]) + f"\n... ({len(matches) - limit} more matches)"
        )
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
            if not fpath.is_file() or fpath.stat().st_size == 0:
                continue
            if include:
                import fnmatch

                if not fnmatch.fnmatch(fpath.name, include):
                    continue
            try:
                with fpath.open("r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
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


def _edit_snippet(filepath, new_text, idx, old_len, new_len, context=4):
    line_no = new_text[:idx].count("\n") + 1
    affected_end = new_text[: idx + max(new_len, 1)].count("\n") + 1
    lines = new_text.split("\n")
    start = max(0, line_no - 1 - context)
    end = min(len(lines), affected_end - 1 + context + 1)
    snippet = "\n".join(f"{j + 1}: {lines[j]}" for j in range(start, end))
    return f"{filepath}:{line_no}\n{snippet}"


def _render_diff(old_text, new_text, context=3):
    """Show a git-style colored diff between old and new text blocks."""
    old_lines = old_text.split("\n")
    new_lines = new_text.split("\n")
    if old_lines == new_lines:
        return ""
    parts = []
    show_old = old_lines and old_lines != [""]
    show_new = new_lines and new_lines != [""]
    if show_old:
        for line in old_lines:
            parts.append(f"{C.RED}- {line}{C.RESET}")
    if show_old and show_new:
        parts.append(f"{C.GRAY}───{C.RESET}")
    if show_new:
        for line in new_lines:
            parts.append(f"{C.GREEN}+ {line}{C.RESET}")
    return "\n".join(parts)


_last_edit_diff = None


def _emit_edit_diff():
    global _last_edit_diff
    if _last_edit_diff:
        for line in _last_edit_diff.split("\n"):
            print(f"  {line}")
        _last_edit_diff = None


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
    global _last_edit_diff
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
        idx = new_text.find(new)
        snip = _edit_snippet(filepath, new_text, idx, len(old), len(new))
        diff = _render_diff(old, new)
        _last_edit_diff = diff
        return f"Replaced {count} occurrence(s) in {filepath}\n{snip}"
    idx = text.find(old)
    if idx == -1:
        lines = text.split("\n")
        clue_lines = []
        for old_line in old.strip().split("\n")[:3]:
            stripped = old_line.strip()
            if stripped:
                for i, fline in enumerate(lines):
                    if stripped in fline:
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        ctx = "\n".join(
                            f"{j + 1}: {lines[j]}" for j in range(start, end)
                        )
                        clue_lines.append(f"  Near line {i + 1}:\n{ctx}")
                        break
        hint = ""
        if clue_lines:
            hint = "\nClosest matches in file:\n" + "\n".join(clue_lines[:2])
        hint += "\nSuggestion: include 2-3 lines of context BEFORE and AFTER the text to replace to make the match unique."
        return f"Error: could not find:\n{old[:500]}{hint}"
    last_idx = text.rfind(old)
    if idx != last_idx:
        suggestions = []
        for i, line in enumerate(text.split("\n")):
            if old.strip() in line:
                start = max(0, i - 1)
                end = min(len(text.split("\n")), i + 2)
                ctx = "\n".join(
                    f"{j + 1}: {text.split(chr(10))[j]}" for j in range(start, end)
                )
                suggestions.append(f"  Match at line {i + 1}:\n{ctx}")
                if len(suggestions) >= 2:
                    break
        hint = "\n" + "\n".join(suggestions) if suggestions else ""
        return (
            f"Error: multiple matches found.{hint}"
            f"\nSuggestion: use replaceAll=true to replace all, "
            f"or include 2-3 lines of context BEFORE and AFTER to make the match unique."
        )
    new_text = text[:idx] + new + text[idx + len(old) :]
    p.write_text(new_text, encoding="utf-8")
    snip = _edit_snippet(filepath, new_text, idx, len(old), len(new))
    diff = _render_diff(old, new)
    _last_edit_diff = diff
    return f"Replaced 1 occurrence in {filepath}\n{snip}"


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
    depth = int(os.environ.get("TBOT_DEPTH", "0"))
    if depth >= 3:
        return "Error: máxima profundidad de subagente (3) alcanzada. No se puede lanzar más tareas anidadas."
    desc = args.get("description", "task")
    prompt = args.get("prompt", "")
    if not prompt:
        return "Error: prompt is required"
    cfg = load_cfg()
    cfg["api_key"] = resolve_key(cfg)
    cmd = [sys.executable, sys.argv[0], "-m", cfg["model"], "-x", prompt]
    if not cfg.get("tools_enabled", True):
        cmd.append("--no-tools")
    if cfg.get("trust_mode"):
        cmd.append("--trust")
    env = {**os.environ, "TBOT_DEPTH": str(depth + 1)}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        out = r.stdout
        if r.stderr:
            out += "\n--- stderr ---\n" + r.stderr[-1000:]
        out += f"\n--- exit code: {r.returncode} ---"
        if r.returncode != 0:
            return f"Task '{desc}' failed (exit {r.returncode}):\n{out[:3000]}"
        return f"Task '{desc}' returned:\n{_truncate_output(out.strip())}"
    except subprocess.TimeoutExpired:
        return f"Task '{desc}' timed out after 300s"
    except Exception as e:
        return f"Task '{desc}' error: {e}"


def handle_webfetch(args):
    url = _pick(args, "url")
    if not url:
        return "Error: url is required"
    fmt = args.get("format", "markdown")
    timeout = min(args.get("timeout", 30), 120)
    if not url.startswith(("http://", "https://")):
        return "URL must start with http:// or https://"
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; tbot/1.0; +https://github.com/user/tbot)",
            },
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Error fetching URL: {e}"
    if fmt == "html":
        return _truncate_output(resp.text)
    if fmt == "text":
        text = _clean_html_text(resp.text)
        return _truncate_output(text)
    text = _clean_html_text(resp.text, min_line_length=40)
    return text or "(no readable content found)"


# --- todowrite state (loop prevention) ---
_todo_prev_fingerprint = None
_todo_noop_count = 0
_todo_last_call_time = 0
_todo_has_write_since_last = (
    False  # set by execute_tool_calls when edit/write/bash runs
)


def _todo_fingerprint(todos):
    return tuple(
        sorted(
            (t.get("content", ""), t.get("status", ""), t.get("priority", ""))
            for t in todos
        )
    )


def handle_todowrite(args):
    global \
        _todo_prev_fingerprint, \
        _todo_noop_count, \
        _todo_last_call_time, \
        _todo_has_write_since_last, \
        _todo_blocked_count

    if _todo_blocked_count >= 999:
        return "todowrite BLOQUEADO permanentemente — usa edit/write para modificar TASK.md."

    now = time.monotonic()

    # --- rate limiter: <15s since last call = rapid-fire loop ---
    if 0 < now - _todo_last_call_time < 15:
        _todo_blocked_count += 1
        _todo_last_call_time = now
        if _todo_blocked_count >= 3:
            _todo_blocked_count = 999
            return (
                "todowrite BLOCKED for this round — 3 consecutive rapid-fire calls detected. "
                "Use edit/write to modify TASK.md directly."
            )
        return (
            "TOOL LOOP BLOCKED: todowrite called <15s after previous call. "
            "TASK.md was NOT updated. Use edit/write/bash to make progress."
        )
    _todo_last_call_time = now
    _todo_blocked_count = 0

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
        status_map = {
            "pending": " ",
            "in_progress": "~",
            "completed": "x",
            "cancelled": "-",
        }
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


def _display_task_md():
    """Read TASK.md and print it with ANSI-colored markdown rendering."""
    task_file = CURRENT_DIR / "TASK.md"
    if not task_file.exists():
        return
    content = task_file.read_text(encoding="utf-8")
    print(f"\n  {C.BOLD}{C.GREEN}═══ TASK.md ═══{C.RESET}")
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            print()
            continue
        # Heading
        if stripped.startswith("# ") or stripped.startswith("#"):
            print(f"  {C.BOLD}{C.CYAN}{line}{C.RESET}")
        # Checkbox items
        elif stripped.startswith("- [") and "]" in stripped:
            rest = stripped[6:] if len(stripped) > 6 else ""
            if stripped.startswith("- [x]"):
                print(f"  {C.GREEN}✔{C.RESET} {rest}")
            elif stripped.startswith("- [~]") or stripped.startswith("- [-]"):
                print(f"  {C.YELLOW}◷{C.RESET} {rest}")
            elif stripped.startswith("- [ ]"):
                print(f"  {C.GRAY}○{C.RESET} {rest}")
            else:
                print(f"  {line}")
        # HTML comment
        elif stripped.startswith("<!--"):
            print(f"  {C.GRAY}{line}{C.RESET}")
        else:
            print(f"  {line}")
    print()


def _resolve_ddg_url(url):
    """Resolve DuckDuckGo redirect URLs to the actual target."""
    url = url.strip()
    if url.startswith("//"):
        url = "https:" + url
    m = re.search(r"[?&]uddg=([^&]+)", url)
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
        sess.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://duckduckgo.com/",
                "DNT": "1",
            }
        )
        sess.get("https://duckduckgo.com/", timeout=8)
        resp = sess.post(
            "https://html.duckduckgo.com/html/", data={"q": query}, timeout=10
        )
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
        title = re.sub(r"<[^>]+>", "", tm.group(1)).strip() if tm else ""
        sm = re.search(
            r'class="result__snippet"[^>]*>(.*?)</(?:span|div)>', block, re.DOTALL
        )
        snippet = re.sub(r"<[^>]+>", "", sm.group(1)).strip() if sm else ""
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
                for line in page_text.split("\n")[:10]:
                    out.append(f"  {line.strip()}")
    return "\n".join(out)


def handle_skill(args):
    name = args.get("name", "")
    if not name:
        return "Error: name is required"
    skills = load_skills()
    for n, desc, schema, doc, _entry in skills:
        if n == name:
            return f"Skill '{name}' found. Use `skill_{name}` to load instructions into context."
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
            text = text[:idx] + new_block + text[idx + len(old_block) :]
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
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*$", name):
        return f"{C.RED}invalid skill name — use letters, numbers, underscores{C.RESET}"
    desc = args.get("description", name)
    content = args.get("content", "")
    schema = args.get(
        "schema",
        {
            "type": "object",
            "properties": {"input": {"type": "string", "description": "Input"}},
            "required": ["input"],
        },
    )
    ensure_skills_dir()
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


def handle_create_skill_tool(args):
    """Handler for the `create_skill` tool — model-driven skill creation."""
    name = args.get("name", "")
    if not name:
        return "Error: name is required"
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*$", name):
        return f"Error: invalid skill name — use letters, numbers, underscores, hyphens"

    description = args.get("description", name)
    content = args.get("content", "")
    schema = args.get(
        "schema",
        {
            "type": "object",
            "properties": {"input": {"type": "string", "description": "Input"}},
            "required": ["input"],
        },
    )
    replace = args.get("replace", False)
    files = args.get("files", {})

    # Validate that content has YAML frontmatter
    meta = _parse_skill_text(content)
    if not meta:
        return (
            "Error: content no tiene frontmatter YAML válido (debe comenzar con ---\\n...\\n---\\n). "
            "Carga skill_skill-guide() para ver el formato exacto."
        )

    # Validate that frontmatter name matches argument
    fm_name = meta.get("name", "")
    if fm_name and fm_name != name:
        return f"Error: el nombre en el frontmatter ('{fm_name}') no coincide con el argumento name ('{name}')"

    ensure_skills_dir()
    skill_dir = SKILLS_DIR / name
    if skill_dir.exists():
        if not replace:
            return f"Error: skill '{name}' ya existe. Usa replace=true para sobrescribir o elige otro nombre."
        shutil.rmtree(skill_dir)

    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    # Write sibling files if provided
    for fname, fcontent in files.items():
        # Sanitize filename to prevent path traversal
        safe_name = Path(fname).name
        (skill_dir / safe_name).write_text(str(fcontent), encoding="utf-8")

    clear_skill_cache()
    return f"✓ Skill '{name}' creada exitosamente en {skill_dir}/SKILL.md. Disponible para usar con skill_{name}()."


def handle_get_system_info(_args):
    return json.dumps(
        {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "cwd": os.getcwd(),
            "user": os.environ.get("USER") or os.environ.get("USERNAME", ""),
            "home": str(Path.home()),
        },
        indent=2,
    )


# ── RAG (Go binary) ───────────────────────────────────────────


RAG_BIN = None


def _rag_binary():
    global RAG_BIN
    if RAG_BIN is not None:
        return RAG_BIN
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    cwd = Path.cwd()
    candidates = [
        script_dir / "rag" / "rag_bin",
        script_dir / "rag_bin",
        cwd / "rag" / "rag_bin",
        cwd / "rag_bin",
    ]
    for c in candidates:
        if c.exists() and os.access(str(c), os.X_OK):
            RAG_BIN = str(c)
            return RAG_BIN
    which = shutil.which("rag_bin")
    if which:
        RAG_BIN = which
        return RAG_BIN
    # Try auto-compile from the rag/ folder next to tbot.py
    rag_src = script_dir / "rag"
    if rag_src.is_dir() and (rag_src / "go.mod").exists():
        try:
            result = subprocess.run(
                ["go", "build", "-o", "rag_bin", "."],
                cwd=str(rag_src),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                RAG_BIN = str(rag_src / "rag_bin")
                os.chmod(RAG_BIN, 0o755)
                return RAG_BIN
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return None


def _run_rag(args, timeout=30):
    binary = _rag_binary()
    if not binary:
        return {"error": "rag binary not found — run 'cd rag && go build -o rag_bin .'"}
    try:
        r = subprocess.run(
            [binary] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(CURRENT_DIR),
        )
        if r.returncode != 0:
            return {"error": r.stderr.strip() or f"exit {r.returncode}"}
        if not r.stdout:
            return {"ok": True}
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            return {"raw": r.stdout.strip(), "stderr": r.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"error": "rag command timed out"}
    except FileNotFoundError:
        return {"error": "rag binary not found"}
    except Exception as e:
        return {"error": str(e)}


def handle_rag_index(args):
    path = args.get("path", ".")
    t0 = time.time()
    result = _run_rag(["index", path], timeout=120)
    elapsed = time.time() - t0
    if "error" in result:
        return f"Error indexing: {result['error']}"
    chunks = result.get("chunks", "?")
    files = result.get("files", "?")
    return f"Index built: {files} files, {chunks} chunks for {path} ({elapsed:.1f}s)"


def handle_rag_search(args):
    query = args.get("query", "")
    if not query:
        return "Error: query is required"
    top_k = min(args.get("top_k", 5), 20)
    t0 = time.time()
    result = _run_rag(["search", query, str(top_k)])
    elapsed = time.time() - t0
    if "error" in result:
        return f"Search error: {result['error']}"

    def format_result(r, idx):
        """Format a single search result with snippet text."""
        name = r.get("name", "") or ""
        snippet = r.get("snippet", "") or ""
        text = r.get("text", "") or ""
        score = r.get("score", 0)
        path = r.get("path", "?")
        start = r.get("start", "?")
        end = r.get("end", "?")
        lang = r.get("lang", "")
        parts = [f"[Resultado {idx}] {path}:{start}-{end} | score={score:.2f}"]
        if name:
            parts[0] += f" | «{name}»"
        if lang:
            parts[0] += f" ({lang})"
        # Include snippet (best matching region) — this is what the model needs
        if snippet:
            # Truncate very long snippets to avoid context overflow
            if len(snippet) > 1500:
                snippet = snippet[:1500] + "\n... [truncated]"
            parts.append("  ── contexto ──")
            parts.append(snippet)
            parts.append("  ─────────────")
        # Only include full text if no snippet (fallback for old index)
        elif text:
            if len(text) > 2000:
                text = text[:2000] + "\n... [truncated]"
            parts.append("  ── texto ──")
            parts.append(text)
            parts.append("  ───────────")
        return "\n".join(parts)

    # Case 1: _run_rag already parsed it as list of dicts
    if isinstance(result, list):
        if not result:
            return f"No se encontraron resultados. ({elapsed:.3f}s)"
        out = "\n\n".join(format_result(r, i + 1) for i, r in enumerate(result[:top_k]))
        return f"{out}\n\n({elapsed:.3f}s)"

    # Case 2: raw JSON string that needs parsing
    if "raw" in result:
        raw = result["raw"]
        if raw == "[]":
            return f"No se encontraron resultados. ({elapsed:.3f}s)"
        try:
            data = json.loads(raw)
            if not data:
                return f"No se encontraron resultados. ({elapsed:.3f}s)"
            out = "\n\n".join(format_result(r, i + 1) for i, r in enumerate(data[:top_k]))
            return f"{out}\n\n({elapsed:.3f}s)"
        except json.JSONDecodeError:
            return raw[:2000]
    return str(result)[:2000]


def handle_rag_status(_args):
    result = _run_rag(["status"])
    if "error" in result:
        return f"Index: missing ({result['error']})"
    if isinstance(result, dict):
        if not result.get("exists"):
            return "RAG index: not built yet. Use rag_index tool to build it."
        root = result.get("root_path", "")
        if root and str(_resolve_path(root)) != str(CURRENT_DIR):
            return (
                f"RAG index exists but is for a different project:\n"
                f"  Indexed: {root}\n"
                f"  Current: {CURRENT_DIR}\n"
                f"Use rag_index to build an index for this project."
            )
        return (
            f"RAG index: ready\n"
            f"  Path:   {root}\n"
            f"  Chunks: {result.get('chunks', 0)}\n"
            f"  Files:  {result.get('files', 0)}\n"
            f"  Terms:  {result.get('total_terms', 0)}"
        )
    return f"RAG status: {result}"


# ── Fact Memory handlers ─────────────────────────────────────


def _fact_file_path(store):
    return MEMORY_FILE if store == "memory" else USER_FILE


def _fact_limit(cfg, store):
    return cfg.get("memory_facts_limit", 2200) if store == "memory" else cfg.get("memory_user_limit", 2200)


# Import cfg lazily from the module-level config in main
_cfg_for_memory = None


def _get_memory_gate(cfg):
    return cfg.get("memory_gate", False)


def handle_fact_read(args):
    store = args.get("store", "memory")
    path = _fact_file_path(store)
    content = _read_fact_file(path)
    if not content:
        return f"# {store.upper()} — Perfil del Usuario\n\n(empty — no facts recorded yet)" if store == "user" else f"# MEMORY — Hechos del Entorno\n\n(empty — no facts recorded yet)"
    header = f"=== {path.name} ===\n"
    return header + content


def handle_fact_write(args):
    store = args.get("store", "memory")
    content = args.get("content", "").strip()
    if not content:
        return "Error: content is empty"
    path = _fact_file_path(store)
    # Memory gate: ask for user approval before writing
    if _cfg_for_memory and _cfg_for_memory.get("memory_gate", False):
        print(f"\n\033[33mMemory gate: approve write to {path.name}?\033[0m")
        print(f"  Content: {content[:200]}")
        answer = input("  y/N: ").strip().lower()
        if answer != "y":
            return "Write cancelled by memory gate"
    current = _read_fact_file(path)
    lines = current.split("\n") if current else []
    # Check if content already exists (substring match on non-header lines)
    # Lines that are headers or comments are ignored for matching
    found_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("<!--"):
            if content in stripped:
                found_idx = i
                break
    if found_idx is not None:
        lines[found_idx] = f"- {content}"
    else:
        lines.append(f"- {content}")
        lines.append(f"<!-- added {time.strftime('%Y-%m-%d %H:%M:%S')} -->")
    new_text = "\n".join(lines).strip()
    # Check limit
    limit = _fact_limit(_cfg_for_memory or {}, store)
    if len(new_text) > limit:
        return (
            f"Error: exceeds limit of {limit} chars ({len(new_text)} total). "
            f"Consolidate existing entries first. Current content:\n{new_text}"
        )
    _write_fact_file(path, new_text)
    return f"Written to {path.name} ({len(new_text)}/{limit} chars used)"


def handle_fact_remove(args):
    store = args.get("store", "memory")
    substring = args.get("substring", "").strip()
    if not substring:
        return "Error: substring is required"
    path = _fact_file_path(store)
    # Memory gate: ask for user approval before removing
    if _cfg_for_memory and _cfg_for_memory.get("memory_gate", False):
        print(f"\n\033[33mMemory gate: approve removal from {path.name}?\033[0m")
        print(f"  Substring: {substring[:200]}")
        answer = input("  y/N: ").strip().lower()
        if answer != "y":
            return "Removal cancelled by memory gate"
    current = _read_fact_file(path)
    if not current:
        return f"Error: {path.name} is empty"
    lines = current.split("\n")
    # Find lines matching substring (exclude headers/comments)
    matching = [l for l in lines if substring in l and not l.startswith("#") and not l.startswith("<!--")]
    if len(matching) == 0:
        return f"Error: no entry matches '{substring}'"
    if len(matching) > 1:
        matches_str = "\n".join(matching)
        return f"Error: multiple entries match '{substring}':\n{matches_str}\nUse a more specific substring."
    # Remove the matching line and any following <!-- added --> comment
    remove_idx = lines.index(matching[0])
    removed = [remove_idx]
    if remove_idx + 1 < len(lines) and lines[remove_idx + 1].startswith("<!--"):
        removed.append(remove_idx + 1)
    new_lines = [l for i, l in enumerate(lines) if i not in removed]
    new_text = "\n".join(new_lines).strip()
    _write_fact_file(path, new_text)
    return f"Removed entry from {path.name}"


def handle_episodic_search(args):
    query = args.get("query", "")
    top_k = min(args.get("top_k", 5), 20)
    if not query:
        return "Error: query is required"
    result = _episodic_search(query, top_k=top_k)
    if not result:
        return f"No episodic memories found for: {query}"
    return result


# ── MCP tool handler ──────────────────────────────────────────────


def _init_mcp(cfg):
    """Inicializa el gestor MCP con la configuración actual (síncrono).

    Conecta todos los servidores MCP configurados y muestra un resumen
    limpio antes de que aparezca el prompt.
    """
    global _mcp_manager
    if not _HAS_MCP:
        return

    mcp_configs = cfg.get("mcp_servers", [])
    _mcp_manager = MCPServerManager()
    _mcp_manager.load_configs(mcp_configs)

    if not mcp_configs:
        return

    results = _mcp_manager.connect_all(parallel=True)
    ok = sum(1 for _, s in results if s == "ok")
    total = len(results)
    errs = [n for n, s in results if s not in ("ok", "disabled")]

    if ok:
        mcp_count = len(_get_mcp_tools())
        print(f"  {C.GREEN}MCP ✓{C.RESET} {ok}/{total} servidores, {mcp_count} tools")
    if errs:
        for name, status in results:
            if status not in ("ok", "disabled"):
                print(f"  {C.RED}MCP ✗{C.RESET} {name} — {status}")


def _get_mcp_tools():
    """Retorna tools MCP en formato OpenAI function calling.

    Returns:
        list[dict] — herramientas para agregar a TOOLS
    """
    global _mcp_manager
    if not _HAS_MCP or _mcp_manager is None:
        return []
    try:
        return _mcp_manager.get_all_tools()
    except Exception:
        return []


def _get_mcp_resource_notes():
    """Retorna texto sobre recursos MCP para inyectar en system prompt."""
    global _mcp_manager
    if not _HAS_MCP or _mcp_manager is None:
        return ""
    try:
        return _mcp_manager.get_resource_notes()
    except Exception:
        return ""


def handle_mcp_tool_call(qualified_name, arguments):
    """Handler para tool calls MCP (prefijo 'mcp__')."""
    global _mcp_manager
    if not _HAS_MCP:
        return "Error: MCP no disponible — mcp_client.py no encontrado. Asegúrate de que mcp_client.py está en el mismo directorio que tbot.py."
    if _mcp_manager is None:
        return "Error: MCP no inicializado. Reinicia tbot o ejecuta /mcp reconnect."
    return _mcp_manager.call_tool(qualified_name, arguments)


def handle_mcp_command(args_dict):
    """Handler para el comando interactivo /mcp.

    args_dict: dict con 'action' y 'name' opcional
    """
    global _mcp_manager
    if not _HAS_MCP:
        return "MCP no disponible (mcp_client.py no encontrado)"
    if _mcp_manager is None:
        return "MCP no inicializado — reinicia tbot"

    action = args_dict.get("action", "status")

    if action == "status":
        statuses = _mcp_manager.get_status()
        if not statuses:
            return "No hay servidores MCP configurados. Agrega servidores en ~/.config/tbot/config.json bajo 'mcp_servers'."
        lines = [f"MCP Servers ({len(statuses)}):"]
        for s in statuses:
            status_color = {
                "connected": C.GREEN,
                "disconnected": C.RED,
                "not_initialized": C.YELLOW,
            }.get(s["status"], C.GRAY)
            lines.append(
                f"  {status_color}{s['name']:<20}{C.RESET} "
                f"{s['transport']:<8} "
                f"{status_color}{s['status']:<16}{C.RESET} "
                f"{C.GRAY}{s['tools']} tools, {s['resources']} resources{C.RESET}"
            )
        return "\n".join(lines)

    elif action == "connect":
        name = args_dict.get("name", "")
        if not name:
            # Reconectar todos los que estén desconectados
            count = 0
            results = _mcp_manager.connect_all()
            for n, status in results:
                if status == "ok":
                    count += 1
            return f"Conectados {count} servidor(es) MCP"
        client = _mcp_manager.get_client(name)
        if client:
            if client.connected:
                return f"Servidor '{name}' ya está conectado"
            # Reconnect específico
            from mcp_client import MCPClient
            for cfg in _mcp_manager._configs:
                if cfg["name"] == name:
                    try:
                        new_client = MCPClient.from_config(cfg)
                        new_client.connect()
                        _mcp_manager._clients[name] = new_client
                        return f"Servidor '{name}' reconectado"
                    except Exception as e:
                        return f"Error conectando '{name}': {e}"
            return f"Servidor '{name}' no encontrado en configuración"
        return f"Servidor '{name}' no encontrado"

    elif action == "disconnect":
        name = args_dict.get("name", "")
        if not name:
            return "Uso: /mcp disconnect <name>"
        client = _mcp_manager.get_client(name)
        if client:
            client.close()
            return f"Servidor '{name}' desconectado"
        return f"Servidor '{name}' no encontrado"

    elif action == "reconnect":
        name = args_dict.get("name", "")
        if name:
            client = _mcp_manager.get_client(name)
            if client:
                client.close(terminate=True)
                return handle_mcp_command({"action": "connect", "name": name})
            return f"Servidor '{name}' no encontrado"
        # Reconectar todos
        _mcp_manager.close_all()
        results = _mcp_manager.connect_all()
        ok = sum(1 for _, s in results if s == "ok")
        return f"Reconectados {ok}/{len(results)} servidores MCP"

    elif action == "discover":
        name = args_dict.get("name", "")
        if name:
            client = _mcp_manager.get_client(name)
            if client and client.connected:
                _mcp_manager.discover_all()
                return f"Rediscovery completado para '{name}': {len(client.tools)} tools, {len(client.resources)} resources"
            return f"Servidor '{name}' no encontrado o no conectado"
        _mcp_manager.discover_all()
        total_tools = sum(len(c.tools) for c in _mcp_manager._clients.values() if c.connected)
        total_resources = sum(len(c.resources) for c in _mcp_manager._clients.values() if c.connected)
        return f"Rediscovery completado: {total_tools} tools, {total_resources} resources"

    return f"Acción desconocida: {action}"


def handle_mcp_manage(args):
    """Gestiona la configuración de servidores MCP en ~/.config/tbot/config.json.

    Actions: list, add, remove, modify
    """
    action = args.get("action", "list")

    cfg = load_cfg()
    mcp_servers = cfg.get("mcp_servers", [])

    if action == "list":
        if not mcp_servers:
            return "No hay servidores MCP configurados.\n\nUsa `mcp_manage` con action='add' para agregar uno.\nLa configuración se guarda en ~/.config/tbot/config.json bajo 'mcp_servers'."
        lines = [f"Servidores MCP configurados ({len(mcp_servers)}):"]
        for s in mcp_servers:
            name = s.get("name", "?")
            transport = s.get("transport", "stdio")
            enabled = s.get("enabled", True)
            cmd = s.get("command", s.get("url", ""))
            status = "enabled" if enabled else "disabled"
            lines.append(f"  [{status}] {name} ({transport}) — {cmd}")
        return "\n".join(lines)

    elif action == "add":
        name = args.get("name", "").strip()
        if not name:
            return "Error: 'name' es requerido para agregar un servidor."
        # Check duplicate
        for s in mcp_servers:
            if s["name"] == name:
                return f"Error: ya existe un servidor llamado '{name}'. Usa action='modify' para modificarlo."

        transport = args.get("transport", "stdio")
        entry = {
            "name": name,
            "transport": transport,
            "enabled": args.get("enabled", True),
        }

        if transport == "stdio":
            if not args.get("command"):
                return "Error: 'command' es requerido para transporte stdio."
            entry["command"] = args["command"]
            entry["args"] = args.get("args", [])
        elif transport == "http":
            if not args.get("url"):
                return "Error: 'url' es requerido para transporte http."
            entry["url"] = args["url"]
        else:
            return f"Error: transporte desconocido '{transport}'. Usa 'stdio' o 'http'."

        if args.get("env"):
            entry["env"] = args["env"]
        if args.get("headers"):
            entry["headers"] = args["headers"]
        if args.get("auto_reconnect") is not None:
            entry["auto_reconnect"] = args["auto_reconnect"]
        if args.get("request_timeout"):
            entry["request_timeout"] = args["request_timeout"]

        mcp_servers.append(entry)
        cfg["mcp_servers"] = mcp_servers
        save_cfg(cfg)
        return f"Servidor MCP '{name}' agregado exitosamente.\n\nImportante: ejecuta `/mcp reconnect` o reinicia tbot para activar el nuevo servidor."

    elif action == "remove":
        name = args.get("name", "").strip()
        if not name:
            return "Error: 'name' es requerido para eliminar un servidor."
        idx = None
        for i, s in enumerate(mcp_servers):
            if s["name"] == name:
                idx = i
                break
        if idx is None:
            return f"Error: no se encontró un servidor llamado '{name}'."
        removed = mcp_servers.pop(idx)
        cfg["mcp_servers"] = mcp_servers
        save_cfg(cfg)
        return f"Servidor MCP '{name}' eliminado."

    elif action == "modify":
        name = args.get("name", "").strip()
        if not name:
            return "Error: 'name' es requerido para modificar un servidor."
        idx = None
        for i, s in enumerate(mcp_servers):
            if s["name"] == name:
                idx = i
                break
        if idx is None:
            return f"Error: no se encontró un servidor llamado '{name}'."

        entry = mcp_servers[idx]

        if "transport" in args:
            entry["transport"] = args["transport"]
        if "enabled" in args:
            entry["enabled"] = args["enabled"]
        if "command" in args:
            entry["command"] = args["command"]
        if "args" in args:
            entry["args"] = args["args"]
        if "url" in args:
            entry["url"] = args["url"]
        if "env" in args:
            entry["env"] = args["env"]
        if "headers" in args:
            entry["headers"] = args["headers"]
        if "auto_reconnect" in args:
            entry["auto_reconnect"] = args["auto_reconnect"]
        if "request_timeout" in args:
            entry["request_timeout"] = args["request_timeout"]

        cfg["mcp_servers"] = mcp_servers
        save_cfg(cfg)
        return f"Servidor MCP '{name}' modificado.\n\nImportante: ejecuta `/mcp reconnect` o reinicia tbot para aplicar los cambios."

    return f"Acción desconocida: '{action}'. Usa list, add, remove, o modify."


TOOL_HANDLERS = {
    "invalid": handle_invalid,
    "question": handle_question,
    "bash": handle_bash,
    "read": handle_read,
    "glob": handle_glob,
    "grep": handle_grep,
    "edit": handle_edit,
    "write": handle_write,
    "task": handle_task,
    "webfetch": handle_webfetch,
    "todowrite": handle_todowrite,
    "websearch": handle_websearch,
    "skill": handle_skill,
    "rag_index": handle_rag_index,
    "rag_search": handle_rag_search,
    "rag_status": handle_rag_status,
    "create_skill": handle_create_skill_tool,
    "fact_read": handle_fact_read,
    "fact_write": handle_fact_write,
    "fact_remove": handle_fact_remove,
    "episodic_search": handle_episodic_search,
    "mcp_manage": handle_mcp_manage,
}


# ── Skills (SKILL.md v1 — directory format) ───────────────────

_skill_cache = None


try:
    import yaml

    _has_yaml = True
except ImportError:
    _has_yaml = False


def _parse_skill_text(text, source=None):
    """Parse SKILL.md text with YAML frontmatter (between --- delimiters).
    Returns dict with metadata keys + '_doc' for the body, or None if no frontmatter.
    """
    m = re.match(r"^---\s*\n(.*?)\n(?:---|\.\.\.)\s*\n(.*)", text, re.DOTALL)
    if not m:
        return None
    raw_yaml = m.group(1)
    meta = None
    yaml_error = None
    if _has_yaml:
        try:
            meta = yaml.safe_load(raw_yaml) or {}
        except Exception as e:
            yaml_error = e
    if not meta:
        if yaml_error:
            print(
                f"{C.YELLOW}⚠ YAML parse error in {source or 'SKILL.md'}{C.RESET}"
                f"\n  {C.GRAY}{yaml_error}{C.RESET}"
                f"\n  {C.GRAY}Falling back to manual parser{C.RESET}"
            )
        meta = {}
        for line in raw_yaml.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if val.lower() in ("true", "yes"):
                    val = True
                elif val.lower() in ("false", "no"):
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
    """Parse SKILL.md file. Returns metadata dict or None."""
    text = path.read_text(encoding="utf-8")
    result = _parse_skill_text(text, source=str(path))
    if result is None:
        print(f"{C.YELLOW}⚠ SKILL.md has no valid frontmatter: {path}{C.RESET}")
    return result


def load_skills():
    global _skill_cache
    if _skill_cache is not None:
        return _skill_cache
    ensure_skills_dir()
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
        desc = meta.get("description", "") or (
            meta["_doc"][:80] if meta.get("_doc") else name
        )
        schema = meta.get("schema", default_schema)
        doc = meta.get("_doc")
        if doc:
            skills.append((name, desc, schema, doc, entry.name))
    _skill_cache = skills
    return skills


def ensure_skills_dir():
    if not SKILLS_DIR.is_dir():
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)


def clear_skill_cache():
    global _skill_cache
    _skill_cache = None


SKILL_GUIDE_SKILL = r"""---
name: "skill-guide"
description: "Reference guide for creating, modifying, updating, or fixing tbot skills (SKILL.md files). Load this when you need to create, edit, update, or fix a SKILL.md."
schema:
  type: object
  properties:
    input:
      type: string
      description: "Topic to focus on (frontmatter, schema, dependencies, or leave empty for full guide)"
---

# Skill: skill-guide

## Overview

A tbot skill is a directory under `~/.config/tbot/skills/<name>/` containing a `SKILL.md` file with YAML frontmatter.
Skills are loaded on-demand via the `skill` tool — the model calls `skill_<name>()` to inject instructions into context.

---

## Frontmatter

Every `SKILL.md` must start with YAML frontmatter between `---` delimiters:

```yaml
---
name: "skill-name"
description: "Clear description of when to use this skill"
schema:
  type: object
  properties:
    input:
      type: string
      description: "Input description"
  required: [input]
---
```

### Fields

| Field        | Required | Description                                      |
|-------------|----------|--------------------------------------------------|
| `name`       | yes      | Lowercase, alphanumeric with hyphens. Must match the directory name. |
| `description`| yes      | 1-2 sentence description of when the model should use this skill. |
| `schema`     | no       | JSON Schema for the skill tool's parameters. Omit for simple lookup-only skills. |

---

## Body

After frontmatter, write markdown instructions. This is what gets injected into the model's context when `skill_<name>()` is called.

### Structure

- Start with `# Skill: <name>`
- Use `## Overview` to describe what the skill does
- Use `## Dependencies` to list other skills or packages required
- Use `## Steps` or `## Instructions` for actionable steps
- Reference scripts and file paths using absolute paths

### Example

```markdown
# Skill: my-skill

## Overview
Does X, Y, and Z using the existing tools at `/path/to/tools`.

## Dependencies
This skill depends on `other-skill` — load it with `skill_other-skill()` first.

## Steps
1. Load dependencies with `skill_other-skill()`
2. Read the configuration from `/path/to/config`
3. Run the generator script
4. Output the result
```

---

## Dependencies

List other skills or external tools in a `## Dependencies` section:

```markdown
## Dependencies

- `pip install requests` — installs a Python package
- `npm install axios` — installs a Node.js package
- `other-skill` — another skill that must be loaded first
- `git` (must be available on PATH)
```

tbot auto-installs `pip install` and `npm install` dependencies when the skill is installed from a URL or git repo.

---

## Best Practices

1. **Single responsibility** — each skill does one thing well
2. **Black-box dependencies** — invoke other skills via their CLI or tool, never copy their internal logic
3. **Idempotent** — running the same skill twice produces the same result
4. **Self-contained** — avoid relying on mutable global state
5. **Descriptive name** — the model chooses skills by name + description, so be specific
6. **No code duplication** — use `execFileSync` / `subprocess` to call other skills' CLIs instead of reimplementing their logic"""


def _init_default_skills():
    """Create or sync default skills with the latest embedded content."""
    ensure_skills_dir()
    skill_dir = SKILLS_DIR / "skill-guide"
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md.write_text(SKILL_GUIDE_SKILL, encoding="utf-8")
        clear_skill_cache()
    else:
        current = skill_md.read_text(encoding="utf-8")
        if current != SKILL_GUIDE_SKILL:
            skill_md.write_text(SKILL_GUIDE_SKILL, encoding="utf-8")
            clear_skill_cache()


def _install_from_skill_url(skill_url):
    """Install a skill from a direct URL to SKILL.md."""
    ensure_skills_dir()
    try:
        resp = requests.get(skill_url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"{C.RED}download failed: {e}{C.RESET}"
    text = resp.text
    if text.strip().startswith("<!DOCTYPE") or text.strip().startswith("<html"):
        return (
            f"{C.RED}URL returned HTML (web page), not a SKILL.md file{C.RESET}\n"
            f"{C.GRAY}Use a raw URL (e.g. raw.githubusercontent.com/...) or a git repo URL{C.RESET}"
        )
    meta = _parse_skill_text(text)
    if not meta:
        return f"{C.RED}invalid SKILL.md — no frontmatter{C.RESET}"
    name = meta.get("name", "")
    if not name or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*$", name):
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
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
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
            if not name or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*$", name):
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
        r"^##\s+Dependenc(?:ies|ias)\s*\n(.*?)(?=\n##\s|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not m:
        return []
    results = [f"  {C.CYAN}Dependencies:{C.RESET}"]
    for line in m.group(1).split("\n"):
        line = line.strip()
        if not line.startswith("- "):
            continue
        content = line[2:].strip()
        bt = re.match(r"^`([^`]+)`", content)
        cmd_str = (
            bt.group(1)
            if bt
            else content.split(" - ")[0].split(" — ")[0].split(" – ")[0].strip()
        )
        if not cmd_str:
            continue
        if cmd_str.startswith(("pip install", "pip3 install")):
            exe = shutil.which("pip") or shutil.which("pip3")
            if not exe:
                results.append(f"    {C.YELLOW}⚠ pip not found{C.RESET}")
                continue
            args = shlex.split(cmd_str)
            pip_args = [exe, *args[1:]]
            if "--user" not in pip_args:
                pip_args.append("--user")
            if "--break-system-packages" not in pip_args:
                pip_args.append("--break-system-packages")
            try:
                r = subprocess.run(pip_args, capture_output=True, text=True, timeout=60)
                status = (
                    f"{C.GREEN}✓{C.RESET}"
                    if r.returncode == 0
                    else f"{C.RED}✗{C.RESET}"
                )
                if r.returncode != 0:
                    detail = r.stderr.strip()[-200:]
                    results.append(f"    {status} {cmd_str} — {detail}")
                else:
                    results.append(f"    {status} {cmd_str}")
            except Exception as e:
                results.append(f"    {C.RED}✗{C.RESET} {cmd_str}: {e}")
        elif cmd_str.startswith(("npm install", "npm i")):
            exe = shutil.which("npm")
            if not exe:
                results.append(f"    {C.YELLOW}⚠ npm not found{C.RESET}")
                continue
            args = shlex.split(cmd_str)
            try:
                r = subprocess.run(
                    [exe, *args[1:]], capture_output=True, text=True, timeout=120
                )
                status = (
                    f"{C.GREEN}✓{C.RESET}"
                    if r.returncode == 0
                    else f"{C.RED}✗{C.RESET}"
                )
                results.append(f"    {status} {cmd_str}")
            except Exception as e:
                results.append(f"    {C.RED}✗{C.RESET} {cmd_str}: {e}")
        elif cmd_str.startswith(("brew install",)):
            args = shlex.split(cmd_str)
            try:
                r = subprocess.run(args, capture_output=True, text=True, timeout=300)
                status = (
                    f"{C.GREEN}✓{C.RESET}"
                    if r.returncode == 0
                    else f"{C.RED}✗{C.RESET}"
                )
                results.append(f"    {status} {cmd_str}")
            except Exception as e:
                results.append(f"    {C.RED}✗{C.RESET} {cmd_str}: {e}")
        else:
            par = re.search(r"\((`[^`]+`|[^)]+)\)", cmd_str)
            if par:
                binary = par.group(1).strip("`")
                if shutil.which(binary):
                    results.append(f"    {C.GREEN}✓{C.RESET} {binary} found")
                else:
                    results.append(
                        f"    {C.YELLOW}⚠{C.RESET} {binary} not found — install manually"
                    )
            else:
                results.append(f"    {C.YELLOW}⚠{C.RESET} {cmd_str} — skipped")
    return results


_GITHUB_TREE_RE = re.compile(r"https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)")
_GITHUB_RAW_RE = re.compile(
    r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)"
)


def _github_tree_to_raw(url):
    """Convert GitHub tree URL to raw.githubusercontent.com URL for SKILL.md."""
    m = _GITHUB_TREE_RE.match(url)
    if m:
        owner, repo, branch, path = m.group(1), m.group(2), m.group(3), m.group(4)
        return (
            f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/SKILL.md"
        )
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
            resp = requests.get(
                api_url,
                timeout=15,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "tbot/1.0",
                },
            )
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
    is_git = (
        url.endswith(".git")
        or (not has_skill_md and "github.com" in url)
        or url.startswith("git@")
    )
    if is_git:
        return _install_from_git(url)
    if has_skill_md:
        return _install_from_skill_url(url)
    return _install_from_skill_url(url.rstrip("/") + "/SKILL.md")


def skills_to_tools(skills):
    tools = []
    for n, d, s, *_ in skills:
        desc = f"[ONE-TIME] Load instructions for '{n}' skill. Call ONCE, then follow the instructions — do NOT call again. {d}"
        params = dict(s)
        if "required" in params and not params["required"]:
            del params["required"]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"skill_{n}",
                    "description": desc[:500],
                    "parameters": params,
                },
            }
        )
    return tools


def skill_tool_handler(name, args, messages=None):
    for n, desc, schema, doc, entry_name in load_skills():
        if n == name:
            if messages is not None:
                already = any(
                    m.get("role") == "system"
                    and f"## Skill: {name}" in m.get("content", "")
                    for m in messages[-5:]
                )
                if already:
                    return f"Skill '{name}' is already loaded. Follow the instructions already in context."
                skill_dir = SKILLS_DIR / entry_name
                siblings = sorted(
                    f.name
                    for f in skill_dir.iterdir()
                    if f.is_file()
                    and f.suffix in (".md", ".txt", ".py", ".sh", ".json")
                )
                siblings_info = f"\n\nSkill directory: {skill_dir}\n" + (
                    f"Reference files: {', '.join(siblings)}" if siblings else ""
                )
                messages.append(
                    {
                        "role": "system",
                        "content": f"## Skill: {name}\n\n{doc}{siblings_info}",
                    }
                )
                return f"Skill '{name}' instructions loaded. Follow them to complete the task."
            return f"Skill '{name}' found. Use `skill_{name}` to load instructions."
    return f"Skill '{name}' not found"


# ── Model list cache ────────────────────────────────────────

MODELS_CACHE_FILE = CONFIG_DIR / "models.json"
_models_cache = []
_models_cache_time = 0


def _load_models_cache():
    global _models_cache, _models_cache_time
    _models_cache = []
    _models_cache_time = 0
    try:
        if MODELS_CACHE_FILE.exists():
            data = json.loads(MODELS_CACHE_FILE.read_text())
            _models_cache = data.get("models", [])
            _models_cache_time = data.get("time", 0)
    except Exception:
        MODELS_CACHE_FILE.unlink(missing_ok=True)
        _models_cache = []
        _models_cache_time = 0


def _save_models_cache(models):
    global _models_cache, _models_cache_time
    _models_cache = models
    _models_cache_time = time.time()
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_CACHE_FILE.write_text(
            json.dumps({"time": _models_cache_time, "models": models})
        )
    except Exception:
        pass


def _setup_custom_provider(cfg):
    print(f"\n{C.CYAN}── Custom API Provider ──{C.RESET}")
    print(
        f"{C.GRAY}Enter the base URL for your API (e.g. https://api.example.com/v1){C.RESET}"
    )
    raw_url = input(f"{C.GREEN}base_url:{C.RESET} ").strip()
    if not raw_url:
        print(f"{C.RED}cancelled{C.RESET}")
        return
    raw_url = raw_url.rstrip("/")
    if raw_url.endswith("/chat/completions"):
        raw_url = raw_url[: -len("/chat/completions")]
        print(f"{C.GRAY}  → stripped /chat/completions{C.RESET}")
    print(f"{C.GRAY}  → {raw_url}{C.RESET}")
    key = input(f"{C.GREEN}api_key:{C.RESET} ").strip()
    if not key:
        print(f"{C.RED}cancelled{C.RESET}")
        return
    old_prov = cfg.get("provider", "openrouter")
    cfg["provider"] = "custom"
    cfg["custom_url"] = raw_url
    cfg["api_key"] = key
    cfg.pop("_context_length", None)
    save_cfg(cfg)
    _save_models_cache([])
    print(f"{C.GREEN}provider → Custom API{C.RESET}")
    if old_prov != "custom":
        print(f"{C.YELLOW}model reset to default{C.RESET}")
        cfg["model"] = default_cfg()["model"]
        save_cfg(cfg)


def _provider_url(cfg):
    provider = cfg.get("provider", "openrouter")
    if provider == "custom":
        return cfg.get("custom_url", "")
    info = PROVIDERS.get(provider, PROVIDERS["openrouter"])
    return info.get("url", "")


def fetch_models(api_key, max_age=3600, provider="openrouter", custom_url=""):
    global _models_cache, _models_cache_time
    _load_models_cache()
    now = time.time()
    if _models_cache and now - _models_cache_time < max_age:
        return _models_cache
    try:
        if provider == "custom":
            base_url = custom_url
        else:
            info = PROVIDERS.get(provider, PROVIDERS["openrouter"])
            base_url = info.get("url", "")
        if not base_url:
            return _models_cache or []
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.get(base_url + "/models", timeout=10, headers=headers)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            _save_models_cache(data)
            return data
    except Exception:
        pass
    return _models_cache or []


def get_model_context(model_id, api_key, provider="openrouter", custom_url=""):
    for m in fetch_models(api_key, provider=provider, custom_url=custom_url):
        if m.get("id") == model_id:
            ctx = m.get("context_length")
            if ctx:
                return ctx
    return None


def default_cfg():
    return {
        "api_key": "",
        "provider": "openrouter",
        "model": "deepseek/deepseek-v4-flash",
        "temperature": 0.7,
        "system_prompt": "",
        "max_history_chars": 200000,
        "tools_enabled": True,
        "trust_mode": False,
        "resp_color": "95",
        "memory_gate": False,
        "memory_facts_limit": 2200,
        "memory_user_limit": 2200,
        "episodic_retention_days": 90,
        "mcp_servers": [],
    }


def load_cfg():
    if not CONFIG_FILE.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cfg = default_cfg()
        save_cfg(cfg)
        return cfg
    try:
        data = json.loads(CONFIG_FILE.read_text())
        base = default_cfg()
        base.update(data)
        base.pop("max_tokens", None)
        return base
    except Exception:
        return default_cfg()


def save_cfg(cfg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


# ── API ─────────────────────────────────────────────────────────


def resolve_key(cfg):
    provider = cfg.get("provider", "openrouter")
    if provider == "custom":
        key = cfg.get("api_key")
        if not key:
            print(f"{C.YELLOW}No API key configured for Custom API.{C.RESET}")
            try:
                key = input(f"{C.GREEN}Enter API key: {C.RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                sys.exit(1)
            if not key:
                print(f"{C.RED}No key provided.{C.RESET}")
                sys.exit(1)
            cfg["api_key"] = key
            save_cfg(cfg)
        return key or ""
    info = PROVIDERS.get(provider, PROVIDERS["openrouter"])
    env_key = info["env_key"]
    key = cfg.get("api_key") or os.environ.get(env_key)
    if key:
        if not cfg.get("api_key"):
            cfg["api_key"] = key
            save_cfg(cfg)
        return key
    print(f"{C.YELLOW}No API key configured for {info['name']}.{C.RESET}")
    print(f"Set {C.CYAN}{env_key}{C.RESET} environment variable or enter it now.")
    try:
        key = input(f"{C.GREEN}Enter API key: {C.RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    if not key:
        print(f"{C.RED}No key provided.{C.RESET}")
        sys.exit(1)
    cfg["api_key"] = key
    save_cfg(cfg)
    return key


def show_error(title, detail, hint=""):
    width = min(
        72, os.get_terminal_size().columns if hasattr(os, "get_terminal_size") else 72
    )
    print(f"\n{C.RED}╭─{'─' * (width - 4)}─╮{C.RESET}")
    print(
        f"{C.RED}│{C.RESET} {C.BOLD}{C.RED}✗ {title}{C.RESET}{' ' * (width - len(title) - 7)}{C.RED}│{C.RESET}"
    )
    print(f"{C.RED}│{C.RESET} {C.RED}{'─' * (width - 6)}{C.RESET} {C.RED}│{C.RESET}")
    for line in detail.split("\n"):
        wrapped = textwrap.wrap(line, width - 6)
        for w in wrapped or [""]:
            print(
                f"{C.RED}│{C.RESET} {C.GRAY}{w}{C.RESET}{' ' * (width - len(w) - 5)}{C.RED}│{C.RESET}"
            )
    if hint:
        print(f"{C.RED}│{C.RESET} {' ' * (width - 5)}{C.RED}│{C.RESET}")
        for line in hint.split("\n"):
            wrapped = textwrap.wrap(line, width - 6)
            for w in wrapped or [""]:
                print(
                    f"{C.RED}│{C.RESET} {C.YELLOW}{w}{C.RESET}{' ' * (width - len(w) - 5)}{C.RED}│{C.RESET}"
                )
    print(f"{C.RED}╰─{'─' * (width - 4)}─╯{C.RESET}\n")


def _compute_max_tokens(cfg, messages=None):
    ctx = cfg.get("_context_length")
    if not ctx:
        return 16384
    if messages:
        input_chars = sum(_content_str_len(m.get("content", "")) for m in messages)
        estimated_input = input_chars // 4 + 2048
        available = max(1024, ctx - estimated_input)
        return min(16384, available)
    return min(16384, ctx // 8)


def _compute_max_history_chars(cfg):
    ctx = cfg.get("_context_length")
    if ctx:
        return int(ctx * 0.5 * 4)
    return cfg.get("max_history_chars", 200000)


def _provider_name(cfg):
    provider = cfg.get("provider", "openrouter")
    if provider == "custom":
        return "Custom API"
    info = PROVIDERS.get(provider)
    return info["name"] if info else provider


def chat_completion(messages, cfg, stream=True, tools=None):
    api_key = cfg.get("api_key")
    if not api_key:
        provider_name = _provider_name(cfg)
        return {
            "error": "no_api_key",
            "title": "API key not configured",
            "detail": f"No API key found for {provider_name}. Run `/provider` to configure it.",
            "hint": f"Set the API key in the config or use the provider setup command.",
        }

    base_url = _provider_url(cfg)
    if not base_url:
        return {
            "error": "no_url",
            "title": "Provider URL not configured",
            "detail": "No base URL configured for this provider.",
            "hint": "Run `/provider custom` to set up the URL and API key.",
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/user/tbot",
        "X-Title": "tbot",
    }
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"],
        "max_tokens": _compute_max_tokens(cfg, messages),
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools

    provider_name = _provider_name(cfg)
    try:
        resp = requests.post(
            base_url + "/chat/completions",
            headers=headers,
            json=payload,
            stream=stream,
            timeout=10,
        )
    except requests.exceptions.Timeout:
        return {
            "error": "timeout",
            "title": "Connection timed out",
            "detail": f"The request to {provider_name} timed out after 10s.",
            "hint": "Check your internet connection. The client will retry automatically.",
        }
    except requests.exceptions.SSLError as e:
        return {
            "error": "ssl",
            "title": "SSL certificate error",
            "detail": f"Could not verify the SSL certificate: {e}",
            "hint": "Check your system date/time. If on Termux, try: pkg install ca-certificates",
        }
    except requests.exceptions.ConnectionError:
        return {
            "error": "connection",
            "title": f"Could not connect to {provider_name}",
            "detail": "No route to host. Your device may be offline or the service is blocked.",
            "hint": "Check your internet connection with: ping openrouter.ai\nIf on Termux, try: pkg install openssl && pkg reinstall python",
        }
    except requests.exceptions.ProxyError:
        return {
            "error": "proxy",
            "title": "Proxy connection failed",
            "detail": "Could not connect through the configured proxy.",
            "hint": "Check your proxy settings or disable the proxy and try again.",
        }
    except Exception as e:
        return {
            "error": "unknown",
            "title": "Unexpected error",
            "detail": str(e),
            "hint": "This is an unexpected error. Check your setup and try again.",
        }

    if resp.status_code != 200:
        try:
            data = resp.json()
            err = data.get("error", {})
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        except Exception:
            msg = resp.text[:300]
        if cfg.get("provider") == "custom":
            hint = "Check your API key, custom URL, and model name. Ensure the endpoint supports OpenAI-compatible /chat/completions."
        else:
            hint = f"Check your API key, model name, and {provider_name} status."
        return {
            "error": f"http_{resp.status_code}",
            "title": f"HTTP {resp.status_code}",
            "detail": msg,
            "hint": hint,
        }

    return {"stream": resp}


# ── Stream parsing ──────────────────────────────────────────────


def _term_size():
    try:
        return os.get_terminal_size()
    except (ValueError, OSError):
        return None


@functools.lru_cache(maxsize=1)
def _supports_osc8():
    """Detect if the terminal supports OSC 8 hyperlinks via env vars."""
    term_prog = os.environ.get("TERM_PROGRAM", "")
    term = os.environ.get("TERM", "")
    # Known OSC-8-capable terminals
    if term_prog in ("iTerm.app", "WezTerm", "vscode", "Ghostty", "tmux"):
        return True
    if term in ("xterm-kitty", "alacritty", "wezterm", "ghostty"):
        return True
    if os.environ.get("KITTY_WINDOW_ID"):  # Kitty
        return True
    if os.environ.get("WT_SESSION"):  # Windows Terminal
        return True
    if os.environ.get("VTE_VERSION"):  # GNOME Terminal, Konsole, etc.
        return True
    if os.environ.get("TERMINAL_EMULATOR") == "JetBrains-JediTerm":
        return True
    if os.environ.get("ALACRITTY_LOG"):
        return True
    # macOS Terminal.app and unknown terminals → assume no OSC 8
    return False


def _strip_ansi(text):
    """Strip all invisible ANSI escape sequences from text.

    Handles:
    - SGR codes: ``ESC[...m``  (color, bold, italic, etc.)
    - OSC 8 hyperlinks: ``ESC]8;;...ESC\\``
    - Any other OSC sequence: ``ESC]...ESC\\``
    - Other CSI sequences: ``ESC[...``
    """
    ESC = '\033'  # actual ESC byte (0x1b)
    # Strip SGR codes: ESC[<params>m
    s = re.sub(ESC + r'\[[0-9;]*m', '', text)
    # Strip OSC sequences: ESC]<string>ESC\ (e.g. OSC 8 hyperlinks)
    s = re.sub(ESC + r'\].*?' + ESC + r'\\', '', s)
    # Strip any remaining CSI sequences (cursor movement, etc.)
    s = re.sub(ESC + r'\[[0-9;]*[A-Za-z]', '', s)
    # Strip lone ESC characters (just in case)
    s = re.sub(ESC, '', s)
    return s


def _visible_width(s):
    """Return the visible width of string ``s`` in a monospace terminal.

    Follows the same general rules as the standard ``wcwidth`` library, with
    additional heuristics for emoji that are rendered as double-width even
    when their East-Asian Width property is Neutral or Ambiguous.

    Rules applied (in order of precedence):
      1. ANSI escape sequences are stripped first (width 0).
      2. C0 controls (U+0000–U+001F) → 0
         (Tab U+0009 is counted as 0; the terminal handles cursor movement.)
      3. DEL (U+007F) and C1 controls (U+0080–U+009F) → 0
      4. Surrogates (U+D800–U+DFFF) → 0 (*should* never appear in valid text)
      5. Combining marks (Mn, Me) → 0
         (Includes all variation selectors: VS1–VS16, VS17–VS256.)
      6. Format characters (Cf) → 0
         (ZWJ, ZWNJ, LRM, RLM, soft hyphen, interlinear annotation, tags, etc.)
      7. Line/paragraph separators (Zl, Zp) → 0
      8. Hangul Jamo fillers U+115F (initial) and U+1160 (medial) → 0
      9. East-Asian Wide (W) or Fullwidth (F) → 2
     10. Neutral (N) or Ambiguous (A) characters *followed by* U+FE0F
         (Variation Selector-16, which requests emoji presentation) → 2
     11. Space separators (Zs) → 1 (even NBSP and other fixed spaces)
     12. Everything else → 1
    """
    import unicodedata
    s = _strip_ansi(s)          # invisible ANSI sequences → 0 width
    width = 0
    i = 0
    while i < len(s):
        cp = ord(s[i])
        cat = unicodedata.category(s[i])

        # ── always-zero categories ──────────────────────────────
        if cat in ('Mn', 'Me', 'Cf'):            # combining marks, format chars
            i += 1
            continue
        if cat in ('Cc', 'Cs', 'Zl', 'Zp'):       # controls, surrogates, separators
            i += 1
            continue

        # ── Hangul Jamo fillers (invisible in modern usage) ─────
        if cp in (0x115F, 0x1160):
            i += 1
            continue

        # ── East-Asian Width check ──────────────────────────────
        eaw = unicodedata.east_asian_width(s[i])
        if eaw in ('W', 'F'):
            width += 2
            i += 1
            continue

        # ── Emoji with explicit VS16 (U+FE0F) ──────────────────
        # Many characters with EAW=N or EAW=A are rendered as
        # double-width emoji when followed by the variation selector.
        if eaw in ('N', 'A') and i + 1 < len(s) and s[i + 1] == '\ufe0f':
            width += 2
            # Also consume the VS16 (it is Mn and would be skipped anyway)
            i += 2
            continue

        # ── Space separators (Zs) ──────────────────────────────
        # NBSP, en-space, em-space, etc. – all take 1 column.
        if cat == 'Zs':
            width += 1
            i += 1
            continue

        # ── Default: 1 column ───────────────────────────────────
        width += 1
        i += 1

    return width


def _render_inline_fmt(text, base_ansi, reset):
    """Apply ANSI formatting for inline markdown elements (bold, italic, code, links).

    NOTE: The ``osc8=False`` link path does NOT emit OSC-8 hyperlink escape sequences,
    because some terminals/ pipelines may strip the ESC byte and leave raw ``]8;;…``
    visible in table cells.  Without OSC-8 the link text is simply underlined + the URL
    is shown in gray.
    """
    # Strip any pre-existing ANSI escapes (model might inject raw OSC-8, etc.)
    text = _strip_ansi(text)

    osc8 = _supports_osc8()
    # Inline code: `code`
    text = re.sub(r'`([^`]+)`', rf'\033[33m\1{reset}{base_ansi}', text)
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', rf'\033[1m\1{reset}{base_ansi}', text)
    text = re.sub(r'__(.+?)__', rf'\033[1m\1{reset}{base_ansi}', text)
    # Italic: *text* (single asterisk)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', rf'\033[3m\1{reset}{base_ansi}', text)
    # Italic: _text_ (word boundaries)
    text = re.sub(r'(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)', rf'\033[3m\1{reset}{base_ansi}', text)
    # Superscript: ^text^ and HTML <sup>text</sup>
    text = re.sub(r'\^(.+?)\^', rf'\033[33m\1{reset}{base_ansi}', text)
    text = re.sub(r'<sup>(.+?)</sup>', rf'\033[33m\1{reset}{base_ansi}', text, flags=re.IGNORECASE)
    # Subscript: ~text~ and HTML <sub>text</sub> (avoid ~~strikethrough~~)
    text = re.sub(r'(?<!~)~(?!~)(.+?)(?<!~)~(?!~)', rf'\033[34m\1{reset}{base_ansi}', text)
    text = re.sub(r'<sub>(.+?)</sub>', rf'\033[34m\1{reset}{base_ansi}', text, flags=re.IGNORECASE)
    # Links: [text](url)
    if osc8:
        # OSC-8 hyperlink – terminal will render the link clickable
        text = re.sub(
            r'\[([^\]]+)\]\(([^)]+)\)',
            rf'\033]8;;\2\033\\\033[4;34m\1{reset}{base_ansi}\033]8;;\033\\',
            text,
        )
    else:
        # No OSC-8: just show underlined text + gray URL (no escape sequences
        # that could become visible garbage on terminals that strip ESC bytes)
        text = re.sub(
            r'\[([^\]]+)\]\(([^)]+)\)',
            rf'\033[4;34m\1{reset}{base_ansi} \033[90m\2{reset}{base_ansi}',
            text,
        )
    # Strikethrough: ~~text~~
    text = re.sub(r'~~(.+?)~~', rf'\033[9m\1{reset}{base_ansi}', text)
    return text


def _parse_table_row(line):
    """Parse a markdown table row into cells (list of stripped strings)."""
    s = line.strip()
    # Strip leading/trailing pipes
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    return [cell.strip() for cell in s.split('|')]


def _is_table_separator(cells):
    """Return True if all cells look like column separators (---, :---, :--:, ---:)."""
    return all(re.match(r'^:?-{3,}:?$', c) for c in cells if c.strip())


def _render_table(rows, resp_color):
    """Render accumulated markdown table rows with ANSI box-drawing grid.

    Returns a list of strings to print.
    """
    base_ansi = f"\033[{resp_color}m"
    reset = C.RESET
    gray = '\033[90m'

    if not rows:
        return []

    # Parse all rows, find separator position in parsed list
    parsed = []
    sep_pos = None  # position in `parsed` list
    for row in rows:
        cells = _parse_table_row(row)
        if not cells:
            continue
        if _is_table_separator(cells):
            if sep_pos is None:
                sep_pos = len(parsed)
            parsed.append(('sep', cells))
        else:
            # Apply inline formatting to each cell's content
            fmt_cells = [_render_inline_fmt(c, base_ansi, reset) for c in cells]
            parsed.append(('data', fmt_cells))

    if not parsed:
        return [f"{base_ansi}{rows[0]}{reset}"]

    # Determine column count (max across all rows)
    ncols = max(len(cells) for _, cells in parsed)

    # Calculate column widths (strip all invisible ANSI sequences)
    widths = [0] * ncols
    for kind, cells in parsed:
        for i, cell in enumerate(cells):
            clean = _strip_ansi(cell)
            widths[i] = max(widths[i], _visible_width(clean))

    # Pad all cells to uniform column count
    padded = []
    for kind, cells in parsed:
        while len(cells) < ncols:
            cells.append('')
        padded.append((kind, cells))

    # ── Build grid lines ────────────────────────────────────
    H = '─'      # horizontal
    V = gray + '│' + base_ansi
    LT = gray + '┌' + base_ansi
    RT = gray + '┐' + base_ansi
    LB = gray + '└' + base_ansi
    RB = gray + '┘' + base_ansi
    LJ = gray + '├' + base_ansi
    RJ = gray + '┤' + base_ansi
    CR = gray + '┼' + base_ansi

    def _sep_line(left, join, right, fill=H):
        return (left
                + join.join(fill * (w + 2) for w in widths)
                + right)

    def _fmt_cell(text, width, is_header=False):
        style = '\033[1m' if is_header else ''
        text_clean = _strip_ansi(text)
        visible_len = _visible_width(text_clean)
        pad = width - visible_len
        return f" {style}{text}{reset}{' ' * pad} "

    result = []

    # Top border
    result.append(f"{base_ansi}{_sep_line(LT, '┬' if ncols > 1 else '', RT)}{reset}")

    for idx, (kind, cells) in enumerate(padded):
        if kind == 'sep':
            result.append(f"{base_ansi}{_sep_line(LJ, '┼' if ncols > 1 else '', RJ, '─')}{reset}")
        else:
            # Header: first row if no separator, or any data row before the separator
            if sep_pos is None:
                is_header = (idx == 0)
            else:
                is_header = idx < sep_pos
            line = V.join(
                _fmt_cell(cells[i] if i < len(cells) else '', widths[i], is_header=is_header)
                for i in range(ncols)
            )
            result.append(f"{base_ansi}{V}{line}{V}")

    # Bottom border
    result.append(f"{base_ansi}{_sep_line(LB, '┴' if ncols > 1 else '', RB)}{reset}")

    return result


def _render_md_line(line, resp_color, in_code_block):
    """Render a single markdown line with ANSI formatting.

    Returns (rendered_string, updated_in_code_block).
    """
    base_ansi = f"\033[{resp_color}m"
    reset = C.RESET

    # Code block fence detection (```)
    stripped = line.strip()
    if stripped.startswith("```"):
        return f"\033[48;5;236m\033[90m{line}{reset}", not in_code_block

    # Inside code block — cyan on dark background
    if in_code_block:
        return f"\033[48;5;236m\033[36m{line}{reset}", True

    # Empty line
    if not line.strip():
        return "", False

    # ── Block-level elements ────────────────────────────────────

    # Headers (# ## ### etc)
    m = re.match(r'^(#{1,6})\s+(.+)$', line)
    if m:
        level = len(m.group(1))
        hdr_colors = {1: '1;36', 2: '1;34', 3: '1;33', 4: '1;35', 5: '1;32', 6: '1;90'}
        text = _render_inline_fmt(m.group(2), base_ansi, reset)
        return f"\033[{hdr_colors.get(level, '1;36')}m{m.group(1)} {text}{reset}", False

    # Horizontal rules
    if re.match(r'^[-*_]{3,}\s*$', line):
        try:
            width = os.get_terminal_size().columns
        except Exception:
            width = 72
        return f"\033[90m{'─' * min(len(line), width)}{reset}", False

    # Blockquotes
    if line.startswith('>'):
        content = _render_inline_fmt(line[1:].strip(), base_ansi, reset)
        return f"\033[90m│{reset} {content}", False

    # Task list items: - [ ] or - [x]
    task_match = re.match(r'^(\s*)[-*+]\s+\[([ xX])\]\s+(.*)$', line)
    if task_match:
        indent = task_match.group(1)
        checked = task_match.group(2).lower() == 'x'
        text = _render_inline_fmt(task_match.group(3), base_ansi, reset)
        check_char = '✓' if checked else '○'
        check_color = '\033[32m' if checked else '\033[90m'
        return f"{base_ansi}{indent}- {check_color}[{check_char}]{reset} {text}", False

    # Unordered list items: -, *, +
    list_match = re.match(r'^(\s*)([-*+])\s+(.*)$', line)
    if list_match:
        text = _render_inline_fmt(list_match.group(3), base_ansi, reset)
        return f"{base_ansi}{list_match.group(1)}{list_match.group(2)} {text}{reset}", False

    # Ordered list items: 1., 2. etc
    list_match = re.match(r'^(\s*)(\d+\.)\s+(.*)$', line)
    if list_match:
        text = _render_inline_fmt(list_match.group(3), base_ansi, reset)
        return f"{base_ansi}{list_match.group(1)}{list_match.group(2)} {text}{reset}", False

    # Markdown table row: at least 2 pipe characters
    if line.strip().count('|') >= 2:
        return None, False

    # Regular line with inline formatting
    rendered = _render_inline_fmt(line, base_ansi, reset)
    return f"{base_ansi}{rendered}{reset}", False


def parse_stream(resp, resp_color="95"):
    content_parts = []
    reasoning_parts = []
    tool_calls = {}
    token_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cost = 0
    interrupted = False
    _color_printed = False
    line_buf = ""
    in_code_block = False
    table_buf = []

    sock = None
    try:
        conn = getattr(resp.raw, "connection", None)
        sock = getattr(conn, "sock", None) if conn else None
    except Exception:
        pass

    try:
        iterator = resp.iter_lines()
    except Exception:
        return None, None, None, 0, 0, 0, 0, True

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
                usage = data.get("usage")
                if usage:
                    pt = usage.get("prompt_tokens", 0)
                    ct = usage.get("completion_tokens", 0)
                    tt = usage.get("total_tokens", 0)
                    c = usage.get("cost", 0)
                    if pt:
                        prompt_tokens = pt
                    if ct:
                        completion_tokens = ct
                    if tt:
                        total_tokens = tt
                    if c:
                        cost = c
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                # Capture reasoning_content for DeepSeek thinking mode
                rc = delta.get("reasoning_content")
                if rc:
                    reasoning_parts.append(rc)
                c = delta.get("content")
                if c:
                    content_parts.append(c)
                    token_count += 1
                    if not _color_printed:
                        _color_printed = True
                    line_buf += c
                    # Process complete lines from buffer
                    while '\n' in line_buf:
                        idx = line_buf.index('\n')
                        line = line_buf[:idx]
                        line_buf = line_buf[idx + 1:]
                        rendered, in_code_block = _render_md_line(
                            line, resp_color, in_code_block
                        )
                        if rendered is None:
                            # Table row — accumulate
                            table_buf.append(line)
                        else:
                            # Not a table row — flush any pending table first
                            if table_buf:
                                for t_line in _render_table(table_buf, resp_color):
                                    print(t_line, flush=True)
                                table_buf = []
                            if rendered:
                                print(rendered, flush=True)
                            else:
                                print(flush=True)
                    if _log_fh is not None:
                        try:
                            _log_fh.write(c)
                            _log_fh.flush()
                        except Exception:
                            pass
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
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
        interrupted = True

    if _color_printed and not interrupted:
        # Flush remaining partial line in buffer
        if line_buf:
            rendered, in_code_block = _render_md_line(
                line_buf, resp_color, in_code_block
            )
            if rendered is None:
                table_buf.append(line_buf)
            else:
                if table_buf:
                    for t_line in _render_table(table_buf, resp_color):
                        print(t_line, flush=True)
                    table_buf = []
                if rendered:
                    print(rendered, flush=True)
                else:
                    print(flush=True)
        # Flush any table that ends exactly at the last line
        if table_buf:
            for t_line in _render_table(table_buf, resp_color):
                print(t_line, flush=True)
            table_buf = []
        print(C.RESET, end="", flush=True)

    if content_parts and _log_fh is not None and not interrupted:
        try:
            _log_fh.write("\n")
            _log_fh.flush()
        except Exception:
            pass

    if not completion_tokens:
        completion_tokens = token_count
    content = "".join(content_parts)
    reasoning_content = "".join(reasoning_parts) if reasoning_parts else None
    calls = list(tool_calls.values()) if tool_calls else None
    if calls:
        for c in calls:
            _log_write(
                f"tool_call: {c['function']['name']}({c['function']['arguments'][:200]})"
            )
    return (
        content,
        reasoning_content,
        calls,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cost,
        interrupted,
    )


# ── Tool execution ──────────────────────────────────────────────


MAX_TOOLS_PER_ROUND = 10


def execute_tool_calls(tool_calls, messages, cfg):
    global _todo_has_write_since_last
    pending_system = []
    pending_user = []  # user messages (images) to append after ALL tool responses
    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            args = (
                json.loads(tc["function"]["arguments"])
                if tc["function"]["arguments"]
                else {}
            )
        except json.JSONDecodeError:
            args = {}

        args_str = json.dumps(args)[:200]
        desc = args.get("description", "")
        print(f"\n{C.GRAY}── {C.CYAN}{name}{C.RESET} {C.GRAY}{desc or args_str}{C.RESET}")
        _log_write(f"── {name} {args_str}", role="tool", tool_name=name)

        handler = TOOL_HANDLERS.get(name)
        if not handler and name.startswith("skill_"):
            handler = lambda a, _n=name[6:], _msgs=messages: skill_tool_handler(
                _n, a, _msgs
            )
        if not handler and name.startswith("mcp__"):
            handler = lambda a, _qn=name: handle_mcp_tool_call(_qn, a)
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
                before = len(messages)
                try:
                    result = handler(args)
                except KeyboardInterrupt:
                    print(f"\n  {C.YELLOW}cancelled{C.RESET}")
                    return False
                after = len(messages)
                for _ in range(after - before):
                    pending_system.append(messages.pop())
                if len(result) > MAX_TOOL_OUTPUT:
                    result = (
                        result[:MAX_TOOL_OUTPUT]
                        + f"\n... (truncated, {len(result)} total chars)"
                    )
            else:
                result = "TOOL_CALL_DECLINED"

        if name == "edit":
            _emit_edit_diff()
        if name == "todowrite" and result != "TOOL_CALL_DECLINED":
            _display_task_md()
        _log_write(f"→ {result[:1000]}{'...' if len(result) > 1000 else ''}", role="tool", tool_name=name)
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        if name == "read" and result.startswith("data:image/"):
            fpath = args.get("filePath", "?")
            pending_user.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"[read({fpath!r}) returned this image:]",
                        },
                        {"type": "image_url", "image_url": {"url": result}},
                    ],
                }
            )
        if name not in _READ_ONLY_TOOLS and ok:
            _todo_has_write_since_last = True
    # Append user messages (images) AFTER all tool responses to keep tool_calls
    # contiguous — required by providers like DeepSeek
    for msg in pending_user:
        messages.append(msg)
    for msg in pending_system:
        messages.append(msg)
    return True


# ── UI ──────────────────────────────────────────────────────────


def _run_interactive(cmd, cwd, timeout=120):
    """Run a shell command, streaming stdout/stderr in real time, and
    return (captured_output, exit_code).  The output is printed as it
    arrives, then the captured text is sent to the AI."""
    import threading

    out_lines = []
    err_lines = []
    lock = threading.Lock()

    def _reader(stream, lines, color):
        try:
            for line in iter(stream.readline, ""):
                with lock:
                    lines.append(line)
                if color:
                    sys.stdout.write(color + line + C.RESET)
                else:
                    sys.stdout.write(line)
                sys.stdout.flush()
            stream.close()
        except ValueError:
            pass

    try:
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )
    except Exception as e:
        return f"Error: {e}", -1

    tout = threading.Thread(target=_reader, args=(p.stdout, out_lines, None))
    terr = threading.Thread(target=_reader, args=(p.stderr, err_lines, C.RED))
    tout.daemon = True
    terr.daemon = True
    tout.start()
    terr.start()

    timed_out = False
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        p.kill()
        p.wait()

    tout.join()
    terr.join()

    out_b = "".join(out_lines)
    err_b = "".join(err_lines)
    captured = out_b
    if err_b:
        captured += "\n--- stderr ---\n" + err_b

    if timed_out:
        captured += "\n--- exit code: -1 (timed out) ---"
        return captured, -1

    captured += f"\n--- exit code: {p.returncode} ---"
    status_color = C.GREEN if p.returncode == 0 else C.RED
    print(f"  {status_color}exit {p.returncode}{C.RESET}")
    return captured, p.returncode


def _completer(text, state):
    """Tab completion for slash commands and !bash shortcuts."""
    line = readline.get_line_buffer()
    raw = line.lstrip()
    parts = raw.split()
    trailing_space = raw.endswith(" ")

    # ── ! prefix: bash completion via compgen ──
    if raw.startswith("!"):
        try:
            cmd_token = parts[0] if parts else raw
            past_cmd = bool(raw[len(cmd_token) :].strip()) or raw[
                len(cmd_token) :
            ].endswith(" ")
            if not past_cmd:
                prefix = text[1:] if text.startswith("!") else text
                r = subprocess.run(
                    ["bash", "-c", f"compgen -c '{prefix}'"],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                matches = ["!" + c + " " for c in r.stdout.strip().split("\n") if c]
            else:
                prefix = parts[-1] if len(parts) > 1 else ""
                r = subprocess.run(
                    ["bash", "-c", f"compgen -f -- '{prefix}'"],
                    capture_output=True,
                    text=True,
                    timeout=1,
                    cwd=str(CURRENT_DIR),
                )
                matches = []
                for c in r.stdout.strip().split("\n"):
                    c = c.strip()
                    if not c:
                        continue
                    if (CURRENT_DIR / c).is_dir():
                        matches.append(c + "/")
                    else:
                        matches.append(c + " ")
        except Exception:
            matches = []
        return matches[state] if state < len(matches) else None

    # ── Not a slash command → nothing to complete ──
    if not raw.startswith("/"):
        return None

    # ── /command first word ──
    if text.startswith("/"):
        cmd = text[1:]
        if cmd in _COMMANDS:
            matches = [text + " "]
        else:
            matches = ["/" + c + " " for c in _COMMANDS if c.startswith(cmd)]
        return matches[state] if state < len(matches) else None

    # ── Completing arguments of a slash command ──
    if not parts:
        return None
    cmd = parts[0][1:]
    # If a space follows the last token, treat the next argument as empty
    args = parts[1:] if not trailing_space else parts[1:] + [""]
    prefix = text  # "" or partial word

    # ── /preset <subcmd> [name] ──
    if cmd == "preset":
        if not args:
            return None
        if len(args) == 1:
            matches = [s + " " for s in _PRESET_SUBCMDS if s.startswith(prefix)]
        elif args[0] in ("load", "rm", "show", "save"):
            if len(args) > 2 and not trailing_space:
                return None
            presets = list_presets()
            matches = sorted(p["name"] + " " for p in presets if p["name"].startswith(prefix))
        else:
            return None
        return matches[state] if state < len(matches) else None

    # ── /skill <subcmd> [name] ──
    if cmd == "skill":
        if not args:
            return None
        if len(args) == 1:
            matches = [s + " " for s in _SKILL_SUBCMDS if s.startswith(prefix)]
        elif args[0] in ("rm", "show"):
            if len(args) > 2 and not trailing_space:
                return None
            skills = load_skills()
            matches = sorted(n + " " for n, *_ in skills if n.startswith(prefix))
        else:
            return None
        return matches[state] if state < len(matches) else None

    # ── /rag <subcmd> ──
    if cmd == "rag":
        if not args:
            return None
        if len(args) == 1:
            matches = [s + " " for s in _RAG_SUBCMDS if s.startswith(prefix)]
        else:
            return None
        return matches[state] if state < len(matches) else None

    # ── /memory <subcmd> ──
    if cmd == "memory":
        if not args:
            return None
        if len(args) == 1:
            matches = [s + " " for s in _MEMORY_SUBCMDS if s.startswith(prefix)]
        elif args[0] in ("show", "edit") and len(args) == 2:
            matches = [s + " " for s in ("memory", "user") if s.startswith(prefix)]
        elif args[0] == "gate" and len(args) == 2:
            matches = [s + " " for s in ("on", "off") if s.startswith(prefix)]
        elif args[0] == "search" and len(args) == 2:
            # Suggest store names as first arg after "search"
            matches = [s + " " for s in ("memory", "user") if s.startswith(prefix)]
        else:
            return None
        return matches[state] if state < len(matches) else None

    # ── /mcp <subcmd> [name] ──
    if cmd == "mcp":
        if not args:
            return None
        if len(args) == 1:
            matches = [s + " " for s in _MCP_SUBCMDS if s.startswith(prefix)]
            return matches[state] if state < len(matches) else None
        if args[0] in ("connect", "disconnect", "reconnect", "discover") and len(args) > 1:
            if not _mcp_manager:
                return None
            names = [s["name"] + " " for s in _mcp_manager.get_status() if s["name"].startswith(prefix)]
            return names[state] if state < len(names) else None
        return None

    # ── /episodic <subcmd> ──
    if cmd == "episodic":
        if not args:
            return None
        if len(args) == 1:
            matches = [s + " " for s in _EPISODIC_SUBCMDS if s.startswith(prefix)]
        elif args[0] == "prune" and len(args) == 2:
            matches = ["30 ", "60 ", "90 ", "180 "]
            matches = [s for s in matches if s.startswith(prefix)]
        else:
            return None
        return matches[state] if state < len(matches) else None

    # ── /model [name] ──
    if cmd == "model" and args:
        q = prefix.lower()
        models = _models_cache or []
        matches = sorted(
            m.get("id", "") for m in models if q in m.get("id", "").lower()
        )
        if matches:
            matches = [m + " " for m in matches]
            return matches[state] if state < len(matches) else None
        return None

    # ── /provider [name] ──
    if cmd == "provider" and args:
        matches = [p + " " for p in PROVIDERS if p.startswith(prefix)]
        return matches[state] if state < len(matches) else None

    # ── /temp [value] ──
    if cmd == "temp" and args:
        suggestions = ["0", "0.1", "0.3", "0.5", "0.7", "0.8", "1.0", "1.5", "2.0"]
        matches = [s for s in suggestions if s.startswith(prefix)]
        return matches[state] if state < len(matches) else None

    # ── /session [filter] ──
    if cmd == "session" and args:
        sessions = sorted([f.name for f in LOG_DIR.glob("*.log") if f.is_file()])
        matches = [s + " " for s in sessions if s.startswith(prefix)]
        return matches[state] if state < len(matches) else None

    # ── /export <filepath> ──
    if cmd == "export" and args:
        try:
            r = subprocess.run(
                ["bash", "-c", f"compgen -f -- '{prefix}'"],
                capture_output=True,
                text=True,
                timeout=1,
                cwd=str(CURRENT_DIR),
            )
            matches = []
            for c in r.stdout.strip().split("\n"):
                c = c.strip()
                if not c:
                    continue
                if (CURRENT_DIR / c).is_dir():
                    matches.append(c + "/")
                else:
                    matches.append(c + " ")
            return matches[state] if state < len(matches) else None
        except Exception:
            return None

    return None


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
        # Remove / from delimiters so readline completes it as part of the word
        delims = readline.get_completer_delims()
        delims = delims.replace("/", "")
        readline.set_completer_delims(delims)
        # Register completer before binding tab
        readline.set_completer(_completer)
        # Bind Tab — macOS libedit needs different syntax than GNU readline
        is_libedit = "libedit" in (readline.__doc__ or "")
        if is_libedit:
            # libedit on macOS — try syntaxes known to work
            for bind_cmd in (
                'bind ^I rl_complete',
                'tab: complete',
            ):
                try:
                    readline.parse_and_bind(bind_cmd)
                except Exception:
                    continue
        else:
            # GNU readline
            readline.parse_and_bind("tab: complete")
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


def _setup_sigwinch():
    """Handle terminal resize (SIGWINCH) so readline redraws correctly."""
    if readline is None:
        return

    def _on_resize(sig, action):
        try:
            readline.redisplay()
        except Exception:
            pass

    signal.signal(signal.SIGWINCH, _on_resize)


def show_banner(cfg):
    tools_flag = (
        f"{C.CYAN}tools{C.RESET}"
        if cfg["tools_enabled"]
        else f"{C.GRAY}tools off{C.RESET}"
    )
    trust_flag = (
        f"{C.GREEN}trust{C.RESET}"
        if cfg["trust_mode"]
        else f"{C.YELLOW}confirm{C.RESET}"
    )
    ctx = cfg.get("_context_length")
    ctx_str = (
        f"  {C.GRAY}ctx:{C.RESET} {C.MAGENTA}{_fmt_context(ctx)}{C.RESET}"
        if ctx
        else ""
    )
    provider_name = PROVIDERS.get(cfg.get("provider", "openrouter"), {}).get(
        "name", "OpenRouter"
    )
    print(f"\n{C.BOLD}{C.CYAN}  tbot{C.RESET}  {C.GRAY}— {provider_name} CLI{C.RESET}")
    print(f"  {C.GRAY}model:{C.RESET} {C.YELLOW}{cfg['model']}{C.RESET}{ctx_str}")
    print(
        f"  {C.GRAY}prov:{C.RESET}  {C.YELLOW}{cfg.get('provider', 'openrouter')}{C.RESET}"
    )
    print(
        f"  {C.GRAY}temp:{C.RESET}  {C.YELLOW}{cfg['temperature']}{C.RESET}  "
        f"{tools_flag}  {trust_flag}"
    )
    print(
        f"  {C.GRAY}type{C.RESET} {C.BOLD}/help{C.RESET} {C.GRAY}for commands{C.RESET}\n"
    )


def _fmt_context(ctx):
    if ctx >= 1_000_000:
        return f"{ctx // 1_000_000}M"
    if ctx >= 1_000:
        return f"{ctx // 1_000}K"
    return str(ctx)


def _term_width():
    try:
        return os.get_terminal_size().columns
    except (ValueError, OSError):
        return 70


def _render_selector(models, filtered, query, idx, current_id):
    w = _term_width()
    sep = f"{C.GRAY}{'─' * (w - 2)}{C.RESET}"
    buf = [
        f"{C.BOLD}{C.CYAN}model{C.RESET}  {C.YELLOW}{current_id}{C.RESET}  {C.GRAY}({len(filtered)}/{len(models)}){C.RESET}\r\n"
    ]
    buf.append(f"{C.CYAN}filter:{C.RESET} {query if query else ''}\r\n")
    buf.append(f"{sep}\r\n")
    if not filtered:
        buf.append(f"{C.GRAY}no matches{C.RESET}\r\n")
    else:
        start = max(0, idx - 10)
        end = min(len(filtered), start + 20)
        for i in range(start, end):
            m = filtered[i]
            pre = f"{C.CYAN}▸{C.RESET} " if i == idx else "  "
            mid = m.get("id", "?")
            ctx = m.get("context_length", 0)
            cs = f" {C.MAGENTA}{_fmt_context(ctx)}{C.RESET}" if ctx else ""
            try:
                pp = float(m.get("pricing", {}).get("prompt", 0))
                cp = float(m.get("pricing", {}).get("completion", 0))
                cost = (pp + cp) / 0.000001
                ps = f" {C.GRAY}${cost:.2f}{C.RESET}" if cost > 0 else ""
            except (ValueError, TypeError, ZeroDivisionError):
                ps = ""
            bs = ""
            aa = m.get("benchmarks", {}).get("artificial_analysis")
            if aa:
                labels = {
                    "intelligence_index": "int",
                    "coding_index": "code",
                    "agentic_index": "agent",
                }
                parts = []
                for k, lbl in labels.items():
                    v = aa.get(k)
                    if v is not None:
                        parts.append(f"{lbl}:{v}")
                if parts:
                    bs = f" {C.YELLOW}{' '.join(parts)}{C.RESET}"
            buf.append(f"{pre}{mid}{cs}{ps}{bs}\r\n")
    buf.append(f"\r\n{C.GRAY}Ctrl+N/P nav  type filter  ↵ select  Ctrl+C exit{C.RESET}")
    return "".join(buf)


def _read_esc(fd):
    try:
        import select
    except ImportError:
        return ""

    seq = ""
    for _ in range(8):
        try:
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not r:
                break
        except (ValueError, OSError):
            break
        b = sys.stdin.read(1)
        if not b:
            break
        seq += b
    return seq


def model_selector(current_id, api_key, provider="openrouter", custom_url=""):
    try:
        import termios, tty
    except ImportError:
        print(f"\r{C.RED}error: model selector requires a Unix terminal (termios not available){C.RESET}")
        return None

    all_models = fetch_models(api_key, provider=provider, custom_url=custom_url)
    _TBOT_PARAMS = {"temperature", "max_tokens", "tools"}
    has_params = any("supported_parameters" in m for m in all_models)
    if has_params:
        models = [
            m
            for m in all_models
            if _TBOT_PARAMS.issubset(m.get("supported_parameters", []))
        ]
    else:
        models = list(all_models)
    if not models:
        return None
    filtered = list(models)
    query = ""
    idx = 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        termios.tcflush(fd, termios.TCIFLUSH)
    except (OSError, termios.error):
        pass
    try:
        tty.setraw(fd)
        sys.stdout.write("\033[?25l")
        while True:
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(_render_selector(models, filtered, query, idx, current_id))
            sys.stdout.flush()
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                _read_esc(fd)
            elif ch in ("\r", "\n"):
                return filtered[idx]["id"] if filtered else None
            elif ch == "\x03":
                return None
            elif ch == "\x0e":  # Ctrl+N — down
                idx = min(len(filtered) - 1, idx + 1)
            elif ch == "\x10":  # Ctrl+P — up
                idx = max(0, idx - 1)
            elif ch in ("\x7f", "\b"):
                query = query[:-1]
                idx = 0
            elif ch.isprintable():
                query += ch
                idx = 0
            q = query.lower()
            if query:
                filtered = [
                    m
                    for m in models
                    if q in m.get("id", "").lower() or q in m.get("name", "").lower()
                ]
            else:
                filtered = list(models)
    finally:
        sys.stdout.write("\033[?25h")
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()


def session_selector():
    try:
        import termios, tty
    except ImportError:
        print(f"\r{C.RED}error: session selector requires a Unix terminal (termios not available){C.RESET}")
        return None

    all_files = sorted(
        [f for f in LOG_DIR.glob("*.log") if f.is_file()],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not all_files:
        return None

    result = _run_rag(["index", str(LOG_DIR)], timeout=30)
    if isinstance(result, dict) and result.get("error"):
        return None

    query = ""
    results = [
        {"path": f.name, "full_path": str(f), "size": f.stat().st_size, "score": 0}
        for f in all_files
    ]
    idx = 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        termios.tcflush(fd, termios.TCIFLUSH)
    except (OSError, termios.error):
        pass
    try:
        tty.setraw(fd)
        sys.stdout.write("\033[?25l")
        while True:
            w = _term_width()
            sep = f"{C.GRAY}{'─' * (w - 2)}{C.RESET}"
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(
                f"{C.BOLD}{C.CYAN}sessions{C.RESET}  {C.GRAY}({len(results)}/{len(all_files)}){C.RESET}\r\n"
            )
            sys.stdout.write(f"{C.CYAN}filter:{C.RESET} {query if query else ''}\r\n")
            sys.stdout.write(f"{sep}\r\n")
            if not results:
                sys.stdout.write(f"{C.GRAY}no matches{C.RESET}\r\n")
            else:
                start = max(0, idx - 10)
                end = min(len(results), start + 20)
                for i in range(start, end):
                    r = results[i]
                    pre = f"{C.CYAN}▸{C.RESET} " if i == idx else "  "
                    name = r["path"]
                    sz = r["size"]
                    sz_str = (
                        f"{sz}B"
                        if sz < 1024
                        else f"{sz / 1024:.0f}K"
                        if sz < 1024 * 1024
                        else f"{sz / 1024 / 1024:.1f}M"
                    )
                    score_str = (
                        f"  score={C.YELLOW}{r['score']:.1f}{C.RESET}"
                        if r.get("score", 0) > 0
                        else ""
                    )
                    sys.stdout.write(
                        f"{pre}{name}  {C.GRAY}{sz_str}{C.RESET}{score_str}\r\n"
                    )
            sys.stdout.write(
                f"\r\n{C.GRAY}Ctrl+N/P nav  type filter  ↵ view  Ctrl+C exit{C.RESET}"
            )
            sys.stdout.flush()

            ch = sys.stdin.read(1)
            if ch == "\x1b":
                _read_esc(fd)
            elif ch in ("\r", "\n"):
                if results and idx < len(results):
                    return {
                        "path": results[idx]["path"],
                        "full_path": results[idx]["full_path"],
                    }
                return None
            elif ch == "\x03":
                return None
            elif ch == "\x0e":
                if results:
                    idx = min(len(results) - 1, idx + 1)
            elif ch == "\x10":
                if results:
                    idx = max(0, idx - 1)
            elif ch in ("\x7f", "\b"):
                query = query[:-1]
                idx = 0
            elif ch.isprintable():
                query += ch
                idx = 0

            q = query.strip()
            if q:
                rag_out = _run_rag(["search", q, "100"], timeout=10)
                file_scores = {}
                if isinstance(rag_out, list):
                    for r in rag_out:
                        p = r.get("path", "")
                        s = r.get("score", 0)
                        if p:
                            file_scores[p] = max(file_scores.get(p, 0), s)
                elif isinstance(rag_out, dict) and "raw" in rag_out:
                    try:
                        data = json.loads(rag_out["raw"])
                        if isinstance(data, list):
                            for r in data:
                                p = r.get("path", "")
                                s = r.get("score", 0)
                                if p:
                                    file_scores[p] = max(file_scores.get(p, 0), s)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if file_scores:
                    scored = []
                    for f in all_files:
                        s = file_scores.get(f.name, 0)
                        if s > 0:
                            scored.append(
                                {
                                    "path": f.name,
                                    "full_path": str(f),
                                    "size": f.stat().st_size,
                                    "score": s,
                                }
                            )
                    scored.sort(key=lambda x: -x["score"])
                    results = scored
                else:
                    results = []
            else:
                results = [
                    {
                        "path": f.name,
                        "full_path": str(f),
                        "size": f.stat().st_size,
                        "score": 0,
                    }
                    for f in all_files
                ]
            if results and idx >= len(results):
                idx = max(0, len(results) - 1)
    finally:
        sys.stdout.write("\033[?25h")
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()


def _render_provider_selector(providers, filtered, idx, current_id):
    w = _term_width()
    buf = [
        f"{C.BOLD}{C.CYAN}provider{C.RESET}  {C.YELLOW}{current_id}{C.RESET}  {C.GRAY}({len(filtered)}/{len(providers)}){C.RESET}\r\n"
    ]
    buf.append(f"{C.GRAY}{'─' * (w - 2)}{C.RESET}\r\n")
    if not filtered:
        buf.append(f"{C.GRAY}no providers{C.RESET}\r\n")
    else:
        start = max(0, idx - 5)
        end = min(len(filtered), start + 12)
        for i in range(start, end):
            k = filtered[i]
            info = providers[k]
            pre = f"{C.CYAN}▸{C.RESET} " if i == idx else "  "
            name = info["name"]
            env = info.get("env_key", "")
            if env:
                has_env = (
                    C.GREEN + "✓" + C.RESET
                    if os.environ.get(env)
                    else C.GRAY + "✗" + C.RESET
                )
                buf.append(
                    f"{pre}{k:<16} {name:<20} {has_env} {C.GRAY}{env}{C.RESET}\r\n"
                )
            else:
                buf.append(f"{pre}{k:<16} {name:<20}\r\n")
    buf.append(f"\r\n{C.GRAY}↑/↓ nav  ↵ select  Ctrl+C exit{C.RESET}")
    return "".join(buf)


def provider_selector(current_id):
    try:
        import termios, tty
    except ImportError:
        print(f"\r{C.RED}error: provider selector requires a Unix terminal (termios not available){C.RESET}")
        return None

    providers = {k: v for k, v in PROVIDERS.items()}
    sorted_keys = sorted(providers.keys())
    filtered = list(sorted_keys)
    idx = 0
    try:
        idx = filtered.index(current_id)
    except ValueError:
        pass
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        termios.tcflush(fd, termios.TCIFLUSH)
    except (OSError, termios.error):
        pass
    try:
        tty.setraw(fd)
        sys.stdout.write("\033[?25l")
        while True:
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(
                _render_provider_selector(providers, filtered, idx, current_id)
            )
            sys.stdout.flush()
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                _read_esc(fd)
            elif ch in ("\r", "\n"):
                return filtered[idx] if filtered else None
            elif ch == "\x03":
                return None
            elif ch == "\x0e":  # Ctrl+N — down
                idx = min(len(filtered) - 1, idx + 1)
            elif ch == "\x10":  # Ctrl+P — up
                idx = max(0, idx - 1)
    finally:
        sys.stdout.write("\033[?25h")
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()


# ── Presets ──────────────────────────────────────────────


def _presets_dir():
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    return PRESETS_DIR


def list_presets():
    _presets_dir()
    presets = []
    for f in sorted(PRESETS_DIR.iterdir()):
        if f.suffix == ".json" and f.is_file():
            try:
                data = json.loads(f.read_text())
                presets.append({"name": f.stem, "path": f, "data": data})
            except (json.JSONDecodeError, OSError):
                pass
    return presets


def save_preset(name, cfg):
    """Save full config dict as a preset (minus ephemeral fields)."""
    _presets_dir()
    data = {
        k: v
        for k, v in cfg.items()
        if k not in ("_context_length",)
    }
    path = PRESETS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def load_preset_into_cfg(name, cfg):
    """Load a preset into cfg dict in-place. Returns (True, msg) or (False, error)."""
    presets = list_presets()
    match = [p for p in presets if p["name"] == name]
    if not match:
        return False, f"preset '{name}' not found"
    data = match[0]["data"]

    old_provider = cfg.get("provider", "openrouter")
    new_provider = data.get("provider", old_provider)

    # Overwrite all stored keys (preserves ephemeral keys in cfg not in preset)
    cfg.update(data)

    # If provider changed, mirror housekeeping that /provider does
    if new_provider != old_provider:
        cfg.pop("_context_length", None)
        if new_provider != "custom":
            cfg.pop("custom_url", None)
    save_cfg(cfg)

    # Refresh context_length
    ctx = get_model_context(
        cfg["model"],
        cfg.get("api_key", ""),
        cfg.get("provider", "openrouter"),
        cfg.get("custom_url", ""),
    )
    if ctx:
        cfg["_context_length"] = ctx

    return True, f"preset → {name}"


def delete_preset(name):
    path = PRESETS_DIR / f"{name}.json"
    if not path.exists():
        return False, f"preset '{name}' not found"
    path.unlink()
    return True, f"preset '{name}' removed"


def _render_preset_selector(presets, idx):
    w = _term_width()
    buf = [
        f"{C.BOLD}{C.CYAN}preset{C.RESET}  {C.GRAY}({len(presets)} presets){C.RESET}\r\n"
    ]
    buf.append(f"{C.GRAY}{'─' * (w - 2)}{C.RESET}\r\n")
    if not presets:
        buf.append(f"{C.GRAY}no presets — use /preset save <name>{C.RESET}\r\n")
    else:
        start = max(0, idx - 5)
        end = min(len(presets), start + 12)
        for i in range(start, end):
            p = presets[i]
            pre = f"{C.CYAN}▸{C.RESET} " if i == idx else "  "
            prov = p["data"].get("provider", "?")
            model = p["data"].get("model", "?")
            temp = p["data"].get("temperature", "?")
            buf.append(
                f"{pre}{p['name']:<20} {C.YELLOW}{prov:<12}{C.RESET} {C.GREEN}{model}{C.RESET}  {C.GRAY}temp={temp}{C.RESET}\r\n"
            )
    buf.append(f"\r\n{C.GRAY}↑/↓ nav  ↵ load  Ctrl+C exit{C.RESET}")
    return "".join(buf)


def preset_selector():
    try:
        import termios, tty
    except ImportError:
        print(f"\r{C.RED}error: preset selector requires a Unix terminal (termios not available){C.RESET}")
        return None

    presets = list_presets()
    if not presets:
        return None
    idx = 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        termios.tcflush(fd, termios.TCIFLUSH)
    except (OSError, termios.error):
        pass
    try:
        tty.setraw(fd)
        sys.stdout.write("\033[?25l")
        while True:
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(_render_preset_selector(presets, idx))
            sys.stdout.flush()
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                _read_esc(fd)
            elif ch in ("\r", "\n"):
                if presets and idx < len(presets):
                    return presets[idx]["name"]
                return None
            elif ch == "\x03":
                return None
            elif ch == "\x0e":
                idx = min(len(presets) - 1, idx + 1) if presets else 0
            elif ch == "\x10":
                idx = max(0, idx - 1)
    finally:
        sys.stdout.write("\033[?25h")
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()


def print_help():
    print(f"{C.CYAN}Commands:{C.RESET}")
    print(f"  /help              This help")
    print(f"  /new               Reset conversation")
    print(f"  /model [name]      Show or switch model")
    print(f"  /preset [sub]      Save/load/rm/ls/show presets (provider+model pairs)")
    print(f"  /session           Search and load session logs")
    print(f"  /provider [name]   Show or switch provider")
    print(f"  /temp [n]          Show or set temperature")
    print(f"  /sys [prompt]      Show or set system prompt")
    print(f"  /edit  [text]      Multi-line editor (or Ctrl+E / /edit)")
    print(f"  /tools             Toggle tool calling on/off")
    print(f"  /trust             Toggle auto-approve tools")
    print(
        f"  /export <file>      Export session log (use .md/.html for LLM conversion)"
    )
    print(f"  /skills            List installed skills")
    print(f"  /skill add|rm|show  Manage skills")
    print(f"  /mcp               MCP server status")
    print(f"  /mcp connect|disconnect|reconnect [name]  Manage MCP connections")
    print(f"  /mcp discover      Rediscover tools/resources from MCP servers")
    print(f"  /commit <msg>      git add -u && git commit -m '<msg>'")
    print(f"  /exit              Quit")
    print(f"  !<command>         Run bash command and save to conversation")
    print()
    mcp_count = len(_get_mcp_tools()) if _HAS_MCP else 0
    total_tools = len(TOOLS) + mcp_count
    print(f"{C.CYAN}Tools ({total_tools} total, {len(TOOLS)} built-in, {mcp_count} MCP):{C.RESET}")
    for t in TOOLS:
        fn = t["function"]
        print(
            f"  {C.YELLOW}{fn['name']}{C.RESET}  {C.GRAY}{fn['description'].split('.')[0]}.{C.RESET}"
        )
    if mcp_count and _mcp_manager:
        print(f"  {C.GREEN}({mcp_count} MCP tools from {_mcp_manager.connected_count}/{_mcp_manager.total_count} servers){C.RESET}")
    print(
        f"  {C.GRAY}--- skills are injected dynamically via the skill tool ---{C.RESET}"
    )
    print(
        f"  {C.GRAY}--- MCP tools use prefix mcp__<server>__<tool> ---{C.RESET}"
    )


def _render_file_selector(files, filtered, query, idx, selected):
    w = _term_width()
    sep = f"{C.GRAY}{'─' * (w - 2)}{C.RESET}"
    sel_count = len(selected)
    sel_tag = f"  {C.GREEN}{sel_count} selected{C.RESET}" if sel_count else ""
    buf = [
        f"{C.BOLD}{C.CYAN}file picker{C.RESET}  {C.GRAY}({len(filtered)}/{len(files)} files){C.RESET}{sel_tag}\r\n"
    ]
    buf.append(f"{C.CYAN}filter:{C.RESET} {query}\r\n")
    buf.append(f"{sep}\r\n")
    if not filtered:
        buf.append(f"{C.GRAY}no matches{C.RESET}\r\n")
    else:
        start = max(0, idx - 10)
        end = min(len(filtered), start + 20)
        for i in range(start, end):
            f = filtered[i]
            p = f["path"]
            checked = f"{C.GREEN}✓{C.RESET}" if p in selected else " "
            pre = f"{C.CYAN}▸{C.RESET} " if i == idx else "  "
            s = f["size"]
            if s < 1024:
                sz = f"{s}B"
            elif s < 1024 * 1024:
                sz = f"{s / 1024:.0f}K"
            else:
                sz = f"{s / 1024 / 1024:.1f}M"
            buf.append(f"{pre}[{checked}] {p}  {C.GRAY}{sz}{C.RESET}\r\n")
    buf.append(
        f"\r\n{C.GRAY}Space toggle  ↵ confirm  Ctrl+N/P nav  type filter  Ctrl+C cancel{C.RESET}"
    )
    return "".join(buf)


def file_selector(initial_query=""):
    try:
        import termios, tty
    except ImportError:
        print(f"\r{C.RED}error: file selector requires a Unix terminal (termios not available){C.RESET}")
        return None

    all_files = _collect_files()
    if not all_files:
        print(f"\r{C.YELLOW}no files found in project{C.RESET}")
        return None

    filtered = list(all_files)
    query = initial_query
    idx = 0
    selected = set()

    if query:
        q = query.lower()
        filtered = [f for f in all_files if q in f["path"].lower()]
        if idx >= len(filtered):
            idx = max(0, len(filtered) - 1)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        termios.tcflush(fd, termios.TCIFLUSH)
    except (OSError, termios.error):
        pass
    try:
        tty.setraw(fd)
        sys.stdout.write("\033[?25l")
        while True:
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(
                _render_file_selector(all_files, filtered, query, idx, selected)
            )
            sys.stdout.flush()
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                _read_esc(fd)
            elif ch == " ":
                if filtered and idx < len(filtered):
                    p = filtered[idx]["path"]
                    if p in selected:
                        selected.discard(p)
                    else:
                        selected.add(p)
            elif ch in ("\r", "\n"):
                if selected:
                    return ",".join(sorted(selected))
                if filtered:
                    return filtered[idx]["path"]
                return None
            elif ch == "\x03":
                return None
            elif ch == "\x0e":
                idx = min(len(filtered) - 1, idx + 1) if filtered else 0
            elif ch == "\x10":
                idx = max(0, idx - 1)
            elif ch in ("\x7f", "\b"):
                query = query[:-1]
                idx = 0
            elif ch.isprintable():
                query += ch
                idx = 0
            q = query.lower()
            if query:
                filtered = [f for f in all_files if q in f["path"].lower()]
            else:
                filtered = list(all_files)
            if idx >= len(filtered):
                idx = max(0, len(filtered) - 1) if filtered else 0
    finally:
        sys.stdout.write("\033[?25h")
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()


def _expand_file_markers(line):
    parts = []
    last_end = 0
    for m in _FILE_RE.finditer(line):
        parts.append(line[last_end : m.start()])
        filter_text = m.group(1)
        if shutil.which("fzf"):
            path = _fzf_file_selector(initial_query=filter_text)
        else:
            path = file_selector(initial_query=filter_text)
        if path:
            parts.append(path)
        else:
            parts.append(m.group(0))
        last_end = m.end()
    parts.append(line[last_end:])
    return "".join(parts)


def _replace_history_last(line):
    """Replace the last entry in readline history with the given line."""
    if readline is None:
        return
    try:
        length = readline.get_current_history_length()
        if length > 0:
            readline.replace_history_item(length - 1, line)
    except Exception:
        try:
            readline.add_history(line)
        except Exception:
            pass


def _collapse_for_history(text, max_len=200):
    """Collapse multi-line text to a single line for readline history storage."""
    one_line = " ".join(text.split())
    if len(one_line) > max_len:
        one_line = one_line[:max_len] + "..."
    return one_line


def open_editor(initial_text=""):
    editor_cmd = shlex.split(
        os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    )
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w+", delete=False) as f:
        f.write(initial_text)
        f.flush()
        tmp_path = f.name
    try:
        subprocess.run(editor_cmd + [tmp_path], check=True)
        result = Path(tmp_path).read_text(encoding="utf-8")
        return result.rstrip("\n")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"{C.RED}editor error: {e}{C.RESET}")
        return initial_text
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── System prompt loading ───────────────────────────────────────

SYSTEM_PROMPT_DEFAULT = """You are tbot, an interactive CLI tool that helps users with software engineering tasks. Use the tools available to assist the user with their requests.

# Core Rules (highest priority — read in order)
1. RESPOND TO THE USER FIRST. Your text output is the primary channel.
2. Use tools ONLY when necessary to complete a task the user asked for.
3. After tool execution, ALWAYS produce text to communicate results.
4. Be concise but complete — explain what matters, skip what doesn't. No minimum line count.
5. Do not add code explanation summaries unless the user asks.

# Security & Integrity
- NEVER expose secrets, API keys, tokens, or credentials.
- NEVER read, modify, or commit files like `.env`, `credentials.json`, `*.pem`, `*.key`.
- NEVER commit changes unless the user explicitly asks.
- NEVER generate or guess URLs for the user unless you are confident they are legitimate.

# Communication Style
- Output text to communicate with the user; tool results are displayed automatically.
- Your responses render as GitHub-flavored markdown in a terminal.
- Use standard GFM pipe tables for tabular data (e.g., `| col1 | col2 |`). Do NOT use Unicode box-drawing characters (┌─┬┐│└┴┘) in your output — they will be rendered automatically by the terminal.
- Only use emojis if the user explicitly asks. Never use Bash/code comments to communicate.
- If you cannot help, offer alternatives briefly (1-2 sentences). Explain WHY briefly when it helps the user understand.
- Reference code as `file_path:line_number` for clickable navigation.
- When you need clarification, use the `question` tool — do NOT guess critical details.
- Use `webfetch` for reading specific URLs, `websearch` for finding information.

# Coding Conventions
- Understand code conventions before editing. Mimic existing style, libraries, and patterns.
- Check if a library is already used before adding a dependency (check package.json, imports, etc.).
- DO NOT ADD COMMENTS to code unless asked.
- ALWAYS read a file BEFORE editing it (the edit tool enforces this).
- Prefer `edit` for small surgical changes, `write` for new files or complete rewrites.
- After running non-trivial commands, describe what you did and why.

# Tool Usage & Constraints
Available tools: {tools}

General rules:
- Parallelize independent tool calls (multiple in one response). Chain dependent calls sequentially.
- Tool output is truncated at {max_tool_output} characters. If you hit the limit, paginate (read with offset) or refine your query.
- Re-reading a file is allowed — content from previous rounds is stale, re-read is fresh.
- Prefer grep/glob for code search; prefer RAG for semantic/concept search across the codebase.

RAG ({has_rag}):
- `rag_index` builds a BM25 keyword index. Index once before searching.
- `rag_search` searches the index. Use for conceptual queries ("how does auth work?").
- `grep` is better for exact pattern matching, `glob` for filename patterns.

subagent (`task` tool, depth {subagent_depth}/3):
- Use for complex multi-step tasks that need independent research or processing.
- Maximum nesting depth is 3 (current: {subagent_depth}). Do not exceed.

todowrite:
- Call ONCE to plan, then call again ONLY when a task completes or status changes.
- If blocked by loop protection, use edit/write to modify TASK.md directly.

Loop prevention — the system detects and BLOCKS these patterns:
- Calling the same tool with identical arguments 3+ times in a row.
- Alternating tools in a pattern (A-B-A-B-A-B or A-B-C-A-B-C).
- Calling todowrite with the same list back-to-back, or faster than once per 15 seconds.
If blocked, stop and respond with text — do not retry the same sequence.

# Environment
Date: {date}  |  Model: {model}  |  Provider: {provider}  |  Trust mode: {trust_mode}
CWD: {cwd}  |  OS: {os}
Skills: {skills_dir}

Skills are loaded in two steps:
  1. `skill(name)` — checks if a skill exists (returns the loader function name)
  2. `skill_<name>()` — actually loads the skill's instructions into your context
Call `skill_<name>()` directly if you know the skill exists. Available skills: {skills_list}

# MCP (Model Context Protocol)
External tools from MCP servers are available with the prefix `mcp__<server>__<tool>`.
These connect to external services (databases, filesystems, APIs, etc.).
Use them like any other tool — the MCP server handles the execution.
MCP resources (if any) are listed below as hints for what data is available.

Before creating, modifying, updating, or fixing any skill (SKILL.md), first load the `skill-guide` skill with `skill_skill-guide()` to get the format reference and best practices.

═══ MEMORY [{memory_pct}% — {memory_facts_limit} chars max] ═══
{memory_content}

═══ USER PROFILE [{user_pct}% — {memory_user_limit} chars max] ═══
{user_content}

# Fact Memory (hechos persistentes)

Tienes acceso a dos archivos de hechos persistentes:
- **MEMORY.md** — hechos del entorno: proyectos, configuraciones, convenciones aprendidas
- **USER.md** — perfil del usuario: preferencias, estilo de comunicación

Se cargan al inicio de cada sesión y los ves arriba. Para modificarlos:
- `fact_read(store="memory"|"user")` — leer contenido actual
- `fact_write(store, content)` — agregar o reemplazar una entrada
- `fact_remove(store, substring)` — eliminar una entrada por substring

Límite: ~{memory_facts_limit} chars por archivo.

Cuándo guardar automáticamente:
- El usuario dice "recuerda que..." o "toma nota..."
- Descubres un hecho importante del entorno (sistema operativo, herramientas, estructura de proyecto)
- El usuario expresa una preferencia clara ("prefiero X sobre Y")

NO guardes:
- Información trivial o temporal
- Grandes bloques de código o logs
- Cosas que puedes buscar en web

Consolidación: si un archivo está cerca del límite (≥80%), fusiona entradas relacionadas
antes de agregar nuevas.

# Episodic Memory (historial de conversaciones)

Todas las conversaciones pasadas se almacenan en una base de datos SQLite con
búsqueda FTS5. Puedes buscar en ellas con:
- `episodic_search(query, top_k=5)` — busca palabras clave en todo el historial

Úsalo cuando:
- Necesitas recordar una decisión técnica de una sesión anterior
- El usuario menciona algo que discutieron antes
- Quieres saber si ya probaste cierto enfoque

NO lo uses para:
- Información que está en MEMORY.md (usa fact_read)
- Preguntas que puedes responder con las herramientas actuales

# Autonomous Skill Creation

After completing a complex procedure that involved multiple steps, different
tools, and especially if there was trial-and-error (errors, retries, edge cases),
consider creating a skill to preserve what you learned.

## When to create a skill
- The task required 5+ tool calls total
- You used 3+ different tools
- There were errors you had to work around (trial-and-error)
- The result is a reusable procedure (not a one-off)
- No similar skill already exists

## How to create one
1. Load `skill_skill-guide()` for the exact frontmatter format and best practices.
2. Write the skill content including YAML frontmatter.
3. Call `create_skill(name, description, content, replace=false)`.
4. The skill will be available in future sessions via `skill_<name>()`.

## Important
- Do NOT create skills for trivial tasks (1-2 tool calls, no errors).
- Do NOT create skills without checking with `skill("<name>")` first.
- ALWAYS load skill-guide before writing the skill content.
- The skill must be useful for YOU (the model) in the future, not for the user."""


def load_system_prompt(cfg):
    """Load system prompt from file (or config override). Injects dynamic vars."""
    prompt = cfg.get("system_prompt", "")
    if prompt:
        for k, v in _env_vars(cfg).items():
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

    for k, v in _env_vars(cfg).items():
        if "{" + k + "}" in data:
            data = data.replace("{" + k + "}", v)

    # Append MCP resource notes if available
    mcp_notes = _get_mcp_resource_notes()
    if mcp_notes:
        data += "\n\n" + mcp_notes

    return data


def _env_vars(cfg=None):
    skills = load_skills()
    skills_list = ", ".join(n for n, *_ in skills) if skills else "none"
    tool_names = [
        t["function"]["name"] for t in TOOLS if t["function"]["name"] != "invalid"
    ]
    has_rag = "yes" if _rag_binary() else "no"
    depth = os.environ.get("TBOT_DEPTH", "0")
    model = cfg.get("model", "unknown") if cfg else "unknown"
    provider = cfg.get("provider", "unknown") if cfg else "unknown"
    trust_mode = str(cfg.get("trust_mode", False)) if cfg else "unknown"
    # Load memory content for system prompt injection
    memory_content = _read_fact_file(MEMORY_FILE)
    user_content = _read_fact_file(USER_FILE)
    if not memory_content:
        memory_content = "(empty)"
    if not user_content:
        user_content = "(empty)"
    facts_limit = cfg.get("memory_facts_limit", 2200) if cfg else 2200
    user_limit = cfg.get("memory_user_limit", 2200) if cfg else 2200
    memory_pct = min(100, int(len(memory_content) / max(facts_limit, 1) * 100))
    user_pct = min(100, int(len(user_content) / max(user_limit, 1) * 100))
    return {
        "date": time.strftime("%Y-%m-%d"),
        "cwd": str(Path.cwd()),
        "platform": platform.system().lower(),
        "os": f"{platform.system()} {platform.release()}",
        "skills_dir": str(SKILLS_DIR),
        "skills_list": skills_list,
        "tools": ", ".join(tool_names),
        "model": model,
        "provider": provider,
        "trust_mode": trust_mode,
        "max_tool_output": str(MAX_TOOL_OUTPUT),
        "subagent_depth": depth,
        "has_rag": has_rag,
        "memory_content": memory_content,
        "user_content": user_content,
        "memory_facts_limit": str(facts_limit),
        "memory_user_limit": str(user_limit),
        "memory_pct": str(memory_pct),
        "user_pct": str(user_pct),
    }


# ── Main ────────────────────────────────────────────────────────


def _init_messages(cfg):
    prompt = load_system_prompt(cfg)
    return [{"role": "system", "content": prompt}] if prompt else []


def main():
    global _cfg_for_memory
    cfg = load_cfg()
    _cfg_for_memory = cfg
    _init_default_skills()
    ensure_memory_dirs()
    _init_episodic_db()
    cfg["api_key"] = resolve_key(cfg)

    parser = argparse.ArgumentParser(
        description="tbot - Terminal chatbot for OpenRouter"
    )
    parser.add_argument(
        "-m", "--model", help="Model slug (e.g. deepseek/deepseek-chat)"
    )
    parser.add_argument("-t", "--temperature", type=float, help="Temperature 0.0-2.0")
    parser.add_argument("-s", "--system", help="System prompt")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool calling")
    parser.add_argument(
        "--trust", action="store_true", help="Auto-approve tool execution"
    )
    parser.add_argument(
        "-x", "--task", help="Run a single task non-interactively and exit"
    )
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

    # ── non-interactive task mode ──
    if args.task:
        cfg["tools_enabled"] = True
        cfg["trust_mode"] = True
        messages = _init_messages(cfg)
        _log_init()
        atexit.register(_log_close)
        _log_write(f">>> {args.task}", role="user")
        messages.append({"role": "user", "content": args.task})
        send_conversation(messages, cfg)
        return

    _log_init()
    atexit.register(_log_close)
    setup_history()
    atexit.register(save_history)
    _setup_sigwinch()
    # Build prompt: wrap ANSI escapes in \001...\002 so readline knows they are zero-width
    if readline is not None:
        _rl_bold_blue = "\001\033[1;34m\002"
        _rl_reset = "\001\033[0m\002"
        _prompt = f"{_rl_bold_blue}>>>{_rl_reset} "
    else:
        _prompt = f"{C.BOLD}{C.BLUE}>>>{C.RESET} "
    ctx = get_model_context(
        cfg["model"],
        cfg.get("api_key", ""),
        cfg.get("provider", "openrouter"),
        cfg.get("custom_url", ""),
    )
    if ctx:
        cfg["_context_length"] = ctx
    messages = _init_messages(cfg)
    show_banner(cfg)
    # Initialize episodic memory
    ensure_memory_dirs()
    _init_episodic_db()
    _session_state["id"] = time.strftime("%Y%m%d-%H%M%S-") + os.urandom(4).hex()
    try:
        import sqlite3
        conn = sqlite3.connect(str(EPISODIC_DB))
        conn.execute(
            "INSERT INTO sessions (id, started_at, model, provider) VALUES (?, datetime('now'), ?, ?)",
            (_session_state["id"], cfg.get("model", "?"), cfg.get("provider", "?")),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
    # Auto-prune old messages
    try:
        import sqlite3
        retention = cfg.get("episodic_retention_days", 90)
        conn = sqlite3.connect(str(EPISODIC_DB))
        pruned = conn.execute(
            "DELETE FROM messages WHERE created_at < datetime('now', ?)",
            (f"-{retention} days",),
        ).rowcount
        conn.commit()
        conn.close()
        if pruned:
            print(f"{C.GRAY}memory: pruned {pruned} old messages{C.RESET}")
    except Exception:
        pass

    # ── MCP initialization (synchronous — before prompt) ──
    _init_mcp(cfg)
    atexit.register(lambda: _mcp_manager.close_all() if _mcp_manager else None)

    while True:
        try:
            line = input(_prompt)
            save_history()
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
                global _total_tokens, _last_cost, _acc_cost, _todo_blocked_count
                _total_tokens = 0
                _last_cost = 0
                _acc_cost = 0
                _todo_blocked_count = 0
                messages = _init_messages(cfg)
                _doom_trail.clear()
                _read_trail.clear()
                _log_reopen()
                # Initialize episodic session
                _init_episodic_db()
                _session_state["id"] = time.strftime("%Y%m%d-%H%M%S-") + os.urandom(4).hex()
                try:
                    import sqlite3
                    conn = sqlite3.connect(str(EPISODIC_DB))
                    conn.execute(
                        "INSERT INTO sessions (id, started_at, model, provider) VALUES (?, datetime('now'), ?, ?)",
                        (_session_state["id"], cfg.get("model", "?"), cfg.get("provider", "?")),
                    )
                    conn.commit()
                    conn.close()
                except Exception:
                    pass
                print(f"{C.GREEN}reset{C.RESET}")
            elif cmd == "model":
                if arg:
                    cfg["model"] = arg
                    cfg.pop("_context_length", None)
                    save_cfg(cfg)
                    ctx = get_model_context(
                        cfg["model"],
                        cfg.get("api_key", ""),
                        cfg.get("provider", "openrouter"),
                        cfg.get("custom_url", ""),
                    )
                    if ctx:
                        cfg["_context_length"] = ctx
                        print(
                            f"{C.GREEN}model → {cfg['model']} ({_fmt_context(ctx)}){C.RESET}"
                        )
                    else:
                        print(
                            f"{C.GREEN}model → {cfg['model']} (context unknown){C.RESET}"
                        )
                else:
                    selected = model_selector(
                        cfg["model"],
                        cfg.get("api_key", ""),
                        cfg.get("provider", "openrouter"),
                        cfg.get("custom_url", ""),
                    )
                    if selected and selected != cfg["model"]:
                        cfg["model"] = selected
                        cfg.pop("_context_length", None)
                        save_cfg(cfg)
                        ctx = get_model_context(
                            cfg["model"],
                            cfg.get("api_key", ""),
                            cfg.get("provider", "openrouter"),
                            cfg.get("custom_url", ""),
                        )
                        if ctx:
                            cfg["_context_length"] = ctx
                            print(
                                f"{C.GREEN}model → {cfg['model']} ({_fmt_context(ctx)}){C.RESET}"
                            )
                        else:
                            print(f"{C.GREEN}model → {cfg['model']}{C.RESET}")
                    elif selected == cfg["model"]:
                        pass
                    else:
                        print(f"{C.YELLOW}cancelled{C.RESET}")
            elif cmd == "session":
                selected = session_selector()
                if selected:
                    try:
                        text = Path(selected["full_path"]).read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except Exception as e:
                        print(f"{C.RED}cannot read log: {e}{C.RESET}")
                        continue
                    print(
                        f"\n{C.CYAN}── {selected['path']} ({len(text)} chars) ──{C.RESET}"
                    )
                    print(text.rstrip())
                    messages.append({"role": "user", "content": text.rstrip()})
            elif cmd == "preset":
                sub = arg.split(maxsplit=1) if arg else []
                sub_cmd = sub[0].lower() if sub else ""
                sub_arg = sub[1] if len(sub) > 1 else ""
                if not sub_cmd:
                    presets = list_presets()
                    if not presets:
                        print(f"{C.YELLOW}no presets — use /preset save <name>{C.RESET}")
                    else:
                        name = preset_selector()
                        if name:
                            ok, msg = load_preset_into_cfg(name, cfg)
                            if ok:
                                _save_models_cache([])
                                prov = PROVIDERS.get(cfg.get("provider", ""), {}).get(
                                    "name", cfg.get("provider", "?")
                                )
                                print(
                                    f"{C.GREEN}{msg} ({prov} / {cfg['model']}){C.RESET}"
                                )
                            else:
                                print(f"{C.RED}{msg}{C.RESET}")
                        else:
                            print(f"{C.YELLOW}cancelled{C.RESET}")
                elif sub_cmd == "save":
                    if not sub_arg:
                        print(f"{C.YELLOW}usage: /preset save <name>{C.RESET}")
                    else:
                        path = save_preset(sub_arg, cfg)
                        print(f"{C.GREEN}preset '{sub_arg}' saved{C.RESET}")
                elif sub_cmd == "load":
                    if not sub_arg:
                        print(f"{C.YELLOW}usage: /preset load <name>{C.RESET}")
                    else:
                        ok, msg = load_preset_into_cfg(sub_arg, cfg)
                        if ok:
                            _save_models_cache([])
                            prov = PROVIDERS.get(cfg.get("provider", ""), {}).get(
                                "name", cfg.get("provider", "?")
                            )
                            print(
                                f"{C.GREEN}{msg} ({prov} / {cfg['model']}){C.RESET}"
                            )
                        else:
                            print(f"{C.RED}{msg}{C.RESET}")
                elif sub_cmd == "rm":
                    if not sub_arg:
                        print(f"{C.YELLOW}usage: /preset rm <name>{C.RESET}")
                    else:
                        ok, msg = delete_preset(sub_arg)
                        print(f"{C.GREEN if ok else C.RED}{msg}{C.RESET}")
                elif sub_cmd == "ls":
                    presets = list_presets()
                    if not presets:
                        print(f"{C.YELLOW}no presets{C.RESET}")
                        print(f"{C.GRAY}create one: /preset save <name>{C.RESET}")
                    else:
                        print(f"{C.CYAN}presets ({len(presets)}):{C.RESET}")
                        for p in presets:
                            prov = p["data"].get("provider", "?")
                            model = p["data"].get("model", "?")
                            temp = p["data"].get("temperature", "?")
                            print(
                                f"  {C.GREEN}{p['name']:<20}{C.RESET}"
                                f" {C.YELLOW}{prov:<12}{C.RESET}"
                                f" {C.CYAN}{model}{C.RESET}"
                                f"  {C.GRAY}temp={temp}{C.RESET}"
                            )
                elif sub_cmd == "show":
                    if not sub_arg:
                        print(f"{C.YELLOW}usage: /preset show <name>{C.RESET}")
                    else:
                        presets = list_presets()
                        match = [p for p in presets if p["name"] == sub_arg]
                        if not match:
                            print(f"{C.RED}preset '{sub_arg}' not found{C.RESET}")
                        else:
                            print(
                                f"{C.CYAN}── {sub_arg} ──{C.RESET}"
                            )
                            print(
                                json.dumps(match[0]["data"], indent=2, ensure_ascii=False)
                            )
                else:
                    print(f"{C.YELLOW}usage:{C.RESET}")
                    print(f"  /preset               Interactive selector")
                    print(f"  /preset save <name>   Save current config as preset")
                    print(f"  /preset load <name>   Load a preset")
                    print(f"  /preset rm <name>     Delete a preset")
                    print(f"  /preset ls            List presets")
                    print(f"  /preset show <name>   Show preset details")
            elif cmd == "provider":
                if arg:
                    prov = arg.lower().strip()
                    if prov == "custom":
                        _setup_custom_provider(cfg)
                    elif prov in PROVIDERS:
                        old_prov = cfg.get("provider", "openrouter")
                        cfg["provider"] = prov
                        cfg.pop("_context_length", None)
                        cfg.pop("custom_url", None)
                        cfg["api_key"] = ""
                        save_cfg(cfg)
                        cfg["api_key"] = resolve_key(cfg)
                        save_cfg(cfg)
                        _save_models_cache([])
                        info = PROVIDERS[prov]
                        print(f"{C.GREEN}provider → {info['name']}{C.RESET}")
                        if prov != old_prov:
                            print(f"{C.YELLOW}model reset to default{C.RESET}")
                            cfg["model"] = default_cfg()["model"]
                            save_cfg(cfg)
                    else:
                        names = ", ".join(PROVIDERS.keys())
                        print(f"{C.RED}unknown provider. Available: {names}{C.RESET}")
                else:
                    selected = provider_selector(cfg.get("provider", "openrouter"))
                    if selected and selected != cfg.get("provider"):
                        if selected == "custom":
                            _setup_custom_provider(cfg)
                        else:
                            old_prov = cfg.get("provider", "openrouter")
                            cfg["provider"] = selected
                            cfg.pop("_context_length", None)
                            cfg.pop("custom_url", None)
                            cfg["api_key"] = ""
                            save_cfg(cfg)
                            cfg["api_key"] = resolve_key(cfg)
                            save_cfg(cfg)
                            _save_models_cache([])
                            info = PROVIDERS[selected]
                            print(f"{C.GREEN}provider → {info['name']}{C.RESET}")
                            if selected != old_prov:
                                print(f"{C.YELLOW}model reset to default{C.RESET}")
                                cfg["model"] = default_cfg()["model"]
                                save_cfg(cfg)
                    elif selected == cfg.get("provider"):
                        pass
                    else:
                        print(f"{C.YELLOW}cancelled{C.RESET}")
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
                print(
                    f"{C.GREEN}tools {'on' if cfg['tools_enabled'] else 'off'}{C.RESET}"
                )
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
                        messages.append({"role": "user", "content": arg})
                        print(f"{C.GREEN}message added ({len(arg)} chars){C.RESET}")
                        send_conversation(messages, cfg, pop_on_first_error=True)
                    else:
                        messages[last_user]["content"] = arg
                        print(f"{C.GREEN}last message updated ({len(arg)} chars){C.RESET}")
                        send_conversation(messages, cfg, pop_on_first_error=False)
                else:
                    last_user = -1
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i]["role"] == "user":
                            last_user = i
                            break
                    initial = ""
                    print(f"{C.YELLOW}opening editor...{C.RESET}")
                    content = open_editor(initial)
                    if content:
                        # ── Expandir @@ con fzf ──
                        if "@@" in content and shutil.which("fzf"):
                            sel = _fzf_file_selector()
                            if sel:
                                content = content.replace("@@", sel)
                        # ──────────────────────────
                        appended = last_user == -1
                        if last_user != -1:
                            messages[last_user]["content"] = content
                        else:
                            messages.append({"role": "user", "content": content})
                        lines = content.split("\n")
                        print(
                            f"{C.GREEN}message ({len(lines)} lines, {len(content)} chars){C.RESET}"
                        )
                        if len(lines) <= 3:
                            print(content)
                        # Store collapsed version in readline history
                        _replace_history_last(_collapse_for_history(content))
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
                    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*$", name):
                        print(f"{C.RED}invalid skill name{C.RESET}")
                    else:
                        ensure_skills_dir()
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
            elif cmd == "export":
                if not arg:
                    print(f"{C.YELLOW}usage: /export <file>{C.RESET}")
                    continue
                export_path = Path(arg).expanduser().resolve()
                if not _current_log_path or not _current_log_path.exists():
                    print(f"{C.RED}no log file found{C.RESET}")
                    continue
                log_text = _current_log_path.read_text(encoding="utf-8")
                ext = export_path.suffix.lower()
                if ext in (".html", ".md"):
                    print(f"{C.YELLOW}converting to {ext}...{C.RESET}")
                    result = _llm_convert(log_text, ext, cfg)
                    if result is None:
                        print(f"{C.RED}conversion failed{C.RESET}")
                        continue
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    export_path.write_text(result, encoding="utf-8")
                    print(f"{C.GREEN}exported ({ext}) to {export_path}{C.RESET}")
                else:
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(_current_log_path, export_path)
                    print(f"{C.GREEN}exported to {export_path}{C.RESET}")
            elif cmd == "rag":
                sub = arg.split(maxsplit=1) if arg else []
                sub_cmd = sub[0].lower() if sub else ""
                sub_arg = sub[1] if len(sub) > 1 else ""
                if sub_cmd == "index":
                    print(f"{C.YELLOW}Indexing...{C.RESET}")
                    result = _run_rag(["index", sub_arg or "."], timeout=120)
                    if "error" in result:
                        print(f"{C.RED}{result['error']}{C.RESET}")
                    else:
                        print(f"{C.GREEN}Index built{C.RESET}")
                elif sub_cmd == "search":
                    if not sub_arg:
                        print(f"{C.RED}usage: /rag search <query>{C.RESET}")
                    else:
                        result = _run_rag(["search", sub_arg, "5"])
                        if "error" in result:
                            print(f"{C.RED}{result['error']}{C.RESET}")
                        elif isinstance(result, list):
                            for r in result:
                                print(
                                    f"  {C.CYAN}{r.get('path', '?')}:{r.get('start', '?')}{C.RESET} score={r.get('score', 0):.1f} [{r.get('type', '?')}] {r.get('name', '')}"
                                )
                        else:
                            print(str(result)[:2000])
                elif sub_cmd == "status":
                    result = _run_rag(["status"])
                    if isinstance(result, dict) and result.get("exists"):
                        print(f"  Chunks: {result.get('chunks', 0)}")
                        print(f"  Files:  {result.get('files', 0)}")
                        print(f"  Terms:  {result.get('total_terms', 0)}")
                    else:
                        print(f"{C.YELLOW}No index{C.RESET}")
                else:
                    print(f"{C.YELLOW}usage:{C.RESET}")
                    print(f"  /rag index [path]    Build RAG index")
                    print(f"  /rag search <query>  Search codebase")
                    print(f"  /rag status          Show index stats")
            elif cmd == "memory":
                sub = arg.split(maxsplit=1) if arg else []
                sub_cmd = sub[0].lower() if sub else ""
                sub_arg = sub[1] if len(sub) > 1 else ""
                if sub_cmd == "show":
                    if sub_arg and sub_arg in ("memory", "user"):
                        content = handle_fact_read({"store": sub_arg})
                        print(content)
                    elif sub_arg:
                        print(f"{C.RED}invalid store. Use 'memory' or 'user'{C.RESET}")
                    else:
                        print(f"{C.CYAN}── MEMORY.md ──{C.RESET}")
                        print(handle_fact_read({"store": "memory"}))
                        print()
                        print(f"{C.CYAN}── USER.md ──{C.RESET}")
                        print(handle_fact_read({"store": "user"}))
                elif sub_cmd == "edit":
                    store = sub_arg if sub_arg in ("memory", "user") else "memory"
                    path = _fact_file_path(store)
                    ensure_memory_dirs()
                    current = _read_fact_file(path)
                    print(f"{C.YELLOW}opening editor for {path.name}...{C.RESET}")
                    content = open_editor(current)
                    if content is not None:
                        _write_fact_file(path, content)
                        print(f"{C.GREEN}{path.name} updated ({len(content)} chars){C.RESET}")
                    else:
                        print(f"{C.YELLOW}cancelled{C.RESET}")
                elif sub_cmd == "gate":
                    if sub_arg in ("on", "off"):
                        cfg["memory_gate"] = sub_arg == "on"
                        save_cfg(cfg)
                        print(f"{C.GREEN}memory gate {'on' if sub_arg == 'on' else 'off'}{C.RESET}")
                    else:
                        print(f"{C.YELLOW}usage: /memory gate on|off{C.RESET}")
                elif sub_cmd == "search":
                    parts = arg.split(maxsplit=2)
                    store = "both"
                    query = ""
                    if len(parts) == 3:
                        store = parts[1].lower()
                        query = parts[2]
                    elif len(parts) == 2:
                        # Could be "search <store>" or "search <query>"
                        if parts[1] in ("memory", "user"):
                            store = parts[1]
                        else:
                            query = parts[1]
                    if not query:
                        print(f"{C.RED}usage: /memory search [memory|user] <query>{C.RESET}")
                    else:
                        stores = [("memory", MEMORY_FILE), ("user", USER_FILE)]
                        if store != "both":
                            stores = [(s, p) for s, p in stores if s == store]
                        for sname, spath in stores:
                            content = _read_fact_file(spath)
                            if not content:
                                print(f"{C.YELLOW}({sname}: empty){C.RESET}")
                                continue
                            matching = [l for l in content.split("\n")
                                        if query.lower() in l.lower()
                                        and not l.startswith("<!--")]
                            if matching:
                                print(f"{C.CYAN}── {spath.name} ({len(matching)} matches) ──{C.RESET}")
                                for m in matching:
                                    print(f"  {m}")
                            else:
                                print(f"{C.YELLOW}({sname}: no matches){C.RESET}")
                else:
                    print(f"{C.YELLOW}usage:{C.RESET}")
                    print(f"  /memory show [store]     Show MEMORY.md or USER.md (or both)")
                    print(f"  /memory edit [store]     Open editor to edit manually")
                    print(f"  /memory gate on|off      Enable/disable approval gate")
                    print(f"  /memory search [store]   Search facts (case-insensitive)")
            elif cmd == "episodic":
                sub = arg.split(maxsplit=1) if arg else []
                sub_cmd = sub[0].lower() if sub else ""
                sub_arg = sub[1] if len(sub) > 1 else ""
                if sub_cmd == "search":
                    if not sub_arg:
                        print(f"{C.RED}usage: /episodic search <query>{C.RESET}")
                    else:
                        result = _episodic_search(sub_arg, top_k=10)
                        if result:
                            print(result)
                        else:
                            print(f"{C.YELLOW}no results for: {sub_arg}{C.RESET}")
                elif sub_cmd == "stats":
                    try:
                        import sqlite3
                        conn = sqlite3.connect(str(EPISODIC_DB))
                        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
                        messages_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                        size = EPISODIC_DB.stat().st_size if EPISODIC_DB.exists() else 0
                        conn.close()
                        print(f"{C.CYAN}Episodic DB stats:{C.RESET}")
                        print(f"  Sessions: {sessions}")
                        print(f"  Messages: {messages_count}")
                        print(f"  Size:     {size / 1024:.1f} KB")
                    except Exception as e:
                        print(f"{C.RED}error: {e}{C.RESET}")
                elif sub_cmd == "prune":
                    days = 90
                    if sub_arg:
                        try:
                            days = max(1, int(sub_arg))
                        except ValueError:
                            print(f"{C.RED}invalid days{C.RESET}")
                            continue
                    try:
                        import sqlite3
                        conn = sqlite3.connect(str(EPISODIC_DB))
                        deleted = conn.execute(
                            "DELETE FROM messages WHERE created_at < datetime('now', ?)",
                            (f"-{days} days",),
                        ).rowcount
                        conn.commit()
                        conn.close()
                        print(f"{C.GREEN}pruned {deleted} messages older than {days} days{C.RESET}")
                    except Exception as e:
                        print(f"{C.RED}error: {e}{C.RESET}")
                else:
                    print(f"{C.YELLOW}usage:{C.RESET}")
                    print(f"  /episodic search <q>   Search full conversation history (FTS5)")
                    print(f"  /episodic stats        Show database statistics")
                    print(f"  /episodic prune [days] Delete messages older than N days")
            elif cmd == "commit":
                if not arg:
                    print(f"{C.YELLOW}usage: /commit <message>{C.RESET}")
                else:
                    r = subprocess.run(
                        ["git", "add", "-u"],
                        capture_output=True, text=True, timeout=30,
                        cwd=str(CURRENT_DIR),
                    )
                    if r.returncode != 0:
                        print(f"{C.RED}git add -u failed: {r.stderr.strip()}{C.RESET}")
                    else:
                        r2 = subprocess.run(
                            ["git", "commit", "-m", arg],
                            capture_output=True, text=True, timeout=30,
                            cwd=str(CURRENT_DIR),
                        )
                        if r2.returncode == 0:
                            print(f"{C.GREEN}{r2.stdout.strip()}{C.RESET}")
                        else:
                            print(f"{C.RED}{r2.stderr.strip()}{C.RESET}")
            elif cmd == "mcp":
                sub = arg.split(maxsplit=1) if arg else []
                sub_cmd = sub[0].lower() if sub else "status"
                sub_arg = sub[1] if len(sub) > 1 else ""
                result = handle_mcp_command({"action": sub_cmd, "name": sub_arg})
                if isinstance(result, str):
                    print(result)
                else:
                    print(str(result))
            else:
                print(f"{C.RED}unknown: /{cmd}{C.RESET}")
            continue

        # ── bash shortcut (!prefix) ──
        if line.startswith("!"):
            cmd = line[1:].strip()
            if cmd:
                print(f"  {C.GRAY}$ {cmd}{C.RESET}")
                out, rc = _run_interactive(cmd, str(CURRENT_DIR))
                _log_write(f"$ {cmd}")
                _log_write(out)
                messages.append({"role": "user", "content": f"! {cmd}"})
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"```\n$ {cmd}\n{out}\n```",
                    }
                )
            continue

        # ── message ──
        line = _expand_file_markers(line)
        _log_write(f">>> {line}", role="user")
        messages.append({"role": "user", "content": line})
        send_conversation(messages, cfg, pop_on_first_error=True)


MAX_TOOL_ONLY_ROUNDS = 60


def send_conversation(messages, cfg, pop_on_first_error=False):
    try:
        max_chars = _compute_max_history_chars(cfg)
        while (
            sum(_content_str_len(m.get("content", "")) for m in messages) > max_chars
            and sum(1 for m in messages if m["role"] not in ("system",)) > 1
        ):
            for i, m in enumerate(messages):
                if m["role"] == "user":
                    messages.pop(i)
                    if i < len(messages) and messages[i]["role"] == "assistant":
                        messages.pop(i)
                        # Also remove tool messages that belong to this assistant
                        while i < len(messages) and messages[i]["role"] == "tool":
                            messages.pop(i)
                    break
        tools = None
        if cfg["tools_enabled"]:
            tools = list(TOOLS)
            skills = load_skills()
            if skills:
                tools += skills_to_tools(skills)
            # MCP tools
            mcp_tools = _get_mcp_tools()
            if mcp_tools:
                tools += mcp_tools
        max_rounds = cfg.get("max_rounds", 200)
        round_n = 0
        retryable_errors = {"connection", "timeout", "ssl", "proxy"}
        max_retries = 3
        base_delay = 1.0
        max_stream_retries = 5
        stream_retries = 0
        tool_only_rounds = 0
        stuck_rounds = 0
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}cancelled{C.RESET}")
        return
    while round_n < max_rounds:
        _clear_trails()
        round_n += 1
        try:
            for attempt in range(max_retries + 1):
                result = chat_completion(messages, cfg, stream=True, tools=tools)
                if "error" not in result:
                    break
                if attempt < max_retries and result.get("error") in retryable_errors:
                    delay = base_delay * (2**attempt)
                    print(
                        f"\n  {C.YELLOW}Connection lost ({result['error']}), retrying in {delay:.0f}s... (attempt {attempt + 1}/{max_retries}){C.RESET}"
                    )
                    time.sleep(delay)
                    continue
                show_error(
                    result.get("title", "Error"),
                    result.get("detail", result["error"]),
                    result.get("hint", ""),
                )
                if round_n == 1 and pop_on_first_error:
                    messages.pop()
                break
            if "error" in result:
                break
            global _total_tokens, _last_cost, _acc_cost
            content, reasoning_content, tool_calls, pt, ct, _tot, _cost, interrupted = parse_stream(
                result["stream"], resp_color=cfg.get("resp_color", "95")
            )
            result["stream"].close()
            if interrupted:
                stream_retries += 1
                if stream_retries > max_stream_retries:
                    show_error(
                        "Stream keeps failing",
                        "The connection was interrupted 5 times in a row.",
                        "Check your internet connection or try a different model.",
                    )
                    break
                if content:
                    print(
                        f"\n{C.YELLOW}  Stream interrupted, reconnecting... (retry {stream_retries}/{max_stream_retries}){C.RESET}"
                    )
                else:
                    print(
                        f"\n  {C.YELLOW}Stream interrupted, reconnecting... (retry {stream_retries}/{max_stream_retries}){C.RESET}"
                    )
                round_n -= 1
                continue
            if _tot or content or tool_calls:
                stream_retries = 0
            if _tot:
                _total_tokens = _tot
            if _cost:
                _last_cost = _cost
            _acc_cost += _cost
            if tool_calls:
                if content:
                    tool_only_rounds = 0
                    print()
                else:
                    tool_only_rounds += 1
                    if tool_only_rounds >= MAX_TOOL_ONLY_ROUNDS:
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    f"LÍMITE ALCANZADO: Llevas sin responder texto al usuario durante "
                                    f"{MAX_TOOL_ONLY_ROUNDS} rondas. El sistema BLOQUEARÁ cualquier "
                                    "nuevo tool call. DEBES responder ahora directamente al usuario."
                                ),
                            }
                        )
                        break
                # Truncate tool_calls BEFORE building assistant_msg so all tool_calls
                # in the message have corresponding tool responses
                if len(tool_calls) > MAX_TOOLS_PER_ROUND:
                    discarded = tool_calls[MAX_TOOLS_PER_ROUND:]
                    tool_calls = tool_calls[:MAX_TOOLS_PER_ROUND]
                    messages.append(
                        {
                            "role": "system",
                            "content": f"Solo se ejecutaron {MAX_TOOLS_PER_ROUND} de tus tool calls. "
                            f"Los otros {len(discarded)} fueron ignorados. Reduce tool calls paralelos.",
                        }
                    )
                doom_warning = _check_doom_loop(tool_calls)
                if doom_warning:
                    print(f"\n  {C.YELLOW}{doom_warning[:100]}{C.RESET}")
                    messages.append({"role": "system", "content": doom_warning})
                    continue
                # Build assistant message now (after doom check to avoid orphaned tool_calls)
                assistant_msg = {
                    "role": "assistant",
                    "content": content or None,
                }
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in tool_calls
                ]
                before_tc = len(messages)
                messages.append(assistant_msg)
                # Log assistant response to episodic memory
                if content:
                    _log_write(f"assistant: {content[:1000]}", role="assistant")
                if tool_calls:
                    tc_names = ", ".join(tc["function"]["name"] for tc in tool_calls)
                    _log_write(f"assistant → tool_calls: {tc_names}", role="assistant")
                ok = execute_tool_calls(tool_calls, messages, cfg)
                if not ok:
                    # Clean up: remove only the messages added THIS round
                    # (assistant_msg + any tool/user messages appended by execute_tool_calls)
                    while len(messages) > before_tc:
                        messages.pop()
                    break

                # --- stuck round detection (todowrite loops) ---
                last_n = min(len(tool_calls), len(messages))
                blocked = all(
                    "BLOCKED" in m.get("content", "")
                    for m in messages[-last_n:]
                    if m.get("role") == "tool"
                )
                if blocked:
                    stuck_rounds += 1
                    if stuck_rounds >= 3:
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    f"CRITICAL: Tools blocked for {stuck_rounds} consecutive rounds. "
                                    "You MUST stop using tools and respond directly to the user. "
                                    "Do NOT make any more tool calls."
                                ),
                            }
                        )
                        break
                else:
                    stuck_rounds = 0

                continue
            if content:
                tool_only_rounds = 0
                print()
                if _total_tokens:
                    cost_str = _format_cost(_acc_cost) if _acc_cost else ""
                    print(f"{C.GRAY}── {_total_tokens} tokens{cost_str} ──{C.RESET}")
                assistant_msg = {"role": "assistant", "content": content}
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                messages.append(assistant_msg)
                _log_write(f"assistant: {content[:1000]}", role="assistant")
            break
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}cancelled{C.RESET}")
            if round_n == 1 and pop_on_first_error:
                messages.pop()
            break
    if round_n >= max_rounds:
        show_error(
            "Max tool rounds reached",
            f"The model used {max_rounds} consecutive tool calls without producing a final response.",
            "This may indicate a bug in the model or an infinite loop. Try a different model.",
        )


if __name__ == "__main__":
    main()
