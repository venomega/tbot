#!/usr/bin/env python3
"""runner.py — tbot Task Runner (Python reimplementation of agent.sh)

Orquestador de tareas autónomas con pipeline doer → reviewer.
Tipos: onehot (ejecución única) e idle (ejecución periódica).
Soporta fases: pending → running → review → running → completed.
Evaluación automática + notificaciones Termux + integración vía tools en tbot.

Dependencias: python3, termux-notification (opcional), termux-media-player (opcional)
"""

import os
import sys
import json
import time
import uuid
import io
import logging
import logging.handlers
import signal
import argparse
import shutil
import subprocess
import threading
import traceback
import textwrap
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from contextlib import redirect_stdout

# ── Import tbot como librería ──────────────────────────────────────────
_TBOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TBOT_DIR))
import tbot

# ── Constantes ─────────────────────────────────────────────────────────
RUNNER_DIR = Path(os.environ.get("RUNNER_DIR", Path.home() / ".config" / "tbot" / "runner"))
TASKS_FILE = RUNNER_DIR / "tasks.json"
AGENTS_FILE = RUNNER_DIR / "agents.json"
LOCK_FILE = RUNNER_DIR / "runner.lock"
PID_FILE = RUNNER_DIR / "runner.pid"
LOG_FILE = RUNNER_DIR / "runner.log"
HEARTBEAT_FILE = RUNNER_DIR / "heartbeat"
ACTIVE_FILE = RUNNER_DIR / ".running"
PAUSED_FILE = RUNNER_DIR / ".paused"

LOCK_TIMEOUT = 120        # segundos antes de considerar stale
MAX_LOG_SIZE = 5_242_880  # 5MB
MAX_LOG_BACKUPS = 3
TASK_TIMEOUT = 600        # 10 min por tarea
DEFAULT_POLL = 30
DEFAULT_PARALLEL = 3
MAX_RESULT_CHARS = 102_400  # truncar resultados >100KB en JSON
PRUNE_AGE_DAYS = 30

# Herramientas que requieren interacción humana → bloqueadas en modo agente
_BLOCKED_TOOLS = frozenset({"question"})

# Estados válidos del sistema
_VALID_STATUSES = frozenset({"pending", "running", "paused", "completed", "failed", "review"})

# Review prompt por defecto cuando no se especifica uno personalizado
_DEFAULT_REVIEW_PROMPT = """\
Eres un revisor especializado. Tu tarea es verificar que el siguiente resultado
funciona correctamente. Tienes acceso a herramientas (bash, etc.) para probar código.

--- TAREA ORIGINAL ---
{prompt}

--- RESULTADO A REVISAR ---
{result}

Ejecuta las pruebas necesarias para verificar que el código/trabajo funciona.
Luego responde ÚNICAMENTE con:

✅ si todo funciona correctamente
❌ explicando QUÉ falla y QUÉ hay que corregir (máx 200 caracteres)
"""

# ── Logging ────────────────────────────────────────────────────────────
log = logging.getLogger("runner")
_verbose = False


def setup_logging():
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=MAX_LOG_BACKUPS
    )
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    if _verbose:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(ch)


# ── Helpers de tiempo ─────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_epoch() -> int:
    return int(time.time())


def _new_id() -> str:
    return f"t-{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _parse_ts(ts: Optional[str]) -> Optional[float]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except (ValueError, TypeError):
        return None


def _fmt_epoch(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Storage thread-safe con cache ─────────────────────────────────────
class Storage:
    """Persistencia de tareas y agentes en JSON con cache en memoria y lock thread-safe.

    - Cache en memoria: carga una vez, escribe solo cuando hay cambios.
    - RLock para operaciones seguras entre threads.
    - Escritura atómica (tmp + replace).
    """

    def __init__(self):
        RUNNER_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tasks_cache: Optional[list] = None
        self._agents_cache: Optional[dict] = None
        self._tasks_dirty = False
        self._agents_dirty = False
        self._tasks_mtime: Optional[float] = None

    # -- helpers internos con cache -------------------------------------

    def _invalidate_cache_if_changed(self):
        """Si el archivo tasks.json fue modificado externamente, invalida la cache."""
        if self._tasks_cache is None:
            return
        if self._tasks_dirty:
            return  # tenemos cambios sin guardar, no invalidar
        try:
            current_mtime = TASKS_FILE.stat().st_mtime
            if self._tasks_mtime is not None and current_mtime > self._tasks_mtime:
                self._tasks_cache = None
        except OSError:
            pass

    def _load_tasks(self) -> list:
        self._invalidate_cache_if_changed()
        if self._tasks_cache is not None:
            return self._tasks_cache
        if not TASKS_FILE.exists():
            self._tasks_cache = []
            self._tasks_mtime = time.time()
            return self._tasks_cache
        try:
            data = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
            self._tasks_cache = data.get("tasks", [])
            self._tasks_mtime = TASKS_FILE.stat().st_mtime
        except (json.JSONDecodeError, OSError):
            self._tasks_cache = []
            self._tasks_mtime = time.time()
        return self._tasks_cache

    def _save_tasks(self):
        if not self._tasks_dirty or self._tasks_cache is None:
            return
        tmp = TASKS_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"tasks": self._tasks_cache}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(TASKS_FILE)
        self._tasks_dirty = False
        self._tasks_mtime = TASKS_FILE.stat().st_mtime

    def _load_agents(self) -> dict:
        if self._agents_cache is not None:
            return self._agents_cache
        if not AGENTS_FILE.exists():
            self._agents_cache = {}
            return self._agents_cache
        try:
            data = json.loads(AGENTS_FILE.read_text(encoding="utf-8"))
            self._agents_cache = data.get("agents", {})
        except (json.JSONDecodeError, OSError):
            self._agents_cache = {}
        return self._agents_cache

    def _save_agents(self):
        if not self._agents_dirty or self._agents_cache is None:
            return
        tmp = AGENTS_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"agents": self._agents_cache}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(AGENTS_FILE)
        self._agents_dirty = False

    def flush(self):
        """Fuerza escritura a disco de cualquier cambio pendiente."""
        with self._lock:
            self._save_tasks()
            self._save_agents()

    # -- tasks ----------------------------------------------------------

    def add_task(self, task: dict) -> dict:
        with self._lock:
            tasks = self._load_tasks()
            tasks.append(task)
            self._tasks_dirty = True
            self._save_tasks()
        return task

    def get_task(self, task_id: str) -> Optional[dict]:
        with self._lock:
            for t in self._load_tasks():
                if t["id"] == task_id:
                    return dict(t)  # copia para evitar mutaciones externas
        return None

    def update_task(self, task: dict):
        with self._lock:
            tasks = self._load_tasks()
            for i, t in enumerate(tasks):
                if t["id"] == task["id"]:
                    tasks[i] = task
                    self._tasks_dirty = True
                    break
            self._save_tasks()

    def remove_task(self, task_id: str) -> bool:
        with self._lock:
            tasks = self._load_tasks()
            new_tasks = [t for t in tasks if t["id"] != task_id]
            if len(new_tasks) == len(tasks):
                return False
            self._tasks_cache = new_tasks
            self._tasks_dirty = True
            self._save_tasks()
        return True

    def list_tasks(self, status: Optional[str] = None, tag: Optional[str] = None) -> list:
        with self._lock:
            tasks = list(self._load_tasks())  # copia
        if status:
            tasks = [t for t in tasks if t.get("status") == status]
        if tag:
            tasks = [t for t in tasks if tag in t.get("tags", [])]
        return tasks

    def find_tasks_by_prompt(self, prompt_substr: str) -> list:
        """Busca tareas cuyo prompt contenga el substring (útil para evitar duplicados)."""
        with self._lock:
            return [
                t for t in self._load_tasks()
                if prompt_substr.lower() in t.get("prompt", "").lower()
            ]

    def prune_old_tasks(self, max_age_days: int = PRUNE_AGE_DAYS) -> int:
        """Elimina tareas completed/failed con más de max_age_days días."""
        with self._lock:
            tasks = self._load_tasks()
            now = time.time()
            keep = []
            pruned = 0
            for t in tasks:
                if t.get("status") in ("completed", "failed") and t.get("completed_at"):
                    ts = _parse_ts(t["completed_at"])
                    if ts and (now - ts) > max_age_days * 86400:
                        pruned += 1
                        continue
                keep.append(t)
            if pruned:
                self._tasks_cache = keep
                self._tasks_dirty = True
                self._save_tasks()
        return pruned

    # -- agents ---------------------------------------------------------

    def add_agent(self, name: str, system_prompt: str):
        with self._lock:
            agents = self._load_agents()
            agents[name] = {"system_prompt": system_prompt}
            self._agents_dirty = True
            self._save_agents()

    def remove_agent(self, name: str) -> bool:
        with self._lock:
            agents = self._load_agents()
            if name not in agents:
                return False
            del agents[name]
            self._agents_dirty = True
            self._save_agents()
        return True

    def get_agent(self, name: str) -> Optional[dict]:
        with self._lock:
            return self._load_agents().get(name)

    def list_agents(self) -> list:
        with self._lock:
            return list(self._load_agents().items())

    # -- batch operations -----------------------------------------------

    def pause_all(self):
        """Marca todas las tareas 'pending' o 'review' como 'paused'."""
        with self._lock:
            tasks = self._load_tasks()
            changed = False
            for t in tasks:
                if t.get("status") in ("pending", "review"):
                    t["status"] = "paused"
                    changed = True
            if changed:
                self._tasks_dirty = True
                self._save_tasks()

    def resume_all(self):
        """Marca todas las tareas 'paused' como 'pending' (las review vuelven a review)."""
        with self._lock:
            tasks = self._load_tasks()
            changed = False
            for t in tasks:
                if t.get("status") == "paused":
                    # Si estaba en review antes de pausar, lo detectamos por reviewer_agent
                    if t.get("reviewer_agent") and t.get("result"):
                        t["status"] = "review"
                    else:
                        t["status"] = "pending"
                    changed = True
            if changed:
                self._tasks_dirty = True
                self._save_tasks()


# ── Excepción de ejecución ────────────────────────────────────────────
class ExecutionError(Exception):
    """Error durante la ejecución de una tarea (timeout, API error, etc.)."""


class RateLimitError(ExecutionError):
    """Error por rate limiting de la API."""


# ── Task Executor ─────────────────────────────────────────────────────
class TaskExecutor:
    """Ejecuta tareas usando tbot como librería (LLM + tools).

    Mejoras respecto a la versión original:
    - Cache de tools (no se reconstruyen en cada run)
    - Historial podado con contador acumulado O(n) en vez de O(n²)
    - Rate limiting con backoff exponencial
    - Evaluación mejorada con criterios
    - Métricas de ejecución
    """

    _tools_cache: Optional[list] = None
    _tools_cache_version = 0

    def __init__(self, cfg_override: Optional[dict] = None):
        self.cfg = cfg_override or tbot.load_cfg()
        tbot.resolve_key(self.cfg)
        self.cfg["tools_enabled"] = True
        self.cfg["trust_mode"] = True
        self.cfg["resp_color"] = "0"
        tbot._init_default_skills()

    def _build_tools(self) -> list:
        """Construye lista de tools con cache."""
        if self._tools_cache is not None:
            return self._tools_cache
        tools = [
            t for t in tbot.TOOLS
            if t["function"]["name"] not in _BLOCKED_TOOLS
        ]
        skills = tbot.load_skills()
        if skills:
            tools += tbot.skills_to_tools(skills)
        try:
            mcp_tools = tbot._get_mcp_tools()
            if mcp_tools:
                tools += mcp_tools
        except Exception:
            pass
        TaskExecutor._tools_cache = tools
        return tools

    @classmethod
    def invalidate_tools_cache(cls):
        cls._tools_cache = None

    def run(
        self,
        prompt: str,
        agent_name: Optional[str] = None,
        timeout: int = TASK_TIMEOUT,
        max_rounds: int = 200,
        evaluation_criteria: Optional[str] = None,
    ) -> dict:
        """Ejecuta un prompt como tarea LLM con acceso a tools.

        Returns:
            dict con: text, evaluation, success, rounds, elapsed, metrics
        Raises:
            ExecutionError: timeout, API error, etc.
        """
        start_time = time.time()
        result_holder: list = []
        error_holder: list = []

        def worker():
            try:
                result_holder.append(
                    self._run_inner(prompt, agent_name, max_rounds, evaluation_criteria)
                )
            except Exception as e:
                error_holder.append(e)

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        worker_thread.join(timeout)

        if worker_thread.is_alive():
            raise ExecutionError(
                f"Tiempo de ejecución agotado ({timeout}s). "
                "La tarea no completó a tiempo."
            )

        if error_holder:
            raise error_holder[0]

        result = result_holder[0]
        result["elapsed"] = round(time.time() - start_time, 2)
        return result

    def _run_inner(
        self,
        prompt: str,
        agent_name: Optional[str] = None,
        max_rounds: int = 200,
        evaluation_criteria: Optional[str] = None,
    ) -> dict:
        """Bucle interno de conversación."""
        base_sys = tbot.load_system_prompt(self.cfg)
        if agent_name:
            storage = Storage()
            agent_data = storage.get_agent(agent_name)
            if agent_data and agent_data.get("system_prompt"):
                extra = agent_data["system_prompt"]
                base_sys = (base_sys + "\n\n" + extra) if base_sys else extra

        messages: list = []
        if base_sys:
            messages.append({"role": "system", "content": base_sys})
        messages.append({"role": "user", "content": prompt})

        tools = self._build_tools()
        accumulated: list[str] = []
        tool_only_rounds = 0
        total_chars = sum(tbot._content_str_len(m.get("content", "")) for m in messages)
        total_api_calls = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        rate_limit_retries = 0

        for round_n in range(max_rounds):
            # ── Podar historial con contador acumulado O(n) ──
            max_chars = tbot._compute_max_history_chars(self.cfg)
            non_system_count = sum(1 for m in messages if m["role"] != "system")
            while total_chars > max_chars and non_system_count > 1:
                # Buscar primer user message
                for i, m in enumerate(messages):
                    if m["role"] == "user":
                        removed = messages.pop(i)
                        total_chars -= tbot._content_str_len(removed.get("content", ""))
                        non_system_count -= 1
                        # Eliminar assistant + tools que le siguen
                        while i < len(messages) and messages[i]["role"] in ("assistant", "tool"):
                            removed = messages.pop(i)
                            total_chars -= tbot._content_str_len(removed.get("content", ""))
                            non_system_count -= 1 if messages[i-1:i] or True else 0
                        break
                    elif m["role"] == "assistant":
                        # Si el primer non-system es assistant (caso borde), eliminarlo
                        removed = messages.pop(i)
                        total_chars -= tbot._content_str_len(removed.get("content", ""))
                        non_system_count -= 1
                        break

            # ── Llamar API ──
            try:
                api_result = tbot.chat_completion(
                    messages, self.cfg, stream=True, tools=tools
                )
            except Exception as e:
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str or "too many" in err_str:
                    rate_limit_retries += 1
                    if rate_limit_retries > 5:
                        raise RateLimitError(
                            f"Rate limit excedido tras {rate_limit_retries} reintentos"
                        )
                    wait = min(2 ** rate_limit_retries * 5, 120)
                    log.warning(
                        "Rate limit (intento %d/5), esperando %ds...",
                        rate_limit_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                raise ExecutionError(f"Error de API: {str(e)[:500]}")

            total_api_calls += 1

            if "error" in api_result:
                detail = api_result.get("detail", api_result.get("error", "unknown"))
                log.error("API error: %s", str(detail)[:300])
                raise ExecutionError(f"Error de API: {str(detail)[:500]}")

            # Registrar uso de tokens si está disponible
            usage = api_result.get("usage")
            if usage:
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)

            # ── Parsear stream ──
            buf = io.StringIO()
            with redirect_stdout(buf):
                parsed = tbot.parse_stream(
                    api_result["stream"], resp_color="0"
                )

            if parsed is None:
                continue

            content, reasoning, tool_calls, *_rest, interrupted = parsed

            if interrupted:
                log.warning("Stream interrumpido, reintentando...")
                continue

            # ── Tool calls ──
            if tool_calls:
                # Bloquear tools interactivas
                for tc in tool_calls:
                    if tc["function"]["name"] == "question":
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": (
                                "Error: La herramienta 'question' no está disponible "
                                "en modo agente no-interactivo. Resuelve la tarea sin "
                                "pedir información al usuario."
                            ),
                        })

                clean_calls = [
                    tc for tc in tool_calls
                    if tc["function"]["name"] not in _BLOCKED_TOOLS
                ]

                if not clean_calls:
                    continue

                if content:
                    tool_only_rounds = 0
                    log.info(
                        "  [r%d] Thinking: %s...",
                        round_n + 1, content[:150],
                    )
                else:
                    tool_only_rounds += 1
                    if tool_only_rounds >= tbot.MAX_TOOL_ONLY_ROUNDS:
                        messages.append({
                            "role": "system",
                            "content": (
                                f"LÍMITE: {tbot.MAX_TOOL_ONLY_ROUNDS} rondas sin respuesta "
                                "textual. DEBES responder ahora al usuario."
                            ),
                        })
                        break

                # IMPORTANTE: Añadir mensaje assistant con tool_calls ANTES de los tool results
                # El formato API requiere: user → assistant(tool_calls) → tool(result)
                assistant_msg = {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": tc.get("type", "function"),
                            "function": tc["function"],
                        }
                        for tc in clean_calls
                    ],
                }
                # Gemini 3+ thought_signature: preserve extra_content on tool_calls
                for i, tc in enumerate(clean_calls):
                    if "extra_content" in tc:
                        assistant_msg["tool_calls"][i]["extra_content"] = tc["extra_content"]
                messages.append(assistant_msg)

                # Ejecutar tools
                tc_buf = io.StringIO()
                with redirect_stdout(tc_buf):
                    tbot.execute_tool_calls(clean_calls, messages, self.cfg)
                    # Actualizar contador de chars después de tool calls
                    total_chars = sum(
                        tbot._content_str_len(m.get("content", "")) for m in messages
                    )

                names = ", ".join(tc["function"]["name"] for tc in clean_calls)
                log.info("  [r%d] Tools: %s", round_n + 1, names)
                continue

            # ── Respuesta textual final ──
            if content:
                accumulated.append(content)
                messages.append({"role": "assistant", "content": content})
                log.info("  [r%d] Response (%d chars)", round_n + 1, len(content))
                break

            log.info("  [r%d] Empty response, breaking", round_n + 1)
            break

        text = "\n".join(accumulated)
        success = bool(text.strip()) and tool_only_rounds < 60
        evaluation = self._evaluate(prompt, text, success, evaluation_criteria)

        return {
            "text": text,
            "evaluation": evaluation,
            "success": success,
            "rounds": round_n + 1,
            "api_calls": total_api_calls,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "rate_limit_retries": rate_limit_retries,
        }

    def _evaluate(
        self,
        prompt: str,
        result: str,
        simple_ok: bool,
        criteria: Optional[str] = None,
    ) -> str:
        """Evalúa el resultado de la tarea.

        Si se especificaron criterios, hace una evaluación semántica vía LLM.
        Caso contrario, usa reglas heurísticas.
        """
        if not result.strip():
            return "Sin output. La tarea no se ejecutó."

        if criteria:
            # Evaluación semántica con el propio LLM
            eval_prompt = textwrap.dedent(f"""\
            Evalúa si el siguiente resultado cumple con los criterios solicitados.

            CRITERIOS:
            {criteria}

            PROMPT ORIGINAL:
            {prompt[:2000]}

            RESULTADO:
            {result[:3000]}

            Responde solo con:
            ✅ o ❌ seguido de una breve explicación (máx 100 chars).
            """)
            try:
                eval_result = tbot.chat_completion(
                    [
                        {"role": "system", "content": "Eres un evaluador objetivo."},
                        {"role": "user", "content": eval_prompt},
                    ],
                    self.cfg,
                    stream=False,
                    tools=[],
                )
                if "error" not in eval_result:
                    content = eval_result["choices"][0]["message"]["content"]
                    return content.strip()[:200]
            except Exception:
                pass

        if simple_ok:
            return f"Completada ({len(result)} chars)"
        else:
            return "La tarea no produjo una respuesta satisfactoria."


# ── Daemon ────────────────────────────────────────────────────────────
class TaskRunner:
    """Daemon que hace polling de tareas listas y las ejecuta en paralelo.

    Mejoras:
    - Lock vía archivo con PID check + heartbeat
    - Shutdown graceful (espera tareas activas)
    - Pause/Resume global
    - Métricas por tarea
    """

    def __init__(self, poll_interval: int = DEFAULT_POLL, max_parallel: int = DEFAULT_PARALLEL):
        self.poll_interval = poll_interval
        self.max_parallel = max_parallel
        self.storage = Storage()
        self.executor = TaskExecutor()
        self._stop_event = threading.Event()
        self._paused_event = threading.Event()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._running = False
        self._active_futures: list = []
        self._lock_fd: Optional[int] = None

    # -- Lock management -----------------------------------------------

    def _acquire_lock(self) -> bool:
        """Adquiere lock del daemon vía archivo con flock (atómico en Linux)."""
        RUNNER_DIR.mkdir(parents=True, exist_ok=True)
        try:
            # Usar flock para lock atómico
            self._lock_fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o644)
            import fcntl
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                # Lock ocupado, verificar si es stale
                try:
                    pid_data = os.read(self._lock_fd, 32)
                    pid = int(pid_data.strip())
                    os.kill(pid, 0)
                    log.error("Daemon ya activo (pid %d)", pid)
                    os.close(self._lock_fd)
                    self._lock_fd = None
                    return False
                except (OSError, ValueError, ProcessLookupError):
                    log.warning("Lock stale, forzando adquisición...")
                    # PID muerto, podemos tomar el lock (flock se libera al cerrar fd)
                    pass
                return False

            # Escribir nuestro PID
            os.lseek(self._lock_fd, 0, os.SEEK_SET)
            os.write(self._lock_fd, f"{os.getpid()}\n".encode())
            os.truncate(self._lock_fd, os.lseek(self._lock_fd, 0, os.SEEK_CUR))
            ACTIVE_FILE.write_text(str(time.time()))
            return True
        except ImportError:
            # Fallback: mkdir lock si no hay fcntl (ej: Android sin fcntl.flock completo)
            try:
                os.mkdir(str(RUNNER_DIR / "daemon.lock_dir"))
                (RUNNER_DIR / "daemon.lock_dir" / "pid").write_text(str(os.getpid()))
                ACTIVE_FILE.write_text(str(time.time()))
                return True
            except FileExistsError:
                pid_path = RUNNER_DIR / "daemon.lock_dir" / "pid"
                if pid_path.exists():
                    try:
                        pid = int(pid_path.read_text().strip())
                        os.kill(pid, 0)
                        log.error("Daemon ya activo (pid %d)", pid)
                        return False
                    except (OSError, ValueError, ProcessLookupError):
                        shutil.rmtree(str(RUNNER_DIR / "daemon.lock_dir"), ignore_errors=True)
                        return self._acquire_lock()
                return False
        except Exception as e:
            log.error("Error adquiriendo lock: %s", e)
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            return False

    def _release_lock(self):
        if self._lock_fd is not None:
            try:
                import fcntl
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            except Exception:
                pass
            os.close(self._lock_fd)
            self._lock_fd = None
        LOCK_FILE.unlink(missing_ok=True)
        (RUNNER_DIR / "daemon.lock_dir").unlink(missing_ok=True)
        ACTIVE_FILE.unlink(missing_ok=True)

    # -- Heartbeat -----------------------------------------------------

    def _heartbeat_loop(self):
        while self._running:
            try:
                HEARTBEAT_FILE.write_text(str(time.time()))
            except OSError:
                pass
            time.sleep(30)

    # -- Start / Stop --------------------------------------------------

    def start(self) -> bool:
        if not self._acquire_lock():
            return False

        self._running = True
        PID_FILE.write_text(str(os.getpid()))

        # Heartbeat thread
        hb = threading.Thread(target=self._heartbeat_loop, daemon=True, name="heartbeat")
        hb.start()

        # Podar tareas viejas al arrancar
        try:
            pruned = self.storage.prune_old_tasks()
            if pruned:
                log.info("Podadas %d tareas antiguas", pruned)
        except Exception as e:
            log.warning("Error pruning tasks: %s", e)

        log.info(
            "Daemon iniciado (poll=%ds, parallel=%d, model=%s)",
            self.poll_interval,
            self.max_parallel,
            self.executor.cfg.get("model", "?"),
        )

        self._pool = ThreadPoolExecutor(
            max_workers=self.max_parallel,
            thread_name_prefix="agent-worker",
        )

        signal.signal(signal.SIGINT, lambda s, f: self.stop())
        signal.signal(signal.SIGTERM, lambda s, f: self.stop())

        try:
            while not self._stop_event.is_set():
                self._process_ready_tasks()
                self._stop_event.wait(self.poll_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

        return True

    def stop(self):
        log.info("Deteniendo daemon...")
        self._stop_event.set()

    def pause(self):
        """Pausa el procesamiento de nuevas tareas."""
        self._paused_event.set()
        PAUSED_FILE.write_text(_ts())
        self.storage.pause_all()
        log.info("Daemon pausado - no se procesarán nuevas tareas")

    def resume(self):
        """Reanuda el procesamiento de tareas."""
        self._paused_event.clear()
        PAUSED_FILE.unlink(missing_ok=True)
        self.storage.resume_all()
        log.info("Daemon reanudado")

    @property
    def is_paused(self) -> bool:
        # Check both in-memory event AND .paused file (for external pause via tbot tools)
        if self._paused_event.is_set():
            return True
        if PAUSED_FILE.exists():
            # Sync the event with the file
            self._paused_event.set()
            return True
        return False

    def _shutdown(self):
        self._running = False
        if self._pool:
            log.info("Esperando hasta 30s a que terminen tareas activas...")
            self._pool.shutdown(wait=True, timeout=30)
        self._release_lock()
        PID_FILE.unlink(missing_ok=True)
        HEARTBEAT_FILE.unlink(missing_ok=True)
        log.info("Daemon detenido")

    # -- Procesamiento -------------------------------------------------

    def _process_ready_tasks(self):
        if self.is_paused:
            return

        now = _now_epoch()

        # Recoger tareas pending (para ejecutar) y review (para revisar)
        pending_tasks = self.storage.list_tasks(status="pending")
        review_tasks = self.storage.list_tasks(status="review")
        tasks = pending_tasks + review_tasks

        ready: list[dict] = []
        for t in tasks:
            nxt = _parse_ts(t.get("next_run"))
            if nxt is not None and nxt <= now:
                ready.append(t)

        if not ready:
            return

        ready = ready[: self.max_parallel]
        log.info("%d tarea(s) lista(s) para ejecutar", len(ready))

        futures = {}
        for task in ready:
            future = self._pool.submit(self._execute_task, task)
            futures[future] = task

        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except Exception as e:
                log.error("Error en tarea %s: %s", task.get("id", "?")[:12], e)
                log.debug(traceback.format_exc())

    def _execute_task(self, task: dict):
        tid = task["id"][:12]
        ttype = task.get("type", "onehot")
        prompt = task.get("prompt", "")
        agent_name = task.get("agent")
        max_retries = task.get("max_retries", 3)
        interval_secs = task.get("interval_secs", 300)
        old_retries = task.get("retries", 0)
        tags = task.get("tags", [])
        eval_criteria = task.get("evaluation_criteria")
        reviewer_agent = task.get("reviewer_agent")
        original_status = task.get("status", "pending")

        log.info(
            "Procesando %s (%s) [%s] intento %d/%d",
            tid, ttype, original_status, old_retries + 1, max_retries,
        )

        # ── Marcar running ──
        task["status"] = "running"
        try:
            self.storage.update_task(task)
        except Exception as e:
            log.error("Error actualizando estado de %s: %s", tid, e)

        now_epoch = _now_epoch()

        # ── FASE DE EJECUCIÓN (doer) ──
        if original_status == "pending":
            # Incluir feedback de revisiones anteriores si las hay
            effective_prompt = prompt
            if task.get("review_evaluation"):
                feedback = (
                    f"\n\n--- FEEDBACK DE REVISIÓN ANTERIOR ---\n"
                    f"{task['review_evaluation']}\n"
                    f"--- TU RESULTADO ANTERIOR ---\n"
                    f"{task.get('result', '')[:1000]}\n"
                    f"---\n\n"
                    f"Por favor corrige los problemas indicados arriba."
                )
                effective_prompt = prompt + feedback

            # ── Ejecutar ──
            try:
                result = self.executor.run(
                    effective_prompt,
                    agent_name,
                    timeout=TASK_TIMEOUT,
                    evaluation_criteria=eval_criteria,
                )
                result_text = result["text"]
                evaluation = result["evaluation"]
                success = result["success"]
                rounds = result["rounds"]
            except ExecutionError as e:
                result_text = ""
                evaluation = f"Error: {e}"
                success = False
                rounds = 0
            except Exception as e:
                result_text = ""
                evaluation = f"Error inesperado: {e}"
                success = False
                rounds = 0
                log.error("Excepción no capturada en %s: %s", tid, traceback.format_exc())

            # ── Gatekeeper: evaluation_criteria puede anular success ──
            if eval_criteria and evaluation:
                if evaluation.strip().startswith("❌"):
                    log.info(
                        "%s Evaluación falló (❌): %s",
                        tid, evaluation[:120],
                    )
                    success = False

            # Guardar resultado
            task["result"] = result_text[:MAX_RESULT_CHARS] if result_text else ""
            task["evaluation"] = evaluation
            task["last_run"] = _ts()
            task["last_rounds"] = rounds

            # ── Determinar nuevo estado (fase doer) ──
            if success:
                if reviewer_agent:
                    # Pasar a revisión en lugar de completar
                    task["status"] = "review"
                    task["retries"] = old_retries  # mantener retries previos
                    task["completed_at"] = None
                    task["next_run"] = _ts()  # disponible inmediatamente
                    log.info(
                        "%s Ejecución OK — pasando a revisión por '%s'",
                        tid, reviewer_agent,
                    )
                elif ttype == "idle":
                    task["status"] = "pending"
                    task["retries"] = 0
                    task["completed_at"] = None
                    task["next_run"] = _fmt_epoch(now_epoch + interval_secs)
                    log.info(
                        "%s Completada - reprogramando idle en %ds",
                        tid, interval_secs,
                    )
                    self._notify(task["id"], "Tarea completada", prompt[:120])
                    self._bell()
                else:
                    task["status"] = "completed"
                    task["retries"] = old_retries
                    task["next_run"] = None
                    task["completed_at"] = _ts()
                    log.info("%s Tarea completada (%d rounds)", tid, rounds)
                    self._notify(task["id"], "Tarea completada", prompt[:120])
                    self._bell()
            else:
                attempt = old_retries + 1
                if attempt >= max_retries:
                    if ttype == "idle":
                        task["status"] = "pending"
                        task["retries"] = 0
                        task["completed_at"] = None
                        backoff = min(interval_secs * 4, 86400)
                        task["next_run"] = _fmt_epoch(now_epoch + backoff)
                        log.warning(
                            "%s Idle falló tras %d intentos, backoff %ds",
                            tid, attempt, backoff,
                        )
                    else:
                        task["status"] = "failed"
                        task["retries"] = attempt
                        task["next_run"] = None
                        task["completed_at"] = _ts()
                        log.error(
                            "%s Tarea falló tras %d intentos",
                            tid, attempt,
                        )
                    self._notify(
                        task["id"], "Tarea falló",
                        f"{prompt[:120]} ({attempt}/{max_retries})",
                    )
                    self._bell()
                else:
                    task["status"] = "pending"
                    task["retries"] = attempt
                    task["completed_at"] = None
                    retry_delay = max(60, interval_secs) // 2
                    task["next_run"] = _fmt_epoch(now_epoch + retry_delay)
                    log.info(
                        "%s Reintento %d/%d en %ds",
                        tid, attempt, max_retries, retry_delay,
                    )

        # ── FASE DE REVISIÓN (reviewer) ──
        elif original_status == "review":
            if not reviewer_agent:
                # Sin revisor configurado → completar directamente
                task["status"] = "completed"
                task["completed_at"] = _ts()
                task["next_run"] = None
                log.warning(
                    "%s En review pero sin reviewer_agent → completando",
                    tid,
                )
                self._notify(task["id"], "Tarea completada", prompt[:120])
                self._bell()
            else:
                # Construir prompt de revisión
                review_prompt_text = task.get("review_prompt") or _DEFAULT_REVIEW_PROMPT
                review_prompt_actual = review_prompt_text.format(
                    prompt=prompt,
                    result=task.get("result", "")[:3000],
                )

                try:
                    review_result = self.executor.run(
                        review_prompt_actual,
                        reviewer_agent,
                        timeout=TASK_TIMEOUT,
                        evaluation_criteria=None,
                    )
                    review_text = review_result["text"]
                    review_success = review_result["success"]
                    review_rounds = review_result["rounds"]
                except ExecutionError as e:
                    review_text = ""
                    review_success = False
                    review_rounds = 0
                    log.error("%s Error en revisión: %s", tid, e)
                except Exception as e:
                    review_text = ""
                    review_success = False
                    review_rounds = 0
                    log.error(
                        "%s Excepción en revisión: %s",
                        tid, traceback.format_exc(),
                    )

                # Guardar resultado de revisión
                task["review_result"] = review_text[:MAX_RESULT_CHARS] if review_text else ""
                task["last_run"] = _ts()
                task["last_rounds"] = review_rounds

                # Determinar si la revisión fue exitosa
                review_passed = (
                    review_success
                    and review_text.strip()
                    and review_text.strip().startswith("✅")
                )

                if review_passed:
                    task["review_evaluation"] = review_text.strip()
                    task["status"] = "completed"
                    task["retries"] = old_retries
                    task["next_run"] = None
                    task["completed_at"] = _ts()
                    log.info("%s Revisión OK — tarea completada", tid)
                    self._notify(task["id"], "Tarea completada", prompt[:120])
                    self._bell()
                else:
                    # Extraer feedback para el doer
                    review_eval_text = review_text.strip() if review_text.strip() else "❌ Revisión no produjo resultado"
                    # Si el reviewer no empezó con ❌, forzamos❌ para el feedback loop
                    if not review_eval_text.startswith("❌") and not review_eval_text.startswith("✅"):
                        review_eval_text = "❌ " + review_eval_text[:200]
                    task["review_evaluation"] = review_eval_text

                    attempt = old_retries + 1
                    if attempt >= max_retries:
                        task["status"] = "failed"
                        task["retries"] = attempt
                        task["next_run"] = None
                        task["completed_at"] = _ts()
                        log.error(
                            "%s Revisión falló tras %d intentos — tarea fallida",
                            tid, attempt,
                        )
                        self._notify(
                            task["id"], "Tarea falló",
                            f"{prompt[:120]} (revisión {attempt}/{max_retries})",
                        )
                        self._bell()
                    else:
                        task["status"] = "pending"  # volver a doer con feedback
                        task["retries"] = attempt
                        task["completed_at"] = None
                        retry_delay = max(60, interval_secs) // 2
                        task["next_run"] = _fmt_epoch(now_epoch + retry_delay)
                        log.info(
                            "%s Revisión falló — reintento %d/%d en %ds",
                            tid, attempt, max_retries, retry_delay,
                        )

        try:
            self.storage.update_task(task)
        except Exception as e:
            log.error("Error guardando resultado de %s: %s", tid, e)
        log.info("──────────────────────────────────────────")

    def _notify(self, task_id: str, title: str, message: str):
        """Notificación Termux con sonido."""
        try:
            subprocess.run(
                [
                    "termux-notification",
                    "--id", f"tbot-agent-{task_id}",
                    "--title", title,
                    "--content", message[:200],
                    "--priority", "high",
                    "--vibrate", "0,200,100,200",
                    "--action", "am start -n com.termux/.TermuxActivity",
                ],
                timeout=5,
                capture_output=True,
            )
        except Exception:
            pass

        sound_setting = os.environ.get("TERMUX_NOTIFICATION_SOUND", "on")
        if sound_setting == "off":
            return

        try:
            if sound_setting and sound_setting != "on" and Path(sound_setting).is_file():
                sound_file = sound_setting
            else:
                sound_file = "/system/media/audio/notifications/NotificationXylophone.ogg"

            vol_file = RUNNER_DIR / "notif_sound.ogg"
            needs_rebuild = not vol_file.exists() or (
                Path(sound_file).is_file()
                and Path(sound_file).stat().st_mtime > vol_file.stat().st_mtime
            )

            if needs_rebuild and shutil.which("ffmpeg"):
                subprocess.run(
                    ["ffmpeg", "-i", sound_file, "-af", "volume=0.2", "-y", str(vol_file)],
                    timeout=10,
                    capture_output=True,
                )

            play_file = str(vol_file) if vol_file.exists() else sound_file
            if shutil.which("termux-media-player"):
                subprocess.run(
                    ["termux-media-player", "play", play_file],
                    timeout=5,
                    capture_output=True,
                )
        except Exception:
            pass

    def _bell(self):
        """Emit terminal bell (\a) para alertar al usuario."""
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass


# ── CLI ───────────────────────────────────────────────────────────────
def cmd_add(args: argparse.Namespace):
    """Crear una nueva tarea."""
    if args.type not in ("onehot", "idle"):
        print("Error: type debe ser onehot o idle")
        sys.exit(1)
    if not args.prompt:
        print("Error: Se requiere --prompt")
        sys.exit(1)

    storage = Storage()

    # Validar que el agente existe si se especificó
    if args.agent and not storage.get_agent(args.agent):
        print(f"Error: El agente '{args.agent}' no existe. Créalo con 'runner.py agent-add --name ...'")
        sys.exit(1)

    # Validar que el reviewer_agent existe si se especificó
    if args.reviewer_agent and not storage.get_agent(args.reviewer_agent):
        print(f"Error: El agente revisor '{args.reviewer_agent}' no existe. Créalo con 'runner.py agent-add --name ...'")
        sys.exit(1)

    # Detectar duplicados (mismo prompt exacto ya pending)
    if args.skip_duplicates:
        existing = storage.find_tasks_by_prompt(args.prompt)
        existing = [t for t in existing if t.get("status") in ("pending", "running", "review")]
        if existing:
            print(f"Aviso: Ya existe una tarea similar ({existing[0]['id'][:12]}), omitiendo.")
            return

    task = {
        "id": _new_id(),
        "type": args.type,
        "status": "paused" if args.paused else "pending",
        "prompt": args.prompt,
        "agent": args.agent,
        "interval_secs": args.interval,
        "last_run": None,
        "next_run": _ts(),
        "created_at": _ts(),
        "completed_at": None,
        "result": None,
        "evaluation": None,
        "retries": 0,
        "max_retries": args.retries,
        "tags": args.tags or [],
        "evaluation_criteria": args.eval_criteria,
        "reviewer_agent": args.reviewer_agent,
        "review_prompt": args.review_prompt,
        "review_result": None,
        "review_evaluation": None,
    }

    storage.add_task(task)
    print(f"Tarea creada: {task['id']}")
    print(f"   Tipo:    {task['type']}")
    print(f"   Prompt:  {task['prompt'][:80]}...")
    if args.agent:
        print(f"   Agent:   {args.agent}")
    if args.type == "idle":
        print(f"   Cada:    {args.interval}s")
    if args.tags:
        print(f"   Tags:    {', '.join(args.tags)}")
    if args.eval_criteria:
        print(f"   Criterio: {args.eval_criteria[:60]}...")
    if args.reviewer_agent:
        print(f"   Revisor:  {args.reviewer_agent}")
        if args.review_prompt:
            print(f"   Review prompt: {args.review_prompt[:60]}...")
        else:
            print(f"   Review prompt: (default)")
    print()
    print("   Para ejecutar el runner:   ./runner.py run")


def cmd_list(args: argparse.Namespace):
    """Listar tareas."""
    storage = Storage()
    status_filter = args.status  # puede venir de --status o posicional
    tasks = storage.list_tasks(status=status_filter, tag=args.tag)

    if args.json:
        print(json.dumps(tasks, indent=2, ensure_ascii=False))
        return

    print(f"Tareas: {len(tasks)}")
    if not tasks:
        return
    print()
    line = "{:<28} {:<8} {:<10} {:<12} {:<8} {}"
    print(line.format("ID", "TIPO", "ESTADO", "AGENT", "RETRIES", "PROMPT"))
    print(line.format("─" * 28, "─" * 8, "─" * 10, "─" * 12, "─" * 8, "─" * 40))
    for t in tasks:
        tid = t["id"]
        ttype = t.get("type", "?")
        status = t.get("status", "?")
        agent = t.get("agent") or "-"
        retries = f"{t.get('retries', 0)}/{t.get('max_retries', 3)}"
        prompt = t.get("prompt", "")[:60]
        print(line.format(tid, ttype, status, agent, retries, prompt))
    print()
    print("Estados: pending | running | review | paused | completed | failed")


def cmd_show(args: argparse.Namespace):
    """Mostrar detalle de una tarea."""
    task = Storage().get_task(args.task_id)
    if not task:
        print(f"Tarea no encontrada: {args.task_id}")
        sys.exit(1)

    if args.json:
        print(json.dumps(task, indent=2, ensure_ascii=False))
        return

    for k, v in task.items():
        if k == "result" and v and len(str(v)) > 500:
            print(f"{k}: {str(v)[:500]}...")
        elif k == "result" and v:
            print(f"{k}: {v}")
        elif k == "result":
            print(f"{k}: (sin resultado)")
        else:
            print(f"{k}: {v}")


def cmd_remove(args: argparse.Namespace):
    """Eliminar una tarea."""
    ok = Storage().remove_task(args.task_id)
    if ok:
        print(f"Tarea eliminada: {args.task_id}")
    else:
        print(f"Tarea no encontrada: {args.task_id}")


def cmd_edit(args: argparse.Namespace):
    """Modificar campos de una tarea existente."""
    storage = Storage()
    task = storage.get_task(args.task_id)
    if not task:
        print(f"Tarea no encontrada: {args.task_id}")
        sys.exit(1)

    changed = []
    if args.prompt is not None:
        task["prompt"] = args.prompt
        changed.append("prompt")
    if args.interval is not None:
        task["interval_secs"] = args.interval
        changed.append("interval")
    if args.retries is not None:
        task["max_retries"] = args.retries
        changed.append("max_retries")
    if args.agent is not None:
        if args.agent and not storage.get_agent(args.agent):
            print(f"Error: El agente '{args.agent}' no existe.")
            sys.exit(1)
        task["agent"] = args.agent
        changed.append("agent")
    if args.status:
        if args.status in _VALID_STATUSES:
            task["status"] = args.status
            changed.append("status")
        else:
            print(f"Error: Estado inválido '{args.status}'. Válidos: {', '.join(sorted(_VALID_STATUSES))}")
            sys.exit(1)
    if args.tags is not None:
        task["tags"] = args.tags
        changed.append("tags")
    if args.eval_criteria is not None:
        task["evaluation_criteria"] = args.eval_criteria
        changed.append("evaluation_criteria")
    if args.reviewer_agent is not None:
        if args.reviewer_agent and not storage.get_agent(args.reviewer_agent):
            print(f"Error: El agente revisor '{args.reviewer_agent}' no existe.")
            sys.exit(1)
        task["reviewer_agent"] = args.reviewer_agent
        changed.append("reviewer_agent")
    if args.review_prompt is not None:
        task["review_prompt"] = args.review_prompt
        changed.append("review_prompt")

    if not changed:
        print("No se especificaron cambios. Usa --prompt, --interval, --retries, --agent, --status, --tags, --eval-criteria, --reviewer-agent, --review-prompt")
        sys.exit(1)

    storage.update_task(task)
    print(f"Tarea {args.task_id[:12]} actualizada: {', '.join(changed)}")


def cmd_exec(args: argparse.Namespace):
    """Ejecutar una tarea específica (una sola vez, sin loop)."""
    task = Storage().get_task(args.task_id)
    if not task:
        print(f"Tarea no encontrada: {args.task_id}")
        sys.exit(1)

    prompt = task.get("prompt", "")
    agent = task.get("agent")
    eval_criteria = task.get("evaluation_criteria")
    print(f"Ejecutando tarea {args.task_id[:12]} ({task.get('type', '?')})")

    executor = TaskExecutor()
    try:
        result = executor.run(
            prompt, agent,
            timeout=args.timeout,
            max_rounds=args.max_rounds,
            evaluation_criteria=eval_criteria,
        )
    except ExecutionError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print()
    print("═" * 60)
    print("RESULTADO:")
    print("═" * 60)
    print(result["text"][:2000] if result["text"] else "(sin output)")
    if len(result.get("text", "")) > 2000:
        print(f"... ({len(result['text'])} chars totales)")
    print("═" * 60)
    print()
    print(f"Evaluacion: {result['evaluation']}")
    print(f"Rondas:     {result['rounds']}")
    print(f"Elapsed:    {result.get('elapsed', '?')}s")
    print(f"API calls:  {result.get('api_calls', '?')}")
    if result.get("prompt_tokens"):
        print(f"Tokens:     {result['prompt_tokens']} prompt + {result['completion_tokens']} completion")

    runner_inst = TaskRunner()
    runner_inst._notify(
        task["id"],
        "Completada" if result["success"] else "Falló",
        prompt[:100],
    )


def cmd_run(args: argparse.Namespace):
    """Iniciar el runner (bucle de polling y ejecución)."""
    runner = TaskRunner(
        poll_interval=args.interval,
        max_parallel=args.max_parallel,
    )
    if not runner.start():
        print("Error: No se pudo iniciar el runner (ya esta corriendo?)")
        sys.exit(1)


def cmd_status(args: argparse.Namespace):
    """Estado del runner y tareas."""
    storage = Storage()

    if args.json:
        info = {"runner": {"active": False}, "tasks": {}, "agents": 0}
        if PID_FILE.exists():
            try:
                pid = int(PID_FILE.read_text().strip())
                os.kill(pid, 0)
                info["runner"] = {"active": True, "pid": pid}
            except (OSError, ValueError, ProcessLookupError):
                pass
        tasks = storage.list_tasks()
        info["tasks"] = {"total": len(tasks)}
        for status in sorted(_VALID_STATUSES):
            info["tasks"][status] = len([t for t in tasks if t.get("status") == status])
        info["agents"] = len(storage.list_agents())
        print(json.dumps(info, indent=2))
        return

    print("Estado del task runner")
    print()

    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            print(f"   Runner: activo (pid {pid})")
        except (OSError, ValueError, ProcessLookupError):
            print(f"   Runner: no activo (pid obsoleto)")
    else:
        print(f"   Runner: no activo")

    if PAUSED_FILE.exists():
        print(f"   Runner: PAUSADO")

    # Heartbeat
    if HEARTBEAT_FILE.exists():
        try:
            hb_time = float(HEARTBEAT_FILE.read_text().strip())
            age = time.time() - hb_time
            if age < 120:
                print(f"   Heartbeat: ok ({age:.0f}s ago)")
            else:
                print(f"   Heartbeat: ? ({age:.0f}s ago, puede estar colgado)")
        except (OSError, ValueError):
            pass

    print()

    tasks = storage.list_tasks()
    total = len(tasks)
    counts = {}
    for status in sorted(_VALID_STATUSES):
        counts[status] = len([t for t in tasks if t.get("status") == status])

    print(f"   Total:     {total}")
    for status, count in counts.items():
        print(f"   {status.capitalize():>10}: {count}")
    print()

    agents = storage.list_agents()
    print(f"   Agentes:   {len(agents)}")


def cmd_logs(args: argparse.Namespace):
    """Mostrar logs del runner."""
    if not LOG_FILE.exists():
        print("No hay logs disponibles.")
        return

    if args.follow:
        try:
            subprocess.run(["tail", "-f", str(LOG_FILE)], timeout=args.timeout)
        except subprocess.TimeoutExpired:
            pass
        except FileNotFoundError:
            print("tail no disponible, usando modo no-follow")
            args.follow = False
        except KeyboardInterrupt:
            pass

    if not args.follow:
        with open(LOG_FILE) as f:
            lines = f.readlines()
        tail_lines = lines[-args.lines:] if args.lines < len(lines) else lines
        print("".join(tail_lines))


def cmd_pause(args: argparse.Namespace):
    """Pausar el runner o tareas."""
    if args.daemon:
        # Enviar señal al daemon vía archivo de pause
        PAUSED_FILE.write_text(_ts())
        storage = Storage()
        storage.pause_all()
        print("Runner pausado via señal. Las tareas pendientes se reanudaran al hacer resume.")
        print("Ejecuta: ./runner.py resume")
    else:
        storage = Storage()
        task = storage.get_task(args.task_id)
        if not task:
            print(f"Tarea no encontrada: {args.task_id}")
            sys.exit(1)
        if task.get("status") != "pending":
            print(f"La tarea esta en estado '{task.get('status')}', no se puede pausar.")
            sys.exit(1)
        task["status"] = "paused"
        storage.update_task(task)
        print(f"Tarea {args.task_id[:12]} pausada.")


def cmd_resume(args: argparse.Namespace):
    """Reanudar el runner o tareas."""
    if args.daemon or not args.task_id:
        PAUSED_FILE.unlink(missing_ok=True)
        storage = Storage()
        storage.resume_all()
        print("Runner reanudado. Las tareas pausadas volveran a ejecutarse.")
    else:
        storage = Storage()
        task = storage.get_task(args.task_id)
        if not task:
            print(f"Tarea no encontrada: {args.task_id}")
            sys.exit(1)
        if task.get("status") != "paused":
            print(f"La tarea esta en estado '{task.get('status')}', no esta pausada.")
            sys.exit(1)
        task["status"] = "pending"
        task["next_run"] = _ts()  # ejecutar pronto
        storage.update_task(task)
        print(f"Tarea {args.task_id[:12]} reanudada.")


def cmd_agents(args: argparse.Namespace):
    """Listar agentes configurados."""
    agents = Storage().list_agents()

    if args.json:
        data = {name: info for name, info in agents}
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    if not agents:
        print("No hay agentes configurados.")
        return
    for name, data in agents:
        sys_prompt = data.get("system_prompt", "")
        print(f"{name}")
        print(f"   System prompt: {sys_prompt[:100]}...")
        print()


def cmd_agent_add(args: argparse.Namespace):
    """Crear un agente con system prompt personalizado."""
    if not args.name:
        print("Error: Se requiere --name")
        sys.exit(1)
    if not args.system:
        print("Error: Se requiere --system")
        sys.exit(1)
    Storage().add_agent(args.name, args.system)
    print(f"Agente '{args.name}' creado.")


def cmd_agent_rm(args: argparse.Namespace):
    """Eliminar un agente."""
    if Storage().remove_agent(args.name):
        print(f"Agente '{args.name}' eliminado.")
    else:
        print(f"Agente '{args.name}' no encontrado.")


def cmd_flush(args: argparse.Namespace):
    """Forzar escritura de cache a disco."""
    Storage().flush()
    print("Cache escrito a disco.")


def cmd_boot_install(args: argparse.Namespace):
    """Instalar auto-inicio del runner en Termux."""
    runner_py = Path(__file__).resolve()
    home = Path.home()
    method = args.method

    results = []

    # ── Termux:Boot ──
    if method in ("termux-boot", "both"):
        boot_dir = home / ".termux" / "boot"
        boot_dir.mkdir(parents=True, exist_ok=True)
        boot_script = boot_dir / "tbot-runner"
        boot_script.write_text(
            f"""#!/data/data/com.termux/files/usr/bin/bash
# tbot-runner auto-start — installed by runner.py boot-install
# Inicia el runner en background si no está ya corriendo

PID_FILE="{RUNNER_DIR / 'runner.pid'}"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        exit 0  # ya está corriendo
    fi
fi

cd "{runner_py.parent}"
exec python3 "{runner_py}" run --interval 10 &
"""
        )
        boot_script.chmod(0o755)
        results.append(f"Termux:Boot script creado: {boot_script}")

    # ── bashrc ──
    if method in ("bashrc", "both"):
        bashrc = home / ".bashrc"
        marker = "# --- tbot-runner auto-start ---"
        startup_block = f"""
{marker}
if [ -z "$TBOT_RUNNER_DISABLE" ] && command -v python3 >/dev/null 2>&1; then
    PID_FILE="{RUNNER_DIR / 'runner.pid'}"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ! kill -0 "$PID" 2>/dev/null; then
            rm -f "$PID_FILE"
            cd "{runner_py.parent}" && nohup python3 "{runner_py}" run --interval 10 > "{RUNNER_DIR / 'runner-nohup.log'}" 2>&1 &
        fi
    else
        cd "{runner_py.parent}" && nohup python3 "{runner_py}" run --interval 10 > "{RUNNER_DIR / 'runner-nohup.log'}" 2>&1 &
    fi
fi
# --- end tbot-runner auto-start ---
"""
        if bashrc.exists():
            content = bashrc.read_text()
            if marker in content:
                results.append("bashrc: ya tiene auto-start (actualizado)")
                # Replace existing block
                lines = content.split("\n")
                new_lines = []
                skip = False
                for line in lines:
                    if line.strip() == marker:
                        skip = True
                        new_lines.append(line)
                        continue
                    if skip and "# --- end tbot-runner auto-start ---" in line:
                        skip = False
                        new_lines.append(startup_block.strip())
                        continue
                    if not skip:
                        new_lines.append(line)
                bashrc.write_text("\n".join(new_lines))
                results.append("bashrc: bloque reemplazado")
            else:
                with open(str(bashrc), "a") as f:
                    f.write(startup_block)
                results.append(f"bashrc: auto-start añadido a {bashrc}")
        else:
            bashrc.write_text(startup_block.strip() + "\n")
            results.append(f"bashrc: creado {bashrc} con auto-start")

    for r in results:
        print(f"  ✓ {r}")
    print()
    print("Auto-start instalado. El runner arrancará automáticamente al abrir Termux.")
    if method in ("termux-boot", "both"):
        print("  (Requiere Termux:Boot instalado desde F-Droid para arranque al boot del sistema)")
    print()
    print(f"Para desactivar: export TBOT_RUNNER_DISABLE=1")
    print(f"Para desinstalar:  rm -f ~/.termux/boot/tbot-runner  (y editar ~/.bashrc)")


# ── main ──────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="tbot Task Runner - ejecucion autonoma de tareas con pipeline doer+reviewer",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Logs detallados")

    sub = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # add
    p_add = sub.add_parser("add", help="Crear una tarea")
    p_add.add_argument("--type", choices=["onehot", "idle"], required=True)
    p_add.add_argument("--prompt", required=True)
    p_add.add_argument("--agent")
    p_add.add_argument("--interval", type=int, default=300, help="Intervalo en segundos (para tipo idle)")
    p_add.add_argument("--retries", type=int, default=3, help="Maximo de reintentos")
    p_add.add_argument("--tags", nargs="*", default=[], help="Tags para filtrar")
    p_add.add_argument("--eval-criteria", help="Criterios de evaluacion personalizados")
    p_add.add_argument("--reviewer-agent", help="Nombre del agente revisor (activa fase de revision con herramientas)")
    p_add.add_argument("--review-prompt", help="Prompt personalizado para la revision (default: generico con pruebas)")
    p_add.add_argument("--paused", action="store_true", help="Crear en estado paused")
    p_add.add_argument("--skip-duplicates", action="store_true", help="Omitir si ya existe tarea similar")

    # list
    p_list = sub.add_parser("list", help="Listar tareas")
    p_list.add_argument("--status", choices=sorted(_VALID_STATUSES),
                        help="Filtrar por estado")
    p_list.add_argument("--tag", help="Filtrar por tag")
    p_list.add_argument("--json", action="store_true", help="Salida en JSON")
    p_list.add_argument("status_pos", nargs="?", metavar="status",
                        help="Filtro posicional (shortcut para --status)")

    # show
    p_show = sub.add_parser("show", help="Mostrar detalle de tarea")
    p_show.add_argument("task_id")
    p_show.add_argument("--json", action="store_true", help="Salida en JSON")

    # remove
    p_rm = sub.add_parser("remove", help="Eliminar tarea")
    p_rm.add_argument("task_id")

    # edit
    p_edit = sub.add_parser("edit", help="Modificar tarea existente")
    p_edit.add_argument("task_id")
    p_edit.add_argument("--prompt")
    p_edit.add_argument("--interval", type=int)
    p_edit.add_argument("--retries", type=int)
    p_edit.add_argument("--agent")
    p_edit.add_argument("--status", choices=sorted(_VALID_STATUSES))
    p_edit.add_argument("--tags", nargs="*")
    p_edit.add_argument("--eval-criteria")
    p_edit.add_argument("--reviewer-agent")
    p_edit.add_argument("--review-prompt")

    # exec
    p_exec = sub.add_parser("exec", help="Ejecutar tarea (una vez, sin loop)")
    p_exec.add_argument("task_id")
    p_exec.add_argument("--timeout", type=int, default=TASK_TIMEOUT)
    p_exec.add_argument("--max-rounds", type=int, default=200)

    # run (loop principal)
    p_run = sub.add_parser("run", help="Iniciar el runner (bucle de polling + ejecución)")
    p_run.add_argument("--interval", type=int, default=DEFAULT_POLL)
    p_run.add_argument("--max-parallel", type=int, default=DEFAULT_PARALLEL)

    # status
    p_status = sub.add_parser("status", help="Estado del runner y tareas")
    p_status.add_argument("--json", action="store_true", help="Salida en JSON")

    # logs
    p_logs = sub.add_parser("logs", help="Mostrar logs del runner")
    p_logs.add_argument("-f", "--follow", action="store_true", help="Hacer tail -f")
    p_logs.add_argument("-n", "--lines", type=int, default=50, help="Numero de lineas (default: 50)")
    p_logs.add_argument("--timeout", type=int, default=30, help="Timeout para follow mode (s)")

    # pause
    p_pause = sub.add_parser("pause", help="Pausar tarea o runner")
    p_pause.add_argument("task_id", nargs="?", help="ID de la tarea a pausar (omite para pausar runner)")
    p_pause.add_argument("--daemon", action="store_true", help="Pausar el runner completo")

    # resume
    p_resume = sub.add_parser("resume", help="Reanudar tarea o runner")
    p_resume.add_argument("task_id", nargs="?", help="ID de la tarea a reanudar (omite para reanudar runner)")
    p_resume.add_argument("--daemon", action="store_true", help="Reanudar el runner completo")

    # agents
    p_agents = sub.add_parser("agents", help="Listar agentes configurados")
    p_agents.add_argument("--json", action="store_true", help="Salida en JSON")

    # agent-add
    p_agent_add = sub.add_parser("agent-add", help="Crear agente")
    p_agent_add.add_argument("--name", required=True)
    p_agent_add.add_argument("--system", required=True)

    # agent-rm
    p_agent_rm = sub.add_parser("agent-rm", help="Eliminar agente")
    p_agent_rm.add_argument("name")

    # flush
    sub.add_parser("flush", help="Forzar escritura de cache a disco")

    # boot-install
    p_boot = sub.add_parser("boot-install", help="Instalar auto-inicio del runner al arrancar Termux")
    p_boot.add_argument("--method", choices=["termux-boot", "bashrc", "both"],
                        default="both", help="Metodo de auto-inicio")

    return parser


def main():
    global _verbose
    parser = build_parser()
    args = parser.parse_args()

    _verbose = args.verbose
    setup_logging()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Compatibilidad: convertir status posicional a --status para list
    if args.command == "list":
        status_val = getattr(args, "status_pos", None) or args.status
        args.status = status_val

    cmd_map = {
        "add": cmd_add,
        "list": cmd_list,
        "show": cmd_show,
        "remove": cmd_remove,
        "edit": cmd_edit,
        "exec": cmd_exec,
        "run": cmd_run,
        "status": cmd_status,
        "logs": cmd_logs,
        "pause": cmd_pause,
        "resume": cmd_resume,
        "agents": cmd_agents,
        "agent-add": cmd_agent_add,
        "agent-rm": cmd_agent_rm,
        "flush": cmd_flush,
        "boot-install": cmd_boot_install,
    }

    cmd_fn = cmd_map.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
