#!/usr/bin/env python3
"""
mcp_client.py — Cliente MCP (Model Context Protocol) para tbot.

Soporta dos transportes:
  - stdio:   lanza el servidor como subproceso, comunicación por stdin/stdout
  - Streamable HTTP: conexión HTTP con SSE para notificaciones server→client

Basado en la especificación MCP 2025-06-18 (draft).
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Constantes del protocolo ──────────────────────────────────────

MCP_PROTOCOL_VERSION = "2024-11-05"  # versión estable más reciente

MCP_METHODS = {
    # Lifecycle
    "initialize": "initialize",
    "initialized": "notifications/initialized",
    # Tools
    "tools_list": "tools/list",
    "tools_call": "tools/call",
    "tools_list_changed": "notifications/tools/list_changed",
    # Resources
    "resources_list": "resources/list",
    "resources_read": "resources/read",
    "resources_subscribe": "resources/subscribe",
    "resources_unsubscribe": "resources/unsubscribe",
    "resources_templates_list": "resources/templates/list",
    # Prompts
    "prompts_list": "prompts/list",
    "prompts_get": "prompts/get",
    # Logging
    "logging_set_level": "logging/setLevel",
    # Ping / keepalive
    "ping": "ping",
    # Cancellation
    "cancel": "notifications/cancelled",
    # Progress
    "progress": "notifications/progress",
}

# ── Excepciones ──────────────────────────────────────────────────


class MCPError(Exception):
    """Error devuelto por el servidor MCP en una respuesta JSON-RPC con campo 'error'."""

    def __init__(self, code, message, data=None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"[MCP {code}] {message}")


class MCPTimeout(Exception):
    """Timeout esperando respuesta del servidor MCP."""
    pass


class MCPConnectionError(Exception):
    """Error de conexión con el servidor MCP (no responde, no arranca, etc.)."""
    pass


# ── Cliente MCP ──────────────────────────────────────────────────


class MCPClient:
    """Cliente MCP que se conecta a un servidor vía stdio o HTTP.

    Uso básico:
        client = MCPClient("mi-servidor", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/ruta"])
        client.connect()
        tools = client.tools  # lista de herramientas descubiertas
        result = client.call_tool("read_file", {"path": "/ruta/archivo.txt"})
        client.close()
    """

    def __init__(
        self,
        name,
        *,
        command=None,
        args=None,
        url=None,
        transport="stdio",
        env=None,
        headers=None,
        request_timeout=60,
        connect_timeout=15,
        reconnect_delay=1.0,
        max_reconnect_attempts=3,
    ):
        self.name = name
        self.transport = transport  # "stdio" | "http"
        self.command = command
        self.args = args or []
        self.url = url
        self._env = {**os.environ, **(env or {})}
        self._custom_headers = dict(headers or {})
        self.request_timeout = request_timeout
        self.connect_timeout = connect_timeout
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        # Estado interno
        self.process = None  # subprocess.Popen (solo stdio)
        self._msg_id = 0
        self._pending = {}  # msg_id → queue.Queue
        self._lock = threading.Lock()
        self._reader_thread = None
        self._shutdown = False
        self.connected = False
        self._session_id = None  # para HTTP transport

        # Capacidades negociadas
        self.server_info = None
        self.server_capabilities = None
        self.client_capabilities = {
            "roots": {"listChanged": False},
            "sampling": {},
        }

        # Estado descubierto
        self.tools = []       # lista de dicts MCP tool definitions
        self.resources = []   # lista de dicts MCP resource definitions
        self.resource_templates = []  # lista de resource templates
        self.prompts = []     # lista de dicts MCP prompt definitions

        # Callbacks para eventos asíncronos
        self.on_tool_list_changed = None
        self.on_resource_list_changed = None
        self.on_log_message = None

        # Para reconexión automática
        self._reconnect_lock = threading.Lock()
        self._auto_reconnect = False

    # ── Conexión ─────────────────────────────────────────────────

    def connect(self, timeout=None):
        """Inicia la conexión con el servidor MCP (negociación completa).

        Args:
            timeout: timeout en segundos. Si es None, usa connect_timeout (default 15).
        """
        if timeout is None:
            timeout = self.connect_timeout
        if self.transport == "stdio":
            return self._connect_stdio(timeout)
        elif self.transport == "http":
            return self._connect_http(timeout)
        else:
            raise ValueError(f"Transporte no soportado: {self.transport}")

    def _connect_stdio(self, timeout):
        """Lanza el proceso y realiza handshake MCP."""
        if not self.command:
            raise MCPConnectionError("Se requiere 'command' para transporte stdio")

        try:
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self._env,
                bufsize=1,  # line-buffered
            )
        except FileNotFoundError as e:
            raise MCPConnectionError(
                f"Comando no encontrado: {self.command}. ¿Está instalado?"
            ) from e
        except PermissionError as e:
            raise MCPConnectionError(
                f"Sin permiso para ejecutar: {self.command}"
            ) from e
        except OSError as e:
            raise MCPConnectionError(
                f"Error al lanzar proceso: {e}"
            ) from e

        # Hilo lector de stdout
        self._reader_thread = threading.Thread(
            target=self._stdio_reader,
            daemon=True,
            name=f"mcp-reader-{self.name}",
        )
        self._reader_thread.start()

        # Hilo lector de stderr (loggeo)
        self._stderr_thread = threading.Thread(
            target=self._stdio_stderr_reader,
            daemon=True,
            name=f"mcp-stderr-{self.name}",
        )
        self._stderr_thread.start()

        # Handshake MCP
        try:
            self._initialize(timeout)
        except Exception:
            self.close()
            raise

        self.connected = True
        return True

    def _connect_http(self, timeout):
        """Conecta vía Streamable HTTP transport."""
        import requests as _requests

        # Handshake: POST para initialize
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Version": MCP_PROTOCOL_VERSION,
        }
        if self._session_id:
            headers["MCP-Session-ID"] = self._session_id
        # Merge custom headers (e.g. API keys)
        headers.update(self._custom_headers)

        init_msg = self._build_request("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": self.client_capabilities,
            "clientInfo": {"name": "tbot", "version": "1.0"},
        })

        init_id = init_msg["id"]
        self._msg_id = init_id  # sincronizar

        try:
            resp = _requests.post(
                self.url,
                json=init_msg,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
        except _requests.exceptions.ConnectionError as e:
            raise MCPConnectionError(f"No se pudo conectar a {self.url}: {e}") from e
        except _requests.exceptions.Timeout as e:
            raise MCPTimeout(f"Timeout conectando a {self.url}") from e
        except _requests.exceptions.RequestException as e:
            raise MCPConnectionError(f"Error HTTP: {e}") from e

        # Procesar respuesta (puede ser JSON directo o SSE)
        try:
            data = resp.json()
        except json.JSONDecodeError:
            text = resp.text.strip()
            data = self._parse_sse_response(text)
            if data is None:
                raise MCPError(-32700, f"Error decodificando respuesta: {text[:200]}")

        if "error" in data:
            raise MCPError(
                data["error"].get("code", -1),
                data["error"].get("message", "Unknown error"),
                data["error"].get("data"),
            )

        result = data.get("result", {})
        # Extraer session_id del header si existe
        session_id = resp.headers.get("MCP-Session-ID")
        if session_id:
            self._session_id = session_id

        self.server_info = result.get("serverInfo")
        self.server_capabilities = result.get("capabilities", {})

        # Enviar initialized notification
        self._notify("notifications/initialized", {})

        # Descubrir capacidades (igual que en _initialize para stdio)
        caps = self.server_capabilities or {}
        if "tools" in caps:
            self._discover_tools()
        if "resources" in caps:
            self._discover_resources()
            if caps["resources"].get("subscribe"):
                self._discover_resource_templates()
        if "prompts" in caps:
            self._discover_prompts()

        # Iniciar SSE reader si el server soporta streaming
        self._reader_thread = threading.Thread(
            target=self._http_sse_reader,
            daemon=True,
            name=f"mcp-sse-{self.name}",
        )
        self._reader_thread.start()

        self.connected = True
        return True

    # ── Inicialización MCP ───────────────────────────────────────

    def _initialize(self, timeout):
        """Realiza el handshake initialize → initialized."""
        result = self._request("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": self.client_capabilities,
            "clientInfo": {"name": "tbot", "version": "1.0"},
        }, timeout=timeout)

        self.server_info = result.get("serverInfo")
        self.server_capabilities = result.get("capabilities", {})

        # Enviar notificación initialized
        self._notify("notifications/initialized", {})

        # Descubrir capacidades
        caps = self.server_capabilities or {}
        # Nota: capabilities pueden ser dict vacío {}, ej: "tools": {}
        # Usamos "in" en lugar de truthy check porque bool({}) == False
        if "tools" in caps:
            self._discover_tools()
        if "resources" in caps:
            self._discover_resources()
            if caps["resources"].get("subscribe"):
                self._discover_resource_templates()
        if "prompts" in caps:
            self._discover_prompts()

    # ── Descubrimiento ────────────────────────────────────────────

    def _discover_tools(self):
        """Obtiene la lista de herramientas del servidor (con paginación)."""
        self.tools = []
        cursor = None
        while True:
            params = {}
            if cursor:
                params["cursor"] = cursor
            result = self._request("tools/list", params)
            self.tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                break

    def _discover_resources(self):
        """Obtiene la lista de recursos del servidor."""
        self.resources = []
        cursor = None
        while True:
            params = {}
            if cursor:
                params["cursor"] = cursor
            result = self._request("resources/list", params)
            self.resources.extend(result.get("resources", []))
            cursor = result.get("nextCursor")
            if not cursor:
                break

    def _discover_resource_templates(self):
        """Obtiene templates de recursos."""
        self.resource_templates = []
        cursor = None
        while True:
            params = {}
            if cursor:
                params["cursor"] = cursor
            result = self._request("resources/templates/list", params)
            self.resource_templates.extend(result.get("resourceTemplates", []))
            cursor = result.get("nextCursor")
            if not cursor:
                break

    def _discover_prompts(self):
        """Obtiene la lista de prompts del servidor."""
        self.prompts = []
        cursor = None
        while True:
            params = {}
            if cursor:
                params["cursor"] = cursor
            result = self._request("prompts/list", params)
            self.prompts.extend(result.get("prompts", []))
            cursor = result.get("nextCursor")
            if not cursor:
                break

    # ── Tool calls ───────────────────────────────────────────────

    def call_tool(self, name, arguments=None, timeout=None):
        """Invoca una herramienta en el servidor MCP.

        Returns:
            dict con:
              - content: list[TextContent | ImageContent | AudioContent | EmbeddedResource]
              - structuredContent: dict (opcional, salida estructurada)
              - isError: bool (opcional, default False)
        """
        return self._request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        }, timeout=timeout)

    # ── Resource reading ─────────────────────────────────────────

    def read_resource(self, uri):
        """Lee un recurso del servidor por su URI."""
        return self._request("resources/read", {"uri": uri})

    def subscribe_resource(self, uri):
        """Se suscribe a cambios en un recurso."""
        return self._request("resources/subscribe", {"uri": uri})

    def unsubscribe_resource(self, uri):
        """Cancela suscripción a cambios en un recurso."""
        return self._request("resources/unsubscribe", {"uri": uri})

    # ── Prompts ──────────────────────────────────────────────────

    def get_prompt(self, name, arguments=None):
        """Obtiene un prompt del servidor."""
        params = {"name": name}
        if arguments:
            params["arguments"] = arguments
        return self._request("prompts/get", params)

    # ── Ping / Keepalive ─────────────────────────────────────────

    def ping(self):
        """Verifica que el servidor responda."""
        return self._request("ping", {})

    # ── JSON-RPC interno ─────────────────────────────────────────

    def _build_request(self, method, params=None):
        """Construye un mensaje JSON-RPC request."""
        with self._lock:
            self._msg_id += 1
            msg_id = self._msg_id
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": MCP_METHODS.get(method, method),
            "params": params or {},
        }

    def _request(self, method, params=None, timeout=None):
        """Envía un request JSON-RPC y espera respuesta.

        Timeout por defecto: self.request_timeout (60s).
        Lanza MCPError si el server responde con error.
        Lanza MCPTimeout si no hay respuesta en el tiempo límite.
        """
        if timeout is None:
            timeout = self.request_timeout

        msg = self._build_request(method, params)
        msg_id = msg["id"]
        result_queue = queue.Queue()

        with self._lock:
            self._pending[msg_id] = result_queue

        try:
            self._send(msg)
            try:
                response = result_queue.get(timeout=timeout)
            except queue.Empty:
                with self._lock:
                    self._pending.pop(msg_id, None)
                raise MCPTimeout(
                    f"Timeout ({timeout}s) esperando respuesta a '{method}' "
                    f"del servidor MCP '{self.name}'"
                )

            if "error" in response:
                err = response["error"]
                raise MCPError(
                    err.get("code", -1),
                    err.get("message", "Unknown error"),
                    err.get("data"),
                )
            return response.get("result", {})
        except (MCPTimeout, MCPError):
            raise
        except Exception as e:
            raise MCPConnectionError(
                f"Error en comunicación con servidor MCP '{self.name}': {e}"
            ) from e

    def _notify(self, method, params=None):
        """Envía una notificación JSON-RPC (sin esperar respuesta)."""
        msg = {
            "jsonrpc": "2.0",
            "method": MCP_METHODS.get(method, method),
            "params": params or {},
        }
        # Las notificaciones no tienen ID
        try:
            self._send(msg)
        except Exception:
            pass  # Las notificaciones son fire-and-forget

    def _send(self, msg):
        """Envía un mensaje JSON-RPC por el transporte activo."""
        payload = json.dumps(msg, ensure_ascii=False)

        if self.transport == "stdio":
            if self.process and self.process.stdin:
                self.process.stdin.write(payload + "\n")
                self.process.stdin.flush()
            else:
                raise MCPConnectionError("stdin del proceso no disponible")
        elif self.transport == "http":
            self._http_send(payload)
        else:
            raise ValueError(f"Transporte no soportado: {self.transport}")

    def _http_send(self, payload):
        """Envía un mensaje por HTTP (POST)."""
        import requests as _requests

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Version": MCP_PROTOCOL_VERSION,
        }
        if self._session_id:
            headers["MCP-Session-ID"] = self._session_id
        # Merge custom headers (e.g. API keys)
        headers.update(self._custom_headers)

        try:
            resp = _requests.post(
                self.url,
                data=payload,
                headers=headers,
                timeout=self.request_timeout,
            )
            # Si hay respuesta JSON-RPC síncrona, procesarla
            if resp.status_code == 200 and resp.text.strip():
                text = resp.text.strip()
                # Intentar parsear como JSON directo primero
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    # Puede ser SSE stream: "event: message\ndata: {...}\n\n"
                    data = self._parse_sse_response(text)
                if data and "id" in data:
                    msg_id = data.get("id")
                    with self._lock:
                        q = self._pending.pop(msg_id, None)
                    if q:
                        q.put(data)
        except _requests.exceptions.RequestException as e:
            raise MCPConnectionError(f"Error HTTP: {e}") from e

    def _parse_sse_response(self, text):
        """Parsea una respuesta SSE y extrae el primer objeto JSON-RPC."""
        lines = text.split("\n")
        data_lines = []
        for line in lines:
            if line.startswith("data: "):
                data_lines.append(line[6:].strip())
            elif line.strip() == "" and data_lines:
                # End of SSE event
                payload = "".join(data_lines)
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    data_lines.clear()
                    continue
            elif not line.startswith("event:") and not line.startswith(":"):
                # If we have accumulated data but no blank line yet, keep going
                pass
        # Fallback: try to parse everything as a single JSON
        if data_lines:
            try:
                return json.loads("".join(data_lines))
            except json.JSONDecodeError:
                pass
        return None

    # ── Readers (threads) ────────────────────────────────────────

    def _stdio_reader(self):
        """Hilo que lee líneas JSON-RPC de stdout del proceso."""
        while not self._shutdown:
            try:
                if self.process and self.process.stdout:
                    line = self.process.stdout.readline()
                else:
                    break
                if not line:
                    # EOF — proceso terminó
                    if not self._shutdown:
                        self._handle_disconnect()
                    break
                line = line.strip()
                if not line:
                    continue
                msg = json.loads(line)
                self._dispatch_message(msg)
            except json.JSONDecodeError:
                continue  # ignorar líneas malformadas
            except (ValueError, EOFError, OSError):
                if not self._shutdown:
                    self._handle_disconnect()
                break
            except Exception:
                # Loggear pero no morir
                continue

    def _stdio_stderr_reader(self):
        """Hilo que lee stderr del proceso para logging."""
        while not self._shutdown:
            try:
                if self.process and self.process.stderr:
                    line = self.process.stderr.readline()
                else:
                    break
                if not line:
                    break
                # Los servidores MCP pueden loguear por stderr en formato:
                # {"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info","logger":"...","data":"..."}}
                # o simplemente texto plano
                line = line.rstrip()
                if line:
                    try:
                        msg = json.loads(line)
                        if msg.get("method") == "notifications/message":
                            self._handle_log_message(msg.get("params", {}))
                            continue
                    except json.JSONDecodeError:
                        pass
                    # Loggear como texto plano
                    if self.on_log_message:
                        self.on_log_message(self.name, "info", line)
            except Exception:
                break

    def _http_sse_reader(self):
        """Hilo que lee eventos SSE del servidor HTTP."""
        import requests as _requests

        while not self._shutdown:
            try:
                headers = {"Accept": "application/json, text/event-stream", "MCP-Version": MCP_PROTOCOL_VERSION}
                if self._session_id:
                    headers["MCP-Session-ID"] = self._session_id
                headers.update(self._custom_headers)
                resp = _requests.get(
                    self.url,
                    headers=headers,
                    stream=True,
                    timeout=(5, 30),
                )
                resp.raise_for_status()

                event = ""
                for chunk in resp.iter_lines():
                    if self._shutdown:
                        break
                    if not chunk:
                        continue
                    try:
                        line = chunk.decode("utf-8")
                    except (UnicodeDecodeError, AttributeError):
                        continue

                    if line.startswith("data: "):
                        event += line[6:]
                    elif line.startswith("event: "):
                        event = ""  # empezar nuevo evento
                    elif line == "" and event:
                        # Fin del evento
                        try:
                            msg = json.loads(event)
                            self._dispatch_message(msg)
                        except json.JSONDecodeError:
                            pass
                        event = ""
            except _requests.exceptions.RequestException:
                if not self._shutdown:
                    time.sleep(self.reconnect_delay)
            except Exception:
                if not self._shutdown:
                    time.sleep(self.reconnect_delay)

    # ── Dispatch de mensajes ─────────────────────────────────────

    def _dispatch_message(self, msg):
        """Enruta un mensaje JSON-RPC entrante."""
        # Si tiene id, es respuesta a un request nuestro
        msg_id = msg.get("id")
        if msg_id is not None:
            with self._lock:
                q = self._pending.pop(msg_id, None)
            if q:
                q.put(msg)
            return

        # Es una notificación del servidor
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method == "notifications/tools/list_changed":
            if self.connected:
                self._discover_tools()
            if self.on_tool_list_changed:
                self.on_tool_list_changed(self.name, self.tools)

        elif method == "notifications/resources/list_changed":
            if self.connected:
                self._discover_resources()
            if self.on_resource_list_changed:
                self.on_resource_list_changed(self.name, self.resources)

        elif method == "notifications/message":
            self._handle_log_message(params)

        elif method == "notifications/progress":
            # Progreso de operación en curso — podríamos notificar
            pass

    def _handle_log_message(self, params):
        """Maneja una notificación de log del servidor."""
        level = params.get("level", "info")
        logger = params.get("logger", "")
        data = params.get("data", "")
        if self.on_log_message:
            self.on_log_message(self.name, level, f"[{logger}] {data}")

    def _handle_disconnect(self):
        """Maneja desconexión inesperada del servidor."""
        self.connected = False
        # Notificar a todos los pending queues
        with self._lock:
            pending = list(self._pending.items())
            self._pending.clear()
        for msg_id, q in pending:
            q.put({"error": {"code": -32000, "message": "Server disconnected"}})

        # Intentar reconexión automática si está habilitada
        if self._auto_reconnect:
            threading.Thread(
                target=self._reconnect_loop,
                daemon=True,
                name=f"mcp-reconnect-{self.name}",
            ).start()

    def _reconnect_loop(self):
        """Intenta reconectar al servidor en caso de desconexión."""
        if not self._reconnect_lock.acquire(blocking=False):
            return  # ya hay un intento de reconexión en curso
        try:
            for attempt in range(self.max_reconnect_attempts):
                if self._shutdown:
                    return
                time.sleep(self.reconnect_delay * (2 ** attempt))
                try:
                    self.close(terminate=True)
                    self.connect()
                    self._auto_reconnect = True  # mantener flag
                    return
                except Exception:
                    continue
        finally:
            self._reconnect_lock.release()

    # ── Shutdown ─────────────────────────────────────────────────

    def close(self, terminate=False):
        """Cierra la conexión con el servidor MCP.

        Args:
            terminate: si True, mata el proceso (SIGKILL en Unix).
                       si False, cierra stdin (SIGTERM si el server no respeta EOF).
        """
        self._shutdown = True

        if self.transport == "stdio" and self.process:
            try:
                # Cerrar stdin primero (señal de shutdown para el server)
                if self.process.stdin:
                    self.process.stdin.close()
                if terminate:
                    self.process.kill()
                else:
                    self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            except Exception:
                pass
            self.process = None

        # Liberar pending queues
        with self._lock:
            pending = list(self._pending.items())
            self._pending.clear()
        for msg_id, q in pending:
            q.put({"error": {"code": -32000, "message": "Client shutting down"}})

        self.connected = False

    # ── Utilidades para conversión a OpenAI tool schema ──────────

    def to_openai_tools(self):
        """Convierte las herramientas MCP descubiertas al formato OpenAI function calling.

        Returns:
            list[dict] — herramientas en formato {"type": "function", "function": {...}}
        """
        openai_tools = []
        for tool in self.tools:
            mcp_name = tool.get("name", "")
            if not mcp_name:
                continue

            # Prefijo mcp:<server_name> para evitar colisiones
            qualified_name = f"mcp__{self.name}__{mcp_name}"

            desc = tool.get("description", f"MCP tool '{mcp_name}' from server '{self.name}'")
            input_schema = tool.get("inputSchema", {"type": "object", "properties": {}})

            # El inputSchema de MCP es JSON Schema; lo usamos directamente
            # como el schema de parámetros de OpenAI function calling
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": qualified_name,
                    "description": desc,
                    "parameters": input_schema,
                },
            })
        return openai_tools

    def to_openai_resources_notes(self):
        """Genera texto sobre recursos disponibles para inyectar en system prompt."""
        if not self.resources and not self.resource_templates:
            return ""
        lines = [f"\n## Recursos MCP del servidor '{self.name}'"]
        if self.resources:
            lines.append("\n### Recursos estáticos")
            for r in self.resources:
                uri = r.get("uri", "")
                desc = r.get("description", "")
                mime = r.get("mimeType", "")
                name = r.get("name", uri)
                lines.append(f"- `{uri}` — {name}")
                if desc:
                    lines.append(f"  {desc}")
                if mime:
                    lines.append(f"  (tipo: {mime})")
        if self.resource_templates:
            lines.append("\n### Templates de recursos")
            for t in self.resource_templates:
                uri_template = t.get("uriTemplate", "")
                desc = t.get("description", "")
                name = t.get("name", uri_template)
                lines.append(f"- `{uri_template}` — {name}")
                if desc:
                    lines.append(f"  {desc}")
        return "\n".join(lines)

    # ── Factory method ───────────────────────────────────────────

    @classmethod
    def from_config(cls, config):
        """Crea una instancia desde un dict de configuración.

        Config esperado:
            {
                "name": str,
                "transport": "stdio" | "http",
                "command": str,           # para stdio
                "args": list[str],        # para stdio (opcional)
                "url": str,               # para http
                "env": dict,              # opcional
                "headers": dict,          # opcional — headers HTTP personalizados
                "enabled": bool,
                "auto_reconnect": bool,
                "request_timeout": int,
            }
        """
        return cls(
            name=config["name"],
            command=config.get("command"),
            args=config.get("args", []),
            url=config.get("url"),
            transport=config.get("transport", "stdio"),
            env=config.get("env"),
            headers=config.get("headers"),
            request_timeout=config.get("request_timeout", 60),
            connect_timeout=config.get("connect_timeout", 15),
        )


# ── Gestor de servidores MCP ─────────────────────────────────────


class MCPServerManager:
    """Gestiona múltiples conexiones MCP.

    Se encarga de:
      - Arrancar/conectar todos los servidores configurados
      - Proveer una vista unificada de herramientas
      - Enrutar tool calls al servidor correcto
      - Monitorear estado y reconectar
    """

    def __init__(self):
        self._clients = {}  # name → MCPClient
        self._configs = []  # lista de configs originales

    @property
    def clients(self):
        return dict(self._clients)

    @property
    def connected_count(self):
        return sum(1 for c in self._clients.values() if c.connected)

    @property
    def total_count(self):
        """Número total de servidores configurados (no solo conectados)."""
        return len(self._configs)

    def load_configs(self, configs):
        """Carga configuraciones desde la config de tbot."""
        self._configs = list(configs)

    def connect_all(self, parallel=True, max_workers=8):
        """Conecta todos los servidores configurados y habilitados.

        Args:
            parallel: si True, conecta en paralelo usando ThreadPoolExecutor.
            max_workers: máximo de hilos para conexiones paralelas.

        Returns:
            list[tuple[str, str]] — lista de (name, "ok" | error_msg)
        """
        enabled_configs = []
        results = []
        for cfg in self._configs:
            if not cfg.get("enabled", True):
                results.append((cfg["name"], "disabled"))
            else:
                enabled_configs.append(cfg)

        if not enabled_configs:
            return results

        if not parallel or len(enabled_configs) == 1:
            # Conexión secuencial (un solo servidor o modo no paralelo)
            for cfg in enabled_configs:
                try:
                    client = MCPClient.from_config(cfg)
                    client.connect()
                    if cfg.get("auto_reconnect", True):
                        client._auto_reconnect = True
                    self._clients[cfg["name"]] = client
                    results.append((cfg["name"], "ok"))
                except Exception as e:
                    results.append((cfg["name"], str(e)))
            return results

        # Conexión paralela
        def _connect_one(cfg):
            name = cfg["name"]
            try:
                client = MCPClient.from_config(cfg)
                client.connect()
                if cfg.get("auto_reconnect", True):
                    client._auto_reconnect = True
                return (name, "ok", client)
            except Exception as e:
                return (name, str(e), None)

        with ThreadPoolExecutor(max_workers=min(max_workers, len(enabled_configs))) as pool:
            futures = {pool.submit(_connect_one, cfg): cfg["name"] for cfg in enabled_configs}
            for future in as_completed(futures):
                name, status, client = future.result()
                if client is not None:
                    self._clients[name] = client
                results.append((name, status))

        # Preservar orden original de configuración
        ordered = []
        for cfg in self._configs:
            for r in results:
                if r[0] == cfg["name"]:
                    ordered.append(r)
                    break
        return ordered

    def get_client(self, server_name):
        """Obtiene un cliente por nombre."""
        return self._clients.get(server_name)

    def get_all_tools(self):
        """Retorna todas las herramientas de todos los servidores en formato OpenAI.

        Returns:
            list[dict] — herramientas combinadas de todos los servidores
        """
        all_tools = []
        for name, client in self._clients.items():
            if client.connected:
                all_tools.extend(client.to_openai_tools())
        return all_tools

    def get_resource_notes(self):
        """Retorna notas sobre recursos disponibles de todos los servidores."""
        notes = []
        for name, client in self._clients.items():
            if client.connected:
                n = client.to_openai_resources_notes()
                if n:
                    notes.append(n)
        return "\n".join(notes)

    def find_tool_server(self, qualified_name):
        """Dado un nombre cualificado 'mcp__server__tool', retorna (server_name, tool_name)."""
        if not qualified_name.startswith("mcp__"):
            return None, None
        parts = qualified_name.split("__", 2)
        if len(parts) < 3:
            return None, None
        return parts[1], parts[2]

    def call_tool(self, qualified_name, arguments=None):
        """Invoca una tool en el servidor MCP correcto.

        Args:
            qualified_name: str en formato 'mcp__<server>__<tool>'
            arguments: dict de argumentos

        Returns:
            str — resultado formateado para el modelo
        """
        server_name, tool_name = self.find_tool_server(qualified_name)
        if not server_name:
            return f"Error: formato de tool name inválido: {qualified_name}"

        client = self._clients.get(server_name)
        if not client:
            return f"Error: servidor MCP '{server_name}' no encontrado"
        if not client.connected:
            return f"Error: servidor MCP '{server_name}' no está conectado"

        try:
            result = client.call_tool(tool_name, arguments)
        except MCPError as e:
            return f"Error MCP del servidor '{server_name}' llamando a '{tool_name}': {e.message} (código {e.code})"
        except MCPTimeout:
            return f"Timeout del servidor MCP '{server_name}' llamando a '{tool_name}'"
        except MCPConnectionError as e:
            return f"Error de conexión con servidor MCP '{server_name}': {e}"
        except Exception as e:
            return f"Error inesperado del servidor MCP '{server_name}': {e}"

        # Formatear resultado para el modelo
        return self._format_tool_result(result, server_name, tool_name)

    def _format_tool_result(self, result, server_name, tool_name):
        """Formatea el resultado de un tool call MCP para consumo del LLM."""
        content = result.get("content", [])
        is_error = result.get("isError", False)
        structured = result.get("structuredContent")

        parts = []
        if is_error:
            parts.append(f"[Error del servidor MCP '{server_name}' en '{tool_name}']")

        # Contenido textual
        text_parts = []
        for item in content:
            item_type = item.get("type", "")
            if item_type == "text":
                text_parts.append(item.get("text", ""))
            elif item_type == "resource":
                resource = item.get("resource", {})
                if "text" in resource:
                    text_parts.append(resource["text"])
                elif "blob" in resource:
                    text_parts.append(f"[blob data: {len(resource['blob'])} bytes]")
            elif item_type == "image":
                text_parts.append(f"[image: {item.get('mimeType', 'unknown')}]")
            elif item_type == "audio":
                text_parts.append(f"[audio: {item.get('mimeType', 'unknown')}]")

        if text_parts:
            parts.append("\n".join(text_parts))

        # Salida estructurada
        if structured:
            parts.append(f"\n(Salida estructurada: {json.dumps(structured, indent=2)})")

        result_text = "\n".join(parts) if parts else "(sin contenido)"
        return f"MCP[{server_name}] {tool_name}:\n{result_text}"

    def discover_all(self):
        """Rediscover tools/resources/prompts de todos los servidores conectados."""
        for client in self._clients.values():
            if client.connected:
                caps = client.server_capabilities or {}
                try:
                    if "tools" in caps:
                        client._discover_tools()
                    if "resources" in caps:
                        client._discover_resources()
                    if "prompts" in caps:
                        client._discover_prompts()
                except Exception:
                    pass

    def close_all(self):
        """Cierra todas las conexiones MCP."""
        for name, client in self._clients.items():
            try:
                client.close()
            except Exception:
                pass
        self._clients.clear()

    def get_status(self):
        """Retorna lista de estados de todos los servidores configurados."""
        statuses = []
        for cfg in self._configs:
            name = cfg["name"]
            client = self._clients.get(name)
            if client and client.connected:
                tools_count = len(client.tools)
                resources_count = len(client.resources)
                statuses.append({
                    "name": name,
                    "transport": cfg.get("transport", "stdio"),
                    "status": "connected",
                    "tools": tools_count,
                    "resources": resources_count,
                    "server_info": client.server_info,
                })
            elif client and not client.connected:
                statuses.append({
                    "name": name,
                    "transport": cfg.get("transport", "stdio"),
                    "status": "disconnected",
                    "tools": 0,
                    "resources": 0,
                })
            else:
                statuses.append({
                    "name": name,
                    "transport": cfg.get("transport", "stdio"),
                    "status": "not_initialized",
                    "tools": 0,
                    "resources": 0,
                })
        return statuses
