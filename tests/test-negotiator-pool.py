#!/usr/bin/env python3
import concurrent.futures
import http.client
import http.server
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time


MODEL = "fixture-small:latest"
MODEL_WITHOUT_TAG = "fixture-small"
OTHER_MODEL = "fixture-other:latest"
EMBED_MODEL = "fixture-embed:latest"
REJECT_MODEL = "fixture-reject:latest"
RETRY_MODEL = "fixture-retry:latest"
MODEL_DIGEST = "a" * 64
FIXTURE_MODEL_INFO = {
    "general.architecture": "fixture",
    "fixture.context_length": 262144,
    "fixture.block_count": 2,
    "fixture.embedding_length": 128,
    "fixture.attention.head_count": 2,
    "fixture.attention.head_count_kv": 1,
    "fixture.attention.key_length": 64,
    "fixture.attention.value_length": 64,
}


class Backend(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, tags=None, model_info=None):
        super().__init__(("127.0.0.1", 0), Handler)
        self.models = []
        names = tags if tags is not None else [MODEL]
        self.tags = []
        for item in names:
            if isinstance(item, dict):
                self.tags.append({"digest": MODEL_DIGEST, **item})
            else:
                self.tags.append({
                    "name": item,
                    "model": item,
                    "digest": MODEL_DIGEST,
                    "size": 1024**3,
                    "capabilities": ["completion"],
                })
        self.lock = threading.Lock()
        self.model_info = dict(FIXTURE_MODEL_INFO if model_info is None else model_info)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_json(self, payload, status=200):
        data = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        with self.server.lock:
            if self.path == "/api/ps":
                payload = {"models": list(self.server.models)}
            elif self.path == "/api/tags":
                payload = {"models": list(self.server.tags)}
            elif self.path == "/api/version":
                payload = {"version": "fixture"}
            else:
                self.send_json({"error": "not found"}, 404)
                return
        self.send_json(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/show":
            self.send_json({
                "model_info": self.server.model_info,
                "capabilities": next((tag.get("capabilities", ["completion"])
                                      for tag in self.server.tags
                                      if tag.get("name") == payload.get("model")), ["completion"]),
            })
            return
        with self.server.lock:
            if payload.get("keep_alive") == 0:
                self.server.models = []
        self.send_json({"done": True})

    def log_message(self, _fmt, *_args):
        pass


def free_port():
    candidate = socket.socket()
    candidate.bind(("127.0.0.1", 0))
    port = candidate.getsockname()[1]
    candidate.close()
    return port


def http_json(port, method, path, payload=None, timeout=10, extra_headers=None):
    body = None if payload is None else json.dumps(payload).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    headers.update(extra_headers or {})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    data = json.loads(response.read() or b"{}")
    result = response.status, data, dict(response.getheaders())
    conn.close()
    return result


def http_raw(port, method, path, payload=None, timeout=10, extra_headers=None):
    body = None if payload is None else json.dumps(payload).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    headers.update(extra_headers or {})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    data = response.read()
    result = response.status, data, dict(response.getheaders())
    conn.close()
    return result


def control(path, payload):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(10)
    client.connect(path)
    client.sendall(json.dumps(payload).encode() + b"\n")
    response = json.loads(client.makefile("rb").readline())
    client.close()
    assert response["ok"], response
    return response


def control_raw(path, payload):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(10)
    client.connect(path)
    client.sendall(json.dumps(payload).encode() + b"\n")
    response = json.loads(client.makefile("rb").readline())
    client.close()
    return response


def events(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    except FileNotFoundError:
        return []


def wait_ready(daemon, socket_path, proxy_port):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if daemon.poll() is not None:
            if daemon.stderr.seekable():
                daemon.stderr.seek(0)
            raise RuntimeError(daemon.stderr.read())
        if os.path.exists(socket_path):
            try:
                connection = socket.create_connection(("127.0.0.1", proxy_port), 0.2)
                connection.close()
                return
            except OSError:
                pass
        time.sleep(0.05)
    raise RuntimeError("pool broker did not become ready")


def wait_until(predicate, description, timeout=5):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = predicate()
        except (ConnectionError, OSError):
            last = None
        if last:
            return last
        time.sleep(0.025)
    raise AssertionError(f"timed out waiting for {description}; last={last!r}")


def write_compute_apps(path, rows):
    replacement = path + ".new"
    with open(replacement, "w", encoding="utf-8") as stream:
        for pid, gpu_uuid, used_mib in rows:
            stream.write(f"{pid}, {gpu_uuid}, {used_mib}\n")
    os.replace(replacement, path)


def request_events(path):
    return [event for event in events(path)
            if event["kind"] == "request" and event.get("request_id") is not None]


def chat(
    port,
    model,
    request_id,
    delay=0,
    timeout=10,
    *,
    logical_request_id=None,
    admission_wait_ms=None,
    workload_class="foreground",
    queue_policy="wait",
    mock_body_size=None,
    mock_backend_reset=False,
    mock_partial_stall=False,
    mock_barrier_requests=None,
    gpu_uuids=None,
    client=None,
):
    headers = {
        "X-Ollama-Unify-Workload-Class": workload_class,
        "X-Ollama-Unify-Queue-Policy": queue_policy,
    }
    if logical_request_id is not None:
        headers["X-Ollama-Unify-Logical-Request-Id"] = logical_request_id
    if admission_wait_ms is not None:
        headers["X-Ollama-Unify-Admission-Wait-Ms"] = str(admission_wait_ms)
    if gpu_uuids is not None:
        headers["X-Ollama-Unify-GPU-UUIDs"] = (
            ",".join(gpu_uuids) if gpu_uuids else "none"
        )
    if client is not None:
        headers["X-Ollama-Unify-Client"] = client
    payload = {
        "model": model,
        "stream": False,
        "mock_request_id": request_id,
        "mock_delay": delay,
    }
    if mock_body_size is not None:
        payload["mock_body_size"] = mock_body_size
    if mock_barrier_requests is not None:
        payload["mock_barrier_requests"] = mock_barrier_requests
    if mock_backend_reset:
        payload["mock_backend_reset"] = True
    if mock_partial_stall:
        payload["mock_partial_stall"] = True
    return http_json(
        port, "POST", "/api/chat", payload,
        timeout=timeout, extra_headers=headers,
    )


def resume_chat(
    port,
    logical_request_id,
    timeout=10,
    *,
    admission_wait_ms=None,
    workload_class="foreground",
    queue_policy="wait",
):
    headers = {
        "X-Ollama-Unify-Logical-Request-Id": logical_request_id,
        "X-Ollama-Unify-Resume-Request": "true",
        "X-Ollama-Unify-Workload-Class": workload_class,
        "X-Ollama-Unify-Queue-Policy": queue_policy,
    }
    if admission_wait_ms is not None:
        headers["X-Ollama-Unify-Admission-Wait-Ms"] = str(admission_wait_ms)
    return http_json(
        port, "POST", "/api/chat", None,
        timeout=timeout, extra_headers=headers,
    )


def resume_chat_raw(port, logical_request_id, timeout=10):
    return http_raw(
        port,
        "POST",
        "/api/chat",
        None,
        timeout=timeout,
        extra_headers={
            "X-Ollama-Unify-Logical-Request-Id": logical_request_id,
            "X-Ollama-Unify-Resume-Request": "true",
            "X-Ollama-Unify-Workload-Class": "foreground",
            "X-Ollama-Unify-Queue-Policy": "wait",
        },
    )


def embed(port, model, request_id, timeout=10):
    return http_json(port, "POST", "/api/embed", {
        "model": model,
        "input": "fixture input",
        "mock_request_id": request_id,
    }, timeout=timeout)


def streaming_chat(
    port,
    model,
    request_id,
    logical_request_id,
    timeout=10,
    *,
    chunk_count=None,
    chunk_interval=None,
):
    payload = {
        "model": model,
        "stream": True,
        "mock_stream": True,
        "mock_request_id": request_id,
    }
    if chunk_count is not None:
        payload["mock_stream_count"] = chunk_count
    if chunk_interval is not None:
        payload["mock_stream_interval"] = chunk_interval
    return http_raw(
        port,
        "POST",
        "/api/chat",
        payload,
        timeout=timeout,
        extra_headers={
            "X-Ollama-Unify-Logical-Request-Id": logical_request_id,
            "X-Ollama-Unify-Workload-Class": "foreground",
            "X-Ollama-Unify-Queue-Policy": "wait",
        },
    )


def open_abandonable_chat(
    port, model, request_id, logical_request_id=None, delay=0,
):
    payload = json.dumps({
        "model": model,
        "stream": False,
        "mock_request_id": request_id,
        "mock_delay": delay,
    }).encode()
    connection = socket.create_connection(("127.0.0.1", port), timeout=2)
    connection.sendall(
        b"POST /api/chat HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(payload)}\r\n".encode()
        + (
            f"X-Ollama-Unify-Logical-Request-Id: {logical_request_id}\r\n".encode()
            if logical_request_id else b""
        )
        + b"Connection: close\r\n\r\n"
        + payload
    )
    return connection


def reset_connection(connection):
    connection.setsockopt(
        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
    )
    connection.close()


def incomplete_fixed_length_chat(port, model, request_id, timeout=3):
    payload = json.dumps({
        "model": model,
        "stream": False,
        "mock_request_id": request_id,
        "mock_partial_stall": True,
    }).encode()
    connection = http.client.HTTPConnection(
        "127.0.0.1", port, timeout=timeout
    )
    started = time.monotonic()
    connection.request(
        "POST",
        "/api/chat",
        body=payload,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    partial = b""
    try:
        response.read()
    except http.client.IncompleteRead as exc:
        partial = exc.partial
    else:
        raise AssertionError("incomplete fixed-length response read as complete")
    elapsed = time.monotonic() - started
    connection.close()
    return response.status, partial, elapsed


class PoolHarness:
    def __init__(self, helper, fixture_bin, *, max_servers=1, tags=None,
                 runner_vram_mib=2048, max_queue=64,
                 release_tolerance_mib=256, resume_ttl=1.0,
                 completed_ttl=2.0, completed_max_entries=8,
                 completed_max_body_bytes=1024 * 1024,
                 completed_max_total_bytes=4 * 1024 * 1024,
                 pending_timeout=300.0, revoke_timeout=300.0,
                 client_history_ttl=3600.0, client_lane_history_limit=32,
                 request_activity_ttl=3.0, request_detached_ttl=1.0,
                 request_cancel_grace=0.25,
                 instance_parallel=1,
                 model_gpu_preferences=None, model_context_profiles=None,
                 profile="cuda_triple", selected_gpus=None,
                 cpu_only=False, max_context=0, idle_timeout=30,
                 runner_vram_by_gpu=None, runner_gpu_scope=None,
                 partial_gpu_residency=False, auto_model_context=False,
                 model_info=None, runner_context_length=None):
        self.profile = profile
        self.selected_gpus = selected_gpus or [
            "GPU-large-0", "GPU-large-1", "GPU-large-2",
        ]
        self.helper = helper
        self.fixture_bin = fixture_bin
        self.max_servers = max_servers
        self.tags = tags or [MODEL]
        self.runner_vram_mib = runner_vram_mib
        self.max_queue = max_queue
        self.release_tolerance_mib = release_tolerance_mib
        self.resume_ttl = resume_ttl
        self.completed_ttl = completed_ttl
        self.completed_max_entries = completed_max_entries
        self.completed_max_body_bytes = completed_max_body_bytes
        self.completed_max_total_bytes = completed_max_total_bytes
        self.pending_timeout = pending_timeout
        self.revoke_timeout = revoke_timeout
        self.client_history_ttl = client_history_ttl
        self.client_lane_history_limit = client_lane_history_limit
        self.request_activity_ttl = request_activity_ttl
        self.request_detached_ttl = request_detached_ttl
        self.request_cancel_grace = request_cancel_grace
        self.instance_parallel = instance_parallel
        self.model_gpu_preferences = model_gpu_preferences
        self.model_context_profiles = model_context_profiles
        self.cpu_only = cpu_only
        self.max_context = max_context
        self.idle_timeout = idle_timeout
        self.runner_vram_by_gpu = runner_vram_by_gpu or {}
        self.runner_gpu_scope = runner_gpu_scope
        self.partial_gpu_residency = partial_gpu_residency
        self.auto_model_context = auto_model_context
        self.model_info = model_info
        self.runner_context_length = runner_context_length

    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ollama-unify-pool-case-")
        self.temp_dir = self.temp.name
        self.socket_path = os.path.join(self.temp_dir, "control.sock")
        self.event_log = os.path.join(self.temp_dir, "events.jsonl")
        self.compute_apps = os.path.join(self.temp_dir, "compute-apps.csv")
        self.gpu_usage_dir = os.path.join(self.temp_dir, "gpu-usage")
        os.mkdir(self.gpu_usage_dir)
        write_compute_apps(self.compute_apps, [])
        self.backend = Backend(self.tags, self.model_info)
        threading.Thread(target=self.backend.serve_forever, daemon=True).start()
        self.proxy_port = free_port()
        pool_port = free_port()
        env = os.environ.copy()
        env.update({
            "PATH": self.fixture_bin + os.pathsep + env.get("PATH", ""),
            "MOCK_PROFILE": self.profile,
            "MOCK_OLLAMA_EVENT_LOG": self.event_log,
            "MOCK_NVIDIA_COMPUTE_APPS_FILE": self.compute_apps,
            "MOCK_OLLAMA_GPU_USAGE_DIR": self.gpu_usage_dir,
            "MOCK_OLLAMA_VRAM_MIB": str(self.runner_vram_mib),
            "MOCK_OLLAMA_GPU_USAGE_BY_GPU": json.dumps(self.runner_vram_by_gpu),
            "MOCK_OLLAMA_CPU_ONLY": "1" if self.cpu_only else "0",
            "MOCK_OLLAMA_PARTIAL_GPU_RESIDENCY": "1" if self.partial_gpu_residency else "0",
            "MOCK_OLLAMA_MODEL_DIGESTS": json.dumps({
                str(tag.get("name") or tag.get("model")): tag.get("digest")
                for tag in self.backend.tags
            }),
            "OLLAMA_UNIFY_CONFIG": os.path.join(self.temp_dir, "missing.conf"),
            "OLLAMA_UNIFY_BACKEND": f"127.0.0.1:{self.backend.server_port}",
            "OLLAMA_UNIFY_LISTEN": f"127.0.0.1:{self.proxy_port}",
            "OLLAMA_UNIFY_SOCKET": self.socket_path,
            "OLLAMA_UNIFY_LEASE_STATE": os.path.join(self.temp_dir, "leases.json"),
            "OLLAMA_UNIFY_MODEL_POLICY_STATE": os.path.join(self.temp_dir, "model-gpu-policy.json"),
            "OLLAMA_UNIFY_BACKEND_TYPE": "cuda",
            "OLLAMA_UNIFY_SELECTED_GPUS": ",".join(self.selected_gpus),
            "OLLAMA_UNIFY_MODEL_GPU_PREFERENCES": json.dumps(
                self.model_gpu_preferences or {}
            ),
            "OLLAMA_UNIFY_MODEL_CONTEXT_PROFILES": json.dumps(
                self.model_context_profiles or {}
            ),
            "OLLAMA_UNIFY_MAX_CONTEXT": str(self.max_context),
            # Scheduler fixtures retain their deliberate small fixed memory
            # promises; automatic maximum/KV admission has separate cases.
            "OLLAMA_UNIFY_AUTO_MODEL_CONTEXT": "1" if self.auto_model_context else "0",
            "OLLAMA_UNIFY_POOL_ENABLED": "1",
            "OLLAMA_UNIFY_POOL_MAX_SERVERS": str(self.max_servers),
            "OLLAMA_UNIFY_POOL_MAX_QUEUE": str(self.max_queue),
            "OLLAMA_UNIFY_POOL_RESUME_TTL": str(self.resume_ttl),
            "OLLAMA_UNIFY_COMPLETED_RESPONSE_TTL": str(self.completed_ttl),
            "OLLAMA_UNIFY_COMPLETED_RESPONSE_MAX_ENTRIES": str(
                self.completed_max_entries
            ),
            "OLLAMA_UNIFY_COMPLETED_RESPONSE_MAX_BODY_BYTES": str(
                self.completed_max_body_bytes
            ),
            "OLLAMA_UNIFY_COMPLETED_RESPONSE_MAX_TOTAL_BYTES": str(
                self.completed_max_total_bytes
            ),
            "OLLAMA_UNIFY_POOL_PORT_START": str(pool_port),
            "OLLAMA_UNIFY_POOL_INSTANCE_PARALLEL": str(
                self.instance_parallel
            ),
            "OLLAMA_UNIFY_POOL_IDLE_TIMEOUT": str(self.idle_timeout),
            "OLLAMA_UNIFY_POOL_READY_TIMEOUT": "3",
            "OLLAMA_UNIFY_POOL_LOAD_TIMEOUT": "3",
            "OLLAMA_UNIFY_POOL_VRAM_RESERVE_MIB": "1024",
            "OLLAMA_UNIFY_POOL_MODEL_OVERHEAD_PERCENT": "100",
            "OLLAMA_UNIFY_OLLAMA_BINARY": os.path.join(self.fixture_bin, "ollama"),
            "OLLAMA_UNIFY_DRAIN_TIMEOUT": "3",
            "OLLAMA_UNIFY_REQUEST_ACTIVITY_TTL": str(
                self.request_activity_ttl
            ),
            "OLLAMA_UNIFY_REQUEST_DETACHED_TTL": str(
                self.request_detached_ttl
            ),
            "OLLAMA_UNIFY_REQUEST_CANCEL_GRACE": str(
                self.request_cancel_grace
            ),
            "OLLAMA_UNIFY_PENDING_TIMEOUT": str(self.pending_timeout),
            "OLLAMA_UNIFY_REVOKE_TIMEOUT": str(self.revoke_timeout),
            "OLLAMA_UNIFY_UNLOAD_TIMEOUT": "3",
            "OLLAMA_UNIFY_ANON_POLL": "0.05",
            "OLLAMA_UNIFY_ANON_SETTLE": "0.1",
            "OLLAMA_UNIFY_ANON_MAX_DRAIN": "0.3",
            "OLLAMA_UNIFY_FOREIGN_RELEASE_TOLERANCE_MIB": str(
                self.release_tolerance_mib
            ),
            "OLLAMA_UNIFY_CLIENT_HISTORY_TTL": str(self.client_history_ttl),
            "OLLAMA_UNIFY_CLIENT_LANE_HISTORY_LIMIT": str(
                self.client_lane_history_limit
            ),
        })
        if self.runner_context_length is not None:
            env["MOCK_OLLAMA_CONTEXT_LENGTH"] = str(self.runner_context_length)
        if self.runner_gpu_scope is not None:
            env["MOCK_OLLAMA_GPU_USAGE_SCOPE"] = ",".join(self.runner_gpu_scope)
        # Unread PIPEs can fill during the 100-request test and block the
        # broker's logging thread, making a healthy scheduler time out.
        stdout = open(os.path.join(self.temp_dir, "broker.stdout.log"), "w+")
        stderr = open(os.path.join(self.temp_dir, "broker.stderr.log"), "w+")
        self.daemon = subprocess.Popen(
            [self.helper, "serve"], env=env, stdout=stdout,
            stderr=stderr, text=True,
        )
        self.daemon.stdout, self.daemon.stderr = stdout, stderr
        wait_ready(self.daemon, self.socket_path, self.proxy_port)
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.daemon.terminate()
        try:
            self.daemon.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.daemon.kill()
            self.daemon.wait()
        self.daemon.stdout.close()
        self.daemon.stderr.close()
        self.backend.shutdown()
        self.backend.server_close()
        self.temp.cleanup()

    def status(self):
        return control(self.socket_path, {"action": "status"})

    def capacity(self, model, parallel=1, endpoint=None, gpu_uuids=None):
        payload = {"model": model, "parallel": parallel}
        if endpoint is not None:
            payload["endpoint"] = endpoint
        if gpu_uuids is not None:
            payload["gpu_uuids"] = gpu_uuids
        return http_json(
            self.proxy_port, "POST",
            "/.well-known/ollama-unify-gpu-negotiator/capacity",
            payload,
        )


def test_existing_pool_contract(helper, fixture_bin):
    fake_ollama = os.path.join(fixture_bin, "ollama")
    backend = Backend()
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    proxy_port = free_port()
    pool_port = free_port()

    with tempfile.TemporaryDirectory(prefix="ollama-unify-pool-") as temp_dir:
        socket_path = os.path.join(temp_dir, "control.sock")
        event_log = os.path.join(temp_dir, "events.jsonl")
        env = os.environ.copy()
        env.update({
            "PATH": fixture_bin + os.pathsep + env.get("PATH", ""),
            "MOCK_PROFILE": "cuda_triple",
            "MOCK_OLLAMA_EVENT_LOG": event_log,
            "OLLAMA_UNIFY_CONFIG": os.path.join(temp_dir, "missing.conf"),
            "OLLAMA_UNIFY_BACKEND": f"127.0.0.1:{backend.server_port}",
            "OLLAMA_UNIFY_LISTEN": f"127.0.0.1:{proxy_port}",
            "OLLAMA_UNIFY_SOCKET": socket_path,
            "OLLAMA_UNIFY_LEASE_STATE": os.path.join(temp_dir, "leases.json"),
            "OLLAMA_UNIFY_BACKEND_TYPE": "cuda",
            "OLLAMA_UNIFY_SELECTED_GPUS": "GPU-large-0,GPU-large-1,GPU-large-2",
            "OLLAMA_UNIFY_POOL_ENABLED": "1",
            "OLLAMA_UNIFY_AUTO_MODEL_CONTEXT": "0",
            "OLLAMA_UNIFY_POOL_MAX_SERVERS": "3",
            "OLLAMA_UNIFY_POOL_PORT_START": str(pool_port),
            "OLLAMA_UNIFY_POOL_INSTANCE_PARALLEL": "1",
            "OLLAMA_UNIFY_POOL_IDLE_TIMEOUT": "30",
            "OLLAMA_UNIFY_POOL_READY_TIMEOUT": "3",
            "OLLAMA_UNIFY_POOL_LOAD_TIMEOUT": "3",
            "OLLAMA_UNIFY_POOL_VRAM_RESERVE_MIB": "1024",
            "OLLAMA_UNIFY_POOL_MODEL_OVERHEAD_PERCENT": "100",
            "OLLAMA_UNIFY_OLLAMA_BINARY": fake_ollama,
            "OLLAMA_UNIFY_DRAIN_TIMEOUT": "3",
            "OLLAMA_UNIFY_UNLOAD_TIMEOUT": "3",
            "OLLAMA_UNIFY_ANON_POLL": "0.1",
        })
        daemon = subprocess.Popen(
            [helper, "serve"], env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        try:
            wait_ready(daemon, socket_path, proxy_port)
            discovery_status, discovery, _ = http_json(
                proxy_port, "GET",
                "/.well-known/ollama-unify-gpu-negotiator",
            )
            assert discovery_status == 200, discovery
            assert discovery["parallel_pool"]["private_port_start"] == pool_port
            assert discovery["parallel_pool"]["private_port_end"] == pool_port + 31
            lifecycle = discovery["parallel_pool"]["request_lifecycle"]
            assert lifecycle["activity_ttl_seconds"] == 3
            assert lifecycle["detached_ttl_seconds"] == 30
            assert lifecycle["cancel_grace_seconds"] == 5
            assert lifecycle["tracked"] == 0
            admission_protocol = discovery["parallel_pool"]["admission_protocol"]
            assert admission_protocol["logical_request_header"] == (
                "X-Ollama-Unify-Logical-Request-Id"
            )
            assert admission_protocol["gpu_uuids_header"] == (
                "X-Ollama-Unify-GPU-UUIDs"
            )
            assert admission_protocol["gpu_constraint_semantics"] == (
                "ordered hard allowlist intersected with broker-selected GPUs; "
                "rejected only when the intersection is empty"
            )
            assert admission_protocol["queue_policies"] == ["wait", "yield"]
            assert admission_protocol["retry_after_json_field"] == "retry_after_ms"
            replay_contract = admission_protocol["completed_response_replay"]
            assert replay_contract["schema"] == (
                "io.ollama-unify.completed-response-cache.v1"
            )
            assert replay_contract["content_exposed"] is False
            assert replay_contract["max_entries"] >= 1
            assert replay_contract["max_body_bytes"] >= 0
            assert replay_contract["max_total_bytes"] >= 0
            status, capacity, _ = http_json(
                proxy_port, "POST",
                "/.well-known/ollama-unify-gpu-negotiator/capacity",
                {"model": MODEL, "parallel": 2},
            )
            assert status == 200, capacity
            assert capacity["schema"] == "io.ollama-unify.gpu-negotiator.capacity.v1"
            assert capacity["requested_parallel"] == 2
            assert capacity["admitted_parallel"] == 2
            assert capacity["public_ollama_api"] == f"http://127.0.0.1:{proxy_port}"
            assert len(capacity["lanes"]) == 2
            assert len({lane["gpu_uuid"] for lane in capacity["lanes"]}) == 2
            assert all("port" not in lane and "pid" not in lane and "host" not in lane
                       for lane in capacity["lanes"])
            starts = [event for event in events(event_log) if event["kind"] == "start"]
            assert len(starts) == 2
            assert len({event["host"] for event in starts}) == 2
            assert len({event["gpu"] for event in starts}) == 2

            _, repeated, _ = http_json(
                proxy_port, "POST",
                "/.well-known/ollama-unify-gpu-negotiator/capacity",
                {"model": MODEL, "parallel": 2},
            )
            assert repeated["admitted_parallel"] == 2
            assert len([event for event in events(event_log)
                        if event["kind"] == "start"]) == 2

            def chat(request_id):
                return http_json(proxy_port, "POST", "/api/chat", {
                    "model": MODEL, "stream": False, "mock_request_id": request_id,
                    "mock_delay": 0.5,
                })

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(chat, index) for index in (1, 2)]
                results = [future.result(timeout=5) for future in futures]
            assert all(result[0] == 200 for result in results)
            lane_headers = [result[2]["X-Ollama-Unify-Lane"] for result in results]
            assert len(set(lane_headers)) == 2
            ps_status, running, _ = http_json(proxy_port, "GET", "/api/ps")
            assert ps_status == 200
            assert len(running["models"]) == 2
            assert len({model["ollama_unify_lane"] for model in running["models"]}) == 2

            before = len([event for event in events(event_log) if event["kind"] == "start"])
            status, rejected, _ = http_json(
                proxy_port, "POST",
                "/.well-known/ollama-unify-gpu-negotiator/capacity",
                {"model": MODEL, "parallel": 4},
            )
            assert status == 422, rejected
            assert rejected["retryable"] is False
            assert rejected["reason_code"] == "parallel_exceeds_capacity"
            assert len([event for event in events(event_log)
                        if event["kind"] == "start"]) == before

            acquired = control(socket_path, {
                "action": "acquire", "owner": "pool-fixture",
                "requested_mib": 1024, "ttl": 30,
                "justification": "exercise pool drain behavior",
                "expected_duration_seconds": 300,
            })
            assert len(acquired["stopped_lanes"]) == 2
            assert len([event for event in events(event_log)
                        if event["kind"] == "stop"]) == 2
            token = acquired["lease"]["token"]
            remaining_lanes = control(socket_path, {"action": "status"})[
                "parallel_pool"
            ]["lanes"]
            assert len(remaining_lanes) == 1
            assert remaining_lanes[0]["kind"] == "system"
            metadata_started = time.monotonic()
            tags_status, tags, _ = http_json(proxy_port, "GET", "/api/tags")
            assert tags_status == 200
            assert tags["models"][0]["name"] == MODEL
            pending_ps_status, pending_models, _ = http_json(
                proxy_port, "GET", "/api/ps"
            )
            assert pending_ps_status == 200
            assert pending_models == {"models": []}
            assert time.monotonic() - metadata_started < 1
            control(socket_path, {"action": "ready", "token": token})
            control(socket_path, {"action": "release", "token": token})

            lazy_status, _, lazy_headers = chat(3)
            assert lazy_status == 200
            assert lazy_headers["X-Ollama-Unify-Lane"].startswith("lane-")
            assert len([event for event in events(event_log)
                        if event["kind"] == "start"]) == before + 1
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=10)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait()
            observed = events(event_log)
            started_pids = {event["pid"] for event in observed if event["kind"] == "start"}
            stopped_pids = {event["pid"] for event in observed if event["kind"] == "stop"}
            assert started_pids <= stopped_pids
            backend.shutdown()
            backend.server_close()


def test_model_gpu_preference_selects_requested_fitting_gpu(helper, fixture_bin):
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=1,
        model_gpu_preferences={MODEL: ["GPU-large-1"]},
    ) as harness:
        status, capacity, _ = harness.capacity(MODEL)
        assert status == 200, capacity
        lanes = managed_lanes(harness.status())
        assert len(lanes) == 1
        assert lanes[0]["gpu_uuid"] == "GPU-large-1"
        discovery_status, discovery, _ = http_json(
            harness.proxy_port,
            "GET",
            "/.well-known/ollama-unify-gpu-negotiator",
        )
        assert discovery_status == 200, discovery
        assert discovery["parallel_pool"]["model_gpu_preferences"] == {
            MODEL: ["GPU-large-1"],
        }


def test_managed_lane_rejects_cpu_fallback(helper, fixture_bin):
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=1,
        cpu_only=True,
    ) as harness:
        status, rejected, _ = harness.capacity(
            MODEL, gpu_uuids=["GPU-large-0"]
        )
        assert status == 503, rejected
        assert rejected["retryable"] is False
        assert rejected["reason_code"] == "gpu_runtime_unavailable"
        assert "size_vram=0" in rejected["error"]
        assert managed_lanes(harness.status()) == []


def test_default_context_is_pinned_on_warmup_and_inference(helper, fixture_bin):
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=1,
        max_context=8192,
    ) as harness:
        status, capacity, _ = harness.capacity(MODEL)
        assert status == 200, capacity
        warm = next(
            event for event in events(harness.event_log)
            if event["kind"] == "request"
            and event["model"] == MODEL
            and event["request_id"] is None
        )
        assert warm["options"]["num_ctx"] == 8192

        status, response, _ = chat(
            harness.proxy_port,
            MODEL,
            "default-context-request",
        )
        assert status == 200, response
        request = next(
            event for event in request_events(harness.event_log)
            if event["request_id"] == "default-context-request"
        )
        assert request["options"]["num_ctx"] == 8192


def test_fixed_model_context_is_attested_and_enforced(helper, fixture_bin):
    profile = {
        "context_length": 131072,
        "extra_vram_mib": 8192,
        "model_digest": MODEL_DIGEST,
    }
    tag = {
        "name": MODEL,
        "model": MODEL,
        "digest": MODEL_DIGEST,
        "size": 1024**3,
        "capabilities": ["completion"],
    }
    with PoolHarness(
        helper,
        fixture_bin,
        tags=[tag],
        model_context_profiles={MODEL: profile},
    ) as harness:
        status, discovery, _ = http_json(
            harness.proxy_port,
            "GET",
            "/.well-known/ollama-unify-gpu-negotiator",
        )
        assert status == 200, discovery
        policy = discovery["context_policy"]
        assert policy["default_max_context"] == 0
        assert policy["model_profiles"] == {MODEL: profile}
        assert policy["profile_context_is_fixed"] is True
        assert policy["managed_context_policy"] == "legacy_default"
        assert policy["model_query_parameter"] == "model"

        status, capacity, _ = harness.capacity(MODEL)
        assert status == 200, capacity
        assert capacity["lanes"][0]["context_profile"] == profile
        assert capacity["lanes"][0]["resolved_context_length"] == 131072
        assert capacity["lanes"][0]["reserved_mib"] == 1024 + 1024 + 8192
        warm = next(
            event for event in events(harness.event_log)
            if event["kind"] == "request"
            and event["model"] == MODEL
            and event["request_id"] is None
        )
        assert warm["options"]["num_ctx"] == 131072
        started = next(
            event for event in events(harness.event_log)
            if event["kind"] == "start"
        )
        assert started["context_length"] == 131072

        status, _, _ = http_json(
            harness.proxy_port,
            "POST",
            "/api/chat",
            {
                "model": MODEL,
                "stream": False,
                "mock_request_id": "profiled-request",
                "options": {"num_ctx": 8192},
            },
        )
        assert status == 200
        profiled = next(
            event for event in request_events(harness.event_log)
            if event["request_id"] == "profiled-request"
        )
        assert profiled["options"]["num_ctx"] == 131072

        status, rejected, _ = http_json(
            harness.proxy_port,
            "POST",
            "/api/chat",
            {
                "model": MODEL,
                "stream": False,
                "mock_request_id": "oversized-profiled-request",
                "options": {"num_ctx": 131073},
            },
        )
        assert status == 400, rejected
        assert rejected["reason_code"] == "model_context_policy_exceeded"

    wrong_tag = dict(tag, digest="b" * 64)
    with PoolHarness(
        helper,
        fixture_bin,
        tags=[wrong_tag],
        model_context_profiles={MODEL: profile},
    ) as harness:
        status, rejected, _ = harness.capacity(MODEL)
        assert status == 422, rejected
        assert rejected["reason_code"] == "model_context_identity_mismatch"


def test_capacity_gpu_constraint_is_hard_and_validated(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        initial_status, initial, _ = harness.capacity(MODEL)
        assert initial_status == 200, initial
        assert initial["lanes"][0]["gpu_uuid"] != "GPU-large-1"

        status, capacity, _ = harness.capacity(
            MODEL, gpu_uuids=["GPU-large-1"]
        )
        assert status == 200, capacity
        assert capacity["requested_gpu_uuids"] == ["GPU-large-1"]
        assert [lane["gpu_uuid"] for lane in capacity["lanes"]] == [
            "GPU-large-1"
        ]

        stale_status, stale, _ = harness.capacity(
            MODEL, gpu_uuids=["GPU-retired", "GPU-large-1", "GPU-missing"]
        )
        assert stale_status == 200, stale
        assert stale["requested_gpu_uuids"] == ["GPU-large-1"]
        assert [lane["gpu_uuid"] for lane in stale["lanes"]] == [
            "GPU-large-1"
        ]

        routed_status, routed, _ = chat(
            harness.proxy_port,
            MODEL,
            "stale-constraint-request",
            gpu_uuids=["GPU-retired", "GPU-large-1"],
        )
        assert routed_status == 200, routed
        routed_event = next(
            event for event in request_events(harness.event_log)
            if event["request_id"] == "stale-constraint-request"
        )
        assert routed_event["gpu"] == "GPU-large-1"

        invalid_status, invalid, _ = harness.capacity(
            MODEL, gpu_uuids=["GPU-display", "GPU-retired"]
        )
        assert invalid_status == 422, invalid
        assert invalid["reason_code"] == "gpu_constraint_unavailable"

        empty_status, empty, _ = harness.capacity(MODEL, gpu_uuids=[])
        assert empty_status == 422, empty
        assert empty["reason_code"] == "gpu_constraint_empty"


def test_lease_registration_metadata_is_advisory_and_visible(
    helper, fixture_bin,
):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        basic = control(harness.socket_path, {
            "action": "acquire",
            "owner": "tau-fixture",
            "requested_mib": 1024,
            "ttl": 30,
            "gpu_uuids": ["GPU-large-0"],
        })
        public = basic["public_lease"]
        assert public["justification"] == "not provided"
        assert public["expected_release_at"] is None
        assert public["expected_release_utc"] is None
        assert public["horizon_status"] == "unknown"
        assert len(public["recommendations"]) == 2
        assert basic["lease"]["ttl"] == 30
        assert basic["lease"]["expected_release_at"] == 0.0
        assert basic["lease"]["gpu_uuids"] == ["GPU-large-0"]
        status = harness.status()
        assert status["lease_summaries"][0]["recommendations"] == public["recommendations"]
        discovery = http_json(harness.proxy_port, "GET",
            "/.well-known/ollama-unify-gpu-negotiator")[1]
        assert discovery["active_leases"][0]["recommendations"] == public["recommendations"]
        assert basic["lease"]["token"] not in json.dumps(discovery)
        assert discovery["lease_policy"]["required_acquire_fields"] == ["owner"]
        assert discovery["lease_policy"]["recommended_acquire_fields"] == [
            "justification", "expected_duration_seconds",
        ]
        control(harness.socket_path, {
            "action": "ready", "token": basic["lease"]["token"],
        })
        control(harness.socket_path, {
            "action": "release", "token": basic["lease"]["token"],
        })
        cli = subprocess.run([
            helper, "acquire", "--owner", "portal-cli", "--gpu", "GPU-large-0",
            "--vram-mib", "1024", "--ttl", "30", "--token-only",
        ], env={**os.environ, "OLLAMA_UNIFY_SOCKET": harness.socket_path,
                "OLLAMA_UNIFY_CONFIG": os.path.join(harness.temp_dir, "missing.conf")},
            check=True, capture_output=True, text=True, timeout=10)
        token = cli.stdout.strip()
        assert token.startswith("lease_") and "\n" not in token
        assert "non-blocking" in cli.stderr
        assert "--justification" in cli.stderr and "--expected-duration" in cli.stderr
        control(harness.socket_path, {"action": "release", "token": token})

        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "tau-fixture",
            "requested_mib": 1024,
            "ttl": 30,
            "gpu_uuids": ["GPU-large-0"],
            "justification": "run the Tau simulation workload",
            "expected_duration_seconds": 900,
        })
        assert acquired["public_lease"]["owner"] == "tau-fixture"
        assert acquired["public_lease"]["justification"] == (
            "run the Tau simulation workload"
        )
        assert "token" not in acquired["public_lease"]
        assert acquired["public_lease"]["recommendations"] == []
        status = harness.status()
        assert status["lease_summaries"][0]["expected_release_utc"]
        assert "tau-fixture until" in status["warnings"][0]
        control(harness.socket_path, {
            "action": "release",
            "token": acquired["lease"]["token"],
        })


def test_lazy_inference_gpu_constraint_ignores_wrong_gpu_lane(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=2) as harness:
        initial_status, initial, _ = harness.capacity(MODEL)
        assert initial_status == 200, initial
        wrong_lane = initial["lanes"][0]
        assert wrong_lane["gpu_uuid"] != "GPU-large-1"

        status, payload, headers = chat(
            harness.proxy_port,
            MODEL,
            "hard-gpu-lazy",
            gpu_uuids=["GPU-large-1"],
        )
        assert status == 200, payload
        selected_lane = next(
            lane for lane in managed_lanes(harness.status())
            if lane["id"] == headers["X-Ollama-Unify-Lane"]
        )
        assert selected_lane["gpu_uuid"] == "GPU-large-1"


def test_idle_lane_reservations_prevent_future_gpu_overcommit(helper, fixture_bin):
    models = [f"fixture-reserved-{ordinal}:latest" for ordinal in range(3)]
    tags = [{
        "name": model,
        "model": model,
        "size": 30 * 1024**3,
        "capabilities": ["completion"],
    } for model in models]
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=3,
        tags=tags,
        runner_vram_mib=2048,
        model_gpu_preferences={model: ["GPU-large-1"] for model in models},
    ) as harness:
        for model in models:
            status, capacity, _ = harness.capacity(model)
            assert status == 200, capacity

        lanes = managed_lanes(harness.status())
        assert len(lanes) == 3
        placements = {lane["model"]: lane["gpu_uuid"] for lane in lanes}
        assert placements[models[0]] == "GPU-large-1"
        assert placements[models[1]] == "GPU-large-1"
        # Each fixture runner consumes only 2 GiB, while each lane promises
        # 31 GiB. Physical free memory would incorrectly admit all three on
        # the preferred GPU after an idle unload; durable reservations leave
        # only 18 GiB, so the soft preference must fall back.
        assert placements[models[2]] == "GPU-large-0"


def test_scoped_pending_lease_preserves_unreserved_inference(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=3, tags=[MODEL, OTHER_MODEL],
    ) as harness:
        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "scoped-fixture",
            "requested_mib": 1024,
            "ttl": 30,
            "justification": "exercise scoped inference continuity",
            "expected_duration_seconds": 300,
            "gpu_uuids": ["GPU-large-0", "GPU-large-1"],
        })
        token = acquired["lease"]["token"]
        assert acquired["lease"]["gpu_uuids"] == [
            "GPU-large-0", "GPU-large-1",
        ]
        pending = harness.status()
        assert pending["draining"] is False
        assert pending["leases"][0]["state"] == "pending"

        status, payload, headers = chat(
            harness.proxy_port, MODEL, "during-scoped-pending", timeout=10
        )
        assert status == 200, payload
        assert headers["X-Ollama-Unify-Lane"].startswith("lane-")
        managed = managed_lanes(harness.status())
        assert len(managed) == 1
        assert managed[0]["gpu_uuid"] == "GPU-large-2"
        starts = [event for event in events(harness.event_log)
                  if event["kind"] == "start"]
        assert starts
        assert all(event["gpu"] == "GPU-large-2" for event in starts)

        status, payload, headers = chat(
            harness.proxy_port, OTHER_MODEL, "replace-on-only-free-gpu", timeout=10
        )
        assert status == 200, payload
        assert headers["X-Ollama-Unify-Lane"].startswith("lane-")
        replacement = managed_lanes(harness.status())
        assert len(replacement) == 2
        assert {lane["model"] for lane in replacement} == {MODEL, OTHER_MODEL}
        assert {lane["gpu_uuid"] for lane in replacement} == {"GPU-large-2"}
        lifecycle = events(harness.event_log)
        assert not any(event["kind"] == "stop" for event in lifecycle)
        assert [
            event["gpu"] for event in lifecycle if event["kind"] == "start"
        ] == ["GPU-large-2", "GPU-large-2"]

        control(harness.socket_path, {"action": "ready", "token": token})
        released = control(
            harness.socket_path, {"action": "release", "token": token}
        )
        assert released["released"] == token
        assert harness.status()["draining"] is False


def test_scoped_release_tolerates_baseline_pid_churn_without_global_drain(
    helper, fixture_bin,
):
    with PoolHarness(helper, fixture_bin, max_servers=2, tags=[MODEL]) as harness:
        # A long-running external service can replace a worker PID while its
        # stable CUDA allocation remains identical. The lease baseline must be
        # compared by scoped GPU usage, not by process identity.
        write_compute_apps(harness.compute_apps, [
            (910101, "GPU-large-0", 1024),
        ])
        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "scoped-pid-churn-fixture",
            "requested_mib": 4096,
            "ttl": 30,
            "justification": "exercise scoped PID churn",
            "expected_duration_seconds": 300,
            "gpu_uuids": ["GPU-large-0"],
        })
        token = acquired["lease"]["token"]
        control(harness.socket_path, {"action": "ready", "token": token})

        capacity_status, capacity, _ = harness.capacity(MODEL)
        assert capacity_status == 200, capacity
        lane_ids = [lane["id"] for lane in managed_lanes(harness.status())]
        assert lane_ids

        write_compute_apps(harness.compute_apps, [
            (910202, "GPU-large-0", 1024),
        ])
        released = control(
            harness.socket_path, {"action": "release", "token": token}
        )
        assert released["released"] == token
        assert released["stopped_lanes"] == []
        status = harness.status()
        assert status["draining"] is False
        assert status["leases"] == []
        assert [lane["id"] for lane in managed_lanes(status)] == lane_ids


def test_scoped_release_tolerates_small_foreign_accounting_drift(
    helper, fixture_bin,
):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        write_compute_apps(harness.compute_apps, [
            (920101, "GPU-large-0", 1024),
        ])
        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "scoped-accounting-drift-fixture",
            "requested_mib": 4096,
            "ttl": 30,
            "justification": "exercise accounting drift tolerance",
            "expected_duration_seconds": 300,
            "gpu_uuids": ["GPU-large-0"],
        })
        token = acquired["lease"]["token"]
        control(harness.socket_path, {"action": "ready", "token": token})

        # nvidia-smi accounting for an unrelated resident process can move by
        # a small amount even when the leased workload has fully exited.
        write_compute_apps(harness.compute_apps, [
            (920202, "GPU-large-0", 1126),
        ])
        released = control(
            harness.socket_path, {"action": "release", "token": token}
        )
        assert released["released"] == token
        assert harness.status()["leases"] == []


def test_broker_queue_has_hard_ceiling_during_all_gpu_lease(
    helper, fixture_bin,
):
    with PoolHarness(
        helper, fixture_bin, max_servers=1, max_queue=2,
    ) as harness:
        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "all-gpu-queue-cap-fixture",
            "requested_mib": 4096,
            "ttl": 30,
            "justification": "exercise bounded broker queue",
            "expected_duration_seconds": 300,
            "gpu_uuids": ["GPU-large-0", "GPU-large-1", "GPU-large-2"],
        })
        token = acquired["lease"]["token"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(
                chat, harness.proxy_port, MODEL, "queue-cap-1", 0, 10
            )
            second = executor.submit(
                chat, harness.proxy_port, MODEL, "queue-cap-2", 0, 10
            )
            queue = wait_until(
                lambda: queue_with_depth(harness, 2),
                "broker queue to reach its configured ceiling",
            )
            assert queue["limit"] == 2
            assert sum(queue["phase_counts"].values()) == 2
            assert queue["phase_counts"].get("waiting-lease") == 1

            started = time.monotonic()
            status, payload, headers = chat(
                harness.proxy_port, MODEL, "queue-cap-rejected", timeout=3
            )
            assert status == 503, payload
            assert time.monotonic() - started < 1
            assert "broker queue is full (2/2)" in payload["error"]
            assert payload["retryable"] is True
            assert payload["reason_code"] == "queue_full"
            assert payload["retry_after_ms"] == 2000
            assert headers["Retry-After"] == "2"
            assert headers["X-Ollama-Unify-Retryable"] == "true"
            assert harness.status()["parallel_pool"]["queue"]["depth"] == 2

            released = control(
                harness.socket_path, {"action": "release", "token": token}
            )
            assert released["released"] == token
            assert first.result(timeout=10)[0] == 200
            assert second.result(timeout=10)[0] == 200

        final_queue = harness.status()["parallel_pool"]["queue"]
        assert final_queue["depth"] == 0
        assert final_queue["limit"] == 2
        assert final_queue["rejected_total"] >= 1


def test_scoped_active_lease_shares_stable_vram_with_ollama(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=2, tags=[MODEL],
    ) as harness:
        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "active-shared-fixture",
            "requested_mib": 32768,
            "ttl": 30,
            "justification": "exercise stable active sharing",
            "expected_duration_seconds": 300,
            "gpu_uuids": ["GPU-large-0"],
        })
        token = acquired["lease"]["token"]

        pending_status, pending_payload, _ = harness.capacity(
            MODEL, gpu_uuids=["GPU-large-0"]
        )
        assert pending_status == 503, pending_payload
        assert "free=[]" in pending_payload["error"]

        ready = control(harness.socket_path, {"action": "ready", "token": token})
        assert ready["lease"]["state"] == "active"

        active_status, active_capacity, _ = harness.capacity(
            MODEL, gpu_uuids=["GPU-large-0"]
        )
        assert active_status == 200, active_capacity
        assert active_capacity["admitted_parallel"] == 1
        assert len(active_capacity["lanes"]) == 1
        assert active_capacity["lanes"][0]["gpu_uuid"] == "GPU-large-0"

        response = chat(
            harness.proxy_port, MODEL, "during-active-scoped-lease", timeout=10,
            gpu_uuids=["GPU-large-0"],
        )
        assert response[0] == 200, response[1]
        assert response[2]["X-Ollama-Unify-Lane"].startswith("lane-")

        prepared = control(
            harness.socket_path, {"action": "prepare", "token": token}
        )
        assert prepared["lease"]["state"] == "pending"
        assert prepared["stopped_lanes"] == []
        assert [lane["id"] for lane in managed_lanes(harness.status())] == [
            active_capacity["lanes"][0]["id"]
        ]

        control(harness.socket_path, {"action": "ready", "token": token})
        released = control(
            harness.socket_path, {"action": "release", "token": token}
        )
        assert released["released"] == token


def test_scoped_acquire_preserves_managed_lanes_when_allocation_fits(
    helper, fixture_bin,
):
    with PoolHarness(
        helper, fixture_bin, max_servers=2, tags=[MODEL],
    ) as harness:
        initial_status, initial_capacity, _ = harness.capacity(MODEL)
        assert initial_status == 200, initial_capacity
        lane_id = initial_capacity["lanes"][0]["id"]
        lane_gpu = initial_capacity["lanes"][0]["gpu_uuid"]
        starts_before = len([
            event for event in events(harness.event_log)
            if event["kind"] == "start"
        ])

        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "fitting-scoped-fixture",
            "requested_mib": 32768,
            "ttl": 30,
            "justification": "exercise fitting scoped reservation",
            "expected_duration_seconds": 300,
            "gpu_uuids": [lane_gpu],
        })
        token = acquired["lease"]["token"]
        assert acquired["stopped_lanes"] == []
        pending_lanes = managed_lanes(harness.status())
        assert [lane["id"] for lane in pending_lanes] == [lane_id]

        control(harness.socket_path, {"action": "ready", "token": token})
        response = chat(
            harness.proxy_port, MODEL, "reuse-after-scoped-ready", timeout=10
        )
        assert response[0] == 200, response[1]
        assert response[2]["X-Ollama-Unify-Lane"] == lane_id
        assert len([
            event for event in events(harness.event_log)
            if event["kind"] == "start"
        ]) == starts_before

        released = control(
            harness.socket_path, {"action": "release", "token": token}
        )
        assert released["released"] == token


def exercise_multi_gpu_lease_exclusivity(
    helper, fixture_bin, *, profile, selected_gpus, scope, outside,
):
    """Hold `scope` exclusively while Ollama keeps serving from `outside`.

    `outside` must be the only unscoped GPU the fixture model fits on, so
    the unconstrained request below proves placement never falls back to
    reclaiming or loading on the peer scope.
    """
    with PoolHarness(
        helper, fixture_bin, max_servers=len(selected_gpus) + 2,
        tags=[MODEL, OTHER_MODEL], profile=profile,
        selected_gpus=selected_gpus,
    ) as harness:
        placed = {}
        for gpu_uuid in scope + [outside]:
            status, capacity, _ = harness.capacity(
                MODEL if gpu_uuid != outside else OTHER_MODEL,
                gpu_uuids=[gpu_uuid],
            )
            assert status == 200, capacity
            placed[gpu_uuid] = {
                lane["id"] for lane in capacity["lanes"]
                if lane["gpu_uuid"] == gpu_uuid
            }

        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "tensor-parallel-fixture",
            "requested_mib": 4096,
            "ttl": 30,
            "justification": "exercise exclusive multi-GPU peer scope",
            "expected_duration_seconds": 300,
            "gpu_uuids": scope,
        })
        token = acquired["lease"]["token"]
        assert set(acquired["stopped_lanes"]) == set().union(
            *(placed[gpu_uuid] for gpu_uuid in scope)
        )
        surviving = managed_lanes(harness.status())
        assert {lane["gpu_uuid"] for lane in surviving} == {outside}

        ready = control(harness.socket_path, {"action": "ready", "token": token})
        assert ready["lease"]["state"] == "active"
        lifecycle_at_ready = len(events(harness.event_log))

        for gpu_uuid in scope:
            status, payload, _ = harness.capacity(MODEL, gpu_uuids=[gpu_uuid])
            assert status == 503, payload
            assert "free=[]" in payload["error"]
            status, payload, _ = chat(
                harness.proxy_port, MODEL, f"blocked-{gpu_uuid}", timeout=10,
                gpu_uuids=[gpu_uuid],
            )
            assert status == 503, payload

        # Unconstrained demand for a model with no lane must be placed off
        # the scope, never by reclaiming or loading on the peer GPUs.
        status, payload, headers = chat(
            harness.proxy_port, MODEL, "unconstrained-during-peer-lease",
            timeout=10,
        )
        assert status == 200, payload
        served = next(
            lane for lane in managed_lanes(harness.status())
            if lane["id"] == headers["X-Ollama-Unify-Lane"]
        )
        assert served["gpu_uuid"] == outside
        assert not [
            event for event in events(harness.event_log)[lifecycle_at_ready:]
            if event["kind"] in ("start", "stop")
            and event.get("gpu") in scope
        ]
        assert not [
            lane for lane in managed_lanes(harness.status())
            if lane["gpu_uuid"] in scope
        ]

        released = control(
            harness.socket_path, {"action": "release", "token": token}
        )
        assert released["released"] == token
        status, capacity, _ = harness.capacity(MODEL, gpu_uuids=[scope[-1]])
        assert status == 200, capacity
        assert [lane["gpu_uuid"] for lane in capacity["lanes"]] == [scope[-1]]


def test_multi_gpu_lease_is_exclusive_for_its_lifetime(helper, fixture_bin):
    # Regression for a fatal NVLink Xid 74 host lockup: a tensor-parallel
    # owner held two GPUs while the broker reclaimed and reloaded Ollama
    # lanes on one of them during live peer-to-peer traffic. The rule is
    # topology- and size-agnostic, so exercise equal and uneven arrays.
    exercise_multi_gpu_lease_exclusivity(
        helper, fixture_bin, profile="cuda_triple",
        selected_gpus=["GPU-large-0", "GPU-large-1", "GPU-large-2"],
        scope=["GPU-large-0", "GPU-large-2"], outside="GPU-large-1",
    )
    # 48/24/16/8 GB devices: the scope spans the largest and a small card,
    # and the 8 GB device is too small for the fixture lane, leaving the
    # 24 GB device as the only legal placement.
    exercise_multi_gpu_lease_exclusivity(
        helper, fixture_bin, profile="cuda_uneven",
        selected_gpus=[
            "GPU-uneven-0", "GPU-uneven-1", "GPU-uneven-2", "GPU-uneven-3",
        ],
        scope=["GPU-uneven-0", "GPU-uneven-2"], outside="GPU-uneven-1",
    )


def test_operator_revoke_and_lane_stop(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=2, tags=[MODEL],
    ) as harness:
        status, capacity, _ = harness.capacity(MODEL, gpu_uuids=["GPU-large-1"])
        assert status == 200, capacity
        lane_id = capacity["lanes"][0]["id"]

        missing = control_raw(harness.socket_path, {
            "action": "stop_lane", "lane_id": "lane-missing",
        })
        assert missing["ok"] is False
        base = control_raw(harness.socket_path, {
            "action": "stop_lane", "lane_id": "base",
        })
        assert base["ok"] is False
        assert "broker-managed" in base["error"]

        stopped = control(harness.socket_path, {
            "action": "stop_lane", "lane_id": lane_id,
        })
        assert stopped["stopped_lanes"] == [lane_id]
        assert managed_lanes(harness.status()) == []
        assert any(
            event["kind"] == "stop" and event["gpu"] == "GPU-large-1"
            for event in events(harness.event_log)
        )

        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "operator-revoke-fixture",
            "requested_mib": 1024,
            "ttl": 30,
            "justification": "exercise operator revoke",
            "expected_duration_seconds": 300,
            "gpu_uuids": ["GPU-large-0"],
        })
        token = acquired["lease"]["token"]
        control(harness.socket_path, {"action": "ready", "token": token})
        revoked = control(harness.socket_path, {
            "action": "revoke", "token": token, "reason": "fixture operator",
        })
        assert revoked["lease"]["state"] == "revoking"
        heartbeat = control_raw(harness.socket_path, {
            "action": "heartbeat", "token": token,
        })
        assert heartbeat["ok"] is False
        assert "revoked" in heartbeat["error"]
        blocked_status, blocked, _ = harness.capacity(
            MODEL, gpu_uuids=["GPU-large-0"]
        )
        assert blocked_status == 503, blocked
        assert control_raw(harness.socket_path, {
            "action": "revoke", "token": "lease_unknown",
        })["ok"] is False

        released = control(
            harness.socket_path, {"action": "release", "token": token}
        )
        assert released["released"] == token


def test_client_attribution(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=2, tags=[MODEL, OTHER_MODEL],
    ) as harness:
        status, payload, headers = chat(
            harness.proxy_port, MODEL, "declared-client", client="fixture-app",
        )
        assert status == 200, payload
        first_lane = headers["X-Ollama-Unify-Lane"]
        status, payload, headers = chat(
            harness.proxy_port, MODEL, "declared-client-again",
            client="fixture-app",
        )
        assert status == 200, payload
        status, payload, headers = chat(
            harness.proxy_port, OTHER_MODEL, "undeclared-client",
        )
        assert status == 200, payload
        other_lane = headers["X-Ollama-Unify-Lane"]

        current = harness.status()
        clients = {client["key"]: client for client in current["clients"]}
        declared = clients["app:fixture-app"]
        assert declared["requests"] == 2
        assert declared["models"] == {MODEL: 2}
        assert [usage["id"] for usage in declared["lanes"]] == [first_lane]
        assert declared["lanes"][0]["live"] is True

        undeclared = next(
            client for key, client in clients.items() if key != "app:fixture-app"
        )
        identity = undeclared["identity"]
        assert identity["address"] == "127.0.0.1"
        assert identity["label"] and undeclared["key"] != "app:fixture-app"
        # Socket diagnostics resolve local callers without privileges; the
        # test process owns the connection, so it resolves to exactly us.
        if "uid" in identity:
            assert identity["uid"] == os.getuid()
            assert identity.get("pid") == os.getpid(), identity
            # Hosted runners can execute the caller inside a systemd service;
            # service identity intentionally takes precedence over process name.
            unit = identity.get("unit", "")
            expected_label = unit if unit.endswith(".service") else identity["process"]
            assert identity["label"].startswith(expected_label), identity

        lanes = {lane["id"]: lane for lane in managed_lanes(current)}
        assert lanes[first_lane]["triggered_by"] == {
            "key": "app:fixture-app", "label": declared["identity"]["label"],
        }
        assert lanes[first_lane]["clients"][0]["key"] == "app:fixture-app"
        assert lanes[first_lane]["clients"][0]["requests"] == 2
        assert lanes[other_lane]["triggered_by"]["key"] == undeclared["key"]


def test_client_history_ttl_and_lane_bound(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=1,
        client_history_ttl=60, client_lane_history_limit=2,
    ) as harness:
        lane_ids = []
        for index in range(3):
            status, payload, headers = chat(
                harness.proxy_port, MODEL, f"history-{index}",
                client="history-client",
            )
            assert status == 200, payload
            lane_id = headers["X-Ollama-Unify-Lane"]
            lane_ids.append(lane_id)
            wait_until(
                lambda: all(
                    lane["in_flight"] == 0
                    for lane in managed_lanes(harness.status())
                    if lane["id"] == lane_id
                ),
                "client-history request finalization",
            )
            control(harness.socket_path, {
                "action": "stop_lane", "lane_id": lane_id,
            })

        current = harness.status()
        policy = current["client_history_policy"]
        assert policy == {
            "ttl_seconds": 60.0,
            "max_clients": 256,
            "max_lanes_per_client": 2,
            "max_clients_per_lane": 8,
        }
        history = next(
            client for client in current["clients"]
            if client["key"] == "app:history-client"
        )
        assert history["expires_at"] == history["last_seen"] + 60
        assert [usage["id"] for usage in history["lanes"]] == lane_ids[-2:][::-1]
        assert all(
            usage["expires_at"] == usage["last_seen"] + 60
            for usage in history["lanes"]
        )

    with PoolHarness(
        helper, fixture_bin, max_servers=1, client_history_ttl=0.4,
    ) as harness:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            active = executor.submit(
                chat, harness.proxy_port, MODEL, "active-past-history-ttl",
                1.2, 5, client="active-client",
            )
            wait_until(
                lambda: any(
                    client["key"] == "app:active-client"
                    for client in harness.status()["clients"]
                ),
                "active client attribution",
            )
            time.sleep(0.5)
            assert any(
                client["key"] == "app:active-client"
                for client in harness.status()["clients"]
            ), "an in-flight request must survive history TTL pruning"
            assert active.result(timeout=5)[0] == 200

        time.sleep(0.5)
        expired = harness.status()
        assert not expired["clients"]
        assert all(
            not lane["clients"] for lane in managed_lanes(expired)
        ), "per-lane Used by history must expire with the same TTL"


def model_lane_gpus(harness, model):
    return sorted(
        lane["gpu_uuid"] for lane in managed_lanes(harness.status())
        if lane["model"] == model and lane["state"] == "ready"
    )


def test_operator_model_gpu_policy(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=3, tags=[MODEL],
    ) as harness:
        status, capacity, _ = harness.capacity(MODEL, gpu_uuids=["GPU-large-0"])
        assert status == 200, capacity
        first_lane = capacity["lanes"][0]["id"]

        # An idle lane on a newly disallowed GPU moves to an allowed one.
        moved = control(harness.socket_path, {
            "action": "set_model_gpus", "model": MODEL_WITHOUT_TAG,
            "gpu_uuids": ["GPU-large-1"],
        })
        assert moved["allowed_gpu_uuids"] == ["GPU-large-1"]
        assert moved["stopping_lanes"] == [first_lane]
        wait_until(
            lambda: model_lane_gpus(harness, MODEL) == ["GPU-large-1"],
            "lane moved to the allowed GPU", timeout=20,
        )
        assert harness.status()["model_gpu_policy"] == {MODEL: ["GPU-large-1"]}
        with open(os.path.join(harness.temp_dir, "model-gpu-policy.json"),
                  encoding="utf-8") as stream:
            assert json.load(stream)["models"] == {MODEL: ["GPU-large-1"]}

        status, payload, headers = chat(harness.proxy_port, MODEL, "on-policy")
        assert status == 200, payload
        served = next(lane for lane in managed_lanes(harness.status())
                      if lane["id"] == headers["X-Ollama-Unify-Lane"])
        assert served["gpu_uuid"] == "GPU-large-1"

        status, payload, _ = harness.capacity(MODEL, gpu_uuids=["GPU-large-0"])
        assert status == 409, payload
        assert payload["reason_code"] == "gpu_policy_conflict"
        status, payload, _ = chat(
            harness.proxy_port, MODEL, "off-policy", gpu_uuids=["GPU-large-0"],
        )
        assert status == 409, payload

        for bad in ([], ["GPU-missing"]):
            refused = control_raw(harness.socket_path, {
                "action": "set_model_gpus", "model": MODEL, "gpu_uuids": bad,
            })
            assert refused["ok"] is False, refused

        # A busy lane finishes its request before it moves.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            busy = executor.submit(
                chat, harness.proxy_port, MODEL, "busy-during-move", delay=2,
            )
            wait_until(
                lambda: any(lane["in_flight"] for lane in
                            managed_lanes(harness.status())),
                "request in flight",
            )
            retired = control(harness.socket_path, {
                "action": "set_model_gpus", "model": MODEL,
                "gpu_uuids": ["GPU-large-2"],
            })
            assert retired["retiring_lanes"] and not retired["stopping_lanes"]
            status, payload, _ = busy.result(timeout=20)
            assert status == 200, payload
        wait_until(
            lambda: model_lane_gpus(harness, MODEL) == ["GPU-large-2"],
            "busy lane moved after its request", timeout=20,
        )

        # Allowing every brokered GPU clears the policy.
        cleared = control(harness.socket_path, {
            "action": "set_model_gpus", "model": MODEL,
            "gpu_uuids": ["GPU-large-0", "GPU-large-1", "GPU-large-2"],
        })
        assert cleared["allowed_gpu_uuids"] is None
        assert harness.status()["model_gpu_policy"] == {}
        status, capacity, _ = harness.capacity(MODEL, gpu_uuids=["GPU-large-0"])
        assert status == 200, capacity


def capacity_when_ready(harness, model, parallel=1):
    status, payload, headers = harness.capacity(model, parallel)
    return (status, payload, headers) if status == 200 else None


def managed_lanes(status):
    return [lane for lane in status["parallel_pool"]["lanes"]
            if lane["kind"] == "managed"]


def queue_with_depth(harness, depth):
    queue = harness.status()["parallel_pool"]["queue"]
    return queue if queue["depth"] == depth else None


def test_foreign_gpu_transition_stability(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        status, capacity, _ = harness.capacity(MODEL)
        assert status == 200, capacity
        # Warm-up and normal inference cannot restore native idle expiration.
        assert all(event["keep_alive"] == -1 for event in events(harness.event_log)
                   if event["kind"] == "request")
        result = http_json(harness.proxy_port, "POST", "/api/generate",
                           {"model": MODEL, "prompt": "fixture", "stream": False, "keep_alive": "1s"})
        assert result[0] == 200, result
        assert [event for event in events(harness.event_log) if event["kind"] == "request"][-1]["keep_alive"] == -1
        start = wait_until(
            lambda: next((event for event in events(harness.event_log)
                          if event["kind"] == "start"), None),
            "managed lane start",
        )

        # A managed child must never be classified as an anonymous CUDA process,
        # including in this fixture where no real systemd cgroup is available.
        write_compute_apps(harness.compute_apps, [
            (start["pid"], start["gpu"], 1024),
        ])
        time.sleep(0.3)
        assert not [event for event in events(harness.event_log)
                    if event["kind"] == "stop"]

        # Process churn on an unselected GPU and allocation departure are inert.
        write_compute_apps(harness.compute_apps, [
            (start["pid"], start["gpu"], 1024),
            (910001, "GPU-unselected", 4096),
        ])
        time.sleep(0.3)
        assert not [event for event in events(harness.event_log)
                    if event["kind"] == "stop"]
        write_compute_apps(harness.compute_apps, [])
        time.sleep(0.3)
        assert not [event for event in events(harness.event_log)
                    if event["kind"] == "stop"]

        # Foreign CUDA never triggers weight eviction or a reload. It blocks
        # even a resident lane until the unknown owner's complete scope exits.
        write_compute_apps(harness.compute_apps, [(910002, start["gpu"], 1024)])
        wait_until(lambda: start["gpu"] in harness.status()["unregistered_gpu_quarantine"],
                   "unregistered allocation quarantined")
        status, _, _ = harness.capacity(MODEL, gpu_uuids=[start["gpu"]])
        assert status == 503
        write_compute_apps(harness.compute_apps, [(910002, start["gpu"], 2048)])
        time.sleep(0.3)
        assert not [event for event in events(harness.event_log) if event["kind"] == "stop"]
        assert len([event for event in events(harness.event_log) if event["kind"] == "start"]) == 1
        write_compute_apps(harness.compute_apps, [])
        wait_until(lambda: not harness.status()["unregistered_gpu_quarantine"],
                   "foreign process exit observed before resident lane reuse")
        status, capacity, _ = harness.capacity(MODEL, gpu_uuids=[start["gpu"]])
        assert status == 200, capacity
        assert len([event for event in events(harness.event_log) if event["kind"] == "start"]) == 1


def test_unregistered_peer_work_defers_every_lane_transition(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=4, tags=[MODEL, OTHER_MODEL], idle_timeout=3) as harness:
        pair = ["GPU-large-0", "GPU-large-2"]
        for gpu in pair:
            status, capacity, _ = harness.capacity(MODEL, gpu_uuids=[gpu])
            assert status == 200, capacity
        initial = managed_lanes(harness.status())
        write_compute_apps(harness.compute_apps, [(910003, gpu, 4096) for gpu in pair])
        wait_until(lambda: set(harness.status()["unregistered_gpu_quarantine"]) == set(pair),
                   "both peer GPUs quarantined")
        for gpu in pair:
            status, _, _ = harness.capacity(OTHER_MODEL, gpu_uuids=[gpu])
            assert status == 503
        status, capacity, _ = harness.capacity(OTHER_MODEL, gpu_uuids=["GPU-large-1"])
        assert status == 200, capacity
        # Explicit stop must retain the process and reservation, too.
        result = control_raw(harness.socket_path, {"action": "stop_lane", "lane_id": initial[0]["id"]})
        assert not result["ok"], result
        assert len(managed_lanes(harness.status())) == 3
        assert not [event for event in events(harness.event_log) if event["kind"] == "stop"]
        # Even idle expiry cannot unload until foreign CUDA has gone.
        wait_until(lambda: any(event["kind"] == "stop" and event["gpu"] == "GPU-large-1"
                               for event in events(harness.event_log)), "unrelated idle lane retires", timeout=6)
        stopped_gpus = {event["gpu"] for event in events(harness.event_log) if event["kind"] == "stop"}
        assert not stopped_gpus.intersection(pair), stopped_gpus
        wait_until(lambda: set(lane["gpu_uuid"] for lane in managed_lanes(harness.status())) == set(pair),
                   "unrelated backend process-group exit observed", timeout=3)
        assert set(lane["gpu_uuid"] for lane in managed_lanes(harness.status())) == set(pair)
        write_compute_apps(harness.compute_apps, [])
        wait_until(lambda: not managed_lanes(harness.status()), "deferred lanes retire after peer exit", timeout=12)


def test_native_explicit_unload_retires_backend(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin) as harness:
        code, capacity, _ = harness.capacity(MODEL, gpu_uuids=['GPU-large-0'])
        assert code == 200, capacity
        code, response, _ = http_json(harness.proxy_port, 'POST', '/api/generate',
                                     {'model': MODEL, 'keep_alive': 0, 'stream': False},
                                     extra_headers={'X-Ollama-Unify-GPU-UUIDs': 'GPU-large-0'})
        assert code == 200, response
        wait_until(lambda: any(event['kind'] == 'stop' for event in events(harness.event_log)),
                   'explicit unload retires the empty backend')
        # The fixture logs SIGTERM before the broker observes process-group exit.
        wait_until(lambda: not managed_lanes(harness.status()),
                   'explicit unload observes backend process-group exit')



def test_implicit_latest_uses_one_lane(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        status, capacity, _ = harness.capacity(MODEL_WITHOUT_TAG)
        assert status == 200, capacity
        assert capacity["canonical_model"] == MODEL
        lane_id = capacity["lanes"][0]["id"]

        explicit = chat(harness.proxy_port, MODEL, "explicit-latest")
        implicit = chat(harness.proxy_port, MODEL_WITHOUT_TAG, "implicit-latest")
        assert explicit[0] == implicit[0] == 200
        assert explicit[2]["X-Ollama-Unify-Lane"] == lane_id
        assert implicit[2]["X-Ollama-Unify-Lane"] == lane_id
        assert len([event for event in events(harness.event_log)
                    if event["kind"] == "start"]) == 1
        assert [event["model"] for event in request_events(harness.event_log)] == [
            MODEL, MODEL_WITHOUT_TAG,
        ]


def test_idle_lane_replacement(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=1, tags=[MODEL, OTHER_MODEL]
    ) as harness:
        status, first_capacity, _ = harness.capacity(MODEL)
        assert status == 200, first_capacity
        first_lane = first_capacity["lanes"][0]["id"]
        first = chat(harness.proxy_port, MODEL, "first-model")
        assert first[0] == 200

        second = chat(harness.proxy_port, OTHER_MODEL, "replacement-model", timeout=8)
        assert second[0] == 200, second
        assert second[2]["X-Ollama-Unify-Lane"] != first_lane
        observed = events(harness.event_log)
        assert len([event for event in observed if event["kind"] == "start"]) == 2
        assert len([event for event in observed if event["kind"] == "stop"]) == 1
        lanes = managed_lanes(harness.status())
        assert len(lanes) == 1
        assert lanes[0]["model"] == OTHER_MODEL


def test_background_yield_preserves_idle_lane_at_process_ceiling(helper, fixture_bin):
    for replacement_class, replacement_policy in (
        ("foreground", "yield"), ("background", "wait"),
    ):
        with PoolHarness(
            helper, fixture_bin, max_servers=1, tags=[MODEL, OTHER_MODEL]
        ) as harness:
            status, first, _ = harness.capacity(MODEL)
            assert status == 200, first
            first_lane = first["lanes"][0]["id"]
            started = time.monotonic()
            result = chat(
                harness.proxy_port, OTHER_MODEL, "optional-replacement",
                logical_request_id="turn:optional-replacement",
                workload_class="background", queue_policy="yield",
            )
            assert result[0] == 503, result
            assert result[1]["reason_code"] == "background_capacity_deferred"
            assert result[1]["retryable"] is True
            assert result[2]["X-Ollama-Unify-Retryable"] == "true"
            assert "Retry-After" in result[2]
            assert "admission_retained" not in result[1]
            assert time.monotonic() - started < 2, result
            pool = harness.status()["parallel_pool"]
            assert pool["queue"]["depth"] == 0
            assert pool["queue"]["retained_request_bytes"] == 0
            assert [lane["id"] for lane in managed_lanes(harness.status())] == [first_lane]
            assert not [event for event in events(harness.event_log) if event["kind"] == "stop"]
            assert not any(event.get("request_id") == "optional-replacement"
                           for event in request_events(harness.event_log))
            reused = chat(harness.proxy_port, MODEL, "still-warm")
            assert reused[0] == 200, reused
            assert reused[2]["X-Ollama-Unify-Lane"] == first_lane

            # Only the explicit background/yield combination is non-disruptive.
            replacement = chat(
                harness.proxy_port, OTHER_MODEL, "allowed-replacement",
                workload_class=replacement_class, queue_policy=replacement_policy,
            )
            assert replacement[0] == 200, replacement
            assert replacement[2]["X-Ollama-Unify-Lane"] != first_lane
            assert len([event for event in events(harness.event_log)
                        if event["kind"] == "stop"]) == 1


def test_background_yield_preserves_vram_without_process_overflow(helper, fixture_bin):
    model_size = 60 * 1024**3
    tags = [{"name": model, "model": model, "size": model_size,
             "capabilities": ["completion"]} for model in (MODEL, OTHER_MODEL)]
    with PoolHarness(
        helper, fixture_bin, max_servers=4, tags=tags,
        selected_gpus=["GPU-large-0"],
        runner_vram_mib=model_size // (1024 * 1024) + 1024,
    ) as harness:
        status, first, _ = harness.capacity(MODEL)
        assert status == 200, first
        result = chat(harness.proxy_port, OTHER_MODEL, "optional-vram",
                      workload_class="background", queue_policy="yield")
        assert result[0] == 503, result
        assert result[1]["reason_code"] == "background_capacity_deferred"
        assert result[1]["retryable"] is True
        assert [lane["id"] for lane in managed_lanes(harness.status())] == [first["lanes"][0]["id"]]
        assert not [event for event in events(harness.event_log) if event["kind"] == "stop"]


def test_background_yield_can_cold_load_free_capacity_and_reuse_it(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=2,
                     tags=[MODEL, OTHER_MODEL]) as harness:
        assert harness.capacity(MODEL)[0] == 200
        first = chat(harness.proxy_port, OTHER_MODEL, "optional-free-placement",
                     workload_class="background", queue_policy="yield")
        assert first[0] == 200, first
        reused = chat(harness.proxy_port, OTHER_MODEL, "optional-reuse",
                      workload_class="background", queue_policy="yield")
        assert reused[0] == 200, reused
        assert reused[2]["X-Ollama-Unify-Lane"] == first[2]["X-Ollama-Unify-Lane"]
        assert len([event for event in events(harness.event_log) if event["kind"] == "start"]) == 2
        assert not [event for event in events(harness.event_log) if event["kind"] == "stop"]


def test_live_vram_reclaims_idle_lane_below_process_ceiling(helper, fixture_bin):
    model_size = 60 * 1024**3
    tags = [
        {"name": MODEL, "model": MODEL, "size": model_size,
         "capabilities": ["completion"]},
        {"name": OTHER_MODEL, "model": OTHER_MODEL, "size": model_size,
         "capabilities": ["completion"]},
    ]
    required_mib = (model_size // (1024 * 1024)) + 1024
    with PoolHarness(
        helper, fixture_bin, max_servers=6, tags=tags,
        runner_vram_mib=required_mib,
    ) as harness:
        status, first_capacity, _ = harness.capacity(MODEL, parallel=3)
        assert status == 200, first_capacity
        assert len(first_capacity["lanes"]) == 3

        status, payload, _ = chat(
            harness.proxy_port, OTHER_MODEL, "vram-reclamation", timeout=10
        )
        assert status == 200, payload
        observed = events(harness.event_log)
        assert len([event for event in observed if event["kind"] == "start"]) == 4
        assert len([event for event in observed if event["kind"] == "stop"]) == 1
        lanes = managed_lanes(harness.status())
        assert len(lanes) == 3
        assert [lane["model"] for lane in lanes].count(MODEL) == 2
        assert [lane["model"] for lane in lanes].count(OTHER_MODEL) == 1


def test_lazy_parallel_scaling(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=3) as harness:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    chat, harness.proxy_port, MODEL, f"lazy-{index}", 0.5, 10,
                    mock_barrier_requests=3,
                )
                for index in (1, 2, 3)
            ]
            results = [future.result(timeout=10) for future in futures]
        assert all(result[0] == 200 for result in results)
        assert len({result[2]["X-Ollama-Unify-Lane"] for result in results}) == 3
        assert len([event for event in events(harness.event_log)
                    if event["kind"] == "start"]) == 3
        queue = harness.status()["parallel_pool"]["queue"]
        assert queue["enqueued_total"] == 3
        assert queue["admitted_total"] == 3
        assert queue["cancelled_total"] == 0


def test_fifo_queue_and_metrics(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        status, capacity, _ = harness.capacity(MODEL)
        assert status == 200, capacity
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(
                chat, harness.proxy_port, MODEL, "fifo-1", 0.8, 8
            )
            wait_until(
                lambda: any(event.get("request_id") == "fifo-1"
                            for event in request_events(harness.event_log)),
                "first FIFO request admission",
            )
            second = executor.submit(
                chat, harness.proxy_port, MODEL, "fifo-2", 0.05, 8
            )
            first_queue = wait_until(
                lambda: queue_with_depth(harness, 1),
                "one queued FIFO request",
            )
            third = executor.submit(
                chat, harness.proxy_port, MODEL, "fifo-3", 0.05, 8
            )
            queue = wait_until(
                lambda: queue_with_depth(harness, 2),
                "two queued FIFO requests",
            )
            assert queue["by_model"] == {MODEL: 2}
            assert queue["phase_counts"].get("queued") == 2
            assert queue["oldest_wait_ms"] >= first_queue["oldest_wait_ms"]
            assert [item["position"] for item in queue["requests"]] == [1, 2]
            assert all(item["model"] == MODEL for item in queue["requests"])

            results = [future.result(timeout=8) for future in (first, second, third)]

        assert all(result[0] == 200 for result in results)
        assert [event["request_id"] for event in request_events(
            harness.event_log
        )] == ["fifo-1", "fifo-2", "fifo-3"]
        assert queue_with_depth(harness, 0)
        for result in results:
            assert result[2]["X-Ollama-Unify-Request-Id"]
            assert int(result[2]["X-Ollama-Unify-Queue-Ms"]) >= 0
            assert int(result[2]["X-Ollama-Unify-Queue-Position"]) >= 1
        final_queue = harness.status()["parallel_pool"]["queue"]
        assert final_queue["enqueued_total"] == 3
        assert final_queue["admitted_total"] == 3
        assert final_queue["peak"] >= 2
        assert final_queue["wait_ms_max"] >= final_queue["wait_ms_mean"]


def test_logical_request_resumes_fifo_without_reenqueue(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1, resume_ttl=2.0) as harness:
        status, capacity, _ = harness.capacity(MODEL)
        assert status == 200, capacity
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            active = executor.submit(
                chat, harness.proxy_port, MODEL, "logical-active", 0.8, 8
            )
            wait_until(
                lambda: any(
                    event.get("request_id") == "logical-active"
                    for event in request_events(harness.event_log)
                ),
                "active request before resumable admission",
            )

            timed_out = chat(
                harness.proxy_port,
                MODEL,
                "logical-run",
                0.5,
                3,
                logical_request_id="turn:logical-resume",
                admission_wait_ms=100,
            )
            assert timed_out[0] == 503, timed_out
            timeout_payload = timed_out[1]
            timeout_headers = timed_out[2]
            assert timeout_payload["reason_code"] == "queue_admission_timeout"
            assert timeout_payload["retryable"] is True
            assert timeout_payload["retry_after_ms"] == 2000
            assert timeout_payload["admission_retained"] is True
            assert timeout_payload["logical_request_id"] == "turn:logical-resume"
            assert timeout_payload["queue_position"] == 1
            assert timeout_payload["queue_ticket"] >= 1
            assert timeout_headers["Retry-After"] == "2"
            assert timeout_headers["X-Ollama-Unify-Admission-Retained"] == "true"
            assert timeout_headers["X-Ollama-Unify-Queue-Position"] == "1"
            assert int(timeout_headers["X-Ollama-Unify-Queue-Ticket"]) >= 1
            broker_request_id = timeout_payload["request_id"]
            broker_queue_ticket = timeout_payload["queue_ticket"]

            conflict = chat(
                harness.proxy_port,
                MODEL,
                "logical-run-changed",
                0.5,
                3,
                logical_request_id="turn:logical-resume",
                admission_wait_ms=100,
            )
            assert conflict[0] == 409, conflict
            assert conflict[1]["reason_code"] == "logical_request_conflict"
            assert conflict[1]["retryable"] is False
            assert conflict[1]["request_id"] == broker_request_id
            assert conflict[2]["X-Ollama-Unify-Retryable"] == "false"
            assert "Retry-After" not in conflict[2]

            follower = executor.submit(
                chat, harness.proxy_port, MODEL, "logical-follower", 0.05, 8
            )
            wait_until(
                lambda: queue_with_depth(harness, 2),
                "retained logical waiter and FIFO follower",
            )
            resumed = executor.submit(
                resume_chat,
                harness.proxy_port,
                "turn:logical-resume",
                8,
                admission_wait_ms=2000,
            )
            wait_until(
                lambda: any(
                    event.get("request_id") == "logical-run"
                    for event in request_events(harness.event_log)
                ),
                "resumed logical request admission",
                timeout=3,
            )
            duplicate = chat(
                harness.proxy_port,
                MODEL,
                "logical-run",
                0.5,
                3,
                logical_request_id="turn:logical-resume",
                admission_wait_ms=100,
            )
            assert duplicate[0] == 409, duplicate
            assert duplicate[1]["reason_code"] == "logical_request_in_progress"
            assert duplicate[1]["retryable"] is True
            assert duplicate[1]["request_id"] == broker_request_id

            assert active.result(timeout=8)[0] == 200
            resumed_result = resumed.result(timeout=8)
            assert resumed_result[0] == 200, resumed_result
            assert resumed_result[2]["X-Ollama-Unify-Request-Id"] == broker_request_id
            assert resumed_result[2]["X-Ollama-Unify-Logical-Request-Id"] == (
                "turn:logical-resume"
            )
            assert resumed_result[2]["X-Ollama-Unify-Workload-Class"] == "foreground"
            assert resumed_result[2]["X-Ollama-Unify-Queue-Policy"] == "wait"
            assert int(
                resumed_result[2]["X-Ollama-Unify-Queue-Ticket"]
            ) == broker_queue_ticket
            assert follower.result(timeout=8)[0] == 200

        observed_ids = [
            event["request_id"] for event in request_events(harness.event_log)
        ]
        assert observed_ids == [
            "logical-active", "logical-run", "logical-follower",
        ]
        queue = harness.status()["parallel_pool"]["queue"]
        assert queue["enqueued_total"] == 3
        assert queue["admitted_total"] == 3
        assert queue["admission_timeout_total"] == 1
        assert queue["resumed_total"] == 1
        assert queue["duplicate_total"] == 2
        assert queue["stale_total"] == 0


def test_completed_native_response_replays_without_generation(
    helper, fixture_bin,
):
    logical_id = "turn:completed-native"
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        first = chat(
            harness.proxy_port, MODEL, "completed-native", 0, 8,
            logical_request_id=logical_id,
        )
        replay = resume_chat(harness.proxy_port, logical_id, 8)
        assert first[0] == replay[0] == 200
        assert first[1] == replay[1]
        for header in (
            "Content-Type",
            "X-Mock-Response",
            "X-Ollama-Unify-Lane",
            "X-Ollama-Unify-Request-Id",
            "X-Ollama-Unify-Logical-Request-Id",
            "X-Ollama-Unify-Workload-Class",
            "X-Ollama-Unify-Queue-Policy",
            "X-Ollama-Unify-Queue-Ms",
            "X-Ollama-Unify-Queue-Position",
        ):
            assert first[2][header] == replay[2][header]
        assert "X-Ollama-Unify-Response-Replayed" not in first[2]
        assert replay[2]["X-Ollama-Unify-Response-Replayed"] == "true"
        assert len(replay[2]["X-Ollama-Unify-Completed-Response-Sha256"]) == 64

        conflict = chat(
            harness.proxy_port, MODEL, "completed-native-changed", 0, 8,
            logical_request_id=logical_id,
        )
        assert conflict[0] == 409, conflict
        assert conflict[1]["reason_code"] == "logical_request_conflict"
        assert conflict[1]["retryable"] is False
        assert "Retry-After" not in conflict[2]

        generated = [
            event["request_id"] for event in request_events(harness.event_log)
            if event["request_id"].startswith("completed-native")
        ]
        assert generated == ["completed-native"]
        discovery = http_json(
            harness.proxy_port, "GET",
            "/.well-known/ollama-unify-gpu-negotiator",
        )[1]
        completed = discovery["parallel_pool"]["completed_responses"]
        assert completed["entries"] == 1
        assert completed["retained_entries"] == 1
        assert completed["recorded_total"] == 1
        assert completed["replayed_total"] == 1
        assert completed["conflict_total"] == 1
        serialized = json.dumps(completed, sort_keys=True)
        assert logical_id not in serialized
        assert "completed-native" not in serialized
        assert "fingerprint" not in serialized


def test_completed_streaming_response_replays_exact_body(helper, fixture_bin):
    logical_id = "turn:completed-stream"
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        first = streaming_chat(
            harness.proxy_port, MODEL, "completed-stream", logical_id, 8,
        )
        replay = resume_chat_raw(harness.proxy_port, logical_id, 8)
        assert first[0] == replay[0] == 200
        assert first[1] == replay[1]
        assert first[1].count(b"\n") == 2
        assert first[2]["Content-Type"] == "application/x-ndjson"
        assert replay[2]["Content-Type"] == first[2]["Content-Type"]
        assert replay[2]["X-Mock-Response"] == first[2]["X-Mock-Response"]
        assert replay[2]["X-Ollama-Unify-Response-Replayed"] == "true"
        assert int(replay[2]["Content-Length"]) == len(first[1])
        generated = [
            event["request_id"] for event in request_events(harness.event_log)
            if event["request_id"] == "completed-stream"
        ]
        assert generated == ["completed-stream"]


def test_completed_response_survives_delivery_disconnect(helper, fixture_bin):
    logical_id = "turn:completed-after-disconnect"
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        abandoned = open_abandonable_chat(
            harness.proxy_port,
            MODEL,
            "completed-after-disconnect",
            logical_request_id=logical_id,
            delay=0.25,
        )
        wait_until(
            lambda: any(
                event.get("request_id") == "completed-after-disconnect"
                for event in request_events(harness.event_log)
            ),
            "logical request to reach the backend before disconnect",
        )
        reset_connection(abandoned)
        wait_until(
            lambda: harness.status()["parallel_pool"][
                "completed_responses"
            ]["retained_entries"] == 1,
            "completed response to be retained after delivery disconnect",
        )
        replay = chat(
            harness.proxy_port,
            MODEL,
            "completed-after-disconnect",
            0.25,
            8,
            logical_request_id=logical_id,
        )
        assert replay[0] == 200, replay
        assert replay[2]["X-Ollama-Unify-Response-Replayed"] == "true"
        generated = [
            event["request_id"] for event in request_events(harness.event_log)
            if event["request_id"] == "completed-after-disconnect"
        ]
        assert generated == ["completed-after-disconnect"]
        lifecycle = harness.status()["parallel_pool"]["request_lifecycle"]
        assert lifecycle["expired_total"] == 0


def test_backend_activity_renews_active_request_ttl(helper, fixture_bin):
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=1,
        request_activity_ttl=0.2,
        request_detached_ttl=0.2,
    ) as harness:
        status, body, _ = streaming_chat(
            harness.proxy_port,
            MODEL,
            "renewable-stream",
            "turn:renewable-stream",
            timeout=5,
            chunk_count=8,
            chunk_interval=0.08,
        )
        assert status == 200
        assert body.count(b"\n") == 8
        broker_status = wait_until(
            lambda: (
                current
                if (current := harness.status())["active_requests"] == 0
                else None
            ),
            "renewed streaming request to release",
        )
        lifecycle = broker_status["parallel_pool"]["request_lifecycle"]
        assert lifecycle["expired_total"] == 0
        assert lifecycle["activity_renewed_total"] >= 10
        assert broker_status["active_requests"] == 0
        assert all(
            lane["in_flight"] == 0 for lane in managed_lanes(broker_status)
        )


def test_silent_backend_request_expires_exact_admission(helper, fixture_bin):
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=1,
        request_activity_ttl=0.15,
        request_detached_ttl=0.15,
        request_cancel_grace=0.1,
    ) as harness:
        status, payload, _ = chat(
            harness.proxy_port,
            MODEL,
            "silent-backend",
            delay=1.0,
            timeout=5,
        )
        assert status == 504, (status, payload)
        assert payload["reason_code"] == "backend_activity_timeout"
        recovered = wait_until(
            lambda: (
                current
                if (
                    (current := harness.status())["active_requests"] == 0
                    and all(
                        lane["in_flight"] == 0
                        for lane in managed_lanes(current)
                    )
                )
                else None
            ),
            "expired request ownership to be released",
            timeout=8,
        )
        lifecycle = recovered["parallel_pool"]["request_lifecycle"]
        assert lifecycle["expired_total"] == 1
        assert lifecycle["forced_lane_stop_total"] == 1
        assert lifecycle["forced_release_total"] == 1

        follower_status, follower, _ = chat(
            harness.proxy_port, MODEL, "after-silent-expiry", timeout=8
        )
        assert follower_status == 200, follower
        final = wait_until(
            lambda: (
                current
                if (current := harness.status())["active_requests"] == 0
                else None
            ),
            "follower request to release",
        )
        assert final["active_requests"] == 0
        assert sum(lane["in_flight"] for lane in managed_lanes(final)) == 0


def test_repeated_abandoned_requests_do_not_compound_lanes(
    helper, fixture_bin,
):
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=3,
        request_activity_ttl=1.0,
        request_detached_ttl=0.15,
        request_cancel_grace=0.1,
    ) as harness:
        for attempt in range(1, 4):
            request_id = f"abandoned-active-{attempt}"
            abandoned = open_abandonable_chat(
                harness.proxy_port,
                MODEL,
                request_id,
                logical_request_id=f"turn:{request_id}",
                delay=2.0,
            )
            wait_until(
                lambda: any(
                    event.get("request_id") == request_id
                    for event in request_events(harness.event_log)
                ),
                f"attempt {attempt} to reach its managed backend",
            )
            reset_connection(abandoned)
            recovered = wait_until(
                lambda: (
                    current
                    if (
                        (current := harness.status())["active_requests"] == 0
                        and all(
                            lane["in_flight"] == 0
                            for lane in managed_lanes(current)
                        )
                    )
                    else None
                ),
                f"attempt {attempt} lifecycle expiry",
                timeout=8,
            )
            assert len(managed_lanes(recovered)) <= 1

        follower_status, follower, _ = chat(
            harness.proxy_port, MODEL, "after-abandoned-active", timeout=8
        )
        assert follower_status == 200, follower
        final = wait_until(
            lambda: (
                current
                if (current := harness.status())["active_requests"] == 0
                else None
            ),
            "post-abandonment follower to release",
        )
        lanes = managed_lanes(final)
        lifecycle = final["parallel_pool"]["request_lifecycle"]
        assert len(lanes) == 1
        assert sum(lane["in_flight"] for lane in lanes) == 0
        assert lifecycle["disconnected_total"] >= 3
        assert lifecycle["expired_total"] >= 3
        assert lifecycle["forced_lane_stop_total"] >= 3
        starts = sum(event["kind"] == "start" for event in events(harness.event_log))
        stops = sum(event["kind"] == "stop" for event in events(harness.event_log))
        assert starts - stops == 1


def test_cancelled_shared_lane_blocks_fresh_admission(helper, fixture_bin):
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=2,
        instance_parallel=2,
        request_activity_ttl=2.0,
        request_detached_ttl=0.12,
        request_cancel_grace=0.1,
    ) as harness:
        capacity_status, capacity, _ = harness.capacity(MODEL, parallel=2)
        assert capacity_status == 200, capacity
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            legitimate = executor.submit(
                chat,
                harness.proxy_port,
                MODEL,
                "shared-legitimate",
                0.8,
                8,
            )
            wait_until(
                lambda: any(
                    event.get("request_id") == "shared-legitimate"
                    for event in request_events(harness.event_log)
                ),
                "legitimate request on shared lane",
            )
            abandoned = open_abandonable_chat(
                harness.proxy_port,
                MODEL,
                "shared-abandoned",
                logical_request_id="turn:shared-abandoned",
                delay=2.0,
            )
            wait_until(
                lambda: any(
                    event.get("request_id") == "shared-abandoned"
                    for event in request_events(harness.event_log)
                ),
                "abandoned request on shared lane",
            )
            reset_connection(abandoned)
            wait_until(
                lambda: (
                    current
                    if (
                        (current := harness.status())["parallel_pool"]
                        ["request_lifecycle"]["cancelling"] >= 1
                        and any(
                            lane["state"] == "retiring"
                            for lane in managed_lanes(current)
                        )
                    )
                    else None
                ),
                "shared lane to retire on its first cancellation",
            )
            follower = executor.submit(
                chat,
                harness.proxy_port,
                MODEL,
                "shared-follower",
                0,
                8,
            )
            queued = wait_until(
                lambda: (
                    queue
                    if (
                        (queue := harness.status()["parallel_pool"]["queue"])
                        ["phase_counts"].get(
                            "waiting-request-cancellation", 0
                        ) == 1
                    )
                    else None
                ),
                "follower to wait for cancelled shared lane teardown",
            )
            assert queued["depth"] == 1
            assert not any(
                event.get("request_id") == "shared-follower"
                for event in request_events(harness.event_log)
            )
            starts = sum(
                event["kind"] == "start"
                for event in events(harness.event_log)
            )
            stops = sum(
                event["kind"] == "stop"
                for event in events(harness.event_log)
            )
            assert starts - stops == 1
            legitimate_result = legitimate.result(timeout=8)
            assert legitimate_result[0] == 200, legitimate_result
            follower_result = follower.result(timeout=8)
            assert follower_result[0] == 200, follower_result
            assert (
                follower_result[2]["X-Ollama-Unify-Lane"]
                != legitimate_result[2]["X-Ollama-Unify-Lane"]
            )

        final = wait_until(
            lambda: (
                current
                if (current := harness.status())["active_requests"] == 0
                else None
            ),
            "shared-lane follower release",
            timeout=8,
        )
        assert len(managed_lanes(final)) == 1
        assert all(
            lane["state"] == "ready" and lane["in_flight"] == 0
            for lane in managed_lanes(final)
        )
        starts = sum(
            event["kind"] == "start" for event in events(harness.event_log)
        )
        stops = sum(
            event["kind"] == "stop" for event in events(harness.event_log)
        )
        assert starts - stops == 1


def test_cancellation_induced_eof_is_not_replayable(helper, fixture_bin):
    logical_id = "turn:cancelled-stream"
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=1,
        request_activity_ttl=0.15,
        request_detached_ttl=0.15,
        request_cancel_grace=0.1,
    ) as harness:
        status, body, _ = streaming_chat(
            harness.proxy_port,
            MODEL,
            "cancelled-stream",
            logical_id,
            timeout=5,
            chunk_count=2,
            chunk_interval=1.0,
        )
        assert status == 200
        assert b"chunk-1" in body
        assert b"chunk-2" not in body
        wait_until(
            lambda: (
                current
                if (current := harness.status())["active_requests"] == 0
                else None
            ),
            "cancelled stream admission release",
            timeout=8,
        )
        replay = resume_chat(harness.proxy_port, logical_id, timeout=5)
        assert replay[0] == 409, replay
        assert replay[1]["reason_code"] == "backend_activity_timeout"
        completed = harness.status()["parallel_pool"]["completed_responses"]
        assert completed["entries"] == 0
        generated = [
            event for event in request_events(harness.event_log)
            if event.get("request_id") == "cancelled-stream"
        ]
        assert len(generated) == 1


def test_partial_fixed_length_response_closes_client(helper, fixture_bin):
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=1,
        request_activity_ttl=0.5,
        request_cancel_grace=0.1,
    ) as harness:
        status, partial, elapsed = incomplete_fixed_length_chat(
            harness.proxy_port,
            MODEL,
            "partial-fixed-length",
        )
        assert status == 200
        assert partial
        assert elapsed < 2.0
        final = wait_until(
            lambda: (
                current
                if (current := harness.status())["active_requests"] == 0
                else None
            ),
            "partial-response admission release",
            timeout=8,
        )
        lifecycle = final["parallel_pool"]["request_lifecycle"]
        assert lifecycle["cancelled_total"] == 1
        assert lifecycle["forced_lane_stop_total"] == 1
        assert not managed_lanes(final)


def test_backend_reset_retires_managed_lane(helper, fixture_bin):
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=1,
        request_activity_ttl=0.5,
        request_cancel_grace=0.1,
    ) as harness:
        capacity_status, capacity, _ = harness.capacity(MODEL)
        assert capacity_status == 200, capacity
        old_lane_id = managed_lanes(harness.status())[0]["id"]
        status, payload, _ = chat(
            harness.proxy_port,
            MODEL,
            "backend-reset",
            timeout=5,
            mock_backend_reset=True,
        )
        assert status == 503, (status, payload)
        assert payload["reason_code"] == "backend_transport_failed"
        final = wait_until(
            lambda: (
                current
                if (current := harness.status())["active_requests"] == 0
                else None
            ),
            "reset backend admission release",
            timeout=8,
        )
        lifecycle = final["parallel_pool"]["request_lifecycle"]
        assert lifecycle["cancelled_total"] == 1
        assert lifecycle["forced_lane_stop_total"] == 1
        assert not managed_lanes(final)
        follower = chat(
            harness.proxy_port,
            MODEL,
            "after-backend-reset",
            timeout=8,
        )
        assert follower[0] == 200, follower
        assert follower[2]["X-Ollama-Unify-Lane"] != old_lane_id


def test_completed_oversize_response_fails_closed_without_regeneration(
    helper, fixture_bin,
):
    logical_id = "turn:completed-oversize"
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=1,
        completed_max_body_bytes=128,
        completed_max_total_bytes=512,
    ) as harness:
        first = chat(
            harness.proxy_port, MODEL, "completed-oversize", 0, 8,
            logical_request_id=logical_id,
            mock_body_size=1024,
        )
        assert first[0] == 200, first
        replay = chat(
            harness.proxy_port, MODEL, "completed-oversize", 0, 8,
            logical_request_id=logical_id,
            mock_body_size=1024,
        )
        assert replay[0] == 409, replay
        assert replay[1]["reason_code"] == "completed_response_unavailable"
        assert replay[1]["retryable"] is False
        receipt = replay[1]["completed_response"]
        assert receipt["retained"] is False
        assert receipt["body_bytes"] > 1024
        assert len(receipt["body_sha256"]) == 64
        assert "per_entry_limit" in receipt["unavailable_reason"]
        assert "Retry-After" not in replay[2]
        generated = [
            event["request_id"] for event in request_events(harness.event_log)
            if event["request_id"] == "completed-oversize"
        ]
        assert generated == ["completed-oversize"]
        completed = harness.status()["parallel_pool"]["completed_responses"]
        assert completed["entries"] == 1
        assert completed["unavailable_entries"] == 1
        assert completed["retained_bytes"] == 0
        assert completed["unavailable_total"] == 1


def test_completed_response_cache_ttl_and_bounded_eviction(helper, fixture_bin):
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=1,
        # Eviction must happen before TTL expiry; a 200ms TTL coupled these
        # independent behaviors to the timing of three real HTTP requests.
        completed_ttl=30,
        completed_max_entries=2,
    ) as harness:
        for index in range(1, 4):
            result = chat(
                harness.proxy_port, MODEL, f"completed-evict-{index}", 0, 8,
                logical_request_id=f"turn:completed-evict-{index}",
            )
            assert result[0] == 200, result
        bounded = harness.status()["parallel_pool"]["completed_responses"]
        assert bounded["entries"] == 2
        assert bounded["evicted_total"] == 1
        assert bounded["expired_total"] == 0

        regenerated = chat(
            harness.proxy_port, MODEL, "completed-evict-1", 0, 8,
            logical_request_id="turn:completed-evict-1",
        )
        assert regenerated[0] == 200, regenerated
        assert "X-Ollama-Unify-Response-Replayed" not in regenerated[2]
        generated = [
            event["request_id"] for event in request_events(harness.event_log)
            if event["request_id"] == "completed-evict-1"
        ]
        assert generated == ["completed-evict-1", "completed-evict-1"]

    # Test the actual short expiry separately, without competing max-entry
    # eviction removing the item first.
    with PoolHarness(helper, fixture_bin, max_servers=1, completed_ttl=0.2,
                     completed_max_entries=2) as harness:
        seeded = chat(harness.proxy_port, MODEL, "completed-evict-1", 0, 8,
                      logical_request_id="turn:completed-evict-1")
        assert seeded[0] == 200, seeded
        time.sleep(0.25)
        after_ttl = chat(
            harness.proxy_port, MODEL, "completed-evict-1", 0, 8,
            logical_request_id="turn:completed-evict-1",
        )
        assert after_ttl[0] == 200, after_ttl
        assert "X-Ollama-Unify-Response-Replayed" not in after_ttl[2]
        completed = harness.status()["parallel_pool"]["completed_responses"]
        assert completed["entries"] == 1
        assert completed["expired_total"] >= 1

        generated = [event["request_id"] for event in request_events(harness.event_log)
                     if event["request_id"] == "completed-evict-1"]
        assert generated == ["completed-evict-1", "completed-evict-1"]


def test_detached_logical_request_expires_without_inference(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1, resume_ttl=0.2) as harness:
        assert harness.capacity(MODEL)[0] == 200
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            active = executor.submit(
                chat, harness.proxy_port, MODEL, "stale-active", 0.7, 8
            )
            wait_until(
                lambda: any(
                    event.get("request_id") == "stale-active"
                    for event in request_events(harness.event_log)
                ),
                "active request before stale admission",
            )
            expired = chat(
                harness.proxy_port,
                MODEL,
                "must-expire",
                0,
                3,
                logical_request_id="turn:expires",
                admission_wait_ms=100,
            )
            assert expired[0] == 503, expired
            assert expired[1]["admission_retained"] is True
            wait_until(
                lambda: queue_with_depth(harness, 0),
                "detached logical waiter expiration",
                timeout=2,
            )
            resume = resume_chat(harness.proxy_port, "turn:expires", 3)
            assert resume[0] == 409, resume
            assert resume[1]["reason_code"] == "logical_request_expired"
            assert resume[1]["retryable"] is False
            assert active.result(timeout=8)[0] == 200

        assert not any(
            event.get("request_id") == "must-expire"
            for event in request_events(harness.event_log)
        )
        queue = harness.status()["parallel_pool"]["queue"]
        assert queue["stale_total"] == 1
        assert queue["resumed_total"] == 0


def test_temporary_and_permanent_placement_failures(helper, fixture_bin):
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=1,
        tags=[MODEL, OTHER_MODEL],
    ) as harness:
        assert harness.capacity(MODEL)[0] == 200
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            active = executor.submit(
                chat, harness.proxy_port, MODEL, "temporary-active", 0.8, 8
            )
            wait_until(
                lambda: any(
                    event.get("request_id") == "temporary-active"
                    for event in request_events(harness.event_log)
                ),
                "active lane before temporary placement failure",
            )
            temporary = chat(
                harness.proxy_port,
                OTHER_MODEL,
                "temporary-placement",
                0,
                3,
                logical_request_id="turn:temporary-placement",
                admission_wait_ms=150,
                queue_policy="yield",
            )
            assert temporary[0] == 503, temporary
            assert temporary[1]["reason_code"] == "queue_admission_timeout"
            assert temporary[1]["cause_reason_code"] == (
                "reclaimable_placement_wait"
            )
            assert temporary[1]["retryable"] is True
            assert "admission_retained" not in temporary[1]
            assert active.result(timeout=8)[0] == 200

    oversized = {
        "name": OTHER_MODEL,
        "model": OTHER_MODEL,
        "size": 300 * 1024**3,
        "capabilities": ["completion"],
    }
    with PoolHarness(
        helper, fixture_bin, max_servers=1, tags=[MODEL, oversized]
    ) as harness:
        assert harness.capacity(MODEL)[0] == 200
        permanent = chat(
            harness.proxy_port,
            OTHER_MODEL,
            "permanent-placement",
            timeout=3,
            logical_request_id="turn:permanent-placement",
        )
        assert permanent[0] == 422, permanent
        assert permanent[1]["reason_code"] == "model_exceeds_gpu_capacity"
        assert permanent[1]["retryable"] is False
        assert permanent[2]["X-Ollama-Unify-Retryable"] == "false"
        assert "Retry-After" not in permanent[2]
        assert not any(
            event.get("request_id") == "permanent-placement"
            for event in request_events(harness.event_log)
        )
        assert [lane["model"] for lane in managed_lanes(harness.status())] == [
            MODEL
        ]
        assert not [
            event for event in events(harness.event_log)
            if event["kind"] == "stop"
        ]


def test_ready_model_bypasses_blocked_other_model(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=2, tags=[MODEL, OTHER_MODEL]
    ) as harness:
        assert harness.capacity(MODEL)[0] == 200
        assert harness.capacity(OTHER_MODEL)[0] == 200

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            other_active = executor.submit(
                chat, harness.proxy_port, OTHER_MODEL, "other-active", 1.5, 8
            )
            model_active = executor.submit(
                chat, harness.proxy_port, MODEL, "model-active", 1.0, 8
            )
            wait_until(
                lambda: {
                    event.get("request_id")
                    for event in request_events(harness.event_log)
                } >= {"other-active", "model-active"},
                "both model lanes to become active",
            )

            other_waiting = executor.submit(
                chat, harness.proxy_port, OTHER_MODEL, "other-waiting", 0.05, 8
            )
            wait_until(
                lambda: queue_with_depth(harness, 1),
                "blocked other-model queue head",
            )
            model_waiting = executor.submit(
                chat, harness.proxy_port, MODEL, "model-bypass", 0.05, 8
            )
            wait_until(
                lambda: queue_with_depth(harness, 2),
                "cross-model waiter behind blocked queue head",
            )

            assert model_active.result(timeout=2)[0] == 200
            assert model_waiting.result(timeout=1)[0] == 200
            assert other_active.result(timeout=3)[0] == 200
            assert other_waiting.result(timeout=2)[0] == 200

        request_ids = [
            event["request_id"] for event in request_events(harness.event_log)
        ]
        assert request_ids.index("model-bypass") < request_ids.index("other-waiting")
        assert not [event for event in events(harness.event_log)
                    if event["kind"] == "stop"]
        final_queue = harness.status()["parallel_pool"]["queue"]
        assert final_queue["depth"] == 0
        assert final_queue["timed_out_total"] == 0


def test_queued_disconnect_is_cancelled(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        status, capacity, _ = harness.capacity(MODEL)
        assert status == 200, capacity
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            active = executor.submit(
                chat, harness.proxy_port, MODEL, "active-before-cancel", 1.0, 8
            )
            wait_until(
                lambda: any(event.get("request_id") == "active-before-cancel"
                            for event in request_events(harness.event_log)),
                "active request before cancellation",
            )
            abandoned = open_abandonable_chat(
                harness.proxy_port,
                MODEL,
                "must-not-run",
                logical_request_id="turn:cancelled",
            )
            wait_until(
                lambda: queue_with_depth(harness, 1),
                "abandoned request queue admission",
            )
            reset_connection(abandoned)
            wait_until(
                lambda: queue_with_depth(harness, 0),
                "abandoned request cancellation",
            )
            resume = resume_chat(harness.proxy_port, "turn:cancelled", 3)
            assert resume[0] == 409, resume
            assert resume[1]["reason_code"] == "logical_request_cancelled"
            assert resume[1]["retryable"] is False
            assert active.result(timeout=8)[0] == 200

        time.sleep(0.15)
        assert not any(event.get("request_id") == "must-not-run"
                       for event in request_events(harness.event_log))
        status = harness.status()
        assert status["active_requests"] == 0
        assert status["parallel_pool"]["queue"]["depth"] == 0
        assert status["parallel_pool"]["queue"]["cancelled_total"] == 1
        assert status["parallel_pool"]["queue"]["stale_total"] == 0
        assert status["parallel_pool"]["queue"]["admitted_total"] == 1
        assert all(lane["in_flight"] == 0 for lane in managed_lanes(status))


def test_embedding_model_uses_embedding_warmup(helper, fixture_bin):
    tags = [{
        "name": EMBED_MODEL,
        "model": EMBED_MODEL,
        "size": 1024**3,
        "capabilities": ["embedding"],
    }]
    with PoolHarness(helper, fixture_bin, max_servers=1, tags=tags) as harness:
        capacity_status, capacity, _ = harness.capacity(
            EMBED_MODEL, endpoint="/api/embed"
        )
        assert capacity_status == 200, capacity
        status, payload, headers = embed(
            harness.proxy_port, EMBED_MODEL, "embedding-request", timeout=8
        )
        assert status == 200, payload
        assert headers["X-Ollama-Unify-Lane"].startswith("lane-")
        observed = [event for event in events(harness.event_log)
                    if event["kind"] == "request"
                    and event["model"] == EMBED_MODEL]
        assert [event["path"] for event in observed] == [
            "/api/embed", "/api/embed",
        ]
        assert observed[-1]["request_id"] == "embedding-request"


def test_permanent_failure_does_not_block_fifo(helper, fixture_bin):
    with PoolHarness(
        helper, fixture_bin, max_servers=1, tags=[MODEL, REJECT_MODEL]
    ) as harness:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            rejected = executor.submit(
                chat, harness.proxy_port, REJECT_MODEL, "rejected-model", 0, 8
            )
            wait_until(
                lambda: queue_with_depth(harness, 1),
                "rejected model queue admission",
            )
            accepted = executor.submit(
                chat, harness.proxy_port, MODEL, "after-missing", 0, 8
            )
            rejected_result = rejected.result(timeout=8)
            accepted_result = accepted.result(timeout=8)
        assert rejected_result[0] == 400, rejected_result
        assert "does not support requested operation" in rejected_result[1]["error"]
        assert accepted_result[0] == 200, accepted_result
        queue = harness.status()["parallel_pool"]["queue"]
        assert queue["depth"] == 0
        assert queue["rejected_total"] == 1
        assert queue["admitted_total"] == 1
        assert queue["timed_out_total"] == 0


def test_retryable_warm_failure_stays_queued(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1, tags=[RETRY_MODEL]) as harness:
        status, payload, _ = chat(
            harness.proxy_port, RETRY_MODEL, "after-retry", 0, 10
        )
        assert status == 200, payload
        warm_attempts = [
            event for event in events(harness.event_log)
            if event["kind"] == "request"
            and event["model"] == RETRY_MODEL
            and event["request_id"] is None
        ]
        assert len(warm_attempts) == 2, warm_attempts
        queue = harness.status()["parallel_pool"]["queue"]
        assert queue["rejected_total"] == 0
        assert queue["admitted_total"] == 1


def test_capacity_rejects_unknown_endpoint(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        status, payload, _ = harness.capacity(MODEL, endpoint="/api/unknown")
        assert status == 400, payload
        assert "supported Ollama inference path" in payload["error"]


def test_unknown_body_free_resume_never_enqueues(helper, fixture_bin):
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        before = harness.status()["parallel_pool"]["queue"]
        resumed = resume_chat(harness.proxy_port, "turn:never-accepted", 3)
        after = harness.status()["parallel_pool"]["queue"]
        assert resumed[0] == 409, resumed
        assert resumed[1]["reason_code"] == "logical_request_not_found"
        assert resumed[1]["retryable"] is False
        assert after["depth"] == 0
        assert after["enqueued_total"] == before["enqueued_total"]


def test_feasible_controlled_load_has_no_admission_loss_or_amplification(
    helper, fixture_bin,
):
    """Quantify the broker's bounded behavior without a model or live service."""
    request_count = 100
    concurrent_count = 92
    replay_count = request_count - concurrent_count
    max_queue = 32
    completed_max_entries = 16
    completed_max_total_bytes = 16 * 1024
    with PoolHarness(
        helper,
        fixture_bin,
        max_servers=3,
        max_queue=max_queue,
        completed_ttl=10.0,
        completed_max_entries=completed_max_entries,
        completed_max_body_bytes=2 * 1024,
        completed_max_total_bytes=completed_max_total_bytes,
    ) as harness:
        def submit(index):
            return chat(
                harness.proxy_port,
                MODEL,
                f"controlled-load-{index:03d}",
                0.01,
                10,
                logical_request_id=f"turn:controlled-load:{index:03d}",
                admission_wait_ms=5_000,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(submit, index)
                for index in range(concurrent_count)
            ]
            initial_results = [future.result(timeout=10) for future in futures]

        # These final admissions become deterministic retained replay anchors.
        anchor_results = [
            submit(index)
            for index in range(concurrent_count, request_count)
        ]
        all_results = initial_results + anchor_results
        failed = [result for result in all_results if result[0] != 200]
        assert len(failed) / request_count < 0.01, failed

        replay_results = [
            resume_chat(
                harness.proxy_port,
                f"turn:controlled-load:{index:03d}",
                10,
                admission_wait_ms=5_000,
            )
            for index in range(concurrent_count, request_count)
        ]
        assert all(result[0] == 200 for result in replay_results)
        assert all(
            result[2].get("X-Ollama-Unify-Response-Replayed") == "true"
            for result in replay_results
        )

        generated = [
            event["request_id"]
            for event in request_events(harness.event_log)
            if str(event["request_id"]).startswith("controlled-load-")
        ]
        assert len(generated) == request_count
        assert len(set(generated)) == request_count

        status = harness.status()["parallel_pool"]
        queue = status["queue"]
        completed = status["completed_responses"]
        assert queue["depth"] == 0
        assert queue["peak"] <= max_queue
        assert queue["enqueued_total"] == request_count
        assert queue["admitted_total"] == request_count
        assert queue["rejected_total"] == 0
        assert completed["entries"] <= completed_max_entries
        assert completed["retained_entries"] <= completed_max_entries
        assert completed["retained_bytes"] <= completed_max_total_bytes
        assert completed["replayed_total"] == replay_count


def test_revoked_lease_with_dead_owner_is_abandoned(helper, fixture_bin):
    """A revoked lease whose owner never returns must not pin its GPUs.

    Revocation waits for the owner to free CUDA memory and release. An owner
    that has already died never does either, and revocation had no deadline of
    its own: the reaper's only remedy was to revoke an already-revoking lease,
    which is a no-op. The scope stayed reserved, every admission parked on the
    lease state, and inference hung until the broker was restarted.
    """
    with PoolHarness(
        helper, fixture_bin, max_servers=1,
        pending_timeout=1.0, revoke_timeout=2.0,
    ) as harness:
        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "dead-owner-fixture",
            "requested_mib": 4096,
            "ttl": 1,
            "justification": "exercise dead owner reclamation",
            "expected_duration_seconds": 300,
            "gpu_uuids": ["GPU-large-0", "GPU-large-1", "GPU-large-2"],
        })
        token = acquired["lease"]["token"]

        # The owner dies here: it never calls ready, heartbeat, or release.
        revoked = wait_until(
            lambda: next(
                (lease for lease in harness.status()["leases"]
                 if lease["token"] == token
                 and lease["state"] == "revoking"),
                None,
            ),
            "the pending deadline to revoke the abandoned lease",
            timeout=10,
        )
        # The revoke clock starts on entry to "revoking", not at creation.
        assert revoked["transition_started_at"] >= revoked["created_at"]

        wait_until(
            lambda: harness.status()["leases"] == [],
            "the revoke deadline to abandon the dead owner's lease",
            timeout=15,
        )
        status = harness.status()
        assert status["draining"] is False

        # The reclaimed scope admits inference again without a restart.
        code, payload, _ = chat(harness.proxy_port, MODEL, "after-abandon", 0, 20)
        assert code == 200, payload


def test_forced_release_reclaims_scope_when_owner_never_frees(
    helper, fixture_bin,
):
    """`release --force` recovers a lease whose settle check can never pass.

    Release is gated on foreign CUDA use on the scope returning to its
    pre-lease baseline. An unrelated tenant that arrives on that GPU after
    acquisition is counted against the departed owner forever, so the gate
    can never clear. Without an override the scope stays reserved with no
    supported way to reclaim it.
    """
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        write_compute_apps(harness.compute_apps, [])
        acquired = control(harness.socket_path, {
            "action": "acquire",
            "owner": "never-frees-fixture",
            "requested_mib": 4096,
            "ttl": 30,
            "justification": "exercise forced lease release",
            "expected_duration_seconds": 300,
            "gpu_uuids": ["GPU-large-0"],
        })
        token = acquired["lease"]["token"]
        control(harness.socket_path, {"action": "ready", "token": token})

        # An unrelated tenant lands on the leased GPU after the baseline was
        # taken, and stays. It is not the lease owner's memory, but the
        # release gate cannot tell the difference.
        write_compute_apps(harness.compute_apps, [
            (940101, "GPU-large-0", 4096),
        ])
        refused = control_raw(
            harness.socket_path, {"action": "release", "token": token}
        )
        assert refused["ok"] is False
        assert "baseline" in refused["error"].lower()
        assert harness.status()["leases"] != []

        forced = control(harness.socket_path, {
            "action": "release", "token": token, "force": True,
        })
        assert forced["released"] == token
        status = harness.status()
        assert status["leases"] == []
        assert status["draining"] is False


def test_inference_request_without_json_content_type_is_served(
    helper, fixture_bin,
):
    """A JSON body under another media type must route, not hang.

    Ollama decodes these bodies whatever Content-Type the client declares, so
    a client that omits the header still gets an answer from Ollama directly.
    The broker read the routed model only when the declared type contained
    "json", so the same request arrived with no model, waited for capacity for
    a model named "" that no lane could serve, and hung until the client gave
    up. It also skipped the clamp that pins num_gpu and caps num_ctx, which is
    the bypass the clamp exists to prevent.
    """
    with PoolHarness(helper, fixture_bin, max_servers=1) as harness:
        status, payload, _ = http_json(
            harness.proxy_port, "POST", "/api/chat",
            {
                "model": MODEL,
                "stream": False,
                "mock_request_id": "no-content-type",
                "mock_delay": 0,
                "options": {"num_gpu": 999, "main_gpu": 2},
            },
            timeout=20,
            extra_headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert status == 200, payload
        # The model was recovered from the body, so a lane was placed for it.
        assert [
            lane for lane in managed_lanes(harness.status())
            if lane["model"] == MODEL
        ]


def main():
    helper = os.path.abspath(sys.argv[1])
    fixture_bin = os.path.abspath(sys.argv[2])
    test_existing_pool_contract(helper, fixture_bin)
    test_model_gpu_preference_selects_requested_fitting_gpu(helper, fixture_bin)
    test_managed_lane_rejects_cpu_fallback(helper, fixture_bin)
    test_default_context_is_pinned_on_warmup_and_inference(helper, fixture_bin)
    test_fixed_model_context_is_attested_and_enforced(helper, fixture_bin)
    test_capacity_gpu_constraint_is_hard_and_validated(helper, fixture_bin)
    test_lease_registration_metadata_is_advisory_and_visible(
        helper, fixture_bin,
    )
    test_lazy_inference_gpu_constraint_ignores_wrong_gpu_lane(helper, fixture_bin)
    test_idle_lane_reservations_prevent_future_gpu_overcommit(helper, fixture_bin)
    test_scoped_pending_lease_preserves_unreserved_inference(helper, fixture_bin)
    test_scoped_release_tolerates_baseline_pid_churn_without_global_drain(
        helper, fixture_bin,
    )
    test_scoped_release_tolerates_small_foreign_accounting_drift(
        helper, fixture_bin,
    )
    test_broker_queue_has_hard_ceiling_during_all_gpu_lease(
        helper, fixture_bin,
    )
    test_scoped_active_lease_shares_stable_vram_with_ollama(helper, fixture_bin)
    test_scoped_acquire_preserves_managed_lanes_when_allocation_fits(
        helper, fixture_bin,
    )
    test_multi_gpu_lease_is_exclusive_for_its_lifetime(helper, fixture_bin)
    test_operator_revoke_and_lane_stop(helper, fixture_bin)
    test_client_attribution(helper, fixture_bin)
    test_client_history_ttl_and_lane_bound(helper, fixture_bin)
    test_operator_model_gpu_policy(helper, fixture_bin)
    test_foreign_gpu_transition_stability(helper, fixture_bin)
    test_unregistered_peer_work_defers_every_lane_transition(helper, fixture_bin)
    test_native_explicit_unload_retires_backend(helper, fixture_bin)
    test_implicit_latest_uses_one_lane(helper, fixture_bin)
    test_idle_lane_replacement(helper, fixture_bin)
    test_background_yield_preserves_idle_lane_at_process_ceiling(helper, fixture_bin)
    test_background_yield_preserves_vram_without_process_overflow(helper, fixture_bin)
    test_background_yield_can_cold_load_free_capacity_and_reuse_it(helper, fixture_bin)
    test_live_vram_reclaims_idle_lane_below_process_ceiling(helper, fixture_bin)
    test_lazy_parallel_scaling(helper, fixture_bin)
    test_fifo_queue_and_metrics(helper, fixture_bin)
    test_logical_request_resumes_fifo_without_reenqueue(helper, fixture_bin)
    test_completed_native_response_replays_without_generation(helper, fixture_bin)
    test_completed_streaming_response_replays_exact_body(helper, fixture_bin)
    test_completed_response_survives_delivery_disconnect(helper, fixture_bin)
    test_backend_activity_renews_active_request_ttl(helper, fixture_bin)
    test_silent_backend_request_expires_exact_admission(helper, fixture_bin)
    test_repeated_abandoned_requests_do_not_compound_lanes(helper, fixture_bin)
    test_cancelled_shared_lane_blocks_fresh_admission(helper, fixture_bin)
    test_cancellation_induced_eof_is_not_replayable(helper, fixture_bin)
    test_partial_fixed_length_response_closes_client(helper, fixture_bin)
    test_backend_reset_retires_managed_lane(helper, fixture_bin)
    test_completed_oversize_response_fails_closed_without_regeneration(
        helper, fixture_bin,
    )
    test_completed_response_cache_ttl_and_bounded_eviction(helper, fixture_bin)
    test_detached_logical_request_expires_without_inference(helper, fixture_bin)
    test_temporary_and_permanent_placement_failures(helper, fixture_bin)
    test_ready_model_bypasses_blocked_other_model(helper, fixture_bin)
    test_queued_disconnect_is_cancelled(helper, fixture_bin)
    test_embedding_model_uses_embedding_warmup(helper, fixture_bin)
    test_permanent_failure_does_not_block_fifo(helper, fixture_bin)
    test_retryable_warm_failure_stays_queued(helper, fixture_bin)
    test_capacity_rejects_unknown_endpoint(helper, fixture_bin)
    test_unknown_body_free_resume_never_enqueues(helper, fixture_bin)
    test_feasible_controlled_load_has_no_admission_loss_or_amplification(
        helper, fixture_bin,
    )
    test_inference_request_without_json_content_type_is_served(
        helper, fixture_bin,
    )
    test_revoked_lease_with_dead_owner_is_abandoned(helper, fixture_bin)
    test_forced_release_reclaims_scope_when_owner_never_frees(
        helper, fixture_bin,
    )
    print(
        "negotiator pool integration: PASS "
        "(fit, durable lane reservations, bounded queue, lease accounting tolerance, endpoint-aware warm lanes, "
        "stable watcher, aliases, replacement, per-model FIFO, warm-lane bypass, "
        "resumable logical admission, completed-response replay, cancellation, "
        "terminal admission failures, controlled-load admission and replay bounds, "
        "dead-owner lease reclamation, exclusive multi-GPU leases, "
        "operator revoke and lane stop, client attribution, "
        "operator model GPU policy, "
        "content-type-independent routing)"
    )


if __name__ == "__main__":
    main()
