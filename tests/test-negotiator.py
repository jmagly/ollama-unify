#!/usr/bin/env python3
import http.client
import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time


class Backend(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port=0):
        super().__init__(("127.0.0.1", port), Handler)
        self.models = []
        self.requests = []
        self.lock = threading.Lock()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, payload, status=200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/api/ps":
            self._json({"error": "not found"}, 404)
            return
        with self.server.lock:
            models = list(self.server.models)
        self._json({"models": models})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        with self.server.lock:
            self.server.requests.append((self.path, payload))
            if self.path == "/api/generate":
                if payload.get("keep_alive") == 0:
                    self.server.models = []
                    reason = "unload"
                else:
                    self.server.models = [
                        {
                            "name": payload.get("model", "fixture"),
                            "size": 4096,
                            "size_vram": 2048,
                        }
                    ]
                    reason = "stop"
            else:
                reason = "stop"
        self._json({"done": True, "done_reason": reason})

    def log_message(self, _fmt, *_args):
        pass


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def control(path, payload):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(10)
    client.connect(path)
    client.sendall(json.dumps(payload).encode() + b"\n")
    chunks = []
    while True:
        chunk = client.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    client.close()
    response = json.loads(b"".join(chunks))
    assert response["ok"], response
    return response


def proxy_generate(port, context=262144, admission_wait_ms=None):
    body = json.dumps(
        {
            "model": "fixture",
            "stream": False,
            "options": {"num_gpu": 999, "main_gpu": 2, "num_ctx": context},
        }
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/json"}
    if admission_wait_ms is not None:
        headers["X-Ollama-Unify-Admission-Wait-Ms"] = str(admission_wait_ms)
    conn.request("POST", "/api/generate", body, headers)
    response = conn.getresponse()
    payload = response.read()
    status = response.status
    response_headers = dict(response.getheaders())
    conn.close()
    if admission_wait_ms is None:
        assert status == 200, status
    return status, payload, response_headers


def well_known(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    conn.request("GET", "/.well-known/ollama-unify-gpu-negotiator")
    response = conn.getresponse()
    payload = json.loads(response.read())
    conn.close()
    assert response.status == 200, response.status
    return payload


def main():
    helper = os.path.abspath(sys.argv[1])
    fixture_bin = os.path.abspath(sys.argv[2])
    installer = os.path.abspath(sys.argv[3])
    backend = Backend()
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    proxy_port = free_port()

    with tempfile.TemporaryDirectory(prefix="ollama-unify-negotiator-") as temp_dir:
        socket_path = os.path.join(temp_dir, "control.sock")
        env = os.environ.copy()
        env.update(
            {
                "PATH": fixture_bin + os.pathsep + env.get("PATH", ""),
                "MOCK_PROFILE": "cuda_triple",
                "OLLAMA_UNIFY_CONFIG": os.path.join(temp_dir, "missing.conf"),
                "OLLAMA_UNIFY_BACKEND": f"127.0.0.1:{backend.server_port}",
                "OLLAMA_UNIFY_LISTEN": f"127.0.0.1:{proxy_port}",
                "OLLAMA_UNIFY_SOCKET": socket_path,
                "OLLAMA_UNIFY_LEASE_STATE": os.path.join(temp_dir, "leases.json"),
                "OLLAMA_UNIFY_BACKEND_TYPE": "cuda",
                "OLLAMA_UNIFY_SELECTED_GPUS": "GPU-large-0,GPU-large-1,GPU-large-2",
                "OLLAMA_UNIFY_POOL_ENABLED": "0",
                "OLLAMA_UNIFY_MAX_CONTEXT": "8192",
                "OLLAMA_UNIFY_DRAIN_TIMEOUT": "3",
                "OLLAMA_UNIFY_PENDING_TIMEOUT": "2",
                "OLLAMA_UNIFY_UNLOAD_TIMEOUT": "3",
                "OLLAMA_UNIFY_LEASE_TTL": "30",
                "OLLAMA_UNIFY_HEARTBEAT_RECONNECT_GRACE": "4",
                "OLLAMA_UNIFY_ANON_POLL": "0.1",
                "OLLAMA_UNIFY_ANON_SETTLE": "0.1",
                "OLLAMA_UNIFY_ANON_MAX_DRAIN": "0.5",
            }
        )
        discovery = subprocess.run(
            [helper, "discover"], env=env, check=True, capture_output=True, text=True
        )
        discovery = json.loads(discovery.stdout)
        assert discovery["schema"] == "io.ollama-unify.gpu-negotiator.discovery.v1"
        assert discovery["selected_gpu_count"] == 3
        assert discovery["commands"]["discover"] == "docker gpu discover"
        assert "scope" in discovery["commands"]["manual"]
        assert discovery["lease_policy"]["required_acquire_fields"] == [
            "owner",
        ]
        assert any("Live lease state is unavailable" in warning
                   for warning in discovery["warnings"])
        instructions = subprocess.run(
            [helper, "agent-instructions"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "docker gpu discover" in instructions
        assert "physical GPUs" in instructions
        daemon = subprocess.Popen(
            [helper, "serve"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if daemon.poll() is not None:
                    raise RuntimeError(daemon.stderr.read())
                if os.path.exists(socket_path):
                    try:
                        conn = socket.create_connection(
                            ("127.0.0.1", proxy_port), timeout=0.2
                        )
                        conn.close()
                        break
                    except OSError:
                        pass
                time.sleep(0.05)
            else:
                raise RuntimeError("negotiator did not become ready")

            empty_discovery = well_known(proxy_port)
            assert empty_discovery["active_leases"] == []
            assert empty_discovery["warnings"] == [], empty_discovery["warnings"]
            assert empty_discovery["lease_policy"]["required_acquire_fields"] == [
                "owner",
            ]
            empty_status = control(socket_path, {"action": "status"})
            assert empty_status["lease_summaries"] == []
            assert empty_status["warnings"] == [], empty_status["warnings"]
            empty_cli = subprocess.run(
                [helper, "discover"], env=env, check=True,
                capture_output=True, text=True,
            )
            assert json.loads(empty_cli.stdout)["warnings"] == []

            proxy_generate(proxy_port)
            with backend.lock:
                request = backend.requests[-1][1]
            assert request["options"]["num_gpu"] == -1
            assert "main_gpu" not in request["options"]
            assert request["options"]["num_ctx"] == 8192

            discovery_config = os.path.join(temp_dir, "negotiator.env")
            discovery_manifest = os.path.join(temp_dir, "discovery.json")
            with open(discovery_config, "w", encoding="utf-8") as stream:
                stream.write(
                    f'OLLAMA_UNIFY_LISTEN="127.0.0.1:{proxy_port}"\n'
                )
            refresh_command = [
                "bash",
                "-c",
                'source "$1"; '
                'SAFETY_DISCOVERY_PATH="$2"; '
                'SAFETY_NEGOTIATOR_CONFIG_PATH="$3"; '
                'refresh_installed_gpu_discovery ""',
                "bash",
                installer,
                discovery_manifest,
                discovery_config,
            ]
            subprocess.run(
                refresh_command,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            with open(discovery_manifest, encoding="utf-8") as stream:
                installed_discovery = json.load(stream)
            assert installed_discovery["available"] is True
            assert installed_discovery["backend_available"] is True
            assert installed_discovery["pending_transition_timeout_seconds"] == 2
            assert installed_discovery["heartbeat_reconnect_grace_seconds"] == 4
            assert (
                installed_discovery["parallel_pool"]["admission_protocol"]
                ["logical_request_header"]
                == "X-Ollama-Unify-Logical-Request-Id"
            )

            acquired = subprocess.run(
                [
                    helper,
                    "acquire",
                    "--owner",
                    "fixture",
                    "--justification",
                    "exercise persisted lease lifecycle",
                    "--expected-duration",
                    "300",
                    "--vram-mib",
                    "4096",
                    "--ttl",
                    "30",
                    "--token-only",
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            token = acquired.stdout.strip()
            assert token.startswith("lease_")
            pending_status = control(socket_path, {"action": "status"})
            assert pending_status["leases"][0]["token"] == token
            assert pending_status["leases"][0]["state"] == "pending"
            with backend.lock:
                assert backend.models == []
            pending_discovery = well_known(proxy_port)
            assert pending_discovery["selected_gpu_count"] == 3
            assert pending_discovery["active_leases"][0]["owner"] == "fixture"
            assert pending_discovery["active_leases"][0]["justification"] == (
                "exercise persisted lease lifecycle"
            )
            assert pending_discovery["active_leases"][0]["expected_release_utc"]
            assert "token" not in json.dumps(pending_discovery["active_leases"])
            assert "fixture until" in pending_discovery["warnings"][0]

            bounded_started = time.monotonic()
            bounded_status, bounded_body, bounded_headers = proxy_generate(
                proxy_port, 4096, admission_wait_ms=150,
            )
            assert bounded_status == 503, (bounded_status, bounded_body)
            assert time.monotonic() - bounded_started < 1
            assert bounded_headers["Retry-After"] == "2"
            assert bounded_headers["X-Ollama-Unify-Retryable"] == "true"
            bounded_payload = json.loads(bounded_body)
            assert bounded_payload["request_id"]
            assert "queue" in bounded_payload
            assert bounded_payload["retryable"] is True
            assert bounded_payload["reason_code"] == "queue_admission_timeout"
            assert bounded_payload["retry_after_ms"] == 2000

            ready = control(socket_path, {"action": "ready", "token": token})
            assert ready["lease"]["state"] == "active"
            # Readiness cannot prove placement for a legacy unscoped owner.
            # Keep the host blocked until that owner releases or scopes it.
            assert control(socket_path, {"action": "status"})["draining"] is True
            active_code, active_body, _ = proxy_generate(
                proxy_port, 4096, admission_wait_ms=150,
            )
            assert active_code == 503, (active_code, active_body)
            assert json.loads(active_body)["reason_code"] == "queue_admission_timeout"
            with backend.lock:
                assert backend.models == []

            daemon.terminate()
            daemon.wait(timeout=5)
            daemon = subprocess.Popen(
                [helper, "serve"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if daemon.poll() is not None:
                    raise RuntimeError(daemon.stderr.read())
                if os.path.exists(socket_path):
                    try:
                        restored = control(socket_path, {"action": "status"})
                        break
                    except (ConnectionError, OSError):
                        pass
                time.sleep(0.05)
            else:
                raise RuntimeError("negotiator did not recover after restart")
            assert restored["leases"][0]["token"] == token
            assert restored["leases"][0]["state"] == "active"

            watcher = subprocess.Popen(
                [helper, "heartbeat", token, "--watch", "--interval", "0.1"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            watching = json.loads(watcher.stdout.readline())
            assert watching["watching"] is True
            assert watching["reconnect_grace"] == 4
            heartbeat_before_restart = control(
                socket_path, {"action": "status"}
            )["leases"][0]["heartbeat_at"]
            daemon.terminate()
            daemon.wait(timeout=5)
            time.sleep(0.3)
            assert watcher.poll() is None, (
                "heartbeat watcher exited during bounded broker restart"
            )
            daemon = subprocess.Popen(
                [helper, "serve"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if daemon.poll() is not None:
                    raise RuntimeError(daemon.stderr.read())
                if os.path.exists(socket_path):
                    try:
                        restored = control(socket_path, {"action": "status"})
                        if (
                            restored["leases"][0]["token"] == token
                            and restored["leases"][0]["heartbeat_at"]
                            > heartbeat_before_restart
                        ):
                            break
                    except (ConnectionError, OSError):
                        pass
                time.sleep(0.05)
            else:
                raise RuntimeError(
                    "heartbeat watcher did not renew the persisted lease after restart"
                )
            assert watcher.poll() is None
            watcher.terminate()
            watcher.wait(timeout=5)

            prepared = control(socket_path, {"action": "prepare", "token": token})
            assert prepared["lease"]["state"] == "pending"
            pending_watcher = subprocess.Popen(
                [helper, "heartbeat", token, "--watch", "--interval", "0.1"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            pending_watching = json.loads(pending_watcher.stdout.readline())
            assert pending_watching["watching"] is True
            pending_watcher.wait(timeout=4)
            assert pending_watcher.returncode != 0
            assert "revoked" in pending_watcher.stderr.read().lower()
            revoked_status = control(socket_path, {"action": "status"})
            assert revoked_status["leases"][0]["state"] == "revoking"
            assert revoked_status["draining"] is True
            revoked_started = time.monotonic()
            revoked_code, revoked_body, revoked_headers = proxy_generate(
                proxy_port, 4096, admission_wait_ms=1000,
            )
            assert revoked_code == 503
            assert time.monotonic() - revoked_started < 0.5
            revoked_payload = json.loads(revoked_body)
            assert "revoked" in revoked_payload["error"].lower()
            assert revoked_payload["retryable"] is True
            assert revoked_payload["reason_code"] == "lease_transition"
            assert revoked_payload["retry_after_ms"] == 2000
            assert revoked_headers["Retry-After"] == "2"
            assert revoked_headers["X-Ollama-Unify-Retryable"] == "true"
            released = control(socket_path, {"action": "release", "token": token})
            assert released["released"] == token
            status = control(socket_path, {"action": "status"})
            assert status["leases"] == []
            assert status["draining"] is False
            assert status["backend_available"] is True
            assert len(status["gpus"]) == 3

            scoped_child = subprocess.Popen(
                [
                    helper,
                    "run",
                    "--owner",
                    "restart-safe-child",
                    "--vram-mib",
                    "1",
                    "--ttl",
                    "6",
                    "--gpu",
                    "GPU-large-0",
                    "--ready-timeout",
                    "2",
                    "--ready-command",
                    "true",
                    "--",
                    sys.executable,
                    "-c",
                    "import time; time.sleep(7)",
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                scoped_status = control(socket_path, {"action": "status"})
                scoped_leases = [
                    lease
                    for lease in scoped_status["leases"]
                    if lease["owner"] == "restart-safe-child"
                    and lease["state"] == "active"
                ]
                if scoped_leases:
                    break
                if scoped_child.poll() is not None:
                    raise RuntimeError(scoped_child.stderr.read())
                time.sleep(0.05)
            else:
                raise RuntimeError("scoped child lease did not become active")
            scoped_token = scoped_leases[0]["token"]
            scoped_gpus = scoped_leases[0]["gpu_uuids"]
            scoped_heartbeat = scoped_leases[0]["heartbeat_at"]

            daemon.terminate()
            daemon.wait(timeout=5)
            time.sleep(2.25)
            assert scoped_child.poll() is None, (
                "scoped child exited during negotiator restart grace"
            )
            daemon = subprocess.Popen(
                [helper, "serve"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if daemon.poll() is not None:
                    raise RuntimeError(daemon.stderr.read())
                if os.path.exists(socket_path):
                    try:
                        restored_status = control(
                            socket_path, {"action": "status"}
                        )
                        restored_lease = next(
                            (
                                lease
                                for lease in restored_status["leases"]
                                if lease["token"] == scoped_token
                            ),
                            None,
                        )
                        if (
                            restored_lease is not None
                            and restored_lease["heartbeat_at"] > scoped_heartbeat
                        ):
                            break
                    except (ConnectionError, OSError):
                        pass
                time.sleep(0.05)
            else:
                raise RuntimeError(
                    "scoped child did not renew its exact lease after restart"
                )
            assert restored_lease["gpu_uuids"] == scoped_gpus
            assert scoped_child.wait(timeout=10) == 0, scoped_child.stderr.read()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if not control(socket_path, {"action": "status"})["leases"]:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("scoped child did not release its lease")

            live_acquired = subprocess.run(
                [
                    helper,
                    "acquire",
                    "--owner",
                    "live-scope-fixture",
                    "--justification",
                    "exercise live scoped reservation",
                    "--expected-duration",
                    "300",
                    "--vram-mib",
                    "4096",
                    "--ttl",
                    "30",
                    "--token-only",
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            live_token = live_acquired.stdout.strip()
            blocked_code, _, _ = proxy_generate(
                proxy_port, 4096, admission_wait_ms=100,
            )
            assert blocked_code == 503
            scoped = subprocess.run(
                [
                    helper,
                    "scope",
                    live_token,
                    "--gpu",
                    "GPU-large-0",
                    "--gpu",
                    "GPU-large-1",
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            scoped_payload = json.loads(scoped.stdout)
            assert scoped_payload["lease"]["state"] == "pending"
            assert scoped_payload["lease"]["gpu_uuids"] == [
                "GPU-large-0", "GPU-large-1",
            ]
            live_status = control(socket_path, {"action": "status"})
            assert live_status["draining"] is False
            # A legacy system backend cannot prove it uses only the GPU
            # outside this pending/exclusive scope. Keep inference blocked;
            # an enabled managed pool can route to its exact unreserved UUID.
            blocked_code, _, _ = proxy_generate(
                proxy_port, 4096, admission_wait_ms=100,
            )
            assert blocked_code == 503
            control(socket_path, {"action": "release", "token": live_token})
            proxy_generate(proxy_port, 4096)

            backend_port = backend.server_port
            backend.shutdown()
            backend.server_close()
            dead_status = control(socket_path, {"action": "status"})
            assert dead_status["backend_available"] is False
            assert dead_status["backend_error"]
            assert dead_status["models"] == []
            dead_discovery = well_known(proxy_port)
            assert dead_discovery["available"] is False
            assert dead_discovery["backend_available"] is False
            with open(discovery_manifest, encoding="utf-8") as stream:
                healthy_manifest = stream.read()
            failed_refresh = subprocess.run(
                refresh_command,
                env=env,
                capture_output=True,
                text=True,
            )
            assert failed_refresh.returncode != 0
            with open(discovery_manifest, encoding="utf-8") as stream:
                assert stream.read() == healthy_manifest

            failed_acquire = subprocess.run(
                [helper, "acquire", "--owner", "backend-down",
                 "--justification", "verify backend-down refusal",
                 "--expected-duration", "60", "--vram-mib", "1"],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert failed_acquire.returncode != 0
            assert "backend unavailable" in failed_acquire.stderr.lower()

            conn = http.client.HTTPConnection("127.0.0.1", proxy_port, timeout=2)
            conn.request("GET", "/api/ps")
            unavailable = conn.getresponse()
            unavailable.read()
            conn.close()
            assert unavailable.status == 503

            backend = Backend(backend_port)
            backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
            backend_thread.start()
            recovered_discovery = well_known(proxy_port)
            assert recovered_discovery["available"] is True
            assert recovered_discovery["backend_available"] is True
            proxy_generate(proxy_port, 4096)
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait()
            if daemon.returncode not in (0, -15):
                raise RuntimeError(daemon.stderr.read())
            backend.shutdown()
            backend.server_close()

    print("negotiator integration: PASS (proxy clamp, drain, lease, backend liveness, recovery)")


if __name__ == "__main__":
    main()
