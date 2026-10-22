#!/usr/bin/env python3
"""Exercise client identification helpers without sockets or privileges."""

import importlib.machinery
import importlib.util
import os
import signal
import sys
import tempfile
import threading
import time
from unittest import mock


def load_negotiator(path):
    # Keep the host's broker configuration out of the imported module.
    os.environ["OLLAMA_UNIFY_CONFIG"] = "/nonexistent/ollama-unify-negotiator"
    os.environ["OLLAMA_UNIFY_LEASE_STATE"] = "/nonexistent/leases.json"
    os.environ["OLLAMA_UNIFY_MODEL_POLICY_STATE"] = "/nonexistent/policy.json"
    loader = importlib.machinery.SourceFileLoader("negotiator", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def test_parse_socket_owner(negotiator):
    line = (
        "0      0      127.0.0.1:41994 127.0.0.1:11434 "
        "timer:(keepalive,50sec,0) uid:1005 ino:3196166 sk:6001 "
        "cgroup:/system.slice/matric-tau.service <->\n"
    )
    assert negotiator.parse_socket_owner(line) == {
        "uid": 1005, "inode": 3196166, "cgroup": "/system.slice/matric-tau.service",
    }
    # Older iproute2 omits cgroup; identification still uses the uid.
    assert negotiator.parse_socket_owner(
        "0 0 [::1]:5000 [::1]:11434 uid:0 ino:12 sk:1 <->"
    ) == {"uid": 0, "inode": 12, "cgroup": ""}
    assert negotiator.parse_socket_owner("") is None


def test_cgroup_unit(negotiator):
    assert negotiator.cgroup_unit("/system.slice/matric-tau.service") == (
        "matric-tau.service", "",
    )
    assert negotiator.cgroup_unit(
        "/system.slice/docker-3b73baae0c53aa.scope"
    ) == ("docker-3b73baae0c53aa.scope", "3b73baae0c53aa")
    assert negotiator.cgroup_unit(
        "/user.slice/user-1000.slice/user@1000.service/app.slice/"
        "app-org.gnome.Terminal.slice/vte-spawn-1.scope"
    ) == ("vte-spawn-1.scope", "")
    assert negotiator.cgroup_unit("/") == ("", "")


def test_keys_and_labels(negotiator):
    key = negotiator.client_key_and_label
    assert key({"declared": "eval-runner", "user": "roctinam",
                "container": "ignored"}) == (
        "app:eval-runner", "eval-runner (roctinam)",
    )
    assert key({"container": "voryn", "address": "172.18.0.3"}) == (
        "container:voryn", "container voryn",
    )
    assert key({"unit": "matric-tau.service", "user": "roctinam",
                "process": "python3"}) == (
        "unit:roctinam:matric-tau.service", "matric-tau.service (roctinam)",
    )
    # Interactive scopes are per terminal; the process names the app.
    assert key({"unit": "vte-spawn-1.scope", "user": "roko",
                "process": "python3"}) == (
        "process:roko:python3", "python3 (roko)",
    )
    assert key({"address": "192.168.1.39", "user_agent": "ollama-js/0.5"}) == (
        "remote:192.168.1.39", "192.168.1.39 · ollama-js/0.5",
    )
    assert negotiator.clean_client_text("app\x00\nname " * 40).startswith("app")
    assert len(negotiator.clean_client_text("x" * 500)) == 128


def test_process_names_and_inference(negotiator, tmp):
    import subprocess
    script = os.path.join(tmp, "worker_app.py")
    with open(script, "w", encoding="utf-8") as stream:
        stream.write("import time\ntime.sleep(30)\n")
    child = subprocess.Popen([sys.executable, script, "--token", "secret"])
    try:
        # The kernel fills /proc/PID/cmdline once the exec completes.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with open(f"/proc/{child.pid}/cmdline", "rb") as stream:
                if stream.read():
                    break
            time.sleep(0.02)
        name = negotiator.process_name(child.pid)
        interpreter = os.path.basename(sys.executable)
        assert name == f"{interpreter} worker_app.py", name
        assert "secret" not in name
    finally:
        child.kill()
        child.wait()
    key = negotiator.client_key_and_label
    assert key({"unit": "vte-spawn-1.scope", "user": "roko",
                "candidate_processes": ["bash", "claude", "curl"]}) == (
        "unit:roko:vte-spawn-1.scope", "claude / curl (roko)",
    )
    assert key({"unit": "session-3.scope", "user": "roko",
                "candidate_processes": ["bash", "sshd"]}) == (
        "unit:roko:session-3.scope", "bash / sshd (roko)",
    )


def test_docker_directory_cache(negotiator):
    directory = negotiator.DockerDirectory("/nonexistent/docker.sock")
    # An unreachable Docker API degrades to "unknown", never an error.
    assert directory.lookup(ip="172.18.0.3") == ""
    directory.by_ip = {"172.18.0.3": "voryn"}
    directory.by_id = {"3b73baae0c53aabbcc": "moshi"}
    assert directory.lookup(ip="172.18.0.3") == "voryn"
    assert directory.lookup(container_id="3b73baae0c53") == "moshi"


def test_exact_admission_release_is_idempotent(negotiator):
    broker = negotiator.Broker()
    lane = broker.lanes["base"]
    with broker.cv:
        broker._register_active_request_locked(lane, "old-request", "")
    old = negotiator.Admission(lane, "old-request", "", "", 0, 1, 0)
    broker.proxy_exit(old, "", True)

    with broker.cv:
        broker._register_active_request_locked(lane, "new-request", "")
    new = negotiator.Admission(lane, "new-request", "", "", 0, 1, 0)

    # A delayed finally block from the old handler must not decrement the
    # newer request just because both requests used the same lane.
    broker.proxy_exit(old, "", False)
    assert broker.active_requests == 1
    assert lane.in_flight == 1
    assert list(broker.active_request_records) == ["new-request"]

    broker.proxy_exit(new, "", True)
    assert broker.active_requests == 0
    assert lane.in_flight == 0


def test_failed_lane_stop_retains_reservation(negotiator):
    broker = negotiator.Broker()
    lane = negotiator.Lane(
        "failed-stop",
        "managed",
        "127.0.0.1",
        65530,
        "GPU-test",
        "fixture:latest",
        1,
        4096,
        time.time(),
        time.time(),
        object(),
    )
    with broker.cv:
        broker.lanes[lane.lane_id] = lane
    with (
        mock.patch.object(
            negotiator, "unload_models_at", lambda *_args, **_kwargs: None
        ),
        mock.patch.object(broker, "_terminate_process", lambda _process: False),
        mock.patch.object(negotiator, "process_group_alive", lambda _process: True),
        mock.patch.object(negotiator, "SELECTED_GPUS", ["GPU-test"]),
        mock.patch.object(negotiator, "gpu_snapshot", lambda: [{
            "uuid": "GPU-test", "total_mib": 8192, "free_mib": 8192,
        }]),
        mock.patch.object(negotiator, "foreign_gpu_usage", lambda **_kwargs: {}),
        mock.patch.object(negotiator, "exclusive_cuda_contexts", lambda: {}),
    ):
        failed = broker._stop_lanes([lane], "test failed process group")
        assert failed == [lane]
        assert broker.lanes[lane.lane_id] is lane
        assert lane.retiring is True
        assert lane.reserved_mib == 4096
        placement = broker._placement_devices(set())
        assert len(placement) == 1
        assert placement[0]["reserved_mib"] == 4096
        assert placement[0]["free_mib"] == 4096


def test_terminate_process_kills_stubborn_group_child(negotiator):
    state = {"child_alive": True}
    delivered = []

    class FakeProcess:
        pid = 424242

        def __init__(self):
            self.returncode = None

        def wait(self, timeout):
            assert timeout == 15
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    def killpg(pid, delivered_signal):
        assert pid == FakeProcess.pid
        delivered.append(delivered_signal)
        if delivered_signal == signal.SIGKILL:
            state["child_alive"] = False

    process = FakeProcess()
    with (
        mock.patch.object(negotiator.os, "killpg", killpg),
        mock.patch.object(
            negotiator,
            "process_group_alive",
            lambda _process: state["child_alive"],
        ),
    ):
        stopped = negotiator.Broker._terminate_process(process)
    assert stopped is True
    assert delivered == [signal.SIGTERM, signal.SIGKILL]


def test_completed_admission_has_bounded_terminal_release(negotiator):
    broker = negotiator.Broker()
    lane = broker.lanes["base"]
    with broker.cv:
        broker._register_active_request_locked(
            lane, "terminal-request", "turn:terminal-request"
        )
    admission = negotiator.Admission(
        lane,
        "terminal-request",
        "turn:terminal-request",
        "fingerprint",
        0,
        1,
        0,
    )
    assert broker.request_backend_complete(admission) is True
    with broker.cv:
        broker.active_request_records[
            admission.request_id
        ].expires_at = time.monotonic() - 1
        broker.cv.notify_all()
    watcher = threading.Thread(target=broker.active_request_watchdog)
    watcher.start()
    deadline = time.monotonic() + 2
    while broker.active_requests and time.monotonic() < deadline:
        time.sleep(0.01)
    broker.stopping.set()
    with broker.cv:
        broker.cv.notify_all()
    watcher.join(timeout=2)
    assert not watcher.is_alive()
    assert broker.active_requests == 0
    assert lane.in_flight == 0
    assert broker.request_terminal_release_total == 1
    tombstone = broker.logical_tombstones["turn:terminal-request"]
    assert tombstone.request_id == "terminal-request"
    assert tombstone.reason_code == "completion_finalization_timeout"


def main():
    negotiator = load_negotiator(sys.argv[1])
    test_parse_socket_owner(negotiator)
    test_cgroup_unit(negotiator)
    test_keys_and_labels(negotiator)
    with tempfile.TemporaryDirectory() as tmp:
        test_process_names_and_inference(negotiator, tmp)
    test_docker_directory_cache(negotiator)
    test_exact_admission_release_is_idempotent(negotiator)
    test_failed_lane_stop_retains_reservation(negotiator)
    test_terminate_process_kills_stubborn_group_child(negotiator)
    test_completed_admission_has_bounded_terminal_release(negotiator)
    print("client identity: PASS (socket owner, cgroup units, keys, labels, "
          "process names without arguments, docker directory, exact "
          "admission release, failed-stop reservation retention, stubborn "
          "process-group escalation, bounded terminal release)")


if __name__ == "__main__":
    main()
