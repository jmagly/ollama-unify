#!/usr/bin/env python3
"""CPU-only transport and process-ownership regressions for cancellation."""
import http.client
import importlib.machinery
import importlib.util
import os
import signal
import socket
import subprocess
import sys
import threading
import unittest
from unittest import mock

for key in ('CONFIG', 'LEASE_STATE', 'MODEL_POLICY_STATE'):
    os.environ['OLLAMA_UNIFY_' + key] = '/nonexistent/cancellation-regression'
loader = importlib.machinery.SourceFileLoader('cancel_negotiator', sys.argv.pop(1))
spec = importlib.util.spec_from_loader(loader.name, loader)
n = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = n
loader.exec_module(n)


class StubbornProcess:
    pid = 424242

    def wait(self, timeout):
        raise subprocess.TimeoutExpired('fixture', timeout)

    def poll(self):
        return None


class CancellationTests(unittest.TestCase):
    def test_policy_is_visible_in_discovery_and_status(self):
        for backend, policy in [('cuda', 'sigterm_only'), ('rocm', 'sigterm_then_sigkill')]:
            with mock.patch.object(n, 'BACKEND_TYPE', backend), mock.patch.object(
                n, 'gpu_snapshot', return_value=[]
            ), mock.patch.object(n, 'gpu_health_snapshot', return_value={
                'admission_blocked': False, 'supported': True
            }), mock.patch.object(
                n, 'probe_backend', return_value=mock.Mock(available=True, error=None, checked_at=0)
            ):
                discovery = n.discovery_document()['parallel_pool']['request_lifecycle']
                self.assertEqual(discovery['cancel_process_policy'], policy)
                broker = n.Broker()
                self.assertEqual(broker._active_request_summary_locked()['cancel_process_policy'], policy)

    def request(self):
        broker = n.Broker()
        lane = n.Lane('cancel-test', 'managed', '127.0.0.1', 1,
                      'GPU-fixture', 'fixture:latest', 1, 4096, 0, 0,
                      StubbornProcess())
        with broker.cv:
            broker.lanes[lane.lane_id] = lane
            active = broker._register_active_request_locked(lane, 'request', '')
        admission = n.Admission(lane, 'request', '', '', 0, 0, 1)
        return broker, lane, active, admission

    def test_close_response_keeps_cancellable_socket(self):
        broker, _, active, admission = self.request()
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(2)
        self.addCleanup(listener.close)
        peer = []
        errors = []
        ready = threading.Event()

        def serve():
            try:
                conn, _ = listener.accept()
                peer.append(conn)
                conn.settimeout(2)
                conn.recv(4096)
                conn.sendall(b'HTTP/1.1 200 OK\r\nConnection: close\r\n'
                             b'Content-Length: 100\r\n\r\nx')
            except Exception as exc:
                errors.append(exc)
            finally:
                ready.set()

        server = threading.Thread(target=serve, daemon=True)
        server.start()
        backend = http.client.HTTPConnection('127.0.0.1', listener.getsockname()[1], timeout=2)
        self.addCleanup(backend.close)
        self.assertTrue(broker.request_backend_started(admission, backend))
        backend.connect()
        self.assertTrue(broker.request_backend_connected(admission, backend.sock))
        backend.request('GET', '/')
        response = backend.getresponse()
        self.addCleanup(response.close)
        self.assertTrue(ready.wait(2))
        self.assertFalse(errors)
        self.addCleanup(peer[0].close)
        self.assertTrue(response.will_close)
        self.assertIsNone(backend.sock)
        # Closing HTTPConnection alone leaves the response-owned socket open.
        # Cancellation must still deliver shutdown to that exact transport.
        broker._cancel_backend_transport(active)
        self.assertEqual(peer[0].recv(1), b'')
        server.join(2)
        self.assertFalse(server.is_alive())

    def test_cancellation_before_connect_prevents_dispatch_binding(self):
        broker, _, active, admission = self.request()
        with broker.cv:
            broker._mark_active_cancelling_locked(active, 'client_disconnected', expired=False)
        sock = mock.Mock()
        self.assertFalse(broker.request_backend_connected(admission, sock))
        self.assertIsNone(active.backend_socket)

    def test_graceful_timeout_does_not_kill(self):
        with mock.patch.object(n.os, 'killpg') as kill:
            self.assertFalse(n.Broker._terminate_process(StubbornProcess(), allow_force=False))
        kill.assert_called_once_with(StubbornProcess.pid, signal.SIGTERM)

    def test_surviving_child_does_not_get_hard_killed(self):
        process = mock.Mock(pid=424242)
        process.wait.return_value = 0
        process.poll.return_value = 0
        with mock.patch.object(n.os, 'killpg') as kill, mock.patch.object(
            n, 'process_group_alive', return_value=True
        ):
            self.assertFalse(n.Broker._terminate_process(process, allow_force=False))
        kill.assert_called_once_with(process.pid, signal.SIGTERM)

    def test_cuda_cancel_retains_exact_reservation_on_graceful_failure(self):
        broker, lane, active, _ = self.request()
        with broker.cv:
            broker._mark_active_cancelling_locked(active, 'client_disconnected', expired=False)
            active.lane_stop_started = True
        with mock.patch.object(n, 'BACKEND_TYPE', 'cuda'), mock.patch.object(
            broker, '_require_safe_gpu_transition'
        ), mock.patch.object(n.os, 'killpg') as kill, mock.patch.object(n.LOG, 'error') as log:
            broker._stop_expired_lane(lane)
        kill.assert_called_once_with(StubbornProcess.pid, signal.SIGTERM)
        self.assertIs(broker.lanes[lane.lane_id], lane)
        self.assertIs(broker.active_request_records['request'], active)
        self.assertEqual(lane.in_flight, 1)
        self.assertEqual(lane.reserved_mib, 4096)
        self.assertTrue(lane.retiring)
        self.assertFalse(active.lane_stop_started)
        self.assertIn('process_group_alive', log.call_args.args)
        self.assertIs(log.call_args.args[-1], False)

    def test_transition_contention_is_reported_without_signalling(self):
        broker, lane, active, _ = self.request()
        broker.transition.acquire()
        try:
            with mock.patch.object(n.os, 'killpg') as kill, mock.patch.object(n.LOG, 'error') as log:
                worker = threading.Thread(target=broker._stop_expired_lane, args=(lane,))
                worker.start()
                worker.join(2)
                self.assertFalse(worker.is_alive())
            kill.assert_not_called()
            self.assertIn('transition_busy', log.call_args.args)
            self.assertIs(broker.active_request_records['request'], active)
        finally:
            broker.transition.release()


if __name__ == '__main__':
    unittest.main()
