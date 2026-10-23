#!/usr/bin/env python3
"""CPU-only confidentiality boundary tests against the real HTTP proxy and pool."""
import importlib.machinery
import importlib.util
import json
import http.client
import logging
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import subprocess
import time
import unittest
import stat
from types import SimpleNamespace
from unittest import mock

HELPER = sys.argv.pop(1)
FIXTURE_BIN = sys.argv.pop(1)


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


p = load('confidential_pool', Path(__file__).with_name('test-negotiator-pool.py'))
with mock.patch.dict(os.environ, {'OLLAMA_UNIFY_CONFIG': '/nonexistent/confidential-test'}):
    m = load('confidential_broker', HELPER)
TOKEN = 'T' * 64
AUTH = {'Authorization': 'Bearer ' + TOKEN}


class ConfidentialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.token_path = Path(self.temp.name) / 'token'
        self.token_path.write_text(TOKEN)
        self.token_path.chmod(0o400)
        self.environment = {
            'OLLAMA_UNIFY_CONFIDENTIAL': '1',
            'OLLAMA_UNIFY_PROJECT_ID': 'private-test',
            'OLLAMA_UNIFY_TOKEN_FILE': str(self.token_path),
            'OLLAMA_UNIFY_APPROVED_MODELS': json.dumps({p.MODEL: p.MODEL_DIGEST}),
            'OLLAMA_UNIFY_CLIENT_READ_TIMEOUT': '1',
            'OLLAMA_UNIFY_MAX_CONNECTIONS': '2',
        }

    def tearDown(self):
        self.temp.cleanup()

    def harness(self):
        return p.PoolHarness(HELPER, FIXTURE_BIN)

    def request(self, case, method, path, payload=None, headers=None):
        return p.http_json(case.proxy_port, method, path, payload,
                           extra_headers=AUTH if headers is None else headers)

    def raw(self, case, data):
        conn = socket.create_connection(('127.0.0.1', case.proxy_port), timeout=3)
        conn.sendall(data)
        response = conn.recv(65536)
        conn.close()
        return response

    def test_authenticated_surface_framing_and_model_policy(self):
        with mock.patch.dict(os.environ, self.environment), self.harness() as case:
            # No body bytes sent; authentication must complete immediately.
            started = time.monotonic()
            response = self.raw(case, b'POST /api/chat HTTP/1.1\r\nHost: local\r\nContent-Length: 999\r\n\r\n')
            self.assertIn(b'401', response)
            self.assertLess(time.monotonic() - started, .8)
            self.assertEqual(self.request(case, 'GET', '/healthz')[0], 200)
            self.assertEqual(self.request(case, 'GET', '/healthz', headers={})[0], 401)
            for method, path in [('GET', '/api/tags'), ('GET', '/api/ps'),
                                 ('POST', '/api/pull'), ('POST', p.CAPACITY_PATH if hasattr(p, 'CAPACITY_PATH') else '/.well-known/ollama-unify-gpu-negotiator/capacity'),
                                 ('GET', '/.well-known/ollama-unify-gpu-negotiator')]:
                self.assertEqual(self.request(case, method, path)[0], 403)
            for header in ['X-Ollama-Unify-Logical-Request-Id', 'X-Ollama-Unify-Resume-Request',
                           'X-Ollama-Unify-GPU-UUIDs', 'X-Ollama-Unify-Warm-Admission']:
                self.assertEqual(self.request(case, 'POST', '/api/chat', {'model': p.MODEL},
                                              {**AUTH, header: 'secret-content'})[0], 400)
            for framing in ['Content-Length: -1', 'Content-Length: 1\r\nContent-Length: 1',
                            'Transfer-Encoding: chunked', 'Content-Length: 16777217',
                            'Authorization: Bearer ' + TOKEN]:
                raw = ('POST /api/chat HTTP/1.1\r\nHost: local\r\nAuthorization: Bearer ' + TOKEN + '\r\n' + framing + '\r\n\r\n').encode()
                response = self.raw(case, raw)
                self.assertTrue(any(str(code).encode() in response for code in (400, 401, 413)), response)
            self.assertEqual(self.request(case, 'POST', '/api/chat', {'model': 'unapproved'})[0], 403)
            self.assertEqual(self.request(case, 'POST', '/api/chat', {'model': p.MODEL, 'keep_alive': 0})[0], 400)
            self.assertEqual(self.request(case, 'POST', '/api/chat', {'model': p.MODEL, 'options': {'num_gpu': 0}})[0], 400)
            for alias_payload in [
                    {'model': p.MODEL, 'Model': 'unapproved'},
                    {'model': p.MODEL, 'Options': {'num_gpu': 0}},
                    {'model': p.MODEL, 'optionſ': {'num_gpu': 0}},
                    {'model': p.MODEL, 'Keep_Alive': 0}]:
                self.assertEqual(self.request(case, 'POST', '/api/chat', alias_payload)[0], 400)
            duplicate_body = ('{"model":"' + p.MODEL + '","model":"unapproved"}').encode()
            duplicate_request = ('POST /api/chat HTTP/1.1\r\nHost: local\r\nAuthorization: Bearer ' + TOKEN + '\r\nContent-Length: ' + str(len(duplicate_body)) + '\r\n\r\n').encode() + duplicate_body
            self.assertIn(b'400', self.raw(case, duplicate_request))
            case.backend.tags[0]['digest'] = 'b' * 64
            self.assertEqual(self.request(case, 'POST', '/api/chat', {'model': p.MODEL})[0], 503)

    def test_real_inference_fixed_identity_no_credentials_no_replay(self):
        with mock.patch.dict(os.environ, self.environment), self.harness() as case:
            result = self.request(case, 'POST', '/api/chat',
                                  {'model': p.MODEL, 'stream': False, 'mock_request_id': 'private-test'},
                                  {**AUTH, 'X-Ollama-Unify-Client': 'spoofed',
                                   'Cookie': 'sensitive', 'Proxy-Authorization': 'secret'})
            self.assertEqual(result[0], 200, result)
            event = next(event for event in p.events(case.event_log)
                         if event.get('request_id') == 'private-test')
            for key in ['has_authorization', 'has_proxy_authorization', 'has_cookie']:
                self.assertFalse(event[key], event)
            status = case.status()
            encoded = json.dumps(status)
            self.assertNotIn('spoofed', encoded)
            self.assertIn('project:private-test', encoded)
            self.assertEqual(status['parallel_pool']['logical_in_flight'], 0)
            self.assertEqual(status['parallel_pool']['completed_responses']['entries'], 0)
            self.assertEqual(self.request(case, 'POST', '/api/chat',
                                          {'model': p.MODEL, 'mock_backend_reset': True})[0], 503)
            case.daemon.terminate()
            case.daemon.wait(timeout=10)
            case.daemon.stderr.seek(0)
            output = case.daemon.stderr.read()
            self.assertNotIn(TOKEN, output)
            self.assertNotIn('private-test', output)

    def test_cpu_overflow_header_rejected_before_body(self):
        with mock.patch.dict(os.environ, self.environment), self.harness() as case:
            for value in ['allow', 'deny', 'allow\r\nX-Ollama-Unify-CPU-Offload: allow']:
                request = ('POST /api/chat HTTP/1.1\r\nHost: local\r\nAuthorization: Bearer ' + TOKEN
                    + '\r\nX-Ollama-Unify-CPU-Offload: ' + value
                    + '\r\nContent-Length: 999\r\n\r\n').encode()
                self.assertIn(b'400', self.raw(case, request))
            self.assertEqual(p.events(case.event_log), [])

    def test_internal_cpu_overflow_and_invalid_residency_fail_closed(self):
        broker = object.__new__(m.Broker)
        with mock.patch.object(m, 'CONFIDENTIAL', True):
            with self.assertRaises(m.PermanentCapacityError) as caught:
                broker._ensure_cpu_offload_capacity('approved:latest', 1, 10, {'completion'},
                    '/api/chat', ('GPU-test',), None, None, allow_reclaim=True)
            self.assertEqual(caught.exception.reason_code, 'cpu_offload_forbidden')
        for item in [{}, {'size': 10, 'size_vram': 9}, {'size': 0, 'size_vram': 10},
                     {'size': 10, 'size_vram': float('nan')},
                     {'size': float('inf'), 'size_vram': float('inf')},
                     {'size': True, 'size_vram': 10}]:
            self.assertFalse(m.model_fully_gpu_resident(item))
        self.assertTrue(m.model_fully_gpu_resident({'size': 10, 'size_vram': 10}))

    def test_partial_single_gpu_warmup_cannot_dispatch_private_request(self):
        with mock.patch.dict(os.environ, self.environment), p.PoolHarness(
                HELPER, FIXTURE_BIN, partial_gpu_residency=True) as case:
            result = self.request(case, 'POST', '/api/chat',
                                  {'model': p.MODEL, 'mock_request_id': 'no-partial-dispatch'})
            self.assertEqual(result[0], 503, result)
            self.assertFalse(any(event.get('request_id') == 'no-partial-dispatch'
                                 for event in p.events(case.event_log)))

    def test_dispatch_rechecks_residency_and_rejects_cpu_lane(self):
        original = Path(HELPER).read_text()
        marker = '                    verified = bool(approved_running)'
        self.assertEqual(original.count(marker), 1)
        variants = {
            'partial': original.replace(marker,
                '                    for item in approved_running: item["size_vram"] = 1\n' + marker),
            'cpu-lane': original.replace('        lane = admission.lane\n        try:',
                '        lane = admission.lane\n        lane.cpu_offload = True\n        try:')}
        for name, source in variants.items():
            executable = Path(self.temp.name) / ('dispatch-' + name)
            executable.write_text(source)
            executable.chmod(0o700)
            with self.subTest(name=name), mock.patch.dict(os.environ, self.environment), p.PoolHarness(
                    str(executable), FIXTURE_BIN) as case:
                result = self.request(case, 'POST', '/api/chat',
                                      {'model': p.MODEL, 'mock_request_id': 'no-overflow-dispatch'})
                self.assertIn(result[0], [403, 503], result)
                self.assertFalse(any(event.get('request_id') == 'no-overflow-dispatch'
                                     for event in p.events(case.event_log)))

    def test_actual_worker_digest_mismatch(self):
        original_popen = subprocess.Popen
        def mismatched_worker(*args, **kwargs):
            kwargs['env']['MOCK_OLLAMA_MODEL_DIGESTS'] = json.dumps({p.MODEL: 'b' * 64})
            return original_popen(*args, **kwargs)
        with mock.patch.dict(os.environ, self.environment), mock.patch.object(
                subprocess, 'Popen', side_effect=mismatched_worker), self.harness() as case:
            result = self.request(case, 'POST', '/api/chat',
                                  {'model': p.MODEL, 'mock_request_id': 'must-not-dispatch'})
            self.assertEqual(result[0], 503, result)
            self.assertEqual(result[1]['reason_code'], 'model_digest_mismatch')
            self.assertFalse(any(event.get('request_id') == 'must-not-dispatch'
                                 for event in p.events(case.event_log)))

    def test_bounded_header_body_and_connection_slots(self):
        with mock.patch.dict(os.environ, self.environment), self.harness() as case:
            time.sleep(.15)
            first = socket.create_connection(('127.0.0.1', case.proxy_port), timeout=3)
            second = socket.create_connection(('127.0.0.1', case.proxy_port), timeout=3)
            first.sendall(b'POST /api/chat HTTP/1.1\r\n')
            second.sendall(('POST /api/chat HTTP/1.1\r\nHost: local\r\nAuthorization: Bearer ' + TOKEN + '\r\nContent-Length: 100\r\n\r\n{').encode())
            time.sleep(.15)
            third = socket.create_connection(('127.0.0.1', case.proxy_port), timeout=3)
            self.assertEqual(third.recv(100), b'')
            third.close()
            self.assertEqual(first.recv(100), b'')
            self.assertIn(b'400', second.recv(1000))
            first.close()
            second.close()
            self.assertEqual(self.request(case, 'GET', '/healthz')[0], 200)

    def test_unix_frontend_and_strict_profile_flag(self):
        path = str(Path(self.temp.name) / 'http.sock')
        class UnixConnection(http.client.HTTPConnection):
            def connect(self):
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(2)
                self.sock.connect(path)
        with mock.patch.multiple(m, CONFIDENTIAL=True, CONFIDENTIAL_TOKEN=TOKEN):
            server = m.ThreadingUnixHTTPServer(path, m.ProxyHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                conn = UnixConnection('localhost')
                conn.request('GET', '/healthz', headers=AUTH)
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read()), {'ok': True})
                conn.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
        env = {**os.environ, 'OLLAMA_UNIFY_CONFIG': '/nonexistent/test',
               'OLLAMA_UNIFY_CONFIDENTIAL': 'treu'}
        result = subprocess.run([HELPER, 'serve'], env=env, capture_output=True, timeout=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'invalid confidential profile flag', result.stderr)

    def test_unhandled_exceptions_do_not_write_content(self):
        original = Path(HELPER).read_text()
        marker = "    broker = Broker()\n    control_path ="
        self.assertEqual(original.count(marker), 1)
        sentinel = 'synthetic-private-prompt-DO-NOT-LOG'
        for exception in ('ValueError', 'RuntimeError'):
            injection = (
                "    def fail_thread():\n"
                "        raise ValueError(" + repr(sentinel) + ")\n"
                "    injected_thread = threading.Thread(target=fail_thread)\n"
                "    injected_thread.start()\n"
                "    injected_thread.join()\n"
                "    raise " + exception + "(" + repr(sentinel) + ")\n"
                "    broker = Broker()\n    control_path =")
            executable = Path(self.temp.name) / ('exception-' + exception)
            executable.write_text(original.replace(marker, injection))
            executable.chmod(0o700)
            env = {**os.environ, **self.environment,
                   'OLLAMA_UNIFY_CONFIG': '/nonexistent/exception-test',
                   'OLLAMA_UNIFY_BACKEND_TYPE': 'cuda',
                   'OLLAMA_UNIFY_POOL_ENABLED': '1',
                   'OLLAMA_UNIFY_SELECTED_GPUS': 'GPU-test'}
            result = subprocess.run([str(executable), 'serve'], env=env,
                                    capture_output=True, timeout=3)
            self.assertNotEqual(result.returncode, 0)
            output = (result.stdout + result.stderr).decode()
            self.assertNotIn(sentinel, output)
            self.assertNotIn(TOKEN, output)
            self.assertIn('confidential broker operational event', output)

    def test_systemd_credential_0440_exception_is_exact_and_protected(self):
        directory = '/run/credentials/project-inference.service'
        token_path = directory + '/broker-token'
        def info(mode, uid=0, gid=0, ino=42):
            return SimpleNamespace(st_mode=mode, st_uid=uid, st_gid=gid,
                                   st_ino=ino, st_dev=1)
        token = info(stat.S_IFREG | 0o440)
        ancestors = {name: info(stat.S_IFDIR | (0o550 if name == directory else 0o755))
                     for name in ['/', '/run', '/run/credentials', directory]}
        ancestors[token_path] = token
        with mock.patch.object(m, 'TOKEN_FILE', token_path), mock.patch.dict(
                os.environ, {'CREDENTIALS_DIRECTORY': directory}), mock.patch.object(
                m.os, 'lstat', side_effect=lambda name: ancestors[name]):
            self.assertTrue(m.systemd_credential_token_allowed(token))
            for bad_token in [info(stat.S_IFREG | 0o440, uid=1000),
                              info(stat.S_IFREG | 0o440, gid=1000),
                              info(stat.S_IFREG | 0o444)]:
                self.assertFalse(m.systemd_credential_token_allowed(bad_token))
            for name in ['/', '/run', '/run/credentials', directory]:
                original = ancestors[name]
                for bad in [info(stat.S_IFLNK | 0o777), info(stat.S_IFDIR | 0o775),
                            info(stat.S_IFDIR | 0o755, uid=1000),
                            info(stat.S_IFDIR | 0o755, gid=1000)]:
                    ancestors[name] = bad
                    self.assertFalse(m.systemd_credential_token_allowed(token))
                ancestors[name] = original
            ancestors[token_path] = info(stat.S_IFREG | 0o440, ino=99)
            self.assertFalse(m.systemd_credential_token_allowed(token))
            ancestors[token_path] = token
            for bad_path in ['/tmp/broker-token', directory+'/other',
                             directory+'/../project-inference.service/broker-token']:
                with mock.patch.object(m, 'TOKEN_FILE', bad_path):
                    self.assertFalse(m.systemd_credential_token_allowed(token))
            for bad_dir in ['/tmp/project-inference.service', '/run/credentials/a/b.service',
                            '/run/credentials/../evil.service', '', '/run/credentials/project.socket']:
                with mock.patch.dict(os.environ, {'CREDENTIALS_DIRECTORY': bad_dir}):
                    self.assertFalse(m.systemd_credential_token_allowed(token))

    def test_token_and_configuration_fail_closed(self):
        attrs = dict(CONFIDENTIAL=True, PROJECT_ID='test', TOKEN_FILE=str(self.token_path),
                     POOL_ENABLED=True, SELECTED_GPUS=['GPU-0'], BACKEND_TYPE='cuda',
                     LISTEN_HOST='127.0.0.1', BACKEND_HOST='127.0.0.1')
        with mock.patch.multiple(m, **attrs), mock.patch.dict(os.environ, self.environment):
            m.configure_confidential_profile()
            for mode in [0o644, 0o440]:
                self.token_path.chmod(mode)
                with self.assertRaises(ValueError):
                    m.configure_confidential_profile()
            self.token_path.chmod(0o600)
            m.configure_confidential_profile()
            with mock.patch.object(m, 'POOL_ENABLED', False), self.assertRaises(ValueError):
                m.configure_confidential_profile()
            with mock.patch.dict(os.environ, {'OLLAMA_UNIFY_APPROVED_MODELS': '{}'}), self.assertRaises(ValueError):
                m.configure_confidential_profile()
            with mock.patch.object(m, 'LISTEN_HOST', '0.0.0.0'), self.assertRaises(ValueError):
                m.configure_confidential_profile()
            link = self.token_path.with_name('link')
            link.symlink_to(self.token_path)
            with mock.patch.object(m, 'TOKEN_FILE', str(link)), self.assertRaises(ValueError):
                m.configure_confidential_profile()
            record = logging.LogRecord('test', logging.ERROR, '', 0, 'secret %s', ('payload',), None)
            m.ConfidentialLogFilter().filter(record)
            self.assertNotIn('payload', record.getMessage())


if __name__ == '__main__':
    unittest.main()
