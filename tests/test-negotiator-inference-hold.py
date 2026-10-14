#!/usr/bin/env python3
"""Operator holds must prevent execution even while GPU health is good."""
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ['OLLAMA_UNIFY_CONFIG'] = '/nonexistent/negotiator-config'
os.environ['OLLAMA_UNIFY_LEASE_STATE'] = '/nonexistent/leases.json'
os.environ['OLLAMA_UNIFY_MODEL_POLICY_STATE'] = '/nonexistent/policy.json'
os.environ['OLLAMA_UNIFY_CACHE_RESIDENCY_STATE'] = '/nonexistent/cache.json'
loader = importlib.machinery.SourceFileLoader('hold_negotiator', sys.argv.pop(1))
spec = importlib.util.spec_from_loader(loader.name, loader)
n = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = n
loader.exec_module(n)


class InferenceHoldTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'hold'
        patch = mock.patch.object(n, 'INFERENCE_HOLD_FILE', str(self.path))
        patch.start()
        self.addCleanup(patch.stop)

    def activate(self):
        self.path.write_text('Manual hardware comparison; no model loading.\n')

    def test_absent_hold_preserves_existing_behavior(self):
        self.assertFalse(n.inference_hold_status()['active'])
        n.require_inference_enabled()

    def test_hold_and_release_are_dynamic(self):
        self.activate()
        selected = list(n.SELECTED_GPUS)
        with self.assertRaises(n.PermanentCapacityError) as raised:
            n.require_inference_enabled(request_id='request', logical_request_id='logical')
        self.assertEqual(raised.exception.reason_code, 'operator_inference_hold')
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(raised.exception.request_id, 'request')
        self.path.unlink()
        n.require_inference_enabled()
        self.assertEqual(n.SELECTED_GPUS, selected)

    def test_unreadable_hold_fails_closed(self):
        self.path.mkdir()
        self.assertTrue(n.inference_hold_status()['active'])
        with self.assertRaises(n.PermanentCapacityError):
            n.require_inference_enabled()

    def test_invalid_encoding_fails_closed(self):
        self.path.write_bytes(b'\xff')
        with self.assertRaises(n.PermanentCapacityError):
            n.require_inference_enabled()

    def test_cli_reports_daemon_hold_despite_different_cli_environment(self):
        for active in (True, False):
            with self.subTest(daemon_hold=active):
                daemon_hold = {'active': active, 'path': '/daemon/operator-hold',
                               'reason': 'Operator controls inference'}
                document = {'gpu_health': {'admission_blocked': False, 'supported': True},
                            'parallel_pool': {}, 'warnings': [],
                            'inference_hold': {'active': not active}}
                with mock.patch.object(sys, 'argv', ['broker', 'discover']), mock.patch.object(
                    n, 'discovery_document', return_value=document
                ), mock.patch.object(n, 'send_control', return_value={
                    'leases': [], 'inference_hold': daemon_hold
                }), mock.patch('sys.stdout', new_callable=io.StringIO) as output:
                    self.assertEqual(n.main(), 0)
                self.assertEqual(json.loads(output.getvalue())['inference_hold'], daemon_hold)

    def test_internal_entrypoints_block_before_health_or_spawn(self):
        self.activate()
        broker = n.Broker()
        with mock.patch.object(n, 'gpu_health_snapshot') as health, mock.patch.object(
            n.subprocess, 'Popen'
        ) as spawn:
            operations = [
                lambda: broker.ensure_capacity('qwen:latest', 1),
                lambda: broker.proxy_enter('qwen:latest', True),
                lambda: broker._select_lane_locked('qwen:latest', True),
                lambda: broker._spawn_lane('qwen:latest', 'GPU-selected', 1, set(), '/api/generate'),
            ]
            for operation in operations:
                with self.assertRaises(n.PermanentCapacityError) as raised:
                    operation()
                self.assertEqual(raised.exception.reason_code, 'operator_inference_hold')
            health.assert_not_called()
            spawn.assert_not_called()

    def test_http_routes_block_without_backend_or_capacity_execution(self):
        self.activate()
        for path in (*n.INFERENCE_PATHS, n.CAPACITY_PATH):
            with self.subTest(path=path):
                handler = n.ProxyHandler.__new__(n.ProxyHandler)
                handler.path = path
                handler.command = 'POST'
                body = json.dumps({'model': 'qwen:latest', 'prompt': 'test'}).encode()
                handler.headers = {'Content-Length': str(len(body)), 'Content-Type': 'application/json'}
                handler.rfile = io.BytesIO(body)
                handler.broker = mock.Mock()
                handler._send_capacity_failure = mock.Mock()
                handler._handle()
                failure = handler._send_capacity_failure.call_args.args[0]
                self.assertEqual(failure.reason_code, 'operator_inference_hold')
                self.assertEqual(handler.broker.mock_calls, [])


if __name__ == '__main__':
    unittest.main()
