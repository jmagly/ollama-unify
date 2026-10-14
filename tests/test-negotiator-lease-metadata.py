#!/usr/bin/env python3
"""CPU-only lease metadata and CLI compatibility regressions."""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import sys
import unittest
from unittest import mock

os.environ['OLLAMA_UNIFY_CONFIG'] = '/nonexistent/ollama-unify-negotiator'
os.environ['OLLAMA_UNIFY_LEASE_STATE'] = '/nonexistent/leases.json'
os.environ['OLLAMA_UNIFY_MODEL_POLICY_STATE'] = '/nonexistent/policy.json'
loader = importlib.machinery.SourceFileLoader('metadata_negotiator', sys.argv.pop(1))
spec = importlib.util.spec_from_loader(loader.name, loader)
n = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = n
loader.exec_module(n)


class LeaseMetadataTests(unittest.TestCase):
    def test_policy_distinguishes_requirements_from_recommendations(self):
        policy = n.lease_policy_document()
        self.assertEqual(policy['required_acquire_fields'], ['owner'])
        self.assertEqual(policy['cli_required_options'], ['--owner'])
        self.assertEqual(policy['recommended_acquire_fields'], [
            'justification', 'expected_duration_seconds',
        ])
        self.assertEqual(policy['cli_recommended_options'], [
            '--justification', '--expected-duration',
        ])
        self.assertIn('non-blocking', policy['metadata_advisory'])

    def test_unknown_horizon_is_not_invented_from_heartbeat_ttl(self):
        summary = n.lease_public_summary({
            'owner': 'portal', 'ttl': 120, 'created_at': 100, 'token': 'secret',
        }, now=100)
        self.assertEqual(summary['justification'], 'not provided')
        self.assertIsNone(summary['expected_release_at'])
        self.assertIsNone(summary['seconds_until_expected_release'])
        self.assertEqual(summary['horizon_status'], 'unknown')
        self.assertEqual(len(summary['recommendations']), 2)
        self.assertNotIn('secret', json.dumps(summary))

    def test_provided_metadata_stays_visible_independent_of_ttl(self):
        summary = n.lease_public_summary({
            'owner': 'portal', 'justification': 'web', 'ttl': 120,
            'expected_release_at': 1000,
        }, now=100)
        self.assertEqual(summary['justification'], 'web')
        self.assertEqual(summary['seconds_until_expected_release'], 900)
        self.assertEqual(summary['recommendations'], [])

    def test_cli_acquire_without_metadata_preserves_token_only_output(self):
        result = {'ok': True, 'lease': {'token': 'lease_fixture'},
                  'public_lease': n.lease_public_summary({'owner': 'portal'})}
        with mock.patch.object(sys, 'argv', [
            'broker', 'acquire', '--owner', 'portal', '--gpu', 'GPU-fixture', '--token-only',
        ]), mock.patch.object(n, 'send_control', return_value=result) as control, \
                contextlib.redirect_stdout(io.StringIO()) as stdout, \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(n.main(), 0)
        self.assertEqual(stdout.getvalue(), 'lease_fixture\n')
        self.assertIn('non-blocking', stderr.getvalue())
        self.assertIn('--justification', stderr.getvalue())
        self.assertIn('--expected-duration', stderr.getvalue())
        request = control.call_args.args[0]
        self.assertEqual(request['gpu_uuids'], ['GPU-fixture'])
        self.assertEqual(request['justification'], '')
        self.assertEqual(request['expected_duration_seconds'], 0)

    def test_cli_run_without_metadata_keeps_scoped_readiness_requirement(self):
        with mock.patch.object(sys, 'argv', [
            'broker', 'run', '--owner', 'portal', '--gpu', 'GPU-fixture',
            '--ready-command', 'check-ready', '--', 'portal/start.sh',
        ]), mock.patch.object(n, 'lease_run', return_value=0) as run:
            self.assertEqual(n.main(), 0)
        args = run.call_args.args[0]
        self.assertEqual(args.justification, '')
        self.assertEqual(args.expected_duration, 0)
        self.assertEqual(args.gpu, ['GPU-fixture'])
        self.assertEqual(args.ready_command, 'check-ready')
        self.assertEqual(args.command, ['portal/start.sh'])

    def test_omitted_metadata_does_not_bypass_gpu_health(self):
        broker = n.Broker.__new__(n.Broker)
        with mock.patch.object(n, 'require_gpu_health', side_effect=RuntimeError('recovery required')):
            with self.assertRaisesRegex(RuntimeError, 'recovery required'):
                broker.acquire('portal', 1024, 120, ['GPU-fixture'])

    def test_owner_and_invalid_provided_metadata_remain_rejected(self):
        broker = n.Broker.__new__(n.Broker)
        for owner, purpose, duration, message in [
            ('', '', 0, 'owner'), ('unknown', '', 0, 'owner'),
            ('portal', 'x' * 513, 0, '512'), ('portal', '', -1, 'non-negative'),
        ]:
            with self.subTest(owner=owner, duration=duration):
                with self.assertRaisesRegex(ValueError, message):
                    broker.acquire(owner, 1024, 120, ['GPU-fixture'], purpose, duration)


if __name__ == '__main__':
    unittest.main()
