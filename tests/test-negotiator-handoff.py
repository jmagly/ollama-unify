#!/usr/bin/env python3
"""CPU fixtures only: public control/discovery cooperative ownership transfer."""
import concurrent.futures
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import socket
import sys
import unittest
from unittest import mock

HELPER = sys.argv.pop(1)
FIXTURE_BIN = sys.argv.pop(1)
os.environ['OLLAMA_UNIFY_HANDOFF_TIMEOUT'] = '5'
loader = importlib.machinery.SourceFileLoader(
    'handoff_pool', str(pathlib.Path(__file__).with_name('test-negotiator-pool.py')))
spec = importlib.util.spec_from_loader(loader.name, loader)
p = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = p
loader.exec_module(p)
GPUS = ['GPU-large-0', 'GPU-large-1', 'GPU-large-2']


def acquire(case, owner='requester', gpu=GPUS[0], **overrides):
    return p.control_raw(case.socket_path, {
        'action': 'acquire', 'owner': owner, 'requested_mib': 1024, 'ttl': 60,
        'gpu_uuids': [gpu], 'justification': 'CPU fixture tests cooperative lease ownership',
        'expected_duration_seconds': 60, **overrides})


def incumbent(case, opt_in=True, ready=True):
    response = acquire(case, owner='voice-service', yield_on_request=opt_in)
    assert response['ok'], response
    token = response['lease']['token']
    if ready:
        p.control(case.socket_path, {'action': 'ready', 'token': token})
    return token


class HandoffTests(unittest.TestCase):
    def test_request_waits_for_verified_release_not_revocation(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            token = incumbent(case)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                waiting = executor.submit(acquire, case, justification='', expected_duration_seconds=0)
                p.wait_until(lambda: case.status()['lease_handoff_requests'], 'intent is published')
                status = case.status()
                self.assertEqual(status['leases'][0]['state'], 'active')
                self.assertFalse(waiting.done())
                heartbeat = p.control(case.socket_path, {'action': 'heartbeat', 'token': token})
                self.assertEqual(heartbeat['lease']['state'], 'active')
                public = p.http_json(case.proxy_port, 'GET',
                    '/.well-known/ollama-unify-gpu-negotiator')[1]
                self.assertNotIn(token, json.dumps(public))
                self.assertEqual(public['lease_handoff_requests'][0]['owner'], 'requester')
                self.assertTrue(public['active_leases'][0]['yield_on_request'])
                p.control(case.socket_path, {'action': 'release', 'token': token})
                successor = waiting.result(timeout=5)
            self.assertTrue(successor['ok'], successor)
            self.assertEqual(successor['lease']['gpu_uuids'], [GPUS[0]])
            self.assertEqual(successor['lease']['state'], 'pending')
            self.assertFalse(successor['lease']['yield_on_request'])
            self.assertEqual(len(case.status()['leases']), 1)
            self.assertFalse(case.status()['lease_handoff_requests'])

    def test_ordinary_or_pending_owners_are_not_preempted(self):
        for opt_in, ready in [(False, True), (True, False)]:
            with self.subTest(opt_in=opt_in, ready=ready), p.PoolHarness(HELPER, FIXTURE_BIN) as case:
                token = incumbent(case, opt_in=opt_in, ready=ready)
                self.assertFalse(acquire(case)['ok'])
                self.assertFalse(case.status()['lease_handoff_requests'])
                self.assertEqual(case.status()['leases'][0]['token'], token)

    def test_disconnected_request_cancels_intent_without_revoking_owner(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            token = incumbent(case)
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.connect(case.socket_path)
            connection.sendall(json.dumps({
                'action': 'acquire', 'owner': 'disconnected-requester',
                'gpu_uuids': [GPUS[0]], 'justification': 'CPU cancellation fixture',
                'expected_duration_seconds': 60,
            }).encode() + b'\n')
            p.wait_until(lambda: case.status()['lease_handoff_requests'], 'intent starts')
            connection.close()
            p.wait_until(lambda: not case.status()['lease_handoff_requests'], 'intent cancels')
            self.assertEqual(case.status()['leases'][0]['token'], token)
            self.assertEqual(case.status()['leases'][0]['state'], 'active')

    def test_timeout_retains_owner_and_clears_advisory_reservation(self):
        with mock.patch.dict('os.environ', {'OLLAMA_UNIFY_HANDOFF_TIMEOUT': '1'}), \
                p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            token = incumbent(case)
            response = acquire(case)
            self.assertFalse(response['ok'])
            self.assertIn('timed out', response['error'])
            self.assertFalse(case.status()['lease_handoff_requests'])
            self.assertEqual(case.status()['leases'][0]['token'], token)
            self.assertEqual(case.status()['leases'][0]['state'], 'active')

    def test_metadata_capacity_self_and_multigpu_opt_in_fail_before_yield(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            incumbent(case)
            for changes in [
                {'owner': 'voice-service'}, {'owner': ''}, {'justification': 'x' * 513},
                {'expected_duration_seconds': -1}, {'requested_mib': 10**12},
                {'gpu_uuids': [GPUS[0], GPUS[1]], 'yield_on_request': True},
                {'gpu_uuids': [GPUS[0], 'GPU-unselected']}, {'yield_on_request': 'true'},
            ]:
                with self.subTest(changes=changes):
                    self.assertFalse(acquire(case, **changes)['ok'])
                    self.assertFalse(case.status()['lease_handoff_requests'])
                    self.assertEqual(case.status()['leases'][0]['state'], 'active')

    def test_second_request_cannot_steal_reserved_scope(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            token = incumbent(case)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                waiting = executor.submit(acquire, case)
                p.wait_until(lambda: case.status()['lease_handoff_requests'], 'first request waits')
                self.assertFalse(acquire(case, owner='late-requester')['ok'])
                self.assertEqual(len(case.status()['lease_handoff_requests']), 1)
                p.control(case.socket_path, {'action': 'release', 'token': token})
                self.assertTrue(waiting.result(timeout=5)['ok'])

    def test_owner_opt_in_is_persisted_but_request_intent_is_ephemeral(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            incumbent(case)
            persisted = json.loads(pathlib.Path(case.temp_dir, 'leases.json').read_text())
            self.assertTrue(persisted['leases'][0]['yield_on_request'])
            self.assertNotIn('lease_handoff_requests', persisted)


if __name__ == '__main__':
    unittest.main()
