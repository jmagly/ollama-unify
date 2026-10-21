#!/usr/bin/env python3
"""CPU fixtures only: public control/discovery cooperative ownership transfer."""
import concurrent.futures
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
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
    def test_foreground_supervisor_restarts_only_after_source_exit(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            root = pathlib.Path(case.temp_dir)
            events = root / 'owner-events.jsonl'
            worker = root / 'restartable.py'
            worker.write_text('''import json, os, pathlib, signal, time
root = pathlib.Path(os.environ['OWNER_FIXTURE_ROOT'])
gpu = os.environ['CUDA_VISIBLE_DEVICES']
def event(kind):
    with (root / 'owner-events.jsonl').open('a') as f:
        f.write(json.dumps({'kind': kind, 'gpu': gpu}) + '\\n')
def stop(*_):
    if (root / 'hold-exit').exists():
        event('stopping')
        while (root / 'hold-exit').exists(): time.sleep(.01)
    event('exited')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
event('started')
(root / gpu).touch()
while True: time.sleep(.1)
''')
            env = {**os.environ, 'OWNER_FIXTURE_ROOT': str(root),
                   'PATH': FIXTURE_BIN + os.pathsep + os.environ.get('PATH', ''),
                   'OLLAMA_UNIFY_SOCKET': case.socket_path,
                   'OLLAMA_UNIFY_CONFIG': str(root / 'missing.conf'),
                   'OLLAMA_UNIFY_BACKEND_TYPE': 'cuda',
                   'OLLAMA_UNIFY_SELECTED_GPUS': ','.join(GPUS)}
            command = [HELPER, 'run', '--owner', 'portable-service', '--placement', 'flexible',
                       '--gpu', GPUS[2], '--gpu', GPUS[1], '--priority', '0',
                       '--vram-mib', '1024', '--ttl', '6', '--ready-timeout', '5',
                       '--stop-timeout', '2', '--drain-command',
                       'touch "$OWNER_FIXTURE_ROOT/paused"; test -f "$OWNER_FIXTURE_ROOT/drain-allowed"',
                       '--resume-command', 'touch "$OWNER_FIXTURE_ROOT/resumed"',
                       '--ready-command', 'test -f "$OWNER_FIXTURE_ROOT/$CUDA_VISIBLE_DEVICES"',
                       '--', sys.executable, str(worker)]
            log = (root / 'owner-supervisor.log').open('w')
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            try:
                p.wait_until(lambda: any(l['state'] == 'active' for l in case.status()['leases']),
                             'flexible foreground owner ready')
                cancelled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                cancelled.connect(case.socket_path)
                cancelled.sendall(json.dumps({'action': 'acquire', 'owner': 'cancelled-trainer',
                    'gpu_uuids': [GPUS[0], GPUS[2]], 'priority': 100}).encode() + b'\n')
                p.wait_until(lambda: (root / 'paused').exists(), 'busy drain hook called')
                cancelled.close()
                p.wait_until(lambda: (root / 'resumed').exists(), 'cancelled handoff reopens admission')
                self.assertEqual(len(events.read_text().splitlines()), 1)
                (root / 'drain-allowed').touch()
                (root / 'hold-exit').touch()
                cancelled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                cancelled.connect(case.socket_path)
                cancelled.sendall(json.dumps({'action': 'acquire', 'owner': 'cancel-during-stop',
                    'gpu_uuids': [GPUS[0], GPUS[2]], 'priority': 100}).encode() + b'\n')
                p.wait_until(lambda: 'stopping' in events.read_text(), 'owner began graceful stop')
                cancelled.close()
                p.wait_until(lambda: not case.status()['lease_handoff_requests'], 'stop-time cancellation')
                (root / 'hold-exit').unlink()
                p.wait_until(lambda: len(events.read_text().splitlines()) == 4
                             and case.status()['leases'][0]['state'] == 'active',
                             'original owner reloads its still-reserved scope')
                granted = acquire(case, owner='pinned-trainer', gpu_uuids=[GPUS[0], GPUS[2]], priority=100)
                self.assertTrue(granted['ok'], granted)
                observed = [json.loads(line) for line in events.read_text().splitlines()]
                self.assertEqual(observed, [
                    {'kind': 'started', 'gpu': GPUS[2]},
                    {'kind': 'stopping', 'gpu': GPUS[2]},
                    {'kind': 'exited', 'gpu': GPUS[2]},
                    {'kind': 'started', 'gpu': GPUS[2]},
                    {'kind': 'exited', 'gpu': GPUS[2]},
                    {'kind': 'started', 'gpu': GPUS[1]}])
                leases = case.status()['leases']
                flexible = next(l for l in leases if l['owner'] == 'portable-service')
                self.assertEqual(flexible['gpu_uuids'], [GPUS[1]])
                self.assertEqual(flexible['state'], 'active')
            finally:
                process.terminate()
                try:
                    process.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
            self.assertEqual(process.returncode, 0, (root / 'owner-supervisor.log').read_text())
            self.assertFalse(any(l['owner'] == 'portable-service' for l in case.status()['leases']))

    def flexible(self, case, priority=0, candidates=None):
        response = acquire(case, owner='portable-service', placement='flexible',
                           priority=priority, gpu_uuids=candidates or [GPUS[2], GPUS[1]])
        self.assertTrue(response['ok'], response)
        token = response['lease']['token']
        p.control(case.socket_path, {'action': 'ready', 'token': token})
        return token

    def test_nvlink_group_waits_for_lower_priority_owner_on_reserved_destination(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            token = self.flexible(case)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                waiting = executor.submit(acquire, case, owner='pinned-trainer',
                                          gpu_uuids=[GPUS[0], GPUS[2]], priority=100)
                p.wait_until(lambda: case.status()['lease_handoff_requests'], 'relocation intent')
                public = p.http_json(case.proxy_port, 'GET',
                                     '/.well-known/ollama-unify-gpu-negotiator')[1]
                self.assertNotIn(token, json.dumps(public))
                self.assertEqual(public['active_leases'][0]['placement'], 'flexible')
                heartbeat = p.control(case.socket_path, {'action': 'heartbeat', 'token': token})
                move = heartbeat['relocation']
                self.assertEqual(move['to_gpu_uuid'], GPUS[1])
                self.assertEqual(move['requested_priority'], 100)
                self.assertFalse(acquire(case, owner='target-thief', gpu=GPUS[1])['ok'])
                moved = p.control(case.socket_path, {'action': 'relocate', 'token': token,
                                                    'handoff_id': move['handoff_id']})
                self.assertEqual(moved['lease']['token'], token)
                self.assertEqual(moved['lease']['gpu_uuids'], [GPUS[1]])
                self.assertEqual(moved['lease']['state'], 'pending')
                self.assertFalse(waiting.done())
                p.control(case.socket_path, {'action': 'ready', 'token': token})
                granted = waiting.result(timeout=5)
            self.assertTrue(granted['ok'], granted)
            self.assertEqual(granted['lease']['gpu_uuids'], [GPUS[0], GPUS[2]])
            self.assertEqual(granted['lease']['priority'], 100)
            self.assertEqual(len(case.status()['leases']), 2)
            self.assertFalse(case.status()['lease_handoff_requests'])

    def test_equal_or_lower_priority_never_moves_flexible_owner(self):
        for priority in (-1, 0):
            with self.subTest(priority=priority), p.PoolHarness(HELPER, FIXTURE_BIN) as case:
                token = self.flexible(case)
                self.assertFalse(acquire(case, gpu=GPUS[2], priority=priority)['ok'])
                self.assertFalse(case.status()['lease_handoff_requests'])
                self.assertEqual(case.status()['leases'][0]['token'], token)
                self.assertEqual(case.status()['leases'][0]['gpu_uuids'], [GPUS[2]])

    def test_relocation_refuses_source_cuda_until_it_has_exited(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            token = self.flexible(case)
            p.write_compute_apps(case.compute_apps, [(910050, GPUS[2], 2048)])
            with concurrent.futures.ThreadPoolExecutor() as executor:
                waiting = executor.submit(acquire, case, gpu=GPUS[2], priority=100)
                p.wait_until(lambda: case.status()['lease_handoff_requests'], 'intent')
                move = p.control(case.socket_path, {'action': 'heartbeat', 'token': token})['relocation']
                refused = p.control_raw(case.socket_path, {'action': 'relocate', 'token': token,
                    'handoff_id': move['handoff_id']})
                self.assertFalse(refused['ok'], refused)
                self.assertIn('baseline', refused['error'])
                self.assertEqual(case.status()['leases'][0]['gpu_uuids'], [GPUS[2]])
                self.assertFalse(waiting.done())
                # A tiny context below normal release-accounting tolerance
                # still prevents ownership transfer into a peer-traffic group.
                p.write_compute_apps(case.compute_apps, [(910050, GPUS[2], 64)])
                tiny = p.control_raw(case.socket_path, {'action': 'relocate', 'token': token,
                    'handoff_id': move['handoff_id']})
                self.assertFalse(tiny['ok'], tiny)
                self.assertIn('source CUDA', tiny['error'])
                p.write_compute_apps(case.compute_apps, [])
                p.control(case.socket_path, {'action': 'relocate', 'token': token,
                                             'handoff_id': move['handoff_id']})
                p.control(case.socket_path, {'action': 'ready', 'token': token})
                self.assertTrue(waiting.result(timeout=5)['ok'])

    def test_cancelled_request_releases_only_its_advisory_destination(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            token = self.flexible(case)
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.connect(case.socket_path)
            connection.sendall(json.dumps({'action': 'acquire', 'owner': 'pinned-requester',
                'gpu_uuids': [GPUS[0], GPUS[2]], 'priority': 100}).encode() + b'\n')
            p.wait_until(lambda: case.status()['lease_handoff_requests'], 'intent')
            move = p.control(case.socket_path, {'action': 'heartbeat', 'token': token})['relocation']
            connection.close()
            p.wait_until(lambda: not case.status()['lease_handoff_requests'], 'cancellation')
            self.assertIsNone(p.control(case.socket_path, {'action': 'heartbeat', 'token': token})['relocation'])
            self.assertFalse(p.control_raw(case.socket_path, {'action': 'relocate', 'token': token,
                'handoff_id': move['handoff_id']})['ok'])
            self.assertEqual(case.status()['leases'][0]['gpu_uuids'], [GPUS[2]])
            self.assertTrue(acquire(case, owner='new-destination-owner', gpu=GPUS[1])['ok'])

    def test_relocation_rechecks_destination_for_late_unregistered_cuda(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            token = self.flexible(case)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                waiting = executor.submit(acquire, case, gpu=GPUS[2], priority=100)
                p.wait_until(lambda: case.status()['lease_handoff_requests'], 'intent')
                move = p.control(case.socket_path, {'action': 'heartbeat', 'token': token})['relocation']
                p.write_compute_apps(case.compute_apps, [(910051, GPUS[1], 2048)])
                refused = p.control_raw(case.socket_path, {'action': 'relocate', 'token': token,
                    'handoff_id': move['handoff_id']})
                self.assertFalse(refused['ok'], refused)
                self.assertIn('destination', refused['error'])
                self.assertEqual(case.status()['leases'][0]['gpu_uuids'], [GPUS[2]])
                self.assertFalse(waiting.done())
                p.write_compute_apps(case.compute_apps, [])
                p.control(case.socket_path, {'action': 'relocate', 'token': token,
                    'handoff_id': move['handoff_id']})
                p.control(case.socket_path, {'action': 'ready', 'token': token})
                self.assertTrue(waiting.result(timeout=5)['ok'])

    def test_no_destination_does_not_ask_owner_to_drain(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            token = self.flexible(case, candidates=[GPUS[2]])
            response = acquire(case, gpu=GPUS[2], priority=100)
            self.assertFalse(response['ok'])
            self.assertIn('no eligible destination', response['error'])
            self.assertFalse(case.status()['lease_handoff_requests'])
            self.assertEqual(case.status()['leases'][0]['token'], token)

    def test_fixed_owner_cannot_be_retargeted_or_bypassed_by_priority(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            token = incumbent(case, opt_in=False)
            self.assertFalse(acquire(case, priority=1000)['ok'])
            self.assertFalse(p.control_raw(case.socket_path, {
                'action': 'relocate', 'token': token, 'handoff_id': 'invented'})['ok'])
            self.assertEqual(case.status()['leases'][0]['gpu_uuids'], [GPUS[0]])

    def test_flexible_scope_cannot_widen_or_bypass_reserved_relocation(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            token = self.flexible(case)
            self.assertFalse(p.control_raw(case.socket_path, {'action': 'scope',
                'token': token, 'gpu_uuids': [GPUS[0]]})['ok'])
            self.assertEqual(case.status()['leases'][0]['gpu_uuids'], [GPUS[2]])
            persisted = json.loads(pathlib.Path(case.temp_dir, 'leases.json').read_text())
            self.assertEqual(persisted['leases'][0]['eligible_gpu_uuids'], [GPUS[2], GPUS[1]])
            self.assertEqual(persisted['leases'][0]['placement'], 'flexible')

    def test_invalid_flexibility_and_priority_have_no_side_effects(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            for changes in [{'placement': 'guess'}, {'priority': True}, {'priority': 1001},
                            {'placement': 'flexible', 'gpu_uuids': ['GPU-unselected']},
                            {'placement': 'flexible', 'yield_on_request': True}]:
                with self.subTest(changes=changes):
                    self.assertFalse(acquire(case, **changes)['ok'])
                    self.assertFalse(case.status()['leases'])
                    self.assertFalse(case.status()['lease_handoff_requests'])

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
