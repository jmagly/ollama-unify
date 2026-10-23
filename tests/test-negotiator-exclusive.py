#!/usr/bin/env python3
"""Explicit singleton lease exclusion and lifecycle CPU fixtures."""
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
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


with mock.patch.dict(os.environ, {'OLLAMA_UNIFY_CONFIG': '/nonexistent/exclusive-test',
                                  'OLLAMA_UNIFY_LEASE_STATE': '/nonexistent/exclusive-leases',
                                  'OLLAMA_UNIFY_MODEL_POLICY_STATE': '/nonexistent/exclusive-policy'}):
    n = load('exclusive_broker', HELPER)
p = load('exclusive_pool', Path(__file__).with_name('test-negotiator-pool.py'))
GPU = 'GPU-large-0'
OTHER = 'GPU-large-1'


def acquire(case, **extra):
    return p.control_raw(case.socket_path, {'action': 'acquire', 'owner': 'private-fixture',
        'ttl': 60, 'gpu_uuids': [GPU], 'exclusive': True, **extra})


class ExclusiveLeaseTests(unittest.TestCase):
    def test_singleton_drains_and_stays_excluded_after_ready(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN, max_servers=3) as case:
            code, capacity, _ = case.capacity(p.MODEL, gpu_uuids=[GPU])
            self.assertEqual(code, 200, capacity)
            original = capacity['lanes'][0]['id']
            acquired = acquire(case)
            self.assertTrue(acquired['ok'], acquired)
            self.assertIn(original, acquired['stopped_lanes'])
            self.assertTrue(acquired['lease']['exclusive'])
            token = acquired['lease']['token']
            p.control(case.socket_path, {'action': 'ready', 'token': token})
            self.assertTrue(case.status()['lease_summaries'][0]['exclusive'])
            blocked = case.capacity(p.MODEL, gpu_uuids=[GPU])
            self.assertEqual(blocked[0], 503, blocked)
            allowed = case.capacity(p.MODEL, gpu_uuids=[OTHER])
            self.assertEqual(allowed[0], 200, allowed)
            self.assertTrue(all(GPU not in lane.get('gpu_uuids', [lane.get('gpu_uuid')])
                                for lane in p.managed_lanes(case.status())))
            changed = p.control_raw(case.socket_path, {'action': 'scope', 'token': token,
                                                       'gpu_uuids': [OTHER]})
            self.assertFalse(changed['ok'])
            p.control(case.socket_path, {'action': 'release', 'token': token})
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPU])[0], 200)

    def test_plain_singleton_retains_free_vram_sharing(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            acquired = acquire(case, exclusive=False)
            self.assertTrue(acquired['ok'], acquired)
            token = acquired['lease']['token']
            p.control(case.socket_path, {'action': 'ready', 'token': token})
            self.assertFalse(case.status()['lease_summaries'][0]['exclusive'])
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPU])[0], 200)
            p.control(case.socket_path, {'action': 'release', 'token': token})

    def test_control_strict_boolean_and_no_implicit_migration(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            for invalid in ['true', 'false', 1, 0, None, [], {}]:
                result = acquire(case, exclusive=invalid)
                self.assertFalse(result['ok'], result)
            for extra in [{'placement': 'flexible'}, {'yield_on_request': True},
                          {'evacuation_id': 'must-not-evacuate'}, {'gpu_uuids': []}]:
                self.assertFalse(acquire(case, **extra)['ok'])
            self.assertEqual(case.status()['leases'], [])

    def test_release_and_force_refuse_residual_context_even_zero_mib(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            token = acquire(case)['lease']['token']
            p.control(case.socket_path, {'action': 'ready', 'token': token})
            for pid, used in [(999999, 1), (999999, 0), (case.daemon.pid, 0)]:
                p.write_compute_apps(case.compute_apps, [(pid, GPU, used)])
                for force in [False, True]:
                    result = p.control_raw(case.socket_path,
                        {'action': 'release', 'token': token, 'force': force})
                    self.assertFalse(result['ok'], result)
                    self.assertEqual(case.status()['leases'][0]['token'], token)
            p.write_compute_apps(case.compute_apps, [])
            self.assertTrue(p.control(case.socket_path, {'action': 'release', 'token': token})['ok'])

    def test_cli_exclusive_acquire_and_run(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            env = {**os.environ, 'OLLAMA_UNIFY_CONFIG': '/nonexistent/exclusive-cli',
                   'OLLAMA_UNIFY_SOCKET': case.socket_path}
            result = subprocess.run([HELPER, 'acquire', '--owner', 'exclusive-cli',
                '--gpu', GPU, '--exclusive'], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            lease = json.loads(result.stdout)['lease']
            self.assertTrue(lease['exclusive'])
            p.control(case.socket_path, {'action': 'release', 'token': lease['token']})
            result = subprocess.run([HELPER, 'run', '--owner', 'exclusive-run', '--gpu', GPU,
                '--exclusive', '--ready-command', 'true', '--', sys.executable, '-c',
                'import time; time.sleep(0.3)'], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(case.status()['leases'], [])

    def test_acquire_refuses_existing_foreign_cuda(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            for pid, used in [(999999, 1), (case.daemon.pid, 0)]:
                p.write_compute_apps(case.compute_apps, [(pid, GPU, used)])
                self.assertFalse(acquire(case)['ok'])
                self.assertEqual(case.status()['leases'], [])

    def test_persisted_flag_and_lease_identity_survive_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.multiple(n, LEASE_STATE_PATH=root/'leases.json',
                    MODEL_POLICY_PATH=root/'policy.json', CACHE_STATE_PATH=root/'cache.json',
                    SELECTED_GPUS=[GPU, OTHER], BACKEND_TYPE='cuda'):
                broker = n.Broker()
                now = time.time()
                lease = n.Lease('opaque-test-token', 'private-fixture', 'active', 0,
                    now, now, now, 300, {'123@'+OTHER: 4096}, [GPU], exclusive=True)
                broker.leases[lease.token] = lease
                broker._persist_leases_locked()
                restored = n.Broker()
                actual = restored.leases[lease.token]
                self.assertTrue(actual.exclusive)
                self.assertEqual(n.asdict(actual), n.asdict(lease))
                self.assertEqual(restored._lease_blocked_gpus_locked(), {GPU})
                raw = json.loads((root/'leases.json').read_text())
                raw['leases'][0]['exclusive'] = 'false'
                (root/'leases.json').write_text(json.dumps(raw))
                with self.assertRaises(RuntimeError):
                    n.Broker()

    def test_raw_context_query_keeps_managed_zero_mib_and_rejects_bad_telemetry(self):
        process = mock.Mock(stdout=f'123, {GPU}, 0\n')
        with mock.patch.object(n.subprocess, 'run', return_value=process), mock.patch.object(
                n, 'managed_cgroup_pids', return_value={123}), mock.patch.object(n, 'SELECTED_GPUS', [GPU]):
            self.assertEqual(n.foreign_gpu_usage(strict=True), {})
            self.assertEqual(n.exclusive_cuda_contexts(), {f'123@{GPU}': 0})
            with self.assertRaises(RuntimeError):
                n.require_exclusive_cuda_free([GPU])
            for bad in ['not-a-row', f'123, {GPU}, N/A', f'bad, {GPU}, 0', f'², {GPU}, 0', '123, , 0']:
                process.stdout = bad
                with self.assertRaises(RuntimeError):
                    n.require_exclusive_cuda_free([GPU])
        with mock.patch.object(n.subprocess, 'run', side_effect=OSError('unavailable')):
            with self.assertRaises(RuntimeError):
                n.require_exclusive_cuda_free([GPU])

    def test_expired_exclusive_lease_retained_until_strict_cuda_free(self):
        broker = n.Broker()
        now = time.time()
        lease = n.Lease('opaque-test-token', 'private-fixture', 'revoking', 0,
            now-100, now-100, now-100, 1, {}, [GPU], exclusive=True)
        broker.leases[lease.token] = lease
        with mock.patch.object(n, 'REVOKE_TIMEOUT', 1), mock.patch.object(
                broker, '_persist_leases_locked'):
            for usage in [{f'123@{GPU}': 1}, {f'123@{GPU}': 0}]:
                with mock.patch.object(n, 'exclusive_cuda_contexts', return_value=usage):
                    self.assertFalse(broker._abandon_if_revoke_deadline_expired_locked(lease))
                    self.assertIn(lease.token, broker.leases)
            with mock.patch.object(n, 'exclusive_cuda_contexts', side_effect=RuntimeError('unavailable')):
                self.assertFalse(broker._abandon_if_revoke_deadline_expired_locked(lease))
            with broker.cv, mock.patch.object(n, 'exclusive_cuda_contexts', return_value={}):
                self.assertTrue(broker._abandon_if_revoke_deadline_expired_locked(lease))
            self.assertNotIn(lease.token, broker.leases)


if __name__ == '__main__':
    unittest.main()
