#!/usr/bin/env python3
"""Regressions for free VRAM on a node whose driver requires recovery."""
import importlib.machinery
import importlib.util
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import unittest
from unittest import mock

os.environ['OLLAMA_UNIFY_CONFIG'] = '/nonexistent/ollama-unify-negotiator'
os.environ['OLLAMA_UNIFY_LEASE_STATE'] = '/nonexistent/leases.json'
os.environ['OLLAMA_UNIFY_MODEL_POLICY_STATE'] = '/nonexistent/policy.json'
loader = importlib.machinery.SourceFileLoader('health_negotiator', sys.argv.pop(1))
spec = importlib.util.spec_from_loader(loader.name, loader)
n = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = n
loader.exec_module(n)


class GpuHealthTests(unittest.TestCase):
    def setUp(self):
        self.backend = mock.patch.object(n, 'BACKEND_TYPE', 'cuda')
        self.selected = mock.patch.object(n, 'SELECTED_GPUS', ['GPU-selected'])
        self.backend.start()
        self.selected.start()
        self.cache = mock.patch.object(n, '_GPU_HEALTH_CACHE', None)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.addCleanup(self.backend.stop)
        self.addCleanup(self.selected.stop)

    @contextlib.contextmanager
    def query(self, stdout, returncode=0, stderr=''):
        with mock.patch.object(n, '_GPU_HEALTH_CACHE', None), mock.patch.object(
            n.subprocess, 'run', return_value=subprocess.CompletedProcess([], returncode, stdout, stderr)
        ) as run:
            yield run

    def test_polling_is_coalesced_but_growth_forces_fresh_telemetry(self):
        with self.query('GPU-selected, None\n') as run, mock.patch.object(n.time, 'monotonic', return_value=100) as clock:
            n.require_gpu_health()
            n.require_gpu_health()
            self.assertEqual(run.call_count, 1)
            n.require_gpu_health(refresh=True)
            self.assertEqual(run.call_count, 2)
            clock.return_value = 100.6
            run.return_value.stdout = 'GPU-selected, Reboot\n'
            with self.assertRaises(n.PermanentCapacityError):
                n.require_gpu_health()
            self.assertEqual(run.call_count, 3)

    def test_none_and_unsupported_device_are_admissible(self):
        for action in ('None', 'N/A', '[N/A]'):
            with self.subTest(action=action), self.query('GPU-selected, ' + action + '\n'):
                self.assertFalse(n.gpu_health_snapshot()['admission_blocked'])
                n.require_gpu_health()

    def test_unknown_cuda_and_failed_process_telemetry_block_transitions(self):
        broker = n.Broker()
        for usage in ({'123@GPU-selected': 0}, RuntimeError('telemetry unavailable')):
            with mock.patch.object(n, 'foreign_gpu_usage',
                                   side_effect=usage if isinstance(usage, Exception) else None,
                                   return_value=usage if isinstance(usage, dict) else None):
                self.assertEqual(broker._ollama_blocked_gpus_locked(), {'GPU-selected'})
                with self.assertRaises(n.CapacityError) as raised:
                    broker._require_safe_gpu_transition('GPU-selected')
                self.assertEqual(raised.exception.reason_code, 'gpu_unregistered_workload')

    def test_quarantine_defers_stop_and_cancel_without_losing_reservation(self):
        broker = n.Broker()
        lane = n.Lane('test', 'managed', '127.0.0.1', 1, 'GPU-selected',
                      'fixture:latest', 1, 4096, 0, 0, mock.Mock())
        with mock.patch.object(n, 'foreign_gpu_usage', return_value={'123@GPU-selected': 4096}), mock.patch.object(
            n, 'unload_models_at'
        ) as unload, mock.patch.object(broker, '_terminate_process') as terminate, mock.patch.object(
            n, 'process_group_alive', return_value=True
        ):
            self.assertEqual(broker._stop_lanes([lane], 'fixture'), [lane])
            self.assertIs(broker.lanes['test'], lane)
            self.assertTrue(lane.retiring)
            broker._stop_expired_lane(lane)
            unload.assert_not_called()
            terminate.assert_not_called()

    def test_transition_refreshes_cached_empty_process_inventory(self):
        broker = n.Broker()
        with mock.patch.object(n, 'foreign_gpu_usage', return_value={}) as usage:
            self.assertFalse(broker._unregistered_gpus_locked())
            broker._unregistered_gpus_locked()
            self.assertEqual(usage.call_count, 1)
            usage.return_value = {'123@GPU-selected': 4096}
            with self.assertRaises(n.CapacityError):
                broker._require_safe_gpu_transition('GPU-selected')
            self.assertEqual(usage.call_count, 2)

    def test_system_backend_cannot_bypass_quarantine_but_metadata_stays_available(self):
        broker = n.Broker()
        with mock.patch.object(n, 'POOL_ENABLED', False), mock.patch.object(
            n, 'foreign_gpu_usage', return_value={'123@GPU-selected': 4096}
        ):
            self.assertIsNone(broker._select_lane_locked('fixture:latest', True))
            self.assertIs(broker._select_lane_locked('', False), broker.lanes['base'])

    def test_explicit_native_unload_refreshes_telemetry_then_retires_backend(self):
        broker = n.Broker()
        lane = n.Lane('test', 'managed', '127.0.0.1', 1, 'GPU-selected',
                      'fixture:latest', 1, 4096, 0, 0, mock.Mock())
        body = b'{"model":"fixture:latest","keep_alive":0}'
        with mock.patch.object(n, 'foreign_gpu_usage', return_value={}) as usage:
            broker._unregistered_gpus_locked()
            usage.return_value = {'123@GPU-selected': 4096}
            with self.assertRaises(n.CapacityError):
                broker.prepare_managed_body(lane, '/api/generate', body)
            self.assertFalse(lane.retiring)
            usage.return_value = {}
            prepared = broker.prepare_managed_body(lane, '/api/generate', body)
            self.assertEqual(json.loads(prepared)['keep_alive'], 0)
            self.assertTrue(lane.retiring)

    def test_unknown_memory_size_still_identifies_foreign_context(self):
        with mock.patch.object(n.subprocess, 'run', return_value=subprocess.CompletedProcess(
            [], 0, '123, GPU-selected, [N/A]\n', ''
        )), mock.patch.object(n, 'managed_cgroup_pids', return_value=set()):
            self.assertEqual(n.foreign_gpu_usage(strict=True), {'123@GPU-selected': 0})
        with mock.patch.object(n.subprocess, 'run', side_effect=subprocess.TimeoutExpired('nvidia-smi', 5)):
            with self.assertRaises(RuntimeError):
                n.foreign_gpu_usage(strict=True)

    def test_scoped_prepare_does_not_retire_unrelated_lanes(self):
        broker = n.Broker()
        now = time.time()
        broker.leases['scoped'] = n.Lease('scoped', 'owner', 'active', 1024,
                                         now, now, now, 60, {}, ['GPU-selected'])
        with mock.patch.object(n, 'require_gpu_health'), mock.patch.object(broker, 'begin_drain'), mock.patch.object(
            broker, '_persist_leases_locked'
        ), mock.patch.object(broker, 'stop_pool_lanes', return_value=[]) as stop, mock.patch.object(
            broker, '_unload_base_models', return_value=[]
        ), mock.patch.object(n, 'gpu_snapshot', return_value=[]):
            result = broker.prepare('scoped')
        self.assertEqual(result['lease']['state'], 'pending')
        stop.assert_called_once_with('lease prepare', {'GPU-selected'})

    def test_recovery_actions_block_and_preserve_request_identity(self):
        for action in ('Reset', 'Reboot', 'Drain P2P', 'Drain and Reset', '[Unknown Error]'):
            with self.subTest(action=action), self.query('GPU-selected, ' + action + '\n'):
                with self.assertRaises(n.PermanentCapacityError) as raised:
                    n.require_gpu_health(request_id='request-1', logical_request_id='turn-1')
                exc = raised.exception
                self.assertEqual((exc.status, exc.reason_code, exc.retryable),
                                 (503, 'gpu_recovery_required', False))
                self.assertEqual((exc.request_id, exc.logical_request_id), ('request-1', 'turn-1'))

    def test_reboot_on_unselected_gpu_blocks_selected_gpu(self):
        with self.query('GPU-selected, None\nGPU-display, Reboot\n'):
            self.assertEqual(n.gpu_health_snapshot()['recovery_actions'], {'GPU-display': 'Reboot'})
            with self.assertRaises(n.PermanentCapacityError):
                n.require_gpu_health()
        with self.query('GPU-selected, None\nGPU-display, Reset\n'):
            n.require_gpu_health()

    def test_failed_or_partial_telemetry_is_not_free_capacity(self):
        for stdout, code in [('', 0), ('GPU-other, None\n', 0), ('GPU-selected, None\n', 255)]:
            with self.subTest(stdout=stdout, code=code), self.query(stdout, code):
                with self.assertRaises(n.CapacityError) as raised:
                    n.require_gpu_health()
                self.assertEqual(raised.exception.reason_code, 'gpu_health_unavailable')
        with mock.patch.object(n.subprocess, 'run', side_effect=subprocess.TimeoutExpired('nvidia-smi', 5)):
            self.assertTrue(n.gpu_health_snapshot()['admission_blocked'])

    def test_partial_inventory_still_reports_confirmed_reboot(self):
        with self.query('Unable to determine handle for GPU2\nGPU-selected, Reboot\n', 255):
            with self.assertRaises(n.PermanentCapacityError) as raised:
                n.require_gpu_health()
            self.assertEqual(raised.exception.reason_code, 'gpu_recovery_required')

    def test_older_driver_is_explicitly_unsupported(self):
        with self.query('Field "gpu_recovery_action" is not a valid field to query.\n', 2):
            health = n.gpu_health_snapshot()
            self.assertFalse(health['supported'])
            self.assertFalse(health['admission_blocked'])
            self.assertTrue(n.gpu_health_warnings(health))
        # A generic query failure must not activate the compatibility fallback.
        with self.query('', 2, 'Failed to initialize NVML'):
            self.assertTrue(n.gpu_health_snapshot()['admission_blocked'])

    def test_other_backends_do_not_call_nvidia(self):
        with mock.patch.object(n, 'BACKEND_TYPE', 'rocm'), mock.patch.object(n.subprocess, 'run') as run:
            n.require_gpu_health()
            run.assert_not_called()

    def test_fault_blocks_lease_before_unload_or_reservation(self):
        broker = n.Broker()
        with self.query('GPU-selected, Reboot\n'), mock.patch.object(n, 'unload_all_models') as unload:
            with self.assertRaises(n.PermanentCapacityError):
                broker.acquire('health-regression', 1024, 60, ['GPU-selected'],
                               'Verify recovery refuses new allocations', 60)
            unload.assert_not_called()
        self.assertFalse(broker.leases)
        self.assertFalse(broker.draining)

    def test_fault_blocks_ready_growth_and_lane_start(self):
        broker = n.Broker()
        with self.query('GPU-selected, Reboot\n'), mock.patch.object(n.subprocess, 'Popen') as start:
            for operation in (
                lambda: broker.ready('pending-lease'),
                lambda: broker.prepare('active-lease'),
                lambda: broker.ensure_capacity('fixture:latest', 1),
                lambda: broker._spawn_lane('fixture:latest', 'GPU-selected', 1024, set(), ''),
            ):
                with self.assertRaises(n.PermanentCapacityError):
                    operation()
            start.assert_not_called()

    def test_fault_blocks_inference_but_allows_unload(self):
        broker = n.Broker()
        with mock.patch.object(n, 'POOL_ENABLED', False), self.query('GPU-selected, Reboot\n'), mock.patch.object(
            n, 'foreign_gpu_usage', return_value={}
        ):
            with self.assertRaises(n.PermanentCapacityError):
                broker.proxy_enter('fixture:latest', True)
            self.assertEqual(broker.active_requests, 0)
            admission = broker.proxy_enter('fixture:latest', True, allow_during_drain=True)
            broker.proxy_exit(admission, 'fixture:latest', True)
            self.assertEqual(broker.active_requests, 0)

    def test_fault_while_queued_releases_waiter(self):
        broker = n.Broker()
        healthy = {'admission_blocked': False}
        faulty = {'admission_blocked': True, 'recovery_actions': {'GPU-selected': 'Reboot'}}
        with mock.patch.object(n, 'POOL_ENABLED', True), mock.patch.object(n, 'gpu_health_snapshot', side_effect=[healthy, faulty]):
            with self.assertRaises(n.PermanentCapacityError):
                broker.proxy_enter('fixture:latest', True, request_id='queued-1', logical_request_id='turn-1')
        self.assertFalse(broker.waiters)
        self.assertFalse(broker.logical_in_flight)
        self.assertEqual(broker.retained_request_bytes, 0)

    def test_fault_does_not_block_owner_heartbeat_or_release(self):
        broker = n.Broker()
        now = time.time()
        broker.leases['existing'] = n.Lease('existing', 'owner', 'active', 1024,
                                          now, now, now, 60, {}, ['GPU-selected'])
        with mock.patch.object(n, 'gpu_health_snapshot', side_effect=AssertionError('cleanup must stay available')), mock.patch.object(
            broker, '_persist_leases_locked'
        ), mock.patch.object(n, 'wait_for_foreign_settle'), mock.patch.object(n, 'gpu_snapshot', return_value=[]), mock.patch.object(n, 'host_memory_snapshot', return_value={}):
            self.assertTrue(broker.heartbeat('existing')['ok'])
            self.assertTrue(broker.release('existing')['ok'])
        self.assertFalse(broker.leases)

    def test_cli_discovery_preserves_health_warnings_with_and_without_broker(self):
        health = {'supported': True, 'admission_blocked': True,
                  'recovery_actions': {'GPU-selected': 'Reboot'},
                  'missing_selected_gpu_ids': [], 'error': None}
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable), mock.patch.object(sys, 'argv', ['broker', 'discover']), mock.patch.object(
                n, 'discovery_document', return_value={'gpu_health': health, 'warnings': n.gpu_health_warnings(health)}
            ), mock.patch.object(n, 'send_control', return_value={'leases': []}, side_effect=OSError('offline') if unavailable else None), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(n.main(), 0)
                document = json.loads(output.getvalue())
                self.assertEqual(len(document['warnings']), 2 if unavailable else 1)
                self.assertIn('New GPU work is blocked', document['warnings'][-1])
                if unavailable:
                    self.assertIn('Live lease state is unavailable', document['warnings'][0])

    def test_cli_discovery_uses_daemon_quarantine_instead_of_file_defaults(self):
        healthy = {'supported': True, 'admission_blocked': False,
                   'recovery_actions': {}, 'missing_selected_gpu_ids': [], 'error': None}
        document = {'selected_gpu_ids': ['GPU-selected', 'GPU-quarantined'],
                    'selected_gpu_count': 2, 'gpu_health': {'admission_blocked': True},
                    'parallel_pool': {'enabled': True}, 'gpus': [],
                    'warnings': ['stale fault warning']}
        live = {'leases': [], 'selected_gpu_ids': ['GPU-selected'], 'selected_gpu_count': 1,
                'gpu_health': healthy, 'gpus': [{'uuid': 'GPU-selected'}, {'uuid': 'GPU-display'}],
                'parallel_pool': {'enabled': False}}
        with mock.patch.object(sys, 'argv', ['broker', 'discover']), mock.patch.object(
            n, 'discovery_document', return_value=document
        ), mock.patch.object(n, 'send_control', return_value=live), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(n.main(), 0)
        actual = json.loads(output.getvalue())
        self.assertEqual(actual['selected_gpu_ids'], ['GPU-selected'])
        self.assertEqual(actual['selected_gpu_count'], 1)
        self.assertFalse(actual['parallel_pool']['enabled'])
        self.assertFalse(actual['gpu_health']['admission_blocked'])
        self.assertEqual(actual['warnings'], [])
        self.assertEqual([gpu['selected_for_ollama'] for gpu in actual['gpus']], [True, False])

    def test_empty_leases_are_quiet_but_every_reserved_state_remains_visible(self):
        self.assertEqual(n.lease_visibility_warnings([]), [])
        self.assertEqual(n.lease_policy_document()['required_acquire_fields'], [
            'owner',
        ])
        for state in ('pending', 'active', 'revoking'):
            with self.subTest(state=state):
                warnings = n.lease_visibility_warnings([{
                    'owner': 'fixture-owner', 'state': state,
                    'expected_release_utc': '2026-10-07T18:00:00Z',
                }])
                self.assertEqual(len(warnings), 1)
                self.assertIn('fixture-owner until 2026-10-07T18:00:00Z', warnings[0])


if __name__ == '__main__':
    unittest.main()
