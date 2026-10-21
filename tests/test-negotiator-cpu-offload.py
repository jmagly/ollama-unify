#!/usr/bin/env python3
"""CPU-only offload contract tests; no live inference or GPU loads."""
import concurrent.futures
import email.message
import http.server
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

HELPER = sys.argv.pop(1)
FIXTURE_BIN = sys.argv.pop(1)
os.environ['OLLAMA_UNIFY_CONFIG'] = '/nonexistent/offload-tests'
os.environ['OLLAMA_UNIFY_LEASE_STATE'] = '/nonexistent/offload-leases.json'
os.environ['OLLAMA_UNIFY_MODEL_POLICY_STATE'] = '/nonexistent/offload-policy.json'
os.environ['OLLAMA_UNIFY_CACHE_RESIDENCY_STATE'] = '/nonexistent/offload-cache.json'


def module(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    loader.exec_module(result)
    return result


n = module('cpu_offload_negotiator', HELPER)
p = module('cpu_offload_pool', pathlib.Path(__file__).with_name('test-negotiator-pool.py'))
GPUS = ['GPU-large-0', 'GPU-large-1']
MODEL = 'fixture-offloaded:latest'
MIB = 1024 * 1024


class BudgetTests(unittest.TestCase):
    def group(self, root, relative, used=100, high='max', maximum='max'):
        group = root / relative
        group.mkdir(parents=True, exist_ok=True)
        for name, value in [('current', used * MIB), ('high', high), ('max', maximum)]:
            (group / ('memory.' + name)).write_text(str(value if value == 'max' or name == 'current' else value * MIB))
        return group

    def snapshot(self, root, relative):
        membership = root.parent / 'membership'
        membership.write_text('0::/' + relative + '\n')
        return n.managed_cgroup_memory_headroom(membership, root)['managed_cgroup_headroom_mib']

    def test_actual_parent_and_ancestor_limits_use_the_tightest_headroom(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / 'cgroup'
            root.mkdir()
            self.group(root, 'slice', used=200, high=900, maximum=1000)
            self.group(root, 'slice/broker', used=100, high=1000, maximum=2000)
            self.assertEqual(self.snapshot(root, 'slice/broker'), 700)

    def test_unlimited_and_fileless_root_remain_physically_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / 'cgroup'
            root.mkdir()
            self.group(root, 'broker')
            self.assertEqual(self.snapshot(root, 'broker'), 'max')
            self.assertEqual(self.snapshot(root, ''), 'max')

    def test_missing_nonroot_telemetry_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / 'cgroup'
            root.mkdir()
            self.group(root, 'slice/broker')
            self.assertEqual(self.snapshot(root, 'slice/broker'), 'unknown')

    def test_namespaced_container_root_limits_are_honored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / 'cgroup'
            self.group(root, '', used=200, high=700, maximum=1000)
            self.group(root, 'broker', used=100)
            self.assertEqual(self.snapshot(root, 'broker'), 500)
            self.assertEqual(self.snapshot(root, ''), 500)

    def test_over_limit_is_zero_headroom(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / 'cgroup'
            root.mkdir()
            self.group(root, 'broker', used=1000, high=900)
            self.assertEqual(self.snapshot(root, 'broker'), 0)

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / 'cgroup'
            root.mkdir()
            self.assertEqual(self.snapshot(root, '../other'), 'unknown')

    def test_duplicate_or_unrecognized_offload_header_fails_closed(self):
        handler = object.__new__(n.ProxyHandler)
        for values in [['allow', 'deny'], ['true'], ['']]:
            handler.headers = email.message.Message()
            for value in values:
                handler.headers[n.CPU_OFFLOAD_HEADER] = value
            with self.subTest(values=values), self.assertRaises(n.PermanentCapacityError):
                handler._requested_cpu_offload()


class CpuFixture(unittest.TestCase):
    def setUp(self):
        patches = {
            'SELECTED_GPUS': GPUS, 'MODEL_GPU_PREFERENCES': {}, 'POOL_ENABLED': True,
            'AUTO_MODEL_CONTEXT': False, 'POOL_MAX_SERVERS': 4, 'POOL_INSTANCE_PARALLEL': 1,
            'POOL_HOST_RESERVE_MIB': 2048, 'POOL_VRAM_RESERVE_MIB': 1024,
            'require_inference_enabled': mock.Mock(), 'require_gpu_health': mock.Mock(),
            'effective_model_context_profile': mock.Mock(return_value=None),
            'foreign_gpu_usage': mock.Mock(return_value={}),
            'process_group_alive': mock.Mock(return_value=True),
            'gpu_snapshot': mock.Mock(return_value=[
                {'uuid': gpu, 'total_mib': 81920, 'free_mib': 81920} for gpu in GPUS]),
            'host_memory_snapshot': mock.Mock(return_value={
                'memavailable_mib': 1000000, 'managed_cgroup_headroom_mib': 'max'}),
        }
        for name, value in patches.items():
            patch = mock.patch.object(n, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.broker = n.Broker()
        self.broker._model_profile = mock.Mock(return_value=(93578, {'completion'}))
        self.spawn = mock.Mock(side_effect=self.fake_spawn)
        self.broker._spawn_lane = self.spawn
        self.broker._stop_lanes = mock.Mock(side_effect=self.stop_lanes)

    def fake_spawn(self, model, scope, required, capabilities, path, triggered_by=None,
                   reserved_mib_by_gpu=None, expected_profile=None, **kwargs):
        scope = (scope,) if isinstance(scope, str) else tuple(scope)
        lane_id = 'fixture-' + str(self.broker.next_lane_id)
        self.broker.next_lane_id += 1
        process = mock.Mock()
        process.poll.return_value = None
        lane = n.Lane(lane_id, 'managed', '127.0.0.1', getattr(self, 'backend_port', 1),
                      scope[0], model, 1, required, time.time(), time.time(), process,
                      gpu_uuids=scope, reserved_mib_by_gpu=reserved_mib_by_gpu or {scope[0]: required},
                      cpu_offload=kwargs.get('cpu_offload', False),
                      host_reserved_mib=kwargs.get('host_reserved_mib', 0))
        self.broker.lanes[lane_id] = lane
        return lane

    def stop_lanes(self, lanes, reason):
        for lane in lanes:
            self.assertEqual(lane.in_flight, 0)
            self.broker.lanes.pop(lane.lane_id, None)
        return []

    def capacity(self, scope=(GPUS[0],), **kwargs):
        return self.broker.ensure_capacity(MODEL, 1, '/api/chat', tuple(scope), **kwargs)


class CpuAdmissionTests(CpuFixture):
    def test_default_refuses_oversize_without_spawning(self):
        with self.assertRaises(n.PermanentCapacityError) as raised:
            self.capacity()
        self.assertEqual(raised.exception.reason_code, 'model_exceeds_gpu_capacity')
        self.spawn.assert_not_called()

    def test_opt_in_reserves_exact_singleton_and_whole_host_estimate(self):
        result = self.capacity(cpu_offload=True)
        lane = next(lane for lane in self.broker.lanes.values() if lane.kind == 'managed')
        self.assertEqual(lane.scope, (GPUS[0],))
        self.assertEqual(lane.reserved_mib_by_gpu, {GPUS[0]: 81920})
        self.assertEqual(lane.host_reserved_mib, 93578 + 2048)
        self.assertTrue(result['lanes'][0]['cpu_offload'])
        self.assertTrue(result['lanes'][0]['exclusive'])
        self.assertEqual(self.broker._peer_reserved_gpus_locked(), {GPUS[0]})

    def test_full_gpu_group_is_preferred_when_aggregate_fits(self):
        result = self.capacity(scope=GPUS, cpu_offload=True)
        self.assertFalse(result['lanes'][0]['cpu_offload'])
        self.assertEqual(result['lanes'][0]['gpu_uuids'], GPUS)

    def test_opt_in_can_overflow_multiple_allowed_gpus(self):
        self.broker._model_profile.return_value = (200000, {'completion'})
        result = self.capacity(scope=list(reversed(GPUS)), cpu_offload=True)
        self.assertEqual(result['lanes'][0]['gpu_uuids'], list(reversed(GPUS)))
        self.assertTrue(result['lanes'][0]['cpu_offload'])

    def test_cpu_permission_never_widens_model_or_client_policy(self):
        self.broker.model_gpu_policy[MODEL] = [GPUS[1]]
        with self.assertRaises(n.PermanentCapacityError) as raised:
            self.capacity(cpu_offload=True)
        self.assertEqual(raised.exception.reason_code, 'gpu_policy_conflict')
        self.broker.model_gpu_policy.clear()
        self.broker.client_gpu_policy['client'] = []
        with self.assertRaises(n.PermanentCapacityError) as raised:
            self.capacity(cpu_offload=True, triggered_by={'key': 'client'})
        self.assertEqual(raised.exception.reason_code, 'client_gpu_disabled')
        self.spawn.assert_not_called()

    def test_live_lease_prevents_overflow_without_cpu_fallback(self):
        self.broker.leases['owned'] = mock.Mock(gpu_uuids=[GPUS[0]], state='active')
        with self.assertRaises(n.CapacityError):
            self.capacity(cpu_offload=True)
        self.spawn.assert_not_called()

    def test_missing_or_insufficient_physical_and_cgroup_memory_defer(self):
        for snapshot in [
            {}, {'memavailable_mib': 1000000},
            {'memavailable_mib': 90000, 'managed_cgroup_headroom_mib': 'max'},
            {'memavailable_mib': 1000000, 'managed_cgroup_headroom_mib': 90000},
            {'memavailable_mib': 1000000, 'managed_cgroup_headroom_mib': 0},
        ]:
            with self.subTest(snapshot=snapshot):
                n.host_memory_snapshot.return_value = snapshot
                with self.assertRaises(n.CapacityError) as raised:
                    self.capacity(cpu_offload=True)
                self.assertEqual(raised.exception.reason_code, 'host_memory_unavailable')
        self.spawn.assert_not_called()

    def test_outstanding_host_promises_cannot_be_spent_twice(self):
        self.capacity(cpu_offload=True)
        n.host_memory_snapshot.return_value = {
            'memavailable_mib': 150000, 'managed_cgroup_headroom_mib': 'max'}
        with self.assertRaises(n.CapacityError) as raised:
            self.capacity(scope=[GPUS[1]], cpu_offload=True)
        self.assertEqual(raised.exception.reason_code, 'host_memory_unavailable')
        self.assertEqual(self.spawn.call_count, 1)

    def test_normal_gpu_load_respects_existing_cpu_host_promises(self):
        self.capacity(cpu_offload=True)
        n.host_memory_snapshot.return_value = {
            'memavailable_mib': 96500, 'managed_cgroup_headroom_mib': 'max'}
        self.broker._model_profile.return_value = (2000, {'completion'})
        with self.assertRaises(n.CapacityError) as raised:
            self.broker.ensure_capacity('small:latest', 1, gpu_uuids=(GPUS[1],))
        self.assertEqual(raised.exception.reason_code, 'host_memory_unavailable')
        self.assertEqual(self.spawn.call_count, 1)

    def test_cpu_lane_reuse_is_opt_in_and_larger_scope_prefers_gpu(self):
        self.capacity(cpu_offload=True)
        with self.broker.cv:
            self.assertIsNone(self.broker._select_lane_locked(MODEL, True, (GPUS[0],)))
            self.assertIsNotNone(self.broker._select_lane_locked(MODEL, True, (GPUS[0],), cpu_offload=True))
            self.assertIsNone(self.broker._select_lane_locked(MODEL, True, tuple(GPUS), cpu_offload=True))
        with self.assertRaises(n.PermanentCapacityError):
            self.capacity()
        self.capacity(cpu_offload=True)
        self.assertEqual(self.spawn.call_count, 1)
        result = self.capacity(scope=GPUS, cpu_offload=True)
        self.assertFalse(result['lanes'][0]['cpu_offload'])
        self.assertEqual(self.spawn.call_count, 2)

    def test_idle_singleton_cpu_scope_can_be_reclaimed_for_small_gpu_model(self):
        self.capacity(cpu_offload=True)
        self.broker._model_profile.return_value = (2000, {'completion'})
        result = self.broker.ensure_capacity('small:latest', 1, gpu_uuids=(GPUS[0],))
        self.assertFalse(result['lanes'][0]['cpu_offload'])
        self.assertEqual(self.spawn.call_count, 2)

    def test_unscoped_gpu_preference_order_cannot_strand_cpu_lane(self):
        self.broker._model_profile.return_value = (200000, {'completion'})
        n.MODEL_GPU_PREFERENCES[MODEL] = list(reversed(GPUS))
        self.broker.ensure_capacity(MODEL, 1, cpu_offload=True)
        with self.broker.cv:
            lane = self.broker._select_lane_locked(MODEL, True, cpu_offload=True)
        self.assertEqual(lane.scope, tuple(reversed(GPUS)))

    def test_offload_scope_cannot_issue_full_gpu_warm_certificate(self):
        self.capacity(cpu_offload=True)
        lane = next(lane for lane in self.broker.lanes.values() if lane.kind == 'managed')
        with self.assertRaises(n.WarmPreflightRequired):
            self.broker._warm_runtime_identity_locked(lane)
        with self.assertRaises(n.PermanentCapacityError):
            self.broker.proxy_enter(MODEL, True, cpu_offload=True, warm_admission='certificate')

    def test_background_offload_cannot_evict_resident_lane(self):
        self.broker._model_profile.return_value = (2000, {'completion'})
        self.broker.ensure_capacity('small:latest', 1, gpu_uuids=(GPUS[0],))
        self.broker._model_profile.return_value = (93578, {'completion'})
        with self.assertRaises(n.BackgroundCapacityDeferred):
            self.capacity(cpu_offload=True, allow_reclaim=False)
        self.assertEqual(self.spawn.call_count, 1)
        self.broker._stop_lanes.assert_not_called()

    def test_nonboolean_permissions_and_forced_group_are_rejected(self):
        for value in [None, 'true', 1, []]:
            with self.subTest(value=value), self.assertRaises(n.PermanentCapacityError):
                self.capacity(cpu_offload=value)
        with self.assertRaises(n.PermanentCapacityError):
            self.capacity(scope=GPUS, cpu_offload=True, placement='exclusive_group')
        self.spawn.assert_not_called()


class NativeWarmupTests(CpuFixture):
    def warm(self, *, cpu_only=False, observed=None, host_reservation=100000):
        self.broker._spawn_lane = n.Broker._spawn_lane.__get__(self.broker)
        process = mock.Mock(pid=123456)
        process.poll.return_value = None
        resident = {'name': MODEL, 'size': 90000 * MIB,
                    'size_vram': 0 if cpu_only else 75000 * MIB}
        with mock.patch.object(n.os, 'access', return_value=True), mock.patch.object(
            n.subprocess, 'Popen', return_value=process
        ) as popen, mock.patch.object(self.broker, '_available_port_locked', return_value=30001), mock.patch.object(
            n, 'managed_server_identity', return_value=(123456, 1, 123456)
        ), mock.patch.object(n, 'assert_managed_server_scope'), mock.patch.object(
            n, 'backend_json_at', side_effect=[{}, {}, {'models': [resident]}]
        ) as request, mock.patch.object(n, 'process_gpu_usage', return_value=observed or {GPUS[0]: 75000}), mock.patch.dict(
            n.os.environ, {'GGML_CUDA_ENABLE_UNIFIED_MEMORY': '0', 'GGML_CUDA_REGISTER_HOST': '1'}
        ):
            lane = self.broker._spawn_lane(MODEL, (GPUS[0],), 93578, {'completion'}, '/api/chat',
                                          reserved_mib_by_gpu={GPUS[0]: 81920},
                                          cpu_offload=True, host_reserved_mib=host_reservation)
            self.assertEqual(request.call_args_list[1].args[4]['options']['num_gpu'], -1)
            return lane, popen.call_args.kwargs['env']

    def test_native_partial_residency_requires_gpu_and_exact_scope(self):
        lane, env = self.warm()
        self.assertTrue(lane.cpu_offload)
        self.assertEqual(lane.observed_vram_mib_by_gpu, {GPUS[0]: 75000})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], GPUS[0])
        self.assertNotIn('GGML_CUDA_ENABLE_UNIFIED_MEMORY', env)
        self.assertNotIn('GGML_CUDA_REGISTER_HOST', env)
        self.assertEqual(env['GGML_CUDA_NO_PINNED'], '1')

    def test_cpu_only_native_load_is_rejected(self):
        with self.assertRaises(n.PermanentCapacityError) as raised:
            self.warm(cpu_only=True)
        self.assertEqual(raised.exception.reason_code, 'gpu_runtime_unavailable')

    def test_observed_outside_scope_or_host_overflow_is_rejected(self):
        with self.assertRaises(n.CapacityError):
            self.warm(observed={GPUS[1]: 75000})
        with self.assertRaises(n.PermanentCapacityError) as raised:
            self.warm(host_reservation=1000)
        self.assertEqual(raised.exception.reason_code, 'model_context_memory_exceeded')

    def test_failed_warmup_retains_gpu_and_host_promises_until_process_exit(self):
        self.broker._stop_lanes = mock.Mock(side_effect=lambda lanes, reason: lanes)
        with self.assertRaises(n.CapacityError):
            self.warm(observed={GPUS[1]: 75000})
        lane = next(lane for lane in self.broker.lanes.values() if lane.kind == 'managed')
        self.assertTrue(lane.retiring)
        self.assertEqual(lane.host_reserved_mib, 100000)
        self.assertEqual(lane.reserved_mib_by_gpu, {GPUS[0]: 81920})
        n.host_memory_snapshot.return_value = {'memavailable_mib': 150000,
                                              'managed_cgroup_headroom_mib': 'max'}
        with self.assertRaises(n.CapacityError):
            self.broker._require_cpu_host_budget(100000)


class ProtocolTests(CpuFixture):
    def setUp(self):
        super().setUp()
        self.backend = p.Backend([MODEL])
        self.backend_port = self.backend.server_port
        self.release_backend = threading.Event()
        self.backend_started = threading.Event()
        self.backend_headers = []
        parent = self

        class Inference(p.Handler):
            def do_POST(self):
                parent.backend_headers.append(dict(self.headers))
                parent.backend_started.set()
                parent.release_backend.wait(5)
                super().do_POST()

        self.backend.RequestHandlerClass = Inference
        self.backend_thread = threading.Thread(target=self.backend.serve_forever, daemon=True)
        self.backend_thread.start()
        handler = type('CpuProxy', (n.ProxyHandler,), {'broker': self.broker})
        self.proxy = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self.proxy.daemon_threads = True
        self.proxy_thread = threading.Thread(target=self.proxy.serve_forever, daemon=True)
        self.proxy_thread.start()
        self.scheduler = threading.Thread(target=self.broker.capacity_reconciler, daemon=True)
        self.scheduler.start()
        self.addCleanup(self.close_servers)

    def close_servers(self):
        self.release_backend.set()
        self.broker.stopping.set()
        with self.broker.cv:
            self.broker.cv.notify_all()
        self.scheduler.join(5)
        self.proxy.shutdown()
        self.proxy.server_close()
        self.backend.shutdown()
        self.backend.server_close()

    def chat(self, logical, *, mode='allow', resume=False, wait_ms=None, gpu_scope=GPUS[0]):
        headers = {'X-Ollama-Unify-Logical-Request-Id': logical}
        if gpu_scope is not None:
            headers['X-Ollama-Unify-GPU-UUIDs'] = gpu_scope
        if mode is not None:
            headers[n.CPU_OFFLOAD_HEADER] = mode
        if wait_ms:
            headers['X-Ollama-Unify-Admission-Wait-Ms'] = str(wait_ms)
        if resume:
            headers['X-Ollama-Unify-Resume-Request'] = 'true'
        return p.http_json(self.proxy.server_port, 'POST', '/api/chat',
                           None if resume else {'model': MODEL, 'stream': False},
                           extra_headers=headers)

    def test_second_opt_in_inference_queues_behind_busy_exclusive_scope(self):
        self.broker.model_gpu_policy[MODEL] = [GPUS[0]]
        self.broker.ensure_capacity = mock.Mock(wraps=self.broker.ensure_capacity)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            first = pool.submit(self.chat, 'first', gpu_scope=None)
            self.assertTrue(self.backend_started.wait(3))
            second = pool.submit(self.chat, 'second', gpu_scope=None)
            p.wait_until(lambda: len(self.broker.waiters) == 1, 'second offload waiter')
            time.sleep(.35)
            self.assertFalse(second.done())
            self.assertEqual(self.spawn.call_count, 1)
            self.assertEqual(self.broker.ensure_capacity.call_count, 1)
            self.release_backend.set()
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(second.result()[0], 200)
        self.assertNotIn(n.CPU_OFFLOAD_HEADER, self.backend_headers[0])

    def test_default_request_cannot_consume_cpu_lane(self):
        self.capacity(cpu_offload=True)
        result = self.chat('default', mode=None)
        self.assertEqual(result[0], 422, result)
        self.assertEqual(result[1]['reason_code'], 'model_exceeds_gpu_capacity')
        self.assertFalse(self.backend_started.is_set())

    def test_completed_replay_and_resume_preserve_permission(self):
        self.release_backend.set()
        first = self.chat('completed')
        self.assertEqual(first[0], 200, first)
        self.assertEqual(self.chat('completed', mode='deny')[0], 409)
        self.assertEqual(self.chat('completed', mode='deny', resume=True)[0], 409)
        self.assertEqual(self.chat('completed', mode=None, resume=True)[0], 200)
        self.assertEqual(len(self.backend_headers), 1)

    def test_retained_resume_inherits_permission_and_rejects_change(self):
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            first = pool.submit(self.chat, 'busy')
            self.assertTrue(self.backend_started.wait(3))
            result = self.chat('retained', wait_ms=100)
            self.assertEqual(result[0], 503, result)
            self.assertTrue(result[1]['admission_retained'])
            self.assertEqual(self.chat('retained', mode='deny', resume=True)[0], 409)
            self.release_backend.set()
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(self.chat('retained', mode=None, resume=True)[0], 200)
        self.assertEqual(len(self.backend_headers), 2)

    def test_invalid_header_or_capacity_permission_never_loads(self):
        for value in ['true', 'false', '1', '', 'allow,deny']:
            with self.subTest(value=value):
                self.assertEqual(self.chat('bad-' + value, mode=value)[0], 400)
        self.assertEqual(self.spawn.call_count, 0)
        for value in ['true', 1, None]:
            result = p.http_json(self.proxy.server_port, 'POST', n.CAPACITY_PATH,
                                 {'model': MODEL, 'parallel': 1, 'cpu_offload': value,
                                  'gpu_uuids': [GPUS[0]]})
            self.assertEqual(result[0], 400, result)
        self.assertEqual(self.spawn.call_count, 0)

    def test_capacity_boolean_propagates_and_discovery_advertises_contract(self):
        result = p.http_json(self.proxy.server_port, 'POST', n.CAPACITY_PATH,
                             {'model': MODEL, 'parallel': 1, 'cpu_offload': True,
                              'gpu_uuids': [GPUS[0]]})
        self.assertEqual(result[0], 200, result)
        self.assertTrue(result[1]['lanes'][0]['cpu_offload'])
        with mock.patch.object(n, 'gpu_health_snapshot', return_value={'admission_blocked': False, 'supported': True}), mock.patch.object(
            n, 'probe_backend', return_value=mock.Mock(available=True, error=None, checked_at=0)):
            protocol = n.discovery_document()['parallel_pool']['admission_protocol']
        self.assertEqual(protocol['cpu_offload_header'], n.CPU_OFFLOAD_HEADER)
        self.assertEqual(protocol['cpu_offload_capacity_field'], 'cpu_offload')


if __name__ == '__main__':
    unittest.main()
