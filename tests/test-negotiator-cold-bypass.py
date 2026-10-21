#!/usr/bin/env python3
"""CPU-only fairness checks for cold work on disjoint, unreserved GPUs."""
import concurrent.futures
import http.server
import importlib.machinery
import importlib.util
import os
import pathlib
import sys
import threading
import time
import unittest
from unittest import mock

HELPER = sys.argv.pop(1)
FIXTURE_BIN = sys.argv.pop(1)
os.environ['OLLAMA_UNIFY_CONFIG'] = '/nonexistent/cold-fairness'
os.environ['OLLAMA_UNIFY_LEASE_STATE'] = '/nonexistent/cold-leases.json'
os.environ['OLLAMA_UNIFY_MODEL_POLICY_STATE'] = '/nonexistent/cold-policy.json'
os.environ['OLLAMA_UNIFY_CACHE_RESIDENCY_STATE'] = '/nonexistent/cold-cache.json'


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


n = load('cold_bypass_negotiator', HELPER)
p = load('cold_bypass_pool', pathlib.Path(__file__).with_name('test-negotiator-pool.py'))
GPUS = ['GPU-large-0', 'GPU-large-1', 'GPU-large-2']
PAIR = GPUS[:2]
LARGE = 'fixture-sharded:latest'
LARGE_TAG = {'name': LARGE, 'model': LARGE, 'size': 130 * 1024**3,
             'capabilities': ['completion']}


class ColdSelectionTests(unittest.TestCase):
    def setUp(self):
        for name, value in [('SELECTED_GPUS', GPUS), ('POOL_MAX_SERVERS', 4),
                            ('MODEL_GPU_PREFERENCES', {})]:
            patch = mock.patch.object(n, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.broker = n.Broker()
        self.broker.ensure_capacity = mock.Mock(return_value={'ok': True})
        self.broker.reconcile_retry_at = time.monotonic() + 30
        self.head = self.waiter('head', tuple(PAIR), ticket=10)
        self.head.last_reason_code = 'gpu_peer_group_wait'
        self.candidate = self.waiter('embedding', tuple(reversed(GPUS)), ticket=11)
        self.broker.waiters = [self.head, self.candidate]

    def waiter(self, model, scope, ticket=0):
        return n.QueuedRequest(model + '-request', '', model + '-fingerprint', model,
                               time.monotonic(), time.monotonic() + 30, lambda: True,
                               gpu_uuids=scope, request_path='/api/embed', queue_ticket=ticket)

    def run_once(self):
        self.broker._reconcile_disjoint_cold_waiter()
        return self.broker.ensure_capacity

    def test_disjoint_scope_preserves_original_authority_and_queue_identity(self):
        original_scope = self.candidate.gpu_uuids
        original_retry = self.broker.reconcile_retry_at
        call = self.run_once()
        self.assertEqual(call.call_args.args[:4], ('embedding', 1, '/api/embed', (GPUS[2],)))
        self.assertFalse(call.call_args.kwargs['allow_reclaim'])
        self.assertFalse(call.call_args.kwargs['cpu_offload'])
        self.assertIs(self.candidate.gpu_uuids, original_scope)
        self.assertEqual(self.candidate.queue_ticket, 11)
        self.assertEqual(self.candidate.request_fingerprint, 'embedding-fingerprint')
        self.assertEqual(self.head.phase, 'queued')
        self.assertEqual(self.broker.reconcile_retry_at, original_retry)

    def test_same_model_fifo_cannot_be_bypassed_using_a_different_scope(self):
        self.candidate.model = self.head.model
        self.candidate.gpu_uuids = (GPUS[2],)
        self.run_once().assert_not_called()

    def test_older_intermediate_scope_is_protected(self):
        older = self.waiter('older', (GPUS[2],))
        older.cpu_offload = True
        self.broker.waiters.insert(1, older)
        self.run_once().assert_not_called()

    def test_client_model_and_request_intersection_is_respected(self):
        self.candidate.client = {'key': 'client'}
        self.broker.client_gpu_policy['client'] = [GPUS[0]]
        self.run_once().assert_not_called()
        self.broker.client_gpu_policy['client'] = [GPUS[2]]
        self.broker.model_gpu_policy[n.canonical_model_tag('embedding')] = [GPUS[1]]
        self.run_once().assert_not_called()
        self.broker.model_gpu_policy[n.canonical_model_tag('embedding')] = [GPUS[2]]
        self.run_once().assert_called_once()
        self.assertEqual(self.broker.ensure_capacity.call_args.args[3], (GPUS[2],))

    def test_unscoped_head_protects_every_selected_gpu(self):
        self.head.gpu_uuids = None
        self.run_once().assert_not_called()

    def test_model_policy_can_narrow_head_protection_without_widening_candidate(self):
        self.head.gpu_uuids = None
        self.broker.model_gpu_policy[n.canonical_model_tag('head')] = PAIR
        self.run_once().assert_called_once()
        self.assertEqual(self.broker.ensure_capacity.call_args.args[3], (GPUS[2],))

    def test_cpu_or_warm_certified_candidate_cannot_create_speculative_capacity(self):
        self.candidate.cpu_offload = True
        self.run_once().assert_not_called()
        self.candidate.cpu_offload = False
        self.candidate.warm_admission = 'certificate'
        self.run_once().assert_not_called()

    def test_head_retry_gets_priority_when_its_deadline_arrives(self):
        self.broker.reconcile_retry_at = 0
        self.run_once().assert_not_called()

    def test_global_host_pressure_is_not_a_gpu_bypass_opportunity(self):
        self.head.last_reason_code = 'host_memory_unavailable'
        self.run_once().assert_not_called()

    def test_keep_a_server_slot_for_each_older_model(self):
        with mock.patch.object(n, 'POOL_MAX_SERVERS', 1):
            self.run_once().assert_not_called()
        with mock.patch.object(n, 'POOL_MAX_SERVERS', 2):
            self.run_once().assert_called_once()

    def test_deferral_preserves_foreground_owner_and_head_retry_schedule(self):
        self.broker.ensure_capacity.side_effect = n.BackgroundCapacityDeferred('no free placement')
        retry = self.broker.reconcile_retry_at
        self.run_once().assert_called_once()
        self.assertIn(self.candidate, self.broker.waiters)
        self.assertEqual(self.candidate.phase, 'queued')
        self.assertIsNone(self.candidate.terminal_error)
        self.assertIsNone(self.candidate.last_reason_code)
        self.assertEqual(self.broker.reconcile_retry_at, retry)
        self.assertGreaterEqual(self.candidate.cold_capacity_retry_at, time.monotonic() + 29)
        self.run_once()
        self.assertEqual(self.broker.ensure_capacity.call_count, 1)

    def test_disconnected_candidate_does_not_create_phantom_capacity(self):
        self.candidate.connected = lambda: False
        self.run_once().assert_not_called()

    def test_malformed_speculative_metadata_cannot_stop_the_scheduler(self):
        self.broker.ensure_capacity.side_effect = ValueError('invalid JSON metadata')
        self.run_once().assert_called_once()
        self.assertEqual(self.candidate.phase, 'queued')
        self.assertIsNone(self.candidate.terminal_error)
        self.assertFalse(hasattr(n.CAPACITY_IO_DEADLINE, 'deadline'))

    def test_prequeued_disjoint_waiter_progresses_without_a_new_notification(self):
        self.head.gpu_uuids = (GPUS[0],)
        self.head.last_reason_code = None
        self.candidate.gpu_uuids = (GPUS[2],)
        self.broker.reconcile_retry_at = 0
        arrived = threading.Event()
        self.broker.ensure_capacity.side_effect = lambda *args, **kwargs: arrived.set()
        with mock.patch.object(self.broker, '_ollama_blocked_gpus_locked', return_value={GPUS[0]}):
            worker = threading.Thread(target=self.broker.capacity_reconciler, daemon=True)
            worker.start()
            try:
                self.assertTrue(arrived.wait(1), 'prequeued cold work was parked with the blocked head')
            finally:
                self.broker.stopping.set()
                with self.broker.cv:
                    self.broker.cv.notify_all()
                worker.join(2)

    def test_speculative_budget_is_thread_local_and_restored(self):
        seen = []
        self.broker.ensure_capacity.side_effect = lambda *args, **kwargs: seen.append(
            n.CAPACITY_IO_DEADLINE.deadline - time.monotonic())
        self.run_once()
        self.assertTrue(4.5 < seen[0] <= 5)
        self.assertFalse(hasattr(n.CAPACITY_IO_DEADLINE, 'deadline'))

    def test_lease_transition_lock_is_never_waited_on_for_speculation(self):
        locked = threading.Event()
        release = threading.Event()
        def owner():
            with self.broker.transition:
                locked.set()
                release.wait(2)
        worker = threading.Thread(target=owner)
        worker.start()
        self.assertTrue(locked.wait(1))
        try:
            self.run_once().assert_not_called()
            self.assertEqual(self.candidate.phase, 'queued')
        finally:
            release.set()
            worker.join(2)


class SpeculativeBudgetTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(n.CAPACITY_IO_DEADLINE, 'deadline'):
            del n.CAPACITY_IO_DEADLINE.deadline

    def test_metadata_and_warm_http_calls_share_remaining_budget(self):
        n.CAPACITY_IO_DEADLINE.deadline = time.monotonic() + .2
        response = mock.Mock(status=200)
        response.read.return_value = b'{}'
        response.read1.side_effect = [b'{}', b'']
        response.isclosed.return_value = False
        connection = mock.Mock()
        connection.getresponse.return_value = response
        with mock.patch.object(n.http.client, 'HTTPConnection', return_value=connection) as create:
            self.assertEqual(n.backend_json_at('127.0.0.1', 1, 'GET', '/api/tags', timeout=10), {})
        self.assertTrue(0 < create.call_args.kwargs['timeout'] <= .2)
        n.CAPACITY_IO_DEADLINE.deadline = time.monotonic() - 1
        with self.assertRaises(n.BackgroundCapacityDeferred):
            n.backend_json_at('127.0.0.1', 1, 'GET', '/api/tags')

    def test_failed_warmup_retires_the_owned_process_and_reservation(self):
        process = mock.Mock(pid=123456)
        process.poll.return_value = None
        with mock.patch.object(n, 'SELECTED_GPUS', GPUS), mock.patch.object(
            n, 'require_inference_enabled'
        ), mock.patch.object(n, 'require_gpu_health'), mock.patch.object(
            n, 'foreign_gpu_usage', return_value={}
        ), mock.patch.object(n, 'effective_model_context_profile', return_value=None), mock.patch.object(
            n.os, 'access', return_value=True
        ), mock.patch.object(n.subprocess, 'Popen', return_value=process), mock.patch.object(
            n, 'managed_server_identity', return_value=(123456, 1, 123456)
        ), mock.patch.object(n, 'assert_managed_server_scope'), mock.patch.object(
            n, 'backend_json_at', side_effect=[{}, TimeoutError('cold warm-up timed out')]
        ), mock.patch.object(n, 'process_group_alive', return_value=True), mock.patch.object(
            n, 'unload_models_at', return_value=[]
        ):
            broker = n.Broker()
            broker._available_port_locked = mock.Mock(return_value=30001)
            broker._terminate_process = mock.Mock(return_value=True)
            n.CAPACITY_IO_DEADLINE.deadline = time.monotonic() + 5
            with self.assertRaises(TimeoutError):
                broker._spawn_lane('fixture:latest', (GPUS[2],), 2048,
                                   {'completion'}, '/api/chat')
            broker._terminate_process.assert_called_once_with(process)
            self.assertFalse([lane for lane in broker.lanes.values() if lane.kind == 'managed'])

    def test_slow_response_chunks_cannot_extend_the_absolute_io_budget(self):
        class SlowMetadata(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Length', '11')
                self.end_headers()
                try:
                    for byte in b'{"ok":true}':
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            def log_message(self, *_args):
                pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), SlowMetadata)
        server.daemon_threads = True
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            started = time.monotonic()
            n.CAPACITY_IO_DEADLINE.deadline = started + .06
            with self.assertRaises((n.BackgroundCapacityDeferred, TimeoutError)):
                n.backend_json_at('127.0.0.1', server.server_port, 'GET', '/api/tags', timeout=2)
            self.assertLess(time.monotonic() - started, .25)
        finally:
            server.shutdown()
            server.server_close()


class ColdBypassIntegrationTests(unittest.TestCase):
    def test_blocked_pair_allows_cold_embedding_on_third_gpu_then_head_proceeds(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN, max_servers=4,
                           tags=[LARGE_TAG, {'name': p.EMBED_MODEL, 'model': p.EMBED_MODEL,
                                            'size': 1024**3, 'capabilities': ['embedding']}],
                           runner_vram_by_gpu={GPUS[0]: 62000, GPUS[1]: 69000}) as case:
            lease = p.control(case.socket_path, {'action': 'acquire', 'owner': 'fairness-fixture',
                              'requested_mib': 1024, 'ttl': 30, 'gpu_uuids': [GPUS[1]]})
            token = lease['lease']['token']
            p.control(case.socket_path, {'action': 'ready', 'token': token})
            with concurrent.futures.ThreadPoolExecutor(1) as executor:
                head = executor.submit(p.chat, case.proxy_port, LARGE, 'pair-head', 0, 8,
                                       gpu_uuids=PAIR, logical_request_id='pair-head-logical')
                p.wait_until(lambda: any(item.get('last_reason_code') == 'gpu_peer_group_wait'
                             for item in case.status()['parallel_pool']['queue']['requests']),
                             'pair held by the fixed lease')
                result = p.http_json(case.proxy_port, 'POST', '/api/embed',
                                     {'model': p.EMBED_MODEL, 'input': 'fixture'},
                                     extra_headers={'X-Ollama-Unify-GPU-UUIDs': ','.join(reversed(GPUS))})
                self.assertEqual(result[0], 200, result)
                current = case.status()
                self.assertEqual(current['parallel_pool']['queue']['depth'], 1)
                queued = current['parallel_pool']['queue']['requests'][0]
                self.assertEqual(queued['gpu_uuids'], PAIR)
                self.assertEqual(queued['ticket'], 1)
                lanes = p.managed_lanes(current)
                self.assertEqual([(lane['model'], lane['gpu_uuids']) for lane in lanes],
                                 [(p.EMBED_MODEL, [GPUS[2]])])
                self.assertEqual(current['leases'][0]['owner'], 'fairness-fixture')
                self.assertEqual(current['leases'][0]['state'], 'active')
                self.assertFalse(head.done())
                self.assertFalse([event for event in p.events(case.event_log) if event['kind'] == 'stop'])
                p.control(case.socket_path, {'action': 'release', 'token': token})
                final = head.result(timeout=8)
                self.assertEqual(final[0], 200, final)
                self.assertEqual(final[1]['gpu'], ','.join(PAIR))
                self.assertEqual(case.status()['parallel_pool']['queue']['depth'], 0)

    def test_fully_blocked_singleton_head_does_not_park_unrelated_cold_model(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN, max_servers=3, tags=[p.MODEL, p.OTHER_MODEL]) as case:
            token = p.control(case.socket_path, {'action': 'acquire', 'owner': 'single-fairness-fixture',
                              'requested_mib': 1024, 'ttl': 30, 'gpu_uuids': [GPUS[0]]})['lease']['token']
            with concurrent.futures.ThreadPoolExecutor(1) as executor:
                head = executor.submit(p.chat, case.proxy_port, p.MODEL, 'single-head', 0, 8,
                                       gpu_uuids=[GPUS[0]], logical_request_id='single-head-logical')
                p.wait_until(lambda: any(item.get('last_reason_code') == 'lease_transition'
                             for item in case.status()['parallel_pool']['queue']['requests']),
                             'single GPU blocked by pending lease')
                result = p.chat(case.proxy_port, p.OTHER_MODEL, 'single-bypass', timeout=4,
                                gpu_uuids=[GPUS[1]])
                self.assertEqual(result[0], 200, result)
                self.assertEqual(result[1]['gpu'], GPUS[1])
                self.assertFalse(head.done())
                p.control(case.socket_path, {'action': 'release', 'token': token})
                final = head.result(timeout=8)
                self.assertEqual(final[0], 200, final)
                self.assertEqual(final[1]['gpu'], GPUS[0])


if __name__ == '__main__':
    unittest.main()
