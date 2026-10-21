#!/usr/bin/env python3
"""Client placement through CPU-only broker, HTTP and native-runner fixtures."""
import concurrent.futures
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


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


p = load('client_policy_pool', pathlib.Path(__file__).with_name('test-negotiator-pool.py'))
with mock.patch.dict(os.environ, {
    'OLLAMA_UNIFY_CONFIG': '/nonexistent/client-policy-test',
    'OLLAMA_UNIFY_LEASE_STATE': '/nonexistent/client-policy-leases',
    'OLLAMA_UNIFY_MODEL_POLICY_STATE': '/nonexistent/client-policy-models',
}):
    m = load('client_policy_broker', HELPER)

GPUS = ['GPU-large-0', 'GPU-large-1', 'GPU-large-2']
CLIENT = 'app:placement-fixture'
CAPACITY = '/.well-known/ollama-unify-gpu-negotiator/capacity'


def policy(case, scope, client=CLIENT):
    return p.control(case.socket_path, {
        'action': 'set_client_gpus', 'client_key': client, 'gpu_uuids': scope})


def capacity(case, **extra):
    return p.http_json(case.proxy_port, 'POST', CAPACITY,
                       {'model': p.MODEL, 'parallel': 1, **extra},
                       extra_headers={'X-Ollama-Unify-Client': 'placement-fixture'})


def served_gpu(case, response):
    assert response[0] == 200, response
    return next(lane['gpu_uuid'] for lane in case.status()['parallel_pool']['lanes']
                if lane['id'] == response[2]['X-Ollama-Unify-Lane'])


class PolicyStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.temp.name)
        self.patch = mock.patch.multiple(m, SELECTED_GPUS=list(GPUS),
            MODEL_POLICY_PATH=root/'policy.json', LEASE_STATE_PATH=root/'leases.json',
            CACHE_STATE_PATH=root/'cache.json')
        self.patch.start()
        self.broker = m.Broker()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_single_multiple_none_clear_survive_restart_and_model_save(self):
        for scope in ([GPUS[0]], [GPUS[2], GPUS[0]], []):
            result = self.broker.set_client_gpus(CLIENT, scope)
            self.assertEqual(result['allowed_gpu_uuids'], scope)
            self.assertTrue(result['active_requests_unchanged'])
            self.assertEqual(m.Broker().client_gpu_policy, {CLIENT: scope})
        self.broker.model_gpu_policy[p.MODEL] = [GPUS[1]]
        with self.broker.cv:
            self.broker._persist_model_policy_locked()
        restored = m.Broker()
        self.assertEqual(restored.client_gpu_policy[CLIENT], [])
        self.assertEqual(restored.model_gpu_policy[p.MODEL], [GPUS[1]])
        self.broker.set_client_gpus(CLIENT, list(GPUS))
        self.assertNotIn(CLIENT, m.Broker().client_gpu_policy)
        self.broker.set_client_gpus(CLIENT, [])
        self.broker.set_client_gpus(CLIENT, None)
        self.assertNotIn(CLIENT, m.Broker().client_gpu_policy)

    def test_checkbox_edits_compose_without_waiting_for_tray_refresh(self):
        # Both removals can come from the same stale all-selected tray view.
        self.broker.set_client_gpu(CLIENT, GPUS[0], False)
        self.broker.set_client_gpu(CLIENT, GPUS[2], False)
        self.assertEqual(self.broker.client_gpu_policy[CLIENT], [GPUS[1]])
        # Re-delivery is idempotent and the last checkbox can be removed.
        self.broker.set_client_gpu(CLIENT, GPUS[2], False)
        self.broker.set_client_gpu(CLIENT, GPUS[1], False)
        self.assertEqual(m.Broker().client_gpu_policy[CLIENT], [])
        self.broker.set_client_gpu(CLIENT, GPUS[2], True)
        self.assertEqual(m.Broker().client_gpu_policy[CLIENT], [GPUS[2]])
        for gpu, allowed in ((None, True), ('GPU-unknown', True), (GPUS[0], 'true')):
            with self.assertRaises(ValueError):
                self.broker.set_client_gpu(CLIENT, gpu, allowed)
        self.assertEqual(self.broker.client_gpu_policy[CLIENT], [GPUS[2]])

    def test_invalid_values_do_not_change_policy(self):
        self.broker.set_client_gpus(CLIENT, [GPUS[0]])
        for scope in ('GPU-large-0', {}, [False], [None], ['GPU-display-or-unknown']):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                self.broker.set_client_gpus(CLIENT, scope)
            self.assertEqual(self.broker.client_gpu_policy, {CLIENT: [GPUS[0]]})
        for key in ('', None, 'bad\nkey', 'x'*513):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.broker.set_client_gpus(key, [])

    def test_failed_persistence_restores_effective_policy(self):
        self.broker.set_client_gpus(CLIENT, [])
        with mock.patch.object(self.broker, '_persist_model_policy_locked', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.broker.set_client_gpus(CLIENT, None)
        self.assertEqual(self.broker.client_gpu_policy, {CLIENT: []})
        self.assertEqual(m.Broker().client_gpu_policy, {CLIENT: []})

    def test_expired_client_policy_stays_visible_without_extending_history(self):
        self.broker.clients[CLIENT] = {
            'identity': {'key': CLIENT, 'label': 'Placement fixture'},
            'first_seen': 0, 'last_seen': 0, 'requests': 5, 'models': {}, 'lanes': {}}
        self.broker.set_client_gpus(CLIENT, [])
        with self.broker.cv:
            self.broker._prune_client_history_locked(time.time())
            self.assertNotIn(CLIENT, self.broker.clients)
            row, = self.broker._client_summaries_locked()
        self.assertTrue(row['policy_only'])
        self.assertIsNone(row['expires_at'])
        self.assertEqual(row['identity']['label'], 'Placement fixture')
        row, = m.Broker()._client_summaries_locked()
        self.assertEqual(row['identity']['label'], 'Placement fixture')

    def test_corrupt_saved_policy_does_not_reenable_client(self):
        for value in (None, {'clients': {CLIENT: None}}, {'clients': {CLIENT: ['']}}):
            m.MODEL_POLICY_PATH.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.Broker()

    def test_legacy_backend_cannot_ignore_client_scope(self):
        self.broker.set_client_gpus(CLIENT, [GPUS[0]])
        with mock.patch.object(m, 'POOL_ENABLED', False), mock.patch.object(m, 'require_gpu_health'):
            with self.assertRaises(m.PermanentCapacityError) as caught:
                self.broker.proxy_enter(p.MODEL, True, client={'key': CLIENT})
        self.assertEqual(caught.exception.reason_code, 'pool_disabled')

    def test_capacity_waiting_for_transition_rechecks_policy_before_loading(self):
        self.broker.set_client_gpus(CLIENT, [GPUS[0]])
        checked = threading.Event()
        original = self.broker._capacity_scope_locked

        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            checked.set()
            return result

        with mock.patch.object(m, 'POOL_ENABLED', True), \
                mock.patch.object(m, 'require_inference_enabled'), \
                mock.patch.object(m, 'require_gpu_health'), \
                mock.patch.object(self.broker, '_capacity_scope_locked', side_effect=observe), \
                mock.patch.object(self.broker, '_model_profile') as profile, \
                concurrent.futures.ThreadPoolExecutor() as executor:
            self.broker.transition.acquire()
            try:
                waiting = executor.submit(self.broker.ensure_capacity, p.MODEL, 1,
                                          triggered_by={'key': CLIENT})
                self.assertTrue(checked.wait(3))
                self.broker.set_client_gpus(CLIENT, [])
            finally:
                self.broker.transition.release()
            with self.assertRaises(m.PermanentCapacityError) as caught:
                waiting.result(timeout=3)
            self.assertEqual(caught.exception.reason_code, 'client_gpu_disabled')
            profile.assert_not_called()


class ClientHTTPTests(unittest.TestCase):
    def test_scopes_route_only_target_client_and_intersect_request_and_model(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN, max_servers=3) as case:
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
            initial = p.chat(case.proxy_port, p.MODEL, 'initial', client='placement-fixture')
            self.assertEqual(served_gpu(case, initial), GPUS[0])
            for gpu in (GPUS[0], GPUS[2]):
                changed = p.control(case.socket_path, {
                    'action': 'set_client_gpu', 'client_key': CLIENT,
                    'gpu_uuid': gpu, 'allowed': False})
                self.assertTrue(changed['ok'])
            self.assertEqual(changed['allowed_gpu_uuids'], [GPUS[1]])
            self.assertEqual(served_gpu(case, p.chat(case.proxy_port, p.MODEL, 'scoped', client='placement-fixture')), GPUS[1])
            self.assertEqual(served_gpu(case, p.chat(case.proxy_port, p.MODEL, 'other', client='unaffected')), GPUS[0])
            denied = p.chat(case.proxy_port, p.MODEL, 'header-conflict', client='placement-fixture', gpu_uuids=[GPUS[0]])
            self.assertEqual((denied[0], denied[1]['reason_code']), (409, 'client_gpu_policy_conflict'))
            self.assertEqual(capacity(case, gpu_uuids=[GPUS[0]])[0], 409)
            policy(case, [GPUS[0], GPUS[2]])
            self.assertEqual(served_gpu(case, p.chat(case.proxy_port, p.MODEL, 'second-choice', client='placement-fixture', gpu_uuids=[GPUS[2], GPUS[0]])), GPUS[0])
            p.control(case.socket_path, {'action': 'set_model_gpus', 'model': p.MODEL, 'gpu_uuids': [GPUS[1]]})
            self.assertEqual(capacity(case)[0], 409)

    def test_none_preserves_active_request_shared_lane_and_metadata(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            self.assertEqual(capacity(case, gpu_uuids=[GPUS[0]])[0], 200)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                running = executor.submit(p.chat, case.proxy_port, p.MODEL, 'finishes', delay=1,
                                          client='placement-fixture')
                p.wait_until(lambda: case.status()['active_requests'] == 1, 'active client')
                policy(case, [])
                blocked = p.chat(case.proxy_port, p.MODEL, 'paused', client='placement-fixture')
                self.assertEqual((blocked[0], blocked[1]['reason_code']), (403, 'client_gpu_disabled'))
                self.assertFalse(blocked[1]['retryable'])
                self.assertEqual(running.result(timeout=10)[0], 200)
            self.assertEqual(p.chat(case.proxy_port, p.MODEL, 'other', client='unaffected')[0], 200)
            self.assertEqual(capacity(case, gpu_uuids=[GPUS[0]], warm_admission_proof=True)[0], 403)
            self.assertEqual(p.http_json(case.proxy_port, 'GET', '/api/tags',
                extra_headers={'X-Ollama-Unify-Client': 'placement-fixture'})[0], 200)
            self.assertEqual(len([e for e in p.events(case.event_log) if e['kind']=='start']), 1)
            policy(case, None)
            self.assertEqual(p.chat(case.proxy_port, p.MODEL, 'resumed', client='placement-fixture')[0], 200)

    def test_queue_uses_updated_scope_then_none_rejects_without_loading(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN) as case:
            lease = p.control(case.socket_path, {'action':'acquire', 'owner':'fixture-reservation',
                'gpu_uuids':[GPUS[0]], 'requested_mib':1024, 'ttl':60})['lease']['token']
            policy(case, [GPUS[0]])
            with concurrent.futures.ThreadPoolExecutor() as executor:
                waiting = executor.submit(p.chat, case.proxy_port, p.MODEL, 'retargeted', client='placement-fixture')
                p.wait_until(lambda: case.status()['parallel_pool']['queue']['depth']==1, 'queued on original GPU')
                policy(case, [GPUS[1]])
                self.assertEqual(served_gpu(case, waiting.result(timeout=10)), GPUS[1])
                policy(case, [GPUS[0]])
                paused = executor.submit(p.chat, case.proxy_port, p.MODEL, 'queue-paused', client='placement-fixture')
                p.wait_until(lambda: case.status()['parallel_pool']['queue']['depth']==1, 'queued before pause')
                policy(case, [])
                result = paused.result(timeout=10)
                self.assertEqual((result[0], result[1]['reason_code']), (403, 'client_gpu_disabled'))
                self.assertEqual(case.status()['parallel_pool']['queue']['depth'], 0)
            p.control(case.socket_path, {'action':'release', 'token':lease})

    def test_group_and_warm_proof_do_not_escape_client_scope(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN, max_servers=3, auto_model_context=True,
                         runner_vram_by_gpu={gpu:1024 for gpu in GPUS}) as case:
            policy(case, [GPUS[1]])
            denied = capacity(case, gpu_uuids=GPUS[:2], placement='exclusive_group')
            self.assertEqual(denied[0], 409)
            self.assertEqual([e for e in p.events(case.event_log) if e['kind']=='start'], [])
            policy(case, [GPUS[0]])
            self.assertEqual(capacity(case, gpu_uuids=[GPUS[0]])[0], 200)
            self.assertEqual(capacity(case, gpu_uuids=[GPUS[0]], warm_admission_proof=True)[0], 200)
            policy(case, [GPUS[1]])
            self.assertEqual(capacity(case, gpu_uuids=[GPUS[0]], warm_admission_proof=True)[0], 409)

    def test_existing_warm_certificate_cannot_bypass_changed_client_policy(self):
        with p.PoolHarness(HELPER, FIXTURE_BIN, auto_model_context=True) as case:
            scope = GPUS[:2]
            self.assertEqual(capacity(case, gpu_uuids=scope)[0], 200)
            status, proof, _ = capacity(case, gpu_uuids=scope, warm_admission_proof=True)
            self.assertEqual(status, 200, proof)
            policy(case, [GPUS[1]])
            headers = {'X-Ollama-Unify-Client':'placement-fixture',
                       'X-Ollama-Unify-GPU-UUIDs':','.join(scope),
                       'X-Ollama-Unify-Warm-Admission':json.dumps(proof['warm_admission'])}
            body = {'model':p.MODEL, 'stream':False, 'options':{'num_ctx':262144}}
            denied = p.http_json(case.proxy_port, 'POST', '/api/chat', body, extra_headers=headers)
            self.assertEqual((denied[0], denied[1]['reason_code']), (503, 'warm_preflight_required'))
            policy(case, [])
            paused = p.http_json(case.proxy_port, 'POST', '/api/chat', body, extra_headers=headers)
            self.assertEqual((paused[0], paused[1]['reason_code']), (403, 'client_gpu_disabled'))
            self.assertEqual(len([e for e in p.events(case.event_log) if e['kind']=='start']), 1)


if __name__ == '__main__':
    unittest.main()
