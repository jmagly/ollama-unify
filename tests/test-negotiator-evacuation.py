#!/usr/bin/env python3
"""CPU-only public control/native-HTTP evacuation lifecycle regressions."""
import ast
import concurrent.futures
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

HELPER = sys.argv.pop(1)
FIXTURE_BIN = sys.argv.pop(1)


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


p = load('evacuation_pool', pathlib.Path(__file__).with_name('test-negotiator-pool.py'))
GPUS = ['GPU-large-0', 'GPU-large-1', 'GPU-large-2']


def harness(helper=HELPER, **options):
    defaults = {'max_servers': 4, 'tags': [p.MODEL, p.OTHER_MODEL], 'max_context': 8192}
    defaults.update(options)
    return p.PoolHarness(helper, FIXTURE_BIN, **defaults)


def fault_helper(directory, source):
    path = pathlib.Path(directory) / 'fault-helper.py'
    path.write_text('''#!/usr/bin/env python3
import importlib.machinery, importlib.util, pathlib, sys
loader = importlib.machinery.SourceFileLoader('evacuation_negotiator', %r)
spec = importlib.util.spec_from_loader(loader.name, loader)
n = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = n
loader.exec_module(n)
%s
raise SystemExit(n.main())
''' % (HELPER, source))
    path.chmod(0o755)
    return str(path)


def policy(case, model=p.MODEL, destinations=None, priority=1):
    return p.control(case.socket_path, {'action': 'set_cache_policy', 'model': model,
        'movable': True, 'priority': priority, 'gpu_uuids': destinations or GPUS})


def acquire(case, operation='move-1', **overrides):
    return p.control_raw(case.socket_path, {'action': 'acquire', 'owner': 'evacuation-test',
        'requested_mib': 1024, 'ttl': 30, 'gpu_uuids': [GPUS[0]],
        'justification': 'CPU fixture verifies lower priority cache evacuation',
        'expected_duration_seconds': 30, 'priority': 10,
        'evacuation_id': operation, **overrides})


def starts(case):
    return [e for e in p.events(case.event_log) if e['kind'] == 'start']


class EvacuationHTTPTests(unittest.TestCase):
    def test_explicit_cache_moves_before_pending_grant_and_retries_coalesce(self):
        with harness() as case:
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
            original = p.managed_lanes(case.status())[0]
            policy(case, destinations=[GPUS[1]])
            result = acquire(case)
            self.assertTrue(result['ok'], result)
            self.assertEqual(result['lease']['state'], 'pending')
            lanes = p.managed_lanes(case.status())
            self.assertEqual([(l['model'], l['gpu_uuids']) for l in lanes], [(p.MODEL, [GPUS[1]])])
            self.assertNotEqual(lanes[0]['id'], original['id'])
            self.assertEqual(result['evacuation']['state'], 'lease_pending')
            self.assertEqual(result['evacuation']['moves'][0]['source']['lane_id'], original['id'])
            again = acquire(case)
            self.assertTrue(again['ok'], again)
            self.assertEqual(again['lease']['token'], result['lease']['token'])
            self.assertEqual(len(starts(case)), 2)
            conflict = acquire(case, priority=11)
            self.assertFalse(conflict['ok'])
            ready = p.control(case.socket_path, {'action': 'ready', 'token': result['lease']['token']})
            self.assertEqual(ready['lease']['state'], 'active')
            public = case.status()['cache_residency']
            self.assertNotIn(result['lease']['token'], json.dumps(public))
            self.assertFalse(p.control_raw(case.socket_path, {'action': 'archive_evacuation', 'evacuation_id': 'move-1'})['ok'])
            p.control(case.socket_path, {'action': 'release', 'token': result['lease']['token']})
            archived = p.control(case.socket_path, {'action': 'archive_evacuation', 'evacuation_id': 'move-1'})
            self.assertEqual(archived['archived_evacuation']['state'], 'lease_pending')
            self.assertFalse(case.status()['cache_residency']['operations'])

    def test_undeclared_or_not_lower_priority_cache_is_never_evicted(self):
        for declared in (False, True):
            with self.subTest(declared=declared), harness() as case:
                self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
                original = p.managed_lanes(case.status())[0]['id']
                if declared:
                    policy(case, priority=10)
                result = acquire(case)
                self.assertFalse(result['ok'], result)
                self.assertEqual([l['id'] for l in p.managed_lanes(case.status())], [original])
                self.assertFalse(case.status()['leases'])
                self.assertFalse([e for e in p.events(case.event_log) if e['kind'] == 'stop'])

    def test_actual_busy_request_finishes_before_its_cache_moves(self):
        with harness() as case:
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
            policy(case, destinations=[GPUS[1]])
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                running = executor.submit(p.chat, case.proxy_port, p.MODEL, 'finish-once', .5,
                                          gpu_uuids=[GPUS[0]])
                p.wait_until(lambda: any(l['in_flight'] for l in p.managed_lanes(case.status())), 'actual request starts')
                moving = executor.submit(acquire, case)
                self.assertEqual(running.result(timeout=8)[0], 200)
                self.assertTrue(moving.result(timeout=8)['ok'])
            self.assertEqual([l['gpu_uuids'] for l in p.managed_lanes(case.status())], [[GPUS[1]]])
            self.assertEqual(case.status()['parallel_pool']['request_lifecycle']['cancelled_total'], 0)

    def test_destination_never_widens_hard_model_policy(self):
        with harness() as case:
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
            policy(case, destinations=[GPUS[1]])
            p.control(case.socket_path, {'action': 'set_model_gpus', 'model': p.MODEL, 'gpu_uuids': [GPUS[0]]})
            result = acquire(case)
            self.assertFalse(result['ok'], result)
            self.assertEqual(len(starts(case)), 1)
            self.assertFalse(case.status()['leases'])

    def test_two_distinct_caches_pack_on_one_eligible_destination(self):
        with harness() as case:
            for model in (p.MODEL, p.OTHER_MODEL):
                self.assertEqual(case.capacity(model, gpu_uuids=[GPUS[0]])[0], 200)
                policy(case, model=model, destinations=[GPUS[1]])
            result = acquire(case)
            self.assertTrue(result['ok'], result)
            self.assertEqual(len(result['evacuation']['moves']), 2)
            self.assertEqual({l['model'] for l in p.managed_lanes(case.status())}, {p.MODEL, p.OTHER_MODEL})
            self.assertEqual([l['gpu_uuids'] for l in p.managed_lanes(case.status())], [[GPUS[1]], [GPUS[1]]])

    def test_failed_prepare_preserves_active_external_lease(self):
        with harness() as case:
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
            lease = p.control(case.socket_path, {'action': 'acquire', 'owner': 'existing-owner',
                'requested_mib': 1, 'ttl': 30, 'gpu_uuids': [GPUS[0]],
                'justification': 'CPU fixture checks active resize ownership', 'expected_duration_seconds': 30})['lease']
            p.control(case.socket_path, {'action': 'ready', 'token': lease['token']})
            result = p.control_raw(case.socket_path, {'action': 'prepare', 'token': lease['token'],
                'priority': 10, 'evacuation_id': 'prepare-1'})
            self.assertFalse(result['ok'], result)
            self.assertEqual(case.status()['leases'][0]['state'], 'active')
            self.assertEqual(len(starts(case)), 1)

    def test_cancel_during_destination_validation_rolls_back_without_grant(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = fault_helper(directory, '''
original = n.Broker._cache_lane_identity
def identity(self, lane):
    result = original(self, lane)
    if lane.scope == ('GPU-large-1',):
        self.cancel_evacuation('move-1')
    return result
n.Broker._cache_lane_identity = identity
''')
            with harness(helper=wrapper) as case:
                self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
                original = p.managed_lanes(case.status())[0]['id']
                policy(case, destinations=[GPUS[1]])
                result = acquire(case)
                self.assertFalse(result['ok'], result)
                self.assertFalse(case.status()['leases'])
                self.assertEqual([l['id'] for l in p.managed_lanes(case.status())], [original])
                self.assertEqual(case.status()['cache_residency']['operations'][0]['state'], 'cancelled')

    def test_failed_source_exit_retains_both_reservations_then_restart_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = fault_helper(directory, '''
original_stop = n.Broker._stop_lanes
def stop(self, lanes, reason):
    if reason == 'lower priority cache evacuation':
        return lanes
    return original_stop(self, lanes, reason)
n.Broker._stop_lanes = stop
''')
            archive = pathlib.Path(directory) / 'interrupted-cache.json'
            with harness(helper=wrapper) as case:
                self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
                policy(case, destinations=[GPUS[1]])
                result = acquire(case)
                self.assertFalse(result['ok'], result)
                status = case.status()
                self.assertFalse(status['leases'])
                self.assertEqual(len(p.managed_lanes(status)), 2)
                self.assertTrue(all(l['reserved_mib'] > 0 for l in p.managed_lanes(status)))
                self.assertEqual(status['cache_residency']['operations'][0]['state'], 'restoration_pending')
                archive.write_bytes((pathlib.Path(case.temp_dir) / 'cache-residency.json').read_bytes())
                recovery = p.control_raw(case.socket_path, {'action': 'recover_evacuation', 'evacuation_id': 'move-1'})
                self.assertFalse(recovery['ok'], recovery)
                self.assertEqual(len(starts(case)), 2)
            restart = fault_helper(directory, 'n.CACHE_STATE_PATH = pathlib.Path(%r)' % str(archive))
            with harness(helper=restart) as fresh:
                status = fresh.status()
                self.assertEqual(status['cache_residency']['operations'][0]['state'], 'recovery_pending')
                recovered = p.control(fresh.socket_path, {'action': 'recover_evacuation', 'evacuation_id': 'move-1'})
                self.assertEqual(recovered['evacuation']['state'], 'rolled_back')
                self.assertEqual([l['gpu_uuids'] for l in p.managed_lanes(fresh.status())], [[GPUS[0]]])
                self.assertFalse(fresh.status()['leases'])

    def test_cache_policy_change_across_warm_await_cannot_grant(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = fault_helper(directory, '''
original = n.Broker._cache_lane_identity
def identity(self, lane):
    result = original(self, lane)
    if lane.scope == ('GPU-large-1',):
        self.set_cache_policy(lane.model, False, 100, [])
    return result
n.Broker._cache_lane_identity = identity
''')
            with harness(helper=wrapper) as case:
                self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
                policy(case, destinations=[GPUS[1]])
                self.assertFalse(acquire(case)['ok'])
                self.assertFalse(case.status()['leases'])
                self.assertEqual([l['gpu_uuids'] for l in p.managed_lanes(case.status())], [[GPUS[0]]])

    def test_relocated_cache_does_not_join_original_warm_certificate(self):
        with harness(auto_model_context=True) as case:
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
            proof = p.http_json(case.proxy_port, 'POST', '/.well-known/ollama-unify-gpu-negotiator/capacity',
                {'model': p.MODEL, 'parallel': 1, 'gpu_uuids': [GPUS[0]], 'warm_admission_proof': True})
            self.assertEqual(proof[0], 200, proof)
            policy(case, destinations=[GPUS[1]])
            result = acquire(case)
            self.assertTrue(result['ok'], result)
            rejected = p.http_json(case.proxy_port, 'POST', '/api/chat',
                {'model': p.MODEL, 'stream': False, 'mock_request_id': 'old-cert', 'options': {'num_ctx': 262144}},
                extra_headers={'X-Ollama-Unify-Warm-Admission': json.dumps(proof[1]['warm_admission']),
                    'X-Ollama-Unify-GPU-UUIDs': GPUS[0], 'X-Ollama-Unify-Logical-Request-Id': 'old-cert'})
            self.assertEqual(rejected[0], 503, rejected)
            self.assertEqual(rejected[1]['reason_code'], 'warm_preflight_required')
            self.assertFalse(rejected[1]['backend_started'])
            self.assertEqual(len(starts(case)), 2)

    def test_capacity_claim_excludes_leased_or_unregistered_destination(self):
        for foreign in (False, True):
            with self.subTest(foreign=foreign), harness() as case:
                self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
                policy(case, destinations=[GPUS[1]])
                if foreign:
                    p.write_compute_apps(case.compute_apps, [(os.getpid(), GPUS[1], 1024)])
                else:
                    p.control(case.socket_path, {'action': 'acquire', 'owner': 'peer-owner',
                        'requested_mib': 1, 'ttl': 30, 'gpu_uuids': [GPUS[1]],
                        'justification': 'CPU fixture protects another owner', 'expected_duration_seconds': 30})
                result = acquire(case)
                self.assertFalse(result['ok'], result)
                self.assertEqual(len(starts(case)), 1)
                self.assertEqual([l['gpu_uuids'] for l in p.managed_lanes(case.status())], [[GPUS[0]]])

    def test_bounded_consolidation_moves_off_target_cache_first(self):
        tags = [{'name': p.MODEL, 'size': 60 * 1024**3}, {'name': p.OTHER_MODEL, 'size': 30 * 1024**3}]
        with harness(tags=tags) as case:
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
            self.assertEqual(case.capacity(p.OTHER_MODEL, gpu_uuids=[GPUS[1]])[0], 200)
            policy(case, p.MODEL, [GPUS[1]])
            policy(case, p.OTHER_MODEL, [GPUS[2]])
            result = acquire(case)
            self.assertTrue(result['ok'], result)
            self.assertEqual([m['source']['model'] for m in result['evacuation']['moves']], [p.OTHER_MODEL, p.MODEL])
            self.assertEqual({l['model']: l['gpu_uuids'] for l in p.managed_lanes(case.status())},
                             {p.MODEL: [GPUS[1]], p.OTHER_MODEL: [GPUS[2]]})

    def test_no_destination_does_not_evict_protected_off_target_cache(self):
        tags = [{'name': p.MODEL, 'size': 60 * 1024**3}, {'name': p.OTHER_MODEL, 'size': 30 * 1024**3}]
        with harness(tags=tags) as case:
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
            self.assertEqual(case.capacity(p.OTHER_MODEL, gpu_uuids=[GPUS[1]])[0], 200)
            policy(case, p.MODEL, [GPUS[1]])
            self.assertFalse(acquire(case)['ok'])
            self.assertEqual(len(starts(case)), 2)
            self.assertFalse([e for e in p.events(case.event_log) if e['kind'] == 'stop'])
            self.assertFalse(case.status()['leases'])

    def test_queued_original_work_survives_protected_evacuation_refusal(self):
        with harness(max_servers=1) as case:
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
            policy(case, destinations=[GPUS[1]])
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                first = executor.submit(p.chat, case.proxy_port, p.MODEL, 'first-original', .7, gpu_uuids=[GPUS[0]])
                p.wait_until(lambda: case.status()['active_requests'] == 1, 'first actual request')
                second = executor.submit(p.chat, case.proxy_port, p.MODEL, 'second-original', 0, gpu_uuids=[GPUS[0]])
                p.wait_until(lambda: case.status()['parallel_pool']['queue']['depth'] == 1, 'second actual queued request')
                result = acquire(case)
                self.assertFalse(result['ok'], result)
                self.assertEqual(first.result(timeout=8)[0], 200)
                self.assertEqual(second.result(timeout=8)[0], 200)
            self.assertEqual(len(starts(case)), 1)
            self.assertFalse(case.status()['leases'])

    def test_successful_prepare_preserves_token_and_returns_pending(self):
        with harness() as case:
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
            lease = p.control(case.socket_path, {'action': 'acquire', 'owner': 'resize-owner',
                'requested_mib': 1, 'ttl': 30, 'gpu_uuids': [GPUS[0]],
                'justification': 'CPU checks exact active lease growth', 'expected_duration_seconds': 30})['lease']
            p.control(case.socket_path, {'action': 'ready', 'token': lease['token']})
            policy(case, destinations=[GPUS[1]])
            result = p.control(case.socket_path, {'action': 'prepare', 'token': lease['token'],
                'priority': 10, 'evacuation_id': 'resize-1'})
            self.assertEqual(result['lease']['token'], lease['token'])
            self.assertEqual(result['lease']['state'], 'pending')
            self.assertEqual([l['gpu_uuids'] for l in p.managed_lanes(case.status())], [[GPUS[1]]])
            ready = p.control(case.socket_path, {'action': 'ready', 'token': lease['token']})
            self.assertEqual(ready['lease']['state'], 'active')

    def test_existing_peer_group_is_not_repartitioned_or_retired(self):
        large = 'fixture-peer:latest'
        with harness(tags=[{'name': large, 'size': 100 * 1024**3}],
                     runner_vram_by_gpu={GPUS[0]: 48000, GPUS[1]: 57000}) as case:
            code, capacity, _ = case.capacity(large, gpu_uuids=GPUS[:2])
            self.assertEqual(code, 200, capacity)
            original = p.managed_lanes(case.status())[0]['id']
            policy(case, large, GPUS)
            result = acquire(case)
            self.assertFalse(result['ok'], result)
            self.assertEqual([l['id'] for l in p.managed_lanes(case.status())], [original])
            self.assertEqual(len(starts(case)), 1)
            self.assertFalse(case.status()['leases'])

    def test_startup_pipe_eof_cannot_execute_child_and_ack_preserves_pid(self):
        wrapper = next(ast.literal_eval(node.value) for node in ast.parse(pathlib.Path(HELPER).read_text()).body
                       if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CACHE_STARTUP_WRAPPER' for t in node.targets))
        with tempfile.TemporaryDirectory() as directory:
            marker = pathlib.Path(directory) / 'native-started'
            command = [sys.executable, '-c', "from pathlib import Path; import os; Path(%r).write_text(str(os.getpid()))" % str(marker)]
            for acknowledge in (False, True):
                read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
                try:
                    child = subprocess.Popen([sys.executable, '-c', wrapper, str(read_fd), *command],
                        pass_fds=(read_fd,), start_new_session=True)
                    before = pathlib.Path('/proc/%d/stat' % child.pid).read_text().rsplit(')', 1)[1].split()[19]
                    os.close(read_fd)
                    read_fd = None
                    if acknowledge:
                        os.write(write_fd, b'\x01')
                    os.close(write_fd)
                    write_fd = None
                    self.assertEqual(child.wait(timeout=5), 0 if acknowledge else 125)
                    if acknowledge:
                        self.assertEqual(marker.read_text(), str(child.pid))
                        self.assertTrue(before.isdigit())
                    else:
                        self.assertFalse(marker.exists())
                finally:
                    for fd in (read_fd, write_fd):
                        if fd is not None:
                            os.close(fd)

    def test_pid_journal_failure_cannot_start_unregistered_native_server(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = fault_helper(directory, '''
original_persist = n.Broker._persist_cache_state_locked
failed = False
def persist(self):
    global failed
    if not failed and any(op.get('destination_attempt') for op in self.evacuations.values()):
        failed = True
        raise OSError('CPU injected publication failure')
    return original_persist(self)
n.Broker._persist_cache_state_locked = persist
''')
            with harness(helper=wrapper) as case:
                self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
                policy(case, destinations=[GPUS[1]])
                result = acquire(case)
                self.assertFalse(result['ok'], result)
                self.assertEqual(len(starts(case)), 1)
                self.assertEqual([l['gpu_uuids'] for l in p.managed_lanes(case.status())], [[GPUS[0]]])
                self.assertFalse(case.status()['leases'])

    def test_recovery_excludes_its_own_unpublished_capacity_promise(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = pathlib.Path(directory) / 'recovery.json'
            with harness(tags=[{'name': p.MODEL, 'size': 60 * 1024**3}]) as case:
                self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
                # A real verified source plus a saved recovery reservation.
                # The extra promise represents the not-yet-published restored
                # copy, not a second workload consuming the same capacity.
                policy(case, destinations=[GPUS[1]])
                snapshot = p.managed_lanes(case.status())[0]
                journal = json.loads((pathlib.Path(case.temp_dir) / 'cache-residency.json').read_text())
                wrapper = fault_helper(directory, '''
original_identity = n.Broker._cache_lane_identity
def identity(self, lane):
    result = original_identity(self, lane)
    pathlib.Path(%r).write_text(__import__('json').dumps(result))
    raise n.CapacityError('CPU archives exact source proof')
n.Broker._cache_lane_identity = identity
''' % str(pathlib.Path(directory) / 'identity.json'))
            with harness(helper=wrapper, tags=[{'name': p.MODEL, 'size': 60 * 1024**3}]) as capture:
                self.assertEqual(capture.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
                policy(capture, destinations=[GPUS[1]])
                self.assertFalse(acquire(capture)['ok'])
                journal = json.loads((pathlib.Path(capture.temp_dir) / 'cache-residency.json').read_text())
                identity = json.loads((pathlib.Path(directory) / 'identity.json').read_text())
                operation = journal['operations']['move-1']
                operation['source_identities'] = {identity['lane_id']: identity}
                operation['state'] = 'restoration_pending'
                operation['destination_reservation'] = {GPUS[0]: identity['reserved_mib']}
                archive.write_text(json.dumps(journal))
            recovery = fault_helper(directory, 'n.CACHE_STATE_PATH = pathlib.Path(%r)' % str(archive))
            with harness(helper=recovery, tags=[{'name': p.MODEL, 'size': 60 * 1024**3}]) as fresh:
                result = p.control(fresh.socket_path, {'action': 'recover_evacuation', 'evacuation_id': 'move-1'})
                self.assertEqual(result['evacuation']['state'], 'rolled_back')
                self.assertEqual([l['gpu_uuids'] for l in p.managed_lanes(fresh.status())], [[GPUS[0]]])
                self.assertFalse(fresh.status()['leases'])

    def test_final_commit_reasserts_cancel_policy_and_exact_prepare_owner(self):
        for kind, change in [('acquire', 'cancel'), ('acquire', 'policy'),
                             ('prepare', 'cancel'), ('prepare', 'policy'), ('prepare', 'revoke')]:
            with self.subTest(kind=kind, change=change), tempfile.TemporaryDirectory() as directory:
                wrapper = fault_helper(directory, '''
original_unload = n.Broker._unload_base_models
def unload(self):
    result = original_unload(self)
    operation = self.evacuations.get('move-1')
    if operation and operation['state'] == 'source_retiring':
        if %r == 'cancel':
            self.cancel_evacuation('move-1')
        elif %r == 'policy':
            self.set_cache_policy(%r, True, 2, ['GPU-large-1'])
        else:
            self.revoke(operation['request']['token'], 'CPU exact owner revocation')
    return result
n.Broker._unload_base_models = unload
''' % (change, change, p.MODEL))
                with harness(helper=wrapper) as case:
                    self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
                    if kind == 'prepare':
                        lease = p.control(case.socket_path, {'action': 'acquire', 'owner': 'resize-owner',
                            'requested_mib': 1, 'ttl': 30, 'gpu_uuids': [GPUS[0]],
                            'justification': 'CPU tests final current lease ownership', 'expected_duration_seconds': 30})['lease']
                        p.control(case.socket_path, {'action': 'ready', 'token': lease['token']})
                    policy(case, destinations=[GPUS[1]])
                    result = (acquire(case) if kind == 'acquire' else p.control_raw(case.socket_path,
                        {'action': 'prepare', 'token': lease['token'], 'priority': 10, 'evacuation_id': 'move-1'}))
                    self.assertFalse(result['ok'], result)
                    status = case.status()
                    self.assertNotEqual(status['cache_residency']['operations'][0]['state'], 'lease_pending')
                    if kind == 'acquire':
                        self.assertFalse(status['leases'])
                    else:
                        self.assertEqual(status['leases'][0]['token'], lease['token'])
                        self.assertEqual(status['leases'][0]['state'], 'revoking' if change == 'revoke' else 'active')

    def test_rollback_requires_original_full_native_context_not_only_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = fault_helper(directory, '''
original_stop = n.Broker._stop_lanes
original_backend = n.backend_json_at
restoring = False
def stop(self, lanes, reason):
    global restoring
    result = original_stop(self, lanes, reason)
    if reason == 'lower priority cache evacuation' and not result:
        restoring = True
        self.cancel_evacuation('move-1')
    return result
def backend(host, port, method, path, *args, **kwargs):
    result = original_backend(host, port, method, path, *args, **kwargs)
    if restoring and method == 'GET' and path == '/api/ps':
        for row in result.get('models', []):
            row['context_length'] += 1
    return result
n.Broker._stop_lanes = stop
n.backend_json_at = backend
''')
            with harness(helper=wrapper) as case:
                self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
                policy(case, destinations=[GPUS[1]])
                self.assertFalse(acquire(case)['ok'])
                status = case.status()
                self.assertFalse(status['leases'])
                self.assertEqual(status['cache_residency']['operations'][0]['state'], 'restoration_pending')
                self.assertEqual([l['gpu_uuids'] for l in p.managed_lanes(status)], [[GPUS[1]]])

    def test_initial_journal_failure_removes_only_uncommitted_operation_then_same_id_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = fault_helper(directory, '''
original_persist = n.Broker._persist_cache_state_locked
failed = False
def persist(self):
    global failed
    if not failed and any(op['state'] == 'draining' for op in self.evacuations.values()):
        failed = True
        raise OSError('CPU initial journal write failure')
    return original_persist(self)
n.Broker._persist_cache_state_locked = persist
''')
            with harness(helper=wrapper) as case:
                self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
                policy(case, destinations=[GPUS[1]])
                first = acquire(case)
                self.assertFalse(first['ok'], first)
                self.assertFalse(case.status()['leases'])
                self.assertFalse(case.status()['cache_residency']['operations'])
                self.assertEqual(len(starts(case)), 1)
                second = acquire(case)
                self.assertTrue(second['ok'], second)
                self.assertEqual(second['lease']['state'], 'pending')
                self.assertEqual(len(starts(case)), 2)

    def test_ordinary_priority_metadata_does_not_enable_cache_evacuation(self):
        with harness() as case:
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
            original = p.managed_lanes(case.status())[0]['id']
            result = acquire(case, operation='')
            self.assertTrue(result['ok'], result)
            self.assertEqual(result['lease']['priority'], 10)
            self.assertEqual(result['lease']['evacuation_id'], '')
            self.assertEqual(len(starts(case)), 1)
            self.assertEqual(p.managed_lanes(case.status())[0]['id'], original)
            self.assertEqual(p.managed_lanes(case.status())[0]['gpu_uuids'], [GPUS[0]])
            self.assertFalse(case.status()['cache_residency']['operations'])

    def test_supported_cli_configures_and_requests_evacuating_lease(self):
        with harness() as case:
            self.assertEqual(case.capacity(p.MODEL, gpu_uuids=[GPUS[0]])[0], 200)
            env = {**os.environ, 'OLLAMA_UNIFY_CONFIG': '/nonexistent/evacuation-cli',
                   'OLLAMA_UNIFY_SOCKET': case.socket_path}
            configured = subprocess.run([HELPER, 'set-cache-policy', p.MODEL, '--movable', '--priority', '1', '--gpu', GPUS[1]],
                env=env, check=True, capture_output=True, text=True)
            self.assertTrue(json.loads(configured.stdout)['ok'])
            requested = subprocess.run([HELPER, 'acquire', '--owner', 'cli-owner', '--justification', 'CPU verifies supported CLI transaction',
                '--expected-duration', '30', '--vram-mib', '1024', '--gpu', GPUS[0], '--priority', '10', '--evacuation-id', 'cli-1'],
                env=env, check=True, capture_output=True, text=True)
            result = json.loads(requested.stdout)
            self.assertEqual(result['lease']['state'], 'pending')
            self.assertEqual(result['evacuation']['state'], 'lease_pending')


if __name__ == '__main__':
    unittest.main()
