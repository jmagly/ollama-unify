#!/usr/bin/env python3
"""CPU-only shared VRAM admission, accounting, lifecycle and HTTP regressions."""
import concurrent.futures
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
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


with mock.patch.dict(os.environ, {'OLLAMA_UNIFY_CONFIG':'/nonexistent/shared-lease-fixture'}):
    m = load('shared_lease_broker', HELPER)
p = load('shared_lease_pool', Path(__file__).with_name('test-negotiator-pool.py'))
GPUS = ['GPU-large-0', 'GPU-large-1', 'GPU-large-2']


class SharedBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.usage, self.managed = {}, {}
        self.overhead = 0
        self.patch = mock.patch.multiple(m, SELECTED_GPUS=GPUS, BACKEND_TYPE='cuda',
            LEASE_STATE_PATH=root/'leases.json', MODEL_POLICY_PATH=root/'policy.json',
            CACHE_STATE_PATH=root/'cache.json', ANON_MAX_DRAIN=0.01, ANON_POLL=0.001)
        self.patch.start()
        self.mocks = [mock.patch.object(m, 'gpu_snapshot', side_effect=self.devices),
            mock.patch.object(m, 'foreign_gpu_usage', side_effect=lambda **_:dict(self.usage)),
            mock.patch.object(m, 'exclusive_cuda_contexts', side_effect=lambda:{**self.usage,**self.managed}),
            mock.patch.object(m, 'running_models', return_value=[]),
            mock.patch.object(m, 'host_memory_snapshot', return_value={}),
            mock.patch.object(m, 'require_gpu_health')]
        for patch in self.mocks:patch.start()
        self.broker = m.Broker()

    def tearDown(self):
        for patch in reversed(self.mocks):patch.stop()
        self.patch.stop()
        self.temp.cleanup()

    def devices(self):
        return [dict(uuid=gpu, total_mib=100,
            used_mib=self.overhead+sum(v for k,v in {**self.usage,**self.managed}.items() if k.endswith('@'+gpu)),
            free_mib=100-self.overhead-sum(v for k,v in {**self.usage,**self.managed}.items() if k.endswith('@'+gpu)))
            for gpu in GPUS]

    def test_process_identity_scan_ignores_an_unrelated_exiting_process(self):
        fields=['S','1','222']+['0']*17
        fields[19]='123'
        def stat_file(path, **_):
            if path=='/proc/111/stat':raise ProcessLookupError('exited during read')
            return io.StringIO('222 (server) '+' '.join(fields))
        with mock.patch.object(m.os,'scandir',return_value=[types.SimpleNamespace(name='111'),
                types.SimpleNamespace(name='222')]),mock.patch('builtins.open',side_effect=stat_file):
            identity=m.process_group_identity(types.SimpleNamespace(pid=222,poll=lambda:None))
        self.assertEqual(identity,[{'pid':222,'start_time_ticks':123}])

    def acquire(self, amount, gpu=GPUS[0], **kwargs):
        return self.broker.acquire('fixture', amount, 60, [gpu], **kwargs)['lease']['token']

    def test_concurrent_pending_reservations_have_no_count_limit_and_no_overspend(self):
        def attempt(_):
            try:return self.acquire(10)
            except RuntimeError:return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            granted = list(executor.map(attempt, range(20)))
        self.assertEqual(sum(token is not None for token in granted), 10)
        self.assertEqual(sum(x.requested_mib for x in self.broker.leases.values()), 100)
        self.assertTrue(all(x.state=='pending' for x in self.broker.leases.values()))
        self.assertEqual(len(m.Broker().leases), 10)
        self.acquire(10, GPUS[1])  # Pending loaders do not lock unrelated cards.

    def test_driver_and_unattributed_usage_are_not_available_memory(self):
        self.overhead = 10
        self.acquire(45); self.acquire(45)
        with self.assertRaises(RuntimeError):self.acquire(1)

    def test_capacity_ceiling_survives_a_free_between_telemetry_samples(self):
        self.overhead=10
        first=self.acquire(40)
        self.usage['111@'+GPUS[0]]=20;self.broker.ready(first)
        # A later free does not let a stale process sample widen promises.
        stale=[dict(uuid=GPUS[0],total_mib=100,free_mib=100,used_mib=0)]
        self.assertEqual(self.broker._lease_capacity_devices(stale)[0]['free_mib'],50)
        self.assertEqual(m.Broker().leases[first].capacity_mib,90)

    def test_anonymous_bytes_do_not_hide_inside_a_promised_budget(self):
        self.usage['777@'+GPUS[0]]=20
        first=self.acquire(30);self.broker.ready(first)
        self.acquire(50)
        with self.assertRaises(RuntimeError):self.acquire(1)

    def test_resident_allocations_are_not_double_counted(self):
        first=self.acquire(40)
        self.usage['111@'+GPUS[0]]=40
        self.broker.ready(first)
        second=self.acquire(60)
        self.assertEqual(self.broker.leases[first].cuda_processes, ['111@'+GPUS[0]])
        self.assertIn(second,self.broker.leases)
        with self.assertRaises(RuntimeError):self.acquire(1)

    def test_only_unmaterialized_lane_and_external_promises_are_subtracted(self):
        first=self.acquire(40)
        self.usage['111@'+GPUS[0]]=10;self.broker.ready(first)
        self.managed['222@'+GPUS[0]]=10
        self.broker.lanes['managed']=types.SimpleNamespace(kind='managed',
            scope=(GPUS[0],), protected_scope=(GPUS[0],), cpu_offload=False,
            reserved_mib_by_gpu={GPUS[0]:20},reserved_mib=20)
        available=self.broker._lease_capacity_devices(self.devices())[0]
        self.assertEqual(available['free_mib'],40)
        self.acquire(40)
        self.assertEqual(self.broker._lease_capacity_devices(self.devices())[0]['free_mib'],0)

    def test_fit_does_not_drain_or_interrupt_an_unrelated_active_request(self):
        self.broker.active_requests=1
        self.acquire(20)
        self.assertEqual(self.broker.active_requests,1)
        self.assertFalse(self.broker.draining)

    def test_exclusive_multigpu_unknown_budgets_and_revocations_keep_their_scope(self):
        self.acquire(20)
        for kwargs in ({'exclusive':True},{'requested_gpu_uuids':GPUS[:2]}):
            with self.subTest(kwargs=kwargs),self.assertRaises(RuntimeError):
                if 'requested_gpu_uuids' in kwargs:self.broker.acquire('peer',20,60,GPUS[:2])
                else:self.acquire(20,**kwargs)
        self.acquire(0,GPUS[1])
        with self.assertRaises(RuntimeError):self.acquire(1,GPUS[1])
        private=self.acquire(10,GPUS[2],exclusive=True)
        with self.assertRaises(RuntimeError):self.acquire(1,GPUS[2])
        self.broker.leases[private].state='revoking'
        with self.assertRaises(RuntimeError):self.acquire(1,GPUS[2])

    def test_growth_reserves_new_total_and_preserves_other_owners(self):
        first=self.acquire(20);second=self.acquire(30)
        self.broker.ready(first);self.broker.ready(second)
        self.broker.prepare(first,requested_mib=60)
        self.assertEqual(self.broker.leases[first].requested_mib,60)
        self.assertEqual(self.broker.leases[second].state,'active')
        self.broker.ready(first)
        with self.assertRaises(RuntimeError):self.broker.prepare(first,requested_mib=80)
        self.assertEqual(self.broker.leases[first].requested_mib,60)
        self.assertEqual(self.broker.leases[first].state,'active')
        self.acquire(10)
        with self.assertRaises(RuntimeError):self.acquire(1)

    def test_failed_budget_save_rolls_back_without_losing_other_reservations(self):
        first=self.acquire(20);second=self.acquire(30)
        self.broker.ready(first);self.broker.ready(second)
        with mock.patch.object(self.broker,'_persist_leases_locked',side_effect=OSError('full')):
            with self.assertRaises(OSError):self.broker.prepare(first,requested_mib=40)
        self.assertEqual(self.broker.leases[first].requested_mib,20)
        self.assertEqual(self.broker.leases[first].state,'active')
        self.assertEqual(m.Broker().leases[second].requested_mib,30)

    def test_own_release_does_not_wait_for_a_later_peers_allocation(self):
        first=self.acquire(40)
        self.usage['111@'+GPUS[0]]=20;self.broker.ready(first)
        second=self.acquire(40)
        self.usage['222@'+GPUS[0]]=20;self.broker.ready(second)
        with self.assertRaises(TimeoutError):self.broker.release(first)
        del self.usage['111@'+GPUS[0]]
        self.broker.release(first)
        self.assertIn(second,self.broker.leases)
        self.assertEqual(self.usage,{'222@'+GPUS[0]:20})

    def test_pid_registration_disambiguates_parallel_loaders_without_new_allocation(self):
        first=self.acquire(30);second=self.acquire(30)
        self.usage={'111@'+GPUS[0]:20,'222@'+GPUS[0]:20}
        self.broker.ready(first)
        self.assertIsNone(self.broker.leases[first].cuda_processes)
        with mock.patch.object(m,'workload_contains_pid',side_effect=lambda root,pid:pid==111):
            self.broker.ready(first,workload_pid=os.getpid())
        with mock.patch.object(m,'workload_contains_pid',side_effect=lambda root,pid:pid==222):
            self.broker.ready(second,workload_pid=os.getpid())
        self.assertEqual(self.broker.leases[first].cuda_processes,['111@'+GPUS[0]])
        self.assertEqual(self.broker.leases[second].cuda_processes,['222@'+GPUS[0]])
        self.assertEqual(m.Broker().leases[first].cuda_processes,['111@'+GPUS[0]])

    def test_ambiguous_release_cannot_credit_peer_bytes_or_ignore_tiny_contexts(self):
        peer=self.acquire(30)
        self.usage={'222@'+GPUS[0]:20};self.broker.ready(peer)
        first=self.acquire(30);other=self.acquire(30)
        self.usage['111@'+GPUS[0]]=10
        self.broker.ready(first)
        self.assertIsNone(self.broker.leases[first].cuda_processes)
        for size in (10, 1, 0):
            self.usage['111@'+GPUS[0]]=size
            with self.assertRaises(TimeoutError):self.broker.release(first)
            self.assertIn(first,self.broker.leases)
        del self.usage['111@'+GPUS[0]]
        self.broker.release(first)
        self.assertIn(peer,self.broker.leases)
        self.assertIn(other,self.broker.leases)
        self.assertEqual(self.usage,{'222@'+GPUS[0]:20})

    def test_release_checks_contexts_created_after_readiness(self):
        first=self.acquire(30);self.broker.ready(first)
        peer=self.acquire(30);self.broker.ready(peer)
        self.assertEqual(self.broker.leases[first].cuda_processes,[])
        self.usage['111@'+GPUS[0]]=0
        with self.assertRaises(TimeoutError):self.broker.release(first)
        self.assertIn(first,self.broker.leases)
        self.usage.clear()
        self.broker.release(first)

    def test_malformed_or_missing_memory_telemetry_cannot_admit_new_work(self):
        with mock.patch.object(m,'foreign_gpu_usage',side_effect=RuntimeError('unavailable')):
            with self.assertRaises(RuntimeError):self.acquire(10)
        self.assertFalse(self.broker.leases)


class SharedHTTPTests(unittest.TestCase):
    def test_shared_owners_resize_and_release_without_evicting_fitting_lane(self):
        with p.PoolHarness(HELPER,FIXTURE_BIN) as case:
            tokens=[]
            for i in range(6):
                response=p.control(case.socket_path,{'action':'acquire','owner':f'service-{i}',
                    'requested_mib':4096,'ttl':60,'gpu_uuids':[GPUS[0]]})
                tokens.append(response['lease']['token'])
            for token in tokens:p.control(case.socket_path,{'action':'ready','token':token})
            response,capacity,_=case.capacity(p.MODEL,gpu_uuids=[GPUS[0]])
            self.assertEqual(response,200,capacity)
            lane_id=capacity['lanes'][0]['id']
            prepared=p.control(case.socket_path,{'action':'prepare','token':tokens[0],'requested_mib':8192})
            self.assertEqual(prepared['stopped_lanes'],[])
            p.control(case.socket_path,{'action':'ready','token':tokens[0]})
            p.control(case.socket_path,{'action':'release','token':tokens[0]})
            self.assertEqual(len(case.status()['leases']),5)
            self.assertIn(lane_id,[lane['id'] for lane in case.status()['parallel_pool']['lanes']])
            public=p.http_json(case.proxy_port,'GET','/.well-known/ollama-unify-gpu-negotiator')[1]
            self.assertTrue(all(lease['sharing']=='shared' for lease in public['active_leases']))
            self.assertNotIn(tokens[1],json.dumps(public))


if __name__=='__main__':unittest.main()
