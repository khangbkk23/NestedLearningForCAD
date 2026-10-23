import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hope_cad.runtime.measurement import Measurement,timed
from hope_cad.benchmark.stream import _state_equal
from tests.models.test_consolidation import actor
from tests.models.test_session import stream


class MeasurementContracts(unittest.TestCase):
    def test_nested_host_accounting_and_failed_phase_receipt(self):
        with tempfile.TemporaryDirectory() as t:
            m=Measurement(Path(t),'one','cpu')
            @timed('inner')
            def inner():return 4
            @timed('outer')
            def operation():
                return inner()
            with patch('hope_cad.runtime.measurement.time.monotonic',side_effect=[0.,1.,2.,5.,7.,10.]):
                with m.phase('train','H32_EXACT_seed0'):self.assertEqual(operation(),4)
            row=m.value['phases'][0]
            self.assertEqual(row['wall_seconds'],10.)
            self.assertEqual(row['components']['outer']['inclusive_host_seconds'],6.)
            self.assertEqual(row['components']['outer']['exclusive_host_seconds'],3.)
            self.assertEqual(row['components']['inner']['exclusive_host_seconds'],3.)
            with self.assertRaises(ValueError):
                with m.phase('evaluation'):raise ValueError('fixture')
            self.assertEqual(json.loads(m.path.read_text())['phases'][-1]['status'],'failed')
            self.assertEqual(operation(),4) # Context reset even on failure; no recording outside a phase.
            self.assertEqual(row['components']['outer']['calls'],1)
            with patch('hope_cad.runtime.measurement.resource_usage',side_effect=RuntimeError('fixture')):
                with self.assertRaises(RuntimeError):
                    with m.phase('unavailable'):pass
            self.assertEqual(json.loads(m.path.read_text())['phases'][-1]['status'],'measurement_failed')

    def test_gpu_peaks_reset_for_each_phase_without_using_real_cuda(self):
        with tempfile.TemporaryDirectory() as t,\
             patch('hope_cad.runtime.measurement.torch.cuda.synchronize') as sync,\
             patch('hope_cad.runtime.measurement.torch.cuda.reset_peak_memory_stats') as reset,\
             patch('hope_cad.runtime.measurement.torch.cuda.memory_allocated',return_value=10),\
             patch('hope_cad.runtime.measurement.torch.cuda.memory_reserved',return_value=20),\
             patch('hope_cad.runtime.measurement.resource_usage',return_value=dict(peak_allocated_bytes=30)):
            m=Measurement(Path(t),'fixture','cuda:0')
            with m.phase('train','CADIC_seed0'):pass
            with m.phase('train','H32_EXACT_seed0'):pass
            self.assertEqual(reset.call_count,2);self.assertEqual(sync.call_count,4)
            self.assertEqual([r['job'] for r in m.value['phases']],['CADIC_seed0','H32_EXACT_seed0'])
            self.assertEqual(m.value['phases'][0]['gpu_baseline_bytes'],dict(allocated=10,reserved=20))

    def test_instrumentation_does_not_change_trajectory_or_rng(self):
        import torch,random
        a,b=actor(),actor();xs=stream(48)
        reference=[a.observe(x,str(i)) for i,x in enumerate(xs)]
        with tempfile.TemporaryDirectory() as t:
            meter=Measurement(Path(t),'attempt','cpu')
            before=(random.getstate(),torch.get_rng_state().clone())
            with meter.phase('train','H32_EXACT_seed0'):
                measured=[b.observe(x,str(i)) for i,x in enumerate(xs)]
            self.assertEqual(reference,measured);self.assertTrue(_state_equal(a.state_dict(),b.state_dict()))
            self.assertEqual(before[0],random.getstate());self.assertTrue(torch.equal(before[1],torch.get_rng_state()))
            counters=meter.value['phases'][0]['components']
            for key in ('proposal_update','anchor_risk','learner_fast_and_slow','functional_unroll_and_queries','slow_backward_and_optimizer','bank_hash'):
                self.assertGreater(counters[key]['calls'],0)
