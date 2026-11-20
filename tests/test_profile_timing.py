"""Stdlib resource timing tests; synthetic clock and no Torch import."""
import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('profile_timing',Path(__file__).resolve().parents[1]/'scripts/profile_timing.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ProfileTimingTests(unittest.TestCase):
    def test_cold_start_recorded_once_and_steady_outlier_kept(self):
        now=[0.];durations=iter((99.,1.,7.,2.));sync=[]
        def fn():now[0]+=next(durations)
        got=m.warmed_timing(fn,lambda:sync.append(now[0]),clock=lambda:now[0])
        self.assertEqual(got['warmup_seconds'],[99.])
        self.assertEqual(got['samples_seconds'],[1.,7.,2.])
        self.assertEqual(got['maximum_seconds'],7.)
        self.assertEqual(len(sync),8)

    def test_invalid_timings_fail_closed(self):
        for bad in (0.,-1.,float('nan'),float('inf')):
            times=iter((0.,bad))
            with self.assertRaises(ValueError):m.warmed_timing(lambda:None,lambda:None,clock=lambda:next(times))

    def test_warmup_errors_are_not_hidden(self):
        def bad():raise RuntimeError('synthetic failure')
        with self.assertRaisesRegex(RuntimeError,'synthetic failure'):m.warmed_timing(bad,lambda:None)

    def test_invalid_counts_reject_before_execution(self):
        for k,v in (('warmups',0),('repeats',0),('warmups',True),('repeats',1.5)):
            with self.assertRaises(ValueError):m.warmed_timing(lambda:self.fail('must not execute'),lambda:None,**{k:v})

if __name__=='__main__':unittest.main()
