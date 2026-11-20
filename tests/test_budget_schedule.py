"""Pure stdlib B321 schedule/identity checks; no project conftest or ML imports."""
import importlib.util,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('budget_train',ROOT/'scripts/budget_train.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class BudgetScheduleTests(unittest.TestCase):
    def test_current_batch_distinct_epoch_reproducible(self):
        a=m.Batches(list(range(19)),8,7);b=m.Batches(list(range(19)),8,7)
        aa=[a.next() for _ in range(20)];self.assertEqual(aa,[b.next() for _ in range(20)]);self.assertTrue(all(len(set(x))==8 for x in aa))
    def test_too_large_current_batch_denied(self):
        with self.assertRaises(ValueError):m.Batches([1,2],8,0)
    def test_exact_budget_and_first_stage_counts(self):
        self.assertEqual(667*15,10005);self.assertEqual(667*16+14*667*8*2,160080)
        self.assertEqual(667*16+14*667*8,85376);self.assertEqual(14*667*8,74704)
    def test_role_reservation_disjoint_and_insufficient_denied(self):
        import random
        a=m.Schedule.__new__(m.Schedule);a.reservoir=list(range(24));a.fifo=list(range(16,24))
        a.rng={n:random.Random(i) for i,n in enumerate(('recent','historical','replay','partition'))}
        replay,recent,history,eligible=a.reserve(list(range(24,32)),2)
        self.assertTrue(eligible);self.assertEqual(len(set(replay+recent+history)),12)
        a.reservoir=[1,2]
        with self.assertRaises(ValueError):a.reserve(list(range(24,32)),2)
    def test_current_batch_no_padding(self):
        batch=m.Batches(range(17),16,0)
        self.assertTrue(all(len(set(batch.next()))==16 for _ in range(100)))
if __name__=='__main__':unittest.main()
