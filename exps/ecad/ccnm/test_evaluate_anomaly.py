"""Necessary label-free synthetic checks, no cost benchmark or dataset access."""
import unittest
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
from exps.ecad.r1.distance import configure
from .evaluate_anomaly import ranking_metrics,synthetic_gate,admitted_rows


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure();torch.set_num_threads(2)

    def test_exact_ties_across_scan_boundaries(self):
        s=torch.tensor([3.,3.,2.,2.,2.,1.,0.,0.])
        y=torch.tensor([0,1,0,1,1,0,1,0],dtype=torch.bool)
        for chunk in (1,2,3,100):
            r=ranking_metrics(s,y,scan=chunk)
            self.assertAlmostEqual(r['ap'],average_precision_score(y,s),places=14)
            self.assertAlmostEqual(r['auroc'],roc_auc_score(y,s),places=14)

    def test_degenerate_or_nonfinite_rejected(self):
        with self.assertRaises(ValueError):ranking_metrics(torch.ones(3),torch.zeros(3,dtype=torch.bool))
        with self.assertRaises(ValueError):ranking_metrics(torch.tensor([float('nan'),0.]),torch.tensor([0,1]))

    def test_frozen_scorers_route5_oracle_and_native_geometry(self):
        self.assertEqual(synthetic_gate('cpu')['status'],'PASS')

    def test_whitelist_rejects_unknown_partition(self):
        with self.assertRaises(ValueError):admitted_rows({'evaluations':[]})


if __name__=='__main__':unittest.main(verbosity=2)
