import tempfile
from pathlib import Path
import unittest
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

from training.hope_w7_statistics import (image_counts,exact_bootstrap,pixel_bootstrap,
                                       continual_bootstrap,paired_interval)
from training.hope_w7_io import export_archive,restore_archive,destination


class W7StatisticsContracts(unittest.TestCase):
    def test_whole_image_counts_are_stratified_and_shared_across_uses(self):
        labels=np.array([0,1,0,1,1])
        a=image_counts(labels,2,40);b=image_counts(labels,2,40)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[:,labels==0].sum(1),2)
        np.testing.assert_array_equal(a[:,labels==1].sum(1),3)
    def test_exact_weighted_metrics_match_sklearn_including_ties_and_zero_weights(self):
        scores=np.array([.3,.3,.8,.1,.8]);labels=np.array([0,1,1,0,0])
        weights=np.array([[1,2,0,1,1],[0,3,2,0,2],[1,1,1,1,1]])
        auc,ap=exact_bootstrap(scores,labels,weights)
        for i,w in enumerate(weights):
            self.assertAlmostEqual(auc[i],roc_auc_score(labels,scores,sample_weight=w))
            self.assertAlmostEqual(ap[i],average_precision_score(labels,scores,sample_weight=w))
    def test_pixel_histogram_uses_image_weights_and_passes_exact_32_replicates(self):
        rng=np.random.default_rng(8)
        maps=rng.normal(size=(6,3,3));masks=np.zeros((6,3,3),dtype=np.uint8);masks[3:,:,1]=1
        counts=image_counts([0,0,0,1,1,1],0,40)
        actual,report=pixel_bootstrap(maps,masks,counts)
        expected,exact=pixel_bootstrap(maps,masks,counts,exact=True)
        self.assertEqual(report['validation_replicates'],32)
        self.assertLessEqual(np.max(np.abs(actual[:32]-expected[:32])),2e-4)
        expanded=np.repeat(counts[0],9)
        self.assertAlmostEqual(expected[0],average_precision_score(masks.reshape(-1),maps.reshape(-1),sample_weight=expanded))
    def test_constant_scores_handle_ties_without_division_by_zero(self):
        maps=np.ones((4,2,2));masks=np.zeros((4,2,2),dtype=np.uint8);masks[2:,:,0]=1
        counts=image_counts([0,0,1,1],0,40)
        actual,_=pixel_bootstrap(maps,masks,counts)
        np.testing.assert_allclose(actual,.25)
    def test_forgetting_and_bwt_preserve_negative_forgetting_and_covariance(self):
        matrix=np.full((3,3,2),np.nan)
        matrix[0,0]=[.5,.6];matrix[1,0]=[.6,.7];matrix[1,1]=[.7,.8]
        matrix[2,0]=[.8,.9];matrix[2,1]=[.9,.95];matrix[2,2]=[.7,.8]
        result=continual_bootstrap(matrix)
        np.testing.assert_allclose(result['forgetting'],[-.2,-.175])
        np.testing.assert_allclose(result['bwt'],[.25,.225])
        self.assertEqual(paired_interval(np.arange(2000))['percentiles'],[1.25,98.75])
    def test_zip_and_extracted_restore_verify_bytes_and_reject_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'source').mkdir();(root/'source/a.json').write_text('{"value":1}')
            result=export_archive(root/'source',root/'evidence.zip')
            restore_archive(root/'evidence.zip',root/'dest',schema='hope_w7_evidence_v1')
            self.assertEqual((root/'dest/a.json').read_text(),'{"value":1}')
            (root/'dest/a.json').write_text('wrong')
            with self.assertRaisesRegex(ValueError,'overwrite'):
                restore_archive(root/'evidence.zip',root/'dest',schema='hope_w7_evidence_v1')
            for name in ('../escape','/escape','a\\b','C:/escape'):
                with self.assertRaises(ValueError):destination(root,name)


if __name__=='__main__':unittest.main()
