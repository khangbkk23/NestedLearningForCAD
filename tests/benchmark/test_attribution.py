import copy
import unittest
import numpy as np
from hope_cad.benchmark.attribution import Predictors,strict_json,metrics,joined_rows,IDS


class FixtureArchive:
    arm='NESTED_seed0';legacy=376
    def __init__(self,late_target=0.):
        self.used={};self.rows=[dict(relative_path=f'bottle/train/good/{i:04d}.png',task_id=0,task_name='bottle') for i in range(3599)]
        self.events=[];pending=None;label=0
        for i,row in enumerate(self.rows):
            credit=None
            if pending:
                label+=1;y=.1 if i<3100 else late_target
                credit=dict(decision_identity=pending['identity'],decision_cursor=i-1,c=4.,gamma=0.,accepted=pending['accepted'],prediction=pending['prediction'],decision_d=1.,
                    reward_observed_at=i,Jminus=3+4*y,Jplus=3.,target=y,next_minus=2.5+4*y,next_plus=2.5,anchor_minus=.5,anchor_plus=.5)
            e=dict(cursor=i,identity=row['relative_path'],action='no_change',credit=credit)
            if 13<=i<3224:
                accepted=not (45<=i<1980)
                e.update(action='accept' if accepted else 'reject',accepted=accepted,prediction=.1 if accepted else -.1,
                    gamma=0.,c=4.,decision_d=1.,warmup=label<32,current_minus=1.,current_plus=1.,anchor_minus=.5,anchor_plus=.5,changed_rows=2,proposal=dict(replaced=2))
                pending=e
            else:pending=None
            self.events.append(e)
    def read(self,name):
        self.used[name]='fixture'
        if name.endswith('manifest_train.json'):return dict(online=self.rows)
        if name.endswith('train_complete.json'):return dict(images=3599,counters=dict(censored=0))
        return copy.deepcopy(self.events[int(name.rsplit('/',1)[1][:-5])])


class AttributionContracts(unittest.TestCase):
    def test_strict_json_nonfinite_duplicate_keys(self):
        for raw in ('{"a":1,"a":2}','{"x":NaN}','{"x":1e400}'):
            with self.assertRaises(ValueError):strict_json(raw)
    def test_fallback_bias_and_sum_ridge_oracle(self):
        p=Predictors()
        for i in range(32):
            e=dict(cursor=i,prediction=.2,gamma=i*.01);pred,x=p.predict(e)
            self.assertEqual(pred['OBS-MEAN'],e['gamma'])
            if i:self.assertNotEqual(pred['OBS-NLMS-BIAS'],e['gamma'])
            p.update(e,dict(target=.4+i*.02),pred,x)
        e=dict(cursor=32,prediction=.2,gamma=.32);pred,x=p.predict(e)
        X=np.array([[1,i*.01] for i in range(32)]);y=np.array([.4+i*.02 for i in range(32)])
        beta=np.linalg.solve(X.T@X+np.diag([0,1e-6]),X.T@y)
        self.assertAlmostEqual(pred['OBS-AFFINE-GAMMA'],float(x[2]@beta),12)
    def test_full_trace_future_trap_noop_credit_and_same_cursor_update(self):
        a=list(joined_rows(FixtureArchive(.2)));b=list(joined_rows(FixtureArchive(-.2)))
        self.assertEqual(len(a),3211);self.assertEqual(sum(r['evaluate'] for r in a),3179)
        self.assertEqual([r['predictions'] for r in a[:100]],[r['predictions'] for r in b[:100]])
        self.assertGreater(a[1]['predictions']['OBS-NLMS-BIAS'],0) # previous label resolved before new decision
        self.assertEqual(a[-1]['credit']['reward_observed_at'],3224) # no-change event still resolves label
        f=FixtureArchive();f.events[15]['credit']['decision_cursor']=11
        with self.assertRaises(ValueError):list(joined_rows(f))
    def test_tie_zero_and_empty_metrics(self):
        self.assertIsNone(metrics([],IDS[0])['MSE'])
        row=dict(predictions={IDS[0]:0.},credit=dict(target=0.,c=1.,Jminus=1.,Jplus=1.),decision=dict(gamma=0.),cursor=0)
        m=metrics([row],IDS[0]);self.assertEqual(m['accept_fraction'],1.);self.assertIsNone(m['sign_agreement'])
        self.assertEqual(m['regret_sum'],0.)
