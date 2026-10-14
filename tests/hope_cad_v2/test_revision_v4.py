"""Independent math, identity and data-access regressions for the approved revision."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
import torch
from torch.func import functional_call
from models.hope_cad_v2.visual_memory import VisualMemoryState, scan_chunked, stability_matched_step
from models.hope_cad_v2.reference import scan_reference
from models.hope_cad_v2.cms import masked_teacher_loss
from models.hope_cad_v2.predictor import HOPEFeaturePredictor, HOPEFeaturePredictorV4
from models.hope_cad_v2.visual_block import VisualBlockConfig
from models.hope_cad_v2.state import tensor_digest
from training.hope_outer_learning_v4 import (make_schedule, episode_loss, unroll_support,
    short_validation_components, validation_loss, fit, SHORT_STEPS, export_initialization)
from training.hope_initialization_v4 import validate_initialization
from training.hope_cms_stress_v4 import run_stress_gate
from training.hope_cms_session_v2 import CMSSession
from scripts.hope_cad_v2.synthetic import feature_fixture
from scripts.hope_cad_v2.import_frozen_cache_v4 import import_cache

CONFIG=VisualBlockConfig(4,3,4,2,4,4)


class RevisionContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23); torch.set_num_threads(2)
        self.model=HOPEFeaturePredictorV4(CONFIG,beta=1e-3)
        self.data=feature_fixture(CONFIG)
        self.schedule=make_schedule(self.data,0,steps=2)

    def test_row_retention_oracle_and_all_five_maps_gradients(self):
        z=torch.randn(1,7,3,dtype=torch.float64,requires_grad=True)
        q=torch.randn_like(z,requires_grad=True)
        maps=[(torch.randn(1,3,3,dtype=torch.float64)*.04).requires_grad_() for _ in range(3)]
        maps += [(torch.randn(1,1,3,dtype=torch.float64)*.05).requires_grad_() for _ in range(2)]
        initial=VisualMemoryState(*maps)
        for chunk in (1,3,7,20):
            actual=scan_chunked(z,q,initial,chunk,row_retention=.9)
            oracle=scan_reference(z,q,initial,chunk,row_retention=.9)
            for a,b in zip((actual.output,*actual.final_state.tensors()),(oracle.output,*oracle.final_state.tensors())):
                torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-7)
            variables=(z,q,*maps)
            def gradients(result):
                value=result.output.square().sum()+sum(t.square().sum() for t in result.final_state.tensors())
                first=torch.autograd.grad(value,variables,create_graph=True,retain_graph=True)
                second=torch.autograd.grad(sum(g.square().sum() for g in first),variables,retain_graph=True)
                return first,second
            a,b=gradients(actual),gradients(oracle)
            for groups in zip(a,b):
                for x,y in zip(*groups):
                    torch.testing.assert_close(x,y,atol=1e-9,rtol=1e-7)

    def test_route_specific_retention_and_partial_tail(self):
        from models.hope_cad_v2.visual_memory import FourScanVisualMemory
        memory=FourScanVisualMemory(4,2,(2,3),row_retention=.9).double()
        trace=memory.inspect(torch.randn(1,6,4,dtype=torch.float64))
        for names,chunks in trace.scan_groups:
            length=3 if names[0].startswith('row') else 2
            torch.testing.assert_close(chunks[0].alpha,torch.full_like(chunks[0].alpha,.9**(1/length)))
        z=torch.zeros(1,5,2,dtype=torch.float64)
        initial=VisualMemoryState(*(torch.zeros(1,2,2,dtype=torch.float64) for _ in range(3)),
                                 *(torch.zeros(1,1,2,dtype=torch.float64) for _ in range(2)))
        result=scan_chunked(z,z,initial,3,capture=True,row_retention=.9)
        self.assertEqual(result.chunks[-1].alpha.shape[-1],2)
        self.assertAlmostEqual(result.chunks[-1].alpha[0,0].item(),.9**(1/3))

    def test_retention_finite_zero_saturation_and_near_floor_derivatives(self):
        for value in (0.,.9**(1/28),1.):
            key=torch.tensor([[1.01e-10,0.]],requires_grad=True)
            delta=torch.zeros_like(key,requires_grad=True)
            eta=torch.ones(1,requires_grad=True)
            alpha=torch.tensor([value],requires_grad=True)
            step=stability_matched_step(key,delta,eta,alpha,higher_order_safe=True)
            legacy=stability_matched_step(key,delta,eta,alpha)
            self.assertTrue(torch.equal(step.executed,legacy.executed))
            first=torch.autograd.grad(step.executed.sum(),(key,delta,eta,alpha),create_graph=True)
            # Fixed cotangent, not a squared first-gradient norm: at alpha=0
            # the cap tie can have a representable 1e20 derivative whose square
            # overflows FP32 before differentiating (a different test problem).
            second=torch.autograd.grad(sum(g.sum() for g in first),(key,delta,eta,alpha),allow_unused=True)
            self.assertTrue(all(torch.isfinite(g).all() for g in (*first,*[g for g in second if g is not None])))

    def test_full_grid_v4_float32_float64_parity_and_backward(self):
        from models.hope_cad_v2.visual_block import HOPECADVisualBlock
        model=HOPECADVisualBlock(row_retention=.9)
        reference=copy.deepcopy(model).double()
        x=torch.randn(1,784,768)
        actual=model(x)
        expected=reference(x.double())
        torch.testing.assert_close(actual.double(),expected,atol=1e-5,rtol=1e-4)
        actual.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_v3_defaults_and_cross_revision_weight_rejection(self):
        old=HOPEFeaturePredictor(CONFIG)
        self.assertEqual(old.cms.objective,'intermediate')
        self.assertIsNone(old.visual.memory.row_retention)
        self.assertEqual(self.model.cms.objective,'joint_final')
        with self.assertRaises(ValueError): self.model.load_state_dict(old.state_dict())
        with self.assertRaises(ValueError): old.load_state_dict(self.model.state_dict())
        with self.assertRaises(ValueError): validate_initialization({'schema':'hope_deployment_initialization_v3'}, {})

    def test_joint_final_partials_mean_accumulator_pre_event_16(self):
        model=self.model.double(); cms=model.cms
        x=torch.randn(1,4,3,dtype=torch.float64)
        target=torch.randn(1,4,4,dtype=torch.float64)
        mask=torch.tensor([True,False,True,False])
        state=cms.initial_state(meta=True)
        for t in range(1,17):
            # Independent numerical oracle: local partials hold all other
            # current dictionaries fixed, regardless of their meta ancestry.
            isolated=tuple({k:v.detach().clone().requires_grad_() for k,v in w.items()} for w in state.weights)
            h1=functional_call(cms.levels[0],isolated[0],(x,))
            h2=functional_call(cms.levels[1],isolated[1],(h1,))
            loss=masked_teacher_loss(model.decoder(h2),target,mask)
            expected=[]
            for i,w in enumerate(state.weights):
                grads=torch.autograd.grad(loss,tuple(isolated[i].values()),retain_graph=True)
                acc={k:state.accumulators[i][k]+g for k,g in zip(w,grads)}
                expected.append({k:v-cms.rates[i]*acc[k]/(state.pending[i]+1)
                                 if t%cms.periods[i]==0 else v for k,v in w.items()})
            result=cms.event(x,target,mask,model.decoder,state,meta=True)
            for actual,wanted in zip(result.state.weights,expected):
                for k in actual: torch.testing.assert_close(actual[k],wanted[k],atol=1e-12,rtol=1e-12)
            state=result.state
        self.assertEqual(state.commits,(16,2))

    def test_joint_functional_deployment_parity_and_ownership(self):
        model=self.model
        x,target,mask=self.data.get(self.data.ids('meta_support')[0],0,'meta_support')
        with torch.no_grad(): encoded=model.encode(x)
        a,b=model.cms.initial_state(meta=True),model.cms.initial_state()
        before=tensor_digest((dict(model.named_parameters()),))
        for _ in range(16):
            a=model.cms.event(encoded,target,mask,model.decoder,a,meta=True).state
            b=model.cms.event(encoded,target,mask,model.decoder,b).state
        for w1,w2 in zip(a.weights,b.weights):
            for k in w1: torch.testing.assert_close(w1[k],w2[k],atol=1e-5,rtol=1e-4)
        self.assertTrue(all(v.grad_fn is None for group in b.weights for v in group.values()))
        self.assertEqual(before,tensor_digest((dict(model.named_parameters()),)))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_warm_objective_matches_two_readonly_queries_and_full_gradients(self):
        episode=self.schedule['episodes'][0]
        actual=episode_loss(self.model,self.data,episode)
        fresh=feature_fixture(CONFIG)
        state0=self.model.cms.initial_state(meta=True)
        state16=unroll_support(self.model,fresh,episode['support'],meta=True)
        x,target,mask=fresh.get(episode['query'],episode['query_view'],'meta_query')
        expected=(masked_teacher_loss(self.model.predict(x,state0),target,mask)+
                  masked_teacher_loss(self.model.predict(x,state16),target,mask))/2
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        actual.backward()
        for name,p in self.model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad,name); self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertEqual(self.data.reads['meta_support'],16)
        self.assertEqual(self.data.reads['meta_query'],1)
        self.assertEqual(self.data.reads['normal_probe'],0)
        self.assertEqual(state0.clock,0); self.assertEqual(state16.clock,16)

    def test_exact16_warm_meta_gradient_matches_finite_difference(self):
        # Double precision synthetic provider; unit teacher and role boundary retained.
        model=self.model.double()
        base=self.data.provider
        self.data.provider=lambda identity,view: tuple(t.double() if t.dtype!=torch.bool else t for t in base(identity,view))
        parameter=model.cms.levels[0].mlp[0].weight
        def loss(): return episode_loss(model,self.data,self.schedule['episodes'][0])
        analytic=torch.autograd.grad(loss(),parameter)[0][0,0].item()
        old=parameter[0,0].item(); eps=1e-5
        with torch.no_grad(): parameter[0,0]=old+eps
        plus=loss().item()
        with torch.no_grad(): parameter[0,0]=old-eps
        minus=loss().item()
        with torch.no_grad(): parameter[0,0]=old
        self.assertAlmostEqual(analytic,(plus-minus)/(2*eps),delta=3e-7)

    def test_joint_validation_components_and_static_17_image_privileges(self):
        measured=short_validation_components(self.model,self.data,self.schedule['validation_support'])
        self.assertEqual(measured['loss'],(measured['loss_at_0']+measured['loss_at_16'])/2)
        self.assertEqual(measured['loss_at_0'],validation_loss(self.model,self.data,self.model.cms.initial_state()))
        self.assertEqual(self.data.reads['normal_probe'],0)
        static=HOPEFeaturePredictorV4(CONFIG,'C1')
        fresh=feature_fixture(CONFIG)
        episode_loss(static,fresh,self.schedule['episodes'][0]).backward()
        self.assertEqual(fresh.reads['meta_support'],16); self.assertEqual(fresh.reads['meta_query'],1)
        result=short_validation_components(static,fresh,self.schedule['validation_support'])
        self.assertEqual(result['loss_at_0'],result['loss_at_16'])

    def test_v4_fit_optimizer_rng_exact_resume_and_recipe_rejection(self):
        payloads=[]
        full=fit(self.model,self.data,self.schedule,max_steps=2,checkpoints=(0,1,2),smoke=True,checkpoint_callback=payloads.append)
        self.assertEqual(full['status'],'fit_complete',full['first_failure'])
        resumed_model=HOPEFeaturePredictorV4(CONFIG)
        resumed=fit(resumed_model,feature_fixture(CONFIG),self.schedule,max_steps=2,checkpoints=(0,1,2),smoke=True,resume=payloads[1])
        for key in ('training','short_losses','feature_reads'): self.assertEqual(full[key],resumed[key])
        self.assertEqual(tensor_digest((dict(self.model.named_parameters()),)),tensor_digest((dict(resumed_model.named_parameters()),)))
        self.assertEqual(payloads[1]['schema'],'hope_outer_checkpoint_v4')
        self.assertEqual(SHORT_STEPS,(0,25,50,100,200,300,400))
        with self.assertRaises(ValueError): fit(self.model,self.data,self.schedule,max_steps=200)
        with self.assertRaises(ValueError): fit(HOPEFeaturePredictor(CONFIG),self.data,self.schedule,smoke=True,max_steps=0,checkpoints=(0,))

    def test_export_schema_and_v3_session_restore_rejection(self):
        initial=export_initialization(self.model,{k:'fixture' for k in ('roles_digest','schedule_digest','source_sha256','spec_sha256','feature_identity')})
        self.assertEqual(initial['schema'],'hope_deployment_initialization_v4')
        self.assertEqual(initial['selection_rule_id'],'prefix_warm_joint_stress_v4')
        self.assertEqual(initial['cms']['clock'],0)
        old=CMSSession(HOPEFeaturePredictor(CONFIG),{'same':'identity'})
        new=CMSSession(self.model,{'same':'identity'})
        with self.assertRaises(ValueError): new.restore(old.checkpoint())

    def test_parent_import_copies_only_calibration_cache_and_checks_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); parent=root/'old.zip'; payload=b'frozen-only'
            manifest={'schema':'hope_w1_w6_evidence_v3','files':{'prefix_features.pt':hashlib.sha256(payload).hexdigest()}}
            with zipfile.ZipFile(parent,'w') as z:
                z.writestr('artifact_manifest.json',json.dumps(manifest)); z.writestr('prefix_features.pt',payload)
                z.writestr('selection/M/deployment_initialization.pt',b'forbidden'); z.writestr('probe_features.pt',b'forbidden')
            receipt=import_cache(parent,root/'cache.pt',root/'receipt.json')
            self.assertEqual((root/'cache.pt').read_bytes(),payload)
            self.assertFalse(receipt['learned_weights_imported']); self.assertFalse(receipt['probe_imported'])
            with self.assertRaises(ValueError): import_cache(parent,root/'cache.pt',root/'receipt.json')
            with zipfile.ZipFile(root/'bad.zip','w') as z:
                z.writestr('artifact_manifest.json',json.dumps(manifest)); z.writestr('prefix_features.pt',b'corrupt')
            with self.assertRaises(ValueError): import_cache(root/'bad.zip',root/'bad.pt',root/'bad.json')
            self.assertFalse((root/'bad.pt').exists())

    def test_stress_constants_unchanged_and_full_1024_joint_clock(self):
        from training.hope_cms_stress_v3 import CONSTANTS as old
        from training.hope_cms_stress_v4 import CONSTANTS as new
        self.assertEqual(old,new)
        model=HOPEFeaturePredictorV4(CONFIG,beta=1e-4)
        gate=run_stress_gate(model,self.data)
        self.assertTrue(gate['passed'],gate['first_failure'])
        self.assertEqual(gate['events_completed'],1024); self.assertEqual(gate['commits'],[1024,128])
        self.assertEqual(gate['selection_rule_id'],'prefix_warm_joint_stress_v4')

    def test_selection_filters_without_gate_ranking_and_c4_inherits_beta(self):
        from training.hope_selection_v4 import search
        calls=[]
        def factory(method,beta,selected):
            calls.append((method,beta,selected))
            return HOPEFeaturePredictorV4(CONFIG,method,beta,selected)
        def fitted(model,data,schedule,**kwargs):
            score={1e-4:.2,1e-3:.3,1e-2:.1}[model.cms.rates[0]]
            return dict(status='fit_complete',method=model.dynamics.method,beta=model.cms.rates[0],
                        role_digest=data.manifest['digest'],outer_lr=kwargs['outer_lr'],
                        model_metadata=model.metadata(),
                        shortlist=dict(step=300,loss=score,loss_at_0=score,loss_at_16=score,
                                       weights=copy.deepcopy(model.state_dict())))
        with patch('training.hope_selection_v4.fit',side_effect=fitted), \
             patch('training.hope_selection_v4.run_stress_gate',side_effect=[{'passed':True},{'passed':True},{'passed':False}]) as gate:
            selected=search(factory,self.data,self.schedule,smoke=True)
            self.assertEqual(selected['selected']['beta'],1e-4)
            self.assertEqual(gate.call_count,3)
            self.assertEqual(selected['selection_rule_id'],'prefix_warm_joint_stress_v4')
        calls.clear()
        with patch('training.hope_selection_v4.fit',side_effect=fitted), \
             patch('training.hope_selection_v4.run_stress_gate',return_value={'passed':True}) as gate:
            c4=search(factory,self.data,self.schedule,method='C4',selected_m=selected,smoke=True)
            self.assertEqual(calls,[('C4',1e-4,1e-4)])
            self.assertEqual(c4['selected']['shortlist']['step'],300)
            self.assertEqual(gate.call_count,1)

    def test_real_receipt_rejects_single_horizon_and_wrong_revision(self):
        from training.hope_initialization_v4 import selection_receipt
        model=HOPEFeaturePredictorV4(CONFIG,'C1')
        candidate=dict(status='fit_complete',method='C1',beta=1e-3,outer_lr=1e-4,
                       role_digest='roles',model_metadata=model.metadata(),gate={'passed':True,'status':'not_applicable_static'},
                       shortlist=dict(step=400,loss=.1,loss_at_0=.1,loss_at_16=.1,weights=model.state_dict()))
        outcome=dict(status='selected',smoke=False,selected=candidate,
                     selection_rule_id='prefix_warm_joint_stress_v4',common_seed=0,schedule_digest='schedule')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'selection.pt'; torch.save(outcome,path)
            receipt=selection_receipt(outcome,path)
            self.assertEqual(receipt['selected_step'],400)
            bad=copy.deepcopy(outcome); bad['selected']['shortlist']['loss_at_0']=.2
            with self.assertRaises(ValueError): selection_receipt(bad,path)
            bad=copy.deepcopy(outcome); bad['selected']['model_metadata']['design_id']='hope_cad_v3_masked_teacher_cms'
            with self.assertRaises(ValueError): selection_receipt(bad,path)


if __name__=='__main__': unittest.main()
