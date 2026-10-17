"""Causal one-image fixed-bank credit, atomic learning and read-only raw NN scoring."""
from copy import deepcopy
from dataclasses import asdict
import math,torch
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetConfig
from models.hope_cad_v2.state import finite_tensors,tensor_digest
from .contract import CONTRACT,design_digest,canonical
from .geometry import ForkableCoreset,AnchorReservoir,FixedContextProjection,risk,anchor_risk,advantage,context_statistics
from .learner import UtilityModel,NestedUtilityLearner

MODES=('CADIC','GEOMETRIC','FIRST_ORDER','NESTED','FAST_ONLY','NO_FAST_ADAPTATION','NO_SELF_MODIFICATION','NO_ANCHOR_OBJECTIVE')

class SupportSession:
    def __init__(self,mode='NESTED',*,device='cpu',seed=0,bank_config=None,model_config=None):
        if mode not in MODES:raise ValueError('unregistered support policy')
        self.mode,self.device,self.seed=mode,torch.device(device),seed
        g=CONTRACT['geometry'];self.bank_config=bank_config or CADICPatchCoresetConfig(budget=g['bank_budget'],dim=g['raw_dim'],chunk_size=256,query_chunk_size=256,pair_chunk_size=256,image_neighbors=9)
        self.model_config=dict(model_config or {})
        if mode=='NO_SELF_MODIFICATION':self.model_config['self_modifying']=False
        self.bank=ForkableCoreset(self.bank_config,device=self.device);self.anchors=AnchorReservoir(seed=seed)
        self.learned=mode not in ('CADIC','GEOMETRIC');self.learner=None
        if self.learned:
            devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed);model=UtilityModel(**self.model_config).to(self.device).eval()
            learning_mode=mode if mode in ('FIRST_ORDER','FAST_ONLY','NO_FAST_ADAPTATION') else 'NESTED'
            self.learner=NestedUtilityLearner(model,learning_mode)
        dim=self.model_config.get('context_dim',32)
        self.projection=FixedContextProjection(self.bank_config.dim,dim,device=self.device) if self.learned else None
        self.cursor=0;self.pending=None;self.finalized=False;self.seen_ids=[];self.counters=dict(proposals=0,learned_decisions=0,rejected_learned=0,accepted_proposals=0,noop_proposals=0,labels_completed=0,censored=0)
    @property
    def anchor_weight(self):return 0. if self.mode=='NO_ANCHOR_OBJECTIVE' else 1.
    def _bank_view(self,features):
        view=ForkableCoreset(self.bank_config,device=self.device);view.features=features;return view
    def _resolve(self,features):
        if self.pending is None:return None
        pending=self.pending
        if pending['decision_cursor']+1!=self.cursor or pending['main_version']!=self.bank.features._version:raise ValueError('counterfactual bank changed or future credit chronology violated')
        minus=pending['other'] if pending['accepted'] else self.bank.features
        plus=self.bank.features if pending['accepted'] else pending['other']
        j0=risk(features,self._bank_view(minus))+self.anchor_weight*pending['anchor0']
        j1=risk(features,self._bank_view(plus))+self.anchor_weight*pending['anchor1']
        target=advantage(j0,j1)
        details=self.learner.observe(pending['tokens'].to(self.device),pending['stats'].to(self.device),target,pending['identity'])
        self.counters['labels_completed']+=1;self.pending=None
        return dict(decision_identity=pending['identity'],decision_cursor=pending['decision_cursor'],reward_observed_at=self.cursor,target=target,Jminus=j0,Jplus=j1,prediction=pending['prediction'],accepted=pending['accepted'],learning=details)
    def observe(self,features,identity):
        if self.finalized:raise ValueError('stream finalized, not a resume boundary')
        if not isinstance(identity,str) or not identity or identity in self.seen_ids:raise ValueError('duplicate/missing normal event identity')
        expected_tokens=self.model_config.get('height',28)*self.model_config.get('width',28)
        x=features.detach().to(self.device,dtype=torch.float32)
        if x.shape not in ((expected_tokens,self.bank_config.dim),(1,expected_tokens,self.bank_config.dim)):raise ValueError('one normal image with declared grid/raw dimension required')
        finite_tensors((x,),'normal image features');x=x.reshape(-1,self.bank_config.dim)
        # Raw bank/pending tensors are immutable during this event; only candidate fork is modified.
        backup=dict(bank=self.bank,pending=self.pending,cursor=self.cursor,ids=list(self.seen_ids),counters=dict(self.counters),anchors=self.anchors.state_dict(),learner=self.learner.state_dict() if self.learner else None)
        try:
            credit=self._resolve(x)
            prequential=risk(x,self.bank) if self.bank.count else None
            candidate=self.bank.fork();result=candidate.update(x)
            record=dict(cursor=self.cursor,identity=identity,prequential_risk=prequential,credit=credit,proposal=result,mode=self.mode)
            if not self.bank.is_full:
                self.bank=candidate;record.update(action='bootstrap',accepted=True,prediction=None)
            elif torch.equal(candidate.features,self.bank.features):
                self.counters['noop_proposals']+=1;self.bank=candidate;record.update(action='no_change',accepted=True,prediction=None)
            else:
                self.counters['proposals']+=1
                if self.mode=='CADIC':accepted=True;prediction=None;warm=False
                else:
                    a0=anchor_risk(self.anchors.images,self.bank) if self.anchor_weight else 0.
                    a1=anchor_risk(self.anchors.images,candidate) if self.anchor_weight else 0.
                    stats,known=context_statistics(x,self.bank,candidate,result,self.anchors.seen,self.cursor,a0,a1,self.anchor_weight)
                    if self.learned:
                        tokens=self.projection(x);prediction=self.learner.predict(tokens,stats)
                        if not math.isfinite(prediction):raise FloatingPointError('nonfinite utility admission')
                        warm=self.learner.labels<CONTRACT['learning']['warmup_completed_labels'];accepted=warm or prediction>=0.
                        if not warm:
                            self.counters['learned_decisions']+=1;self.counters['rejected_learned']+=int(not accepted)
                    else:prediction=float(stats[-1]);warm=False;accepted=prediction>=0.
                old=self.bank
                if accepted:self.bank=candidate;self.counters['accepted_proposals']+=1
                if self.learned:
                    self.pending=dict(decision_cursor=self.cursor,identity=identity,accepted=accepted,other=(old.features if accepted else candidate.features),main_version=self.bank.features._version,anchor0=a0,anchor1=a1,tokens=tokens.detach().cpu(),stats=stats.detach().cpu(),prediction=prediction)
                record.update(action='accept' if accepted else 'reject',accepted=accepted,prediction=prediction,warmup=warm)
            if self.mode!='CADIC':self.anchors.update(x,identity)
            self.seen_ids.append(identity);self.cursor+=1
            record.update(bank_count=self.bank.count,learner_labels=self.learner.labels if self.learner else 0,slow_updates=self.learner.slow_updates if self.learner else 0)
            return record
        except BaseException:
            self.bank=backup['bank'];self.pending=backup['pending'];self.cursor=backup['cursor'];self.seen_ids=backup['ids'];self.counters=backup['counters'];self.anchors.load_state_dict(backup['anchors'],self.device)
            if self.learner:self.learner.load_state_dict(backup['learner'])
            raise
    @torch.no_grad()
    def score(self,features):
        if not self.bank.count:raise ValueError('empty bank')
        f=features.to(self.device,dtype=torch.float32);finite_tensors((f,),'score input')
        # No controller prediction, delayed-label resolution or parameter update is allowed here.
        return self.bank.pixel_scores(f)
    def finalize_stream(self):
        if self.finalized:return
        if self.pending is not None:self.counters['censored']+=1;self.pending=None
        self.finalized=True
    def identity(self):return canonical(dict(design_digest=design_digest(),mode=self.mode,seed=self.seed,bank_config=asdict(self.bank_config),model_config=self.model_config))
    def state_dict(self):
        pending=None
        if self.pending:
            pending={k:(v.detach().cpu().clone() if torch.is_tensor(v) else deepcopy(v)) for k,v in self.pending.items() if k!='main_version'}
        return dict(schema='hope_support_session_v5',identity=self.identity(),cursor=self.cursor,finalized=self.finalized,seen_ids=list(self.seen_ids),counters=dict(self.counters),bank=self.bank.state_dict(),bank_sha256=tensor_digest(({'features':self.bank.features},)),anchors=self.anchors.state_dict(),pending=pending,learner=self.learner.state_dict() if self.learner else None)
    def load_state_dict(self,payload):
        if payload['schema']!='hope_support_session_v5' or payload['identity']!=self.identity():raise ValueError('session locked identity changed')
        # Validate on a temporary owner; corrupted input cannot partially load this session.
        bank=ForkableCoreset(self.bank_config,device=self.device);bank.load_state_dict(payload['bank']);finite_tensors((bank.features,),'restored bank')
        if tensor_digest(({'features':bank.features},))!=payload['bank_sha256']:raise ValueError('checkpoint bank checksum mismatch')
        if payload['cursor']!=len(payload['seen_ids']) or len(set(payload['seen_ids']))!=payload['cursor']:raise ValueError('corrupt stream cursor')
        anchors=AnchorReservoir(seed=self.seed);anchors.load_state_dict(payload['anchors'],self.device)
        pending=deepcopy(payload['pending'])
        if pending:
            pending['other']=pending['other'].to(self.device)
            if pending['decision_cursor']+1!=payload['cursor'] or pending['other'].shape!=bank.features.shape or not self.learned or payload['finalized']:raise ValueError('corrupt pending counterfactual')
            finite_tensors((pending['other'],pending['tokens'],pending['stats']),'pending state');pending['main_version']=bank.features._version
        if self.learned:
            # Save rollback for learner validation; bank/owner commit happens only after success.
            previous=self.learner.state_dict()
            try:self.learner.load_state_dict(payload['learner'])
            except BaseException:self.learner.load_state_dict(previous);raise
            if self.learner.labels!=payload['counters']['labels_completed']:self.learner.load_state_dict(previous);raise ValueError('label accounting mismatch')
        self.bank,self.anchors,self.pending=bank,anchors,pending;self.cursor=payload['cursor'];self.finalized=payload['finalized'];self.seen_ids=list(payload['seen_ids']);self.counters=dict(payload['counters'])
    def memory_profile(self):
        # Count unique persistent tensor storages, including optimizer and episode buffers.
        objects=[self.bank.features,self.pending,self.anchors.images]
        if self.projection:objects.append(self.projection.matrix)
        if self.learner:objects.extend([self.learner.model.state_dict(),self.learner.phi,self.learner.optimizer.state_dict(),self.learner.records,self.learner.block_initial])
        visited=set();total=0
        def add(value):
            nonlocal total
            if torch.is_tensor(value):
                key=(str(value.device),value.untyped_storage().data_ptr())
                if key not in visited:visited.add(key);total+=value.untyped_storage().nbytes()
            elif isinstance(value,dict):
                for v in value.values():add(v)
            elif isinstance(value,(tuple,list)):
                for v in value:add(v)
        for value in objects:add(value)
        cache=self.bank.cache_profile()
        return dict(persistent_tensor_bytes=total,bank_bytes=self.bank.feature_bytes,pending_alternative_bank_bytes=self.pending['other'].numel()*self.pending['other'].element_size() if self.pending else 0,derived_pair_cache_bytes=cache['runtime_cache_bytes'],tensor_plus_pair_cache_bytes=total+cache['runtime_cache_bytes'],accounting='includes current/controller/optimizer/episode/anchors/projection/pending tensors and pair-cache layout; excludes other Python objects, shared backbone, allocator and IO peaks')
