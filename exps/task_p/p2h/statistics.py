"""All contrasts use positive reference-minus-candidate retention benefit."""
import math
import numpy as np
from scipy.stats import t


def paired_benefit(reference_excess,candidate_excess,alpha=.05):
    reference=np.asarray(reference_excess,dtype=np.float64)
    candidate=np.asarray(candidate_excess,dtype=np.float64)
    assert reference.shape==candidate.shape and reference.ndim==1 and len(reference)>1
    differences=reference-candidate
    mean=float(differences.mean())
    se=float(differences.std(ddof=1)/math.sqrt(len(differences)))
    width=float(t.ppf(1-alpha/2,len(differences)-1))*se
    p=0. if se==0 and mean!=0 else (1. if se==0 else float(2*t.sf(abs(mean/se),len(differences)-1)))
    return {'benefit':mean,'lower':mean-width,'upper':mean+width,'paired_p':p,
            'n':len(differences),'per_seed':differences.tolist()}


def primary(reference_excess,candidate_excess,new_reference,new_candidate):
    result=paired_benefit(reference_excess,candidate_excess)
    meanE=float(np.mean(reference_excess))
    regression=float(np.mean((np.asarray(new_candidate)-np.asarray(new_reference))/np.maximum(new_reference,1e-8)))
    result.update(excess_E=meanE,required_benefit=.05*meanE,acquisition_regression=regression,
                  passed=meanE>1e-8 and result['benefit']>=.05*meanE and result['lower']>0 and regression<=.01)
    return result


def mde(reference_excess,candidate_excess):
    r=paired_benefit(reference_excess,candidate_excess)
    sd=float(np.std(r['per_seed'],ddof=1))
    absolute=float((t.ppf(.975,9)+t.ppf(.8,9))*sd/math.sqrt(10))
    headroom=float(np.mean(reference_excess))
    return dict(absolute=absolute,relative=None if headroom<=1e-8 else absolute/headroom,
                headroom=headroom,passed=headroom>1e-8 and absolute<=.05*headroom)


def holm(pvalues):
    keys=sorted(pvalues,key=pvalues.get); previous=0.; adjusted={}
    for i,key in enumerate(keys):
        previous=max(previous,min(1.,(len(keys)-i)*pvalues[key]))
        adjusted[key]=previous
    return adjusted
