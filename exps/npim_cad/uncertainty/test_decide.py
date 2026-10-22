import numpy as np
import torch

from exps.npim_cad.uncertainty.calibrate import decide_primary


def base_readout(ratio=1.50, lower=1.30, boot_cv=0.10, offdiag=0.20, ks=0.05):
    return {
        "geometry_raw": {"p95_over_p5": ratio},
        "bootstrap": {"p95_over_p5_interval": [lower, 1.60], "bootstrap_channel_cv_mean": boot_cv},
        "covariance": {"offdiag_energy_fraction": offdiag},
        "fit_witness": {"A2": {"ks": ks}},
        "var_raw": torch.ones(8),
    }


def test_null_isotropic_stops_before_ready():
    decision, reason = decide_primary(base_readout(ratio=1.05, lower=1.01))
    assert decision == "NPIM03_NULL_ISOTROPIC_RESIDUAL"
    assert "rescaling" in reason


def test_ready_requires_all_gates():
    decision, _ = decide_primary(base_readout())
    assert decision == "NPIM03_NORMAL_READOUT_READY_FOR_EXTERNAL_EVAL"


def test_unreliable_bootstrap_offdiag_and_shift():
    assert decide_primary(base_readout(lower=1.04))[0] == "NPIM03_NORMAL_READOUT_STATISTICALLY_UNRELIABLE"
    assert decide_primary(base_readout(offdiag=0.60))[0] == "NPIM03_NORMAL_READOUT_STATISTICALLY_UNRELIABLE"
    assert decide_primary(base_readout(boot_cv=0.50))[0] == "NPIM03_NORMAL_READOUT_STATISTICALLY_UNRELIABLE"
    assert decide_primary(base_readout(ks=0.20))[0] == "NPIM03_NORMAL_READOUT_STATISTICALLY_UNRELIABLE"


def test_nonfinite_variance_fails_closed():
    readout = base_readout()
    readout["var_raw"] = torch.tensor([1.0, float("nan")])
    assert decide_primary(readout)[0] == "NPIM03_NORMAL_READOUT_STATISTICALLY_UNRELIABLE"
