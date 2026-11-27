"""RIGEL-CAD: two-level (fast keys + slow residual whitening) continual anomaly detection under a fixed byte budget."""
from rigel_cad.config import RigelConfig
from rigel_cad.model import BudgetViolation, FastKeyMemory, RigelCAD, RigelState, SlowResidualWhitener

__version__ = '1.1.0'
__all__ = ['RigelCAD', 'RigelConfig', 'RigelState', 'FastKeyMemory', 'SlowResidualWhitener', 'BudgetViolation',
           '__version__']
