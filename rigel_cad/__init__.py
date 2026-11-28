"""RIGEL-CAD: two-level (fast keys + slow residual whitening) continual anomaly detection under a fixed byte budget."""
from rigel_cad.config import RigelConfig
from rigel_cad.model import BudgetViolation, FastKeyMemory, Prepared, RigelCAD, RigelState, SlowResidualWhitener, TaskFit

__version__ = '1.1.1'
__all__ = ['RigelCAD', 'RigelConfig', 'RigelState', 'TaskFit', 'Prepared', 'FastKeyMemory', 'SlowResidualWhitener', 'BudgetViolation',
           '__version__']
