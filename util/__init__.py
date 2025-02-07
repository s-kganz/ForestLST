import warnings
import os

from . import data
from . import daymet
from . import plot

try:
    from . import training_torch as training
    from . import convlstm
except ImportError:
    warnings.warn("PyTorch not found! Module util.training will not be loaded.")

try:
    from . import gcs
except ImportError:
    warnings.warn("GCloud SDK not found! Module util.gcs will not be loaded.")
    
