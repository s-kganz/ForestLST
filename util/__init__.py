import warnings
import os

from . import data
from . import daymet
from . import plot

try:
    from . import training_tf as training
    # Silence tensorflow debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
except ImportError:
    try:
        from . import training_torch as training
    except ImportError:
        warnings.warn("Neither TensorFlow nor PyTorch found! Module util.training will not be loaded.")

try:
    from . import gee
except ImportError:
    warnings.warn("Earth Engine API not found! Module util.gee will not be loaded.")

try:
    from . import gcs
except ImportError:
    warnings.warn("GCloud SDK not found! Module util.gcs will not be loaded.")
    
