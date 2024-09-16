import warnings

from . import data
from . import daymet

try:
    from . import training
    # Silence tensorflow debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
except ImportError:
    warnings.warn("TensorFlow not found! Module util.training will not be loaded.")


try:
    from . import gee
except ImportError:
    warnings.warn("Earth Engine API not found! Module util.gee will not be loaded.")

try:
    from . import gcs
except ImportError:
    warnings.warn("GCloud SDK not found! Module util.gcs will not be loaded.")
    
