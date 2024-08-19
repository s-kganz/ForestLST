import os
# Silence tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from . import data
from . import training

import warnings
try:
    from . import gee
except ImportError:
    warnings.warn("Earth Engine API not found! Module util.gee will not be loaded.")

try:
    from . import gcs
except ImportError:
    warnings.warn("GCloud SDK not found! Module util.gcs will not be loaded.")
    
