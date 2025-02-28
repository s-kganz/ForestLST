'''
Utility functions for low-level array operations.
'''

import numpy as np

def numpy_ffill_1d(arr: np.ndarray) -> np.ndarray:
    '''
    Replace missing values in arr with the leftmost valid value. Modified
    from the below SO thread to work for 1d arrays.
    
    https://stackoverflow.com/questions/41190852/
    most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    '''
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[0]),0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]

def numpy_ewma_vectorized(data: np.ndarray, window: int) -> np.ndarray:
    '''
    Forwards-backwards exponentially weighted average. Modified from the
    below SO thread to handle NAs. This ignores NAs in calculations by
    forward-filling the alst non-NA value.
    
    https://stackoverflow.com/questions/42869495/numpy-version-of-exponential
    -weighted-moving-average-equivalent-to-pandas-ewm
    '''
    mask = np.isnan(data)
    if np.all(mask):
        # Discard all-NA
        return data

    data_ffill = numpy_ffill_1d(data)
    
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data_ffill.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    first_non_nan = np.where(~np.isnan(data_ffill))[0][0]
    offset = data_ffill[first_non_nan]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data_ffill*pw0*scale_arr
    cumsums = np.nancumsum(mult)
    out = offset + cumsums*scale_arr[::-1]

    # Do not introduce unobserved data
    out[mask] = np.nan
    
    return out
