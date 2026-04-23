from scipy.special import logit, expit
import numpy as np
import pandas as pd
from sklego.meta import ZeroInflatedRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    explained_variance_score, 
    roc_auc_score
)
from scipy.stats import spearmanr
import lightgbm as lgb

def safe_logit(x: np.ndarray, min: float=0, max: float=100, eps: float=0.1) -> np.ndarray:
    '''
    Calculates logits for data over the range [min-eps, max+eps] to prevent
    +/-Inf when x == min or x == max.

    Set eps=0 to get plain logits.
    '''
    new_min = min - eps
    new_max = max + eps

    x_scale = (x - new_min) / (new_max - new_min)
    return logit(x_scale)

def safe_inv_logit(x: np.ndarray, min: float=0, max: float=100, eps: float=0.1) -> np.ndarray:
    '''
    Inverse of safe_logit.
    '''
    x_scale = expit(x)
    new_max = max + eps
    new_min = min - eps
    x = (x_scale * (new_max - new_min)) + new_min
    return x

def make_zif_quantile_estimator(classif_args: dict={}, regressor_args: dict={}) -> ZeroInflatedRegressor:
    classifier = lgb.LGBMClassifier(
        verbosity=-1, 
        random_state=1234, 
        **classif_args
    )
    
    quantile_regressor = lgb.LGBMRegressor(
        verbosity=-1, 
        random_state=1234, 
        objective="quantile", 
        alpha=0.5, 
        **regressor_args
    )

    return ZeroInflatedRegressor(
        classifier,
        TransformedTargetRegressor(
            regressor=quantile_regressor,
            func=safe_logit,
            inverse_func=safe_inv_logit
        )
    )

def split_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    return (
        df.drop(columns=target),
        df[target]
    )

def get_results(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y, y_hat)
    nrmse = np.sqrt(mse) / np.std(y)
    r2 = r2_score(y, y_hat)
    ev = explained_variance_score(y, y_hat)
    prop_zero = np.mean(y == 0)
    spearman_test = spearmanr(y, y_hat)
    auc = roc_auc_score(y > 0, y_hat)
    return {
        "n": y.shape[0],
        "mse": mse,
        "nrmse": nrmse,
        "scikit_r2": r2,
        "scikit_exp_var": ev,
        "pearson_r": np.corrcoef(y, y_hat)[0, 1],
        "zero_nonzero_ratio": (prop_zero) / (1-prop_zero),
        "spearman_r": spearman_test.statistic,
        "spearman_p": spearman_test.pvalue,
        "auc": auc
    }

def balance_zeros(df: pd.DataFrame, target: str, nz_ratio: float=1.0) -> pd.DataFrame:
    '''
    Balances a zero-inflated dataset based on the distribution of `df[target]`. The
    resulting dataframe will have `nz_ratio` times as many zeros as nonzero values
    of `target`, sampled randomly. All nonzero rows of `df` are retained.
    '''
    df_zeros = df[df[target] == 0]
    df_nz    = df[df[target] >  0]
    zeros_sample = df_zeros.sample(n=int(df_nz.shape[0]*nz_ratio))

    return pd.concat([df_nz, zeros_sample], axis=0)
    

if __name__ == "__main__":
    x = np.random.uniform(low=0, high=100, size=100)
    logits = safe_logit(x)
    invlogit = safe_inv_logit(logits)
    assert np.allclose(x, invlogit)