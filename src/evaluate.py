import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# MSE computation 
def compute_mse_table(results_store: dict,
                      y_test_store:  dict,
                      models:        list,
                      tickers:       list) -> pd.DataFrame:
    """
    Here we compute absolute out-of-sample MSE for every model and ticker.

    Parameters:
    
    results_store : dict  {ticker: {model: np.ndarray of predictions}}
    y_test_store  : dict  {ticker: np.ndarray of test targets}
    models        : list  of model name strings
    tickers       : list  of ticker strings

    Returns:
    
    pd.DataFrame  shape (n_tickers, n_models)
        Rows = tickers, columns = model names.
    """
    mse_table = {m: [] for m in models}
    for ticker in tickers:
        y_te = y_test_store[ticker]
        for m in models:
            mse_table[m].append(
                mean_squared_error(y_te, results_store[ticker][m])
            )
    return pd.DataFrame(mse_table, index=tickers)


def compute_relative_mse(mse_df: pd.DataFrame,
                          benchmark: str = 'HAR'):
    """
    Here we compute relative MSE with respect to a benchmark model.

    Each entry =  MSE(model) / MSE(benchmark)  for the same ticker.
    
    Parameters:
   
    mse_df    : pd.DataFrame  output of compute_mse_table()
    benchmark : str           column name of the benchmark model

    Returns:
    
    rel_mse     : pd.DataFrame  same shape as mse_df
    rel_mse_avg : pd.Series     cross-sectional average across tickers
    """
    rel_mse     = mse_df.divide(mse_df[benchmark], axis=0)
    rel_mse_avg = rel_mse.mean()
    return rel_mse, rel_mse_avg


# Print a relative MSE table
def print_results_table(rel_mse:     pd.DataFrame,
                         rel_mse_avg: pd.Series,
                         models:      list,
                         tickers:     list) -> None:
    

    # An asterisk marks cells where the model beats HAR.
    
    col_w = 68
    print("\n" + "=" * col_w)
    print("Table 1: One-day-ahead Out-of-Sample Relative MSE (MHAR dataset)")
    print("         HAR = 1.000  |  * = model beats HAR for that stock")
    print("=" * col_w)
    header = f"{'Model':<16}" + "".join([f"{t:>8}" for t in tickers]) + f"{'Avg':>8}"
    print(header)
    print("-" * col_w)
    for m in models:
        row = f"{m:<16}"
        for ticker in tickers:
            v      = rel_mse.loc[ticker, m]
            marker = "*" if v < 1.0 else " "
            row   += f"{v:>7.4f}{marker}"
        row += f"{rel_mse_avg[m]:>8.4f}"
        print(row)
    print("=" * col_w)
    print("Note: values <1 indicate improvement over HAR benchmark.\n")


# Compute ALE 
def compute_ale(model,
                X_sc:        np.ndarray,
                feature_idx: int,
                K:           int = 20):
    
    # Partition feature into K quantile-based intervals
    z_vals = np.percentile(
        X_sc[:, feature_idx],
        np.linspace(0, 100, K + 1)
    )

    ale = np.zeros(K)
    for k in range(K):
        mask = (
            (X_sc[:, feature_idx] >= z_vals[k]) &
            (X_sc[:, feature_idx] <  z_vals[k + 1])
        )
        if mask.sum() < 2:
            continue
        X_lo = X_sc[mask].copy()
        X_hi = X_sc[mask].copy()
        X_lo[:, feature_idx] = z_vals[k]
        X_hi[:, feature_idx] = z_vals[k + 1]
        ale[k] = np.mean(model.predict(X_hi) - model.predict(X_lo))

    # Accumulate and centre
    ale_cum      = np.cumsum(ale)
    ale_centered = ale_cum - np.mean(ale_cum)
    z_mid        = (z_vals[:-1] + z_vals[1:]) / 2

    return z_mid, ale_centered
