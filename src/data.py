import numpy as np
import pandas as pd


# Constants (these need to match the paper)
N_DAYS  = 4257                             # trading days (2001-01-29 to 2017-12-31)
TICKERS = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'GS']
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.10
# TEST_FRAC  = 0.20  (implicit remainder)


# Simulate a daily realised variance series from a log-HAR DGP
def simulate_rv(n_days: int, seed: int = 0) -> np.ndarray:
    """
    Simulate a daily realised variance series (log-HAR DGP):

        RV_t = max( mu_t * exp(eps_t - 0.031),  2.0 )

    where:
        mu_t  = 0.5 + 0.25*RVD + 0.38*RVW + 0.12*RVM
        eps_t ~ N(0, 0.25^2)

    with probability 0.003, eps_t receives an additional upward jump
    drawn from Uniform(0.5, 1.5) to mimic crisis-period volatility spikes.
    The constant -0.031 ≈ -0.5 * 0.25^2 is the log-normal bias correction
    ensuring E[exp(eps_t - 0.031)] ≈ 1.

    The floor of 2.0 prevents degenerate near-zero values.

    Parameters
    
    n_days : int
        Number of trading days to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    
    np.ndarray of shape (n_days,)
        Simulated daily realised variance series.
    """
    np.random.seed(seed)
    rv = np.zeros(n_days)
    rv[:22] = 20.0                   # burn-in: initialise at ~20% annualised vol

    for t in range(22, n_days):
        rvd = rv[t - 1]
        rvw = rv[t - 5:t].mean()
        rvm = rv[t - 22:t].mean()
        mu  = 0.5 + 0.25 * rvd + 0.38 * rvw + 0.12 * rvm
        eps = np.random.normal(0.0, 0.25)
        if np.random.uniform() < 0.003:          # rare crisis spike
            eps += np.random.uniform(0.5, 1.5)
        rv[t] = max(mu * np.exp(eps - 0.031), 2.0)

    return rv


def build_dataset(n_days: int = N_DAYS,
                  tickers: list = TICKERS) -> pd.DataFrame:
    
    # Build a DataFrame of simulated daily RV for all tickers.

    # Each ticker is assigned a distinct random seed so the series are independent but reproducible.

    data = {
        ticker: simulate_rv(n_days, seed=i * 7)
        for i, ticker in enumerate(tickers)
    }
    index = pd.bdate_range(start='2001-01-29', periods=n_days)
    return pd.DataFrame(data, index=index)


# Constructing the features
def build_har_features(rv_series: pd.Series):
    """
    Construct the M_HAR predictor matrix and one-day-ahead target.

    Predictors:
    
    RVD : RV_{t}              daily lag
    RVW : mean(RV_{t-4:t+1}) 5-day (weekly) average
    RVM : mean(RV_{t-21:t+1}) 22-day (monthly) average

    Target:
    
    y : RV_{t+1}  one-day-ahead realised variance

    The first 22 rows are dropped because RVM is not yet fully
    initialised over a 22-day window.

    Parameters:
    
    rv_series : pd.Series  — daily RV for one ticker

    Returns:
   
    X : np.ndarray  shape (T, 3)  — [RVD, RVW, RVM]
    y : np.ndarray  shape (T,)    — RV_{t+1}
    """
    rv = rv_series.values
    n  = len(rv)

    rvd = rv[:-1]
    rvw = np.array([rv[max(0, i - 4):i + 1].mean() for i in range(n - 1)])
    rvm = np.array([rv[max(0, i - 21):i + 1].mean() for i in range(n - 1)])
    y   = rv[1:]
    X   = np.column_stack([rvd, rvw, rvm])

    return X[22:], y[22:]


def build_log_har_features(rv_series: pd.Series):
    """
    Same as build_har_features but computed in log-RV space.

    Used by LogHAR and the neural network, both of which are trained
    on log(RV) and back-transformed to RV space for evaluation.

    Returns:
   
    X_log   : np.ndarray  shape (T, 3)  — [log(RVD), log(RVW), log(RVM)]
    y_log   : np.ndarray  shape (T,)    — log(RV_{t+1})
    y_raw   : np.ndarray  shape (T,)    — RV_{t+1}  (for MSE evaluation)
    """
    rv  = rv_series.values
    lrv = np.log(rv)
    n   = len(rv)

    rvd = lrv[:-1]
    rvw = np.array([lrv[max(0, i - 4):i + 1].mean() for i in range(n - 1)])
    rvm = np.array([lrv[max(0, i - 21):i + 1].mean() for i in range(n - 1)])
    y_log = lrv[1:]
    y_raw = rv[1:]
    X     = np.column_stack([rvd, rvw, rvm])

    return X[22:], y_log[22:], y_raw[22:]


# Training / validation / testing split
def split_data(X: np.ndarray, y: np.ndarray,
               train_frac: float = TRAIN_FRAC,
               val_frac: float   = VAL_FRAC):
    """
    Split fractions:
        Train : 70%  (~2,964 days)
        Val   : 10%  (~424 days)   — used only for hyperparameter tuning
        Test  : 20%  (~847 days)   — held out; never used during estimation

    No shuffling is applied. 

    Parameters:

    X          : np.ndarray  shape (T, p)
    y          : np.ndarray  shape (T,)
    train_frac : float
    val_frac   : float

    Returns:
    
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    n  = len(y)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))

    return (
        X[:t1],  y[:t1],
        X[t1:t2], y[t1:t2],
        X[t2:],  y[t2:]
    )
