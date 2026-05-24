import numpy as np
from numpy.linalg import lstsq


class HAR:
    """
    RV_{t+1} = beta_0 + beta_1*RVD_t + beta_2*RVW_t + beta_3*RVM_t + u_t

    """

    def __init__(self):
        self.beta_ = None

    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray):
        """
        Fit HAR by OLS on the combined train+validation set.

        Parameters:
        
        X_train : (n_train, 3)   — [RVD, RVW, RVM] training features
        y_train : (n_train,)     — training targets
        X_val   : (n_val, 3)     — validation features
        y_val   : (n_val,)       — validation targets (not used for OLS,
                                   included for API consistency)
        """
        X_tv = np.column_stack([
            np.ones(len(X_train) + len(X_val)),
            np.vstack([X_train, X_val])
        ])
        y_tv = np.concatenate([y_train, y_val])
        self.beta_, _, _, _ = lstsq(X_tv, y_tv, rcond=None)
        return self
    
    # Generate one-day-ahead RV forecasts
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Parameters:

        X_test : (n_test, 3)

        Returns:
    
        np.ndarray  shape (n_test,)
        """
        if self.beta_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_te = np.column_stack([np.ones(len(X_test)), X_test])
        return X_te @ self.beta_


class LogHAR:
    """
    log(RV_{t+1}) = b0 + b1*log(RVD_t) + b2*log(RVW_t) + b3*log(RVM_t) + u_t

    Estimated by OLS in log space on the combined train+validation set.
    Forecasts are back-transformed to RV space using the log-normal
    bias correction:

        RV_hat = exp( log_RV_hat + 0.5 * var(train+val residuals) )
    """

    def __init__(self):
        self.beta_      = None
        self.resid_var_ = None

    def fit(self,
            X_train: np.ndarray, y_log_train: np.ndarray,
            X_val:   np.ndarray, y_log_val:   np.ndarray):
        """
        Fit LogHAR by OLS in log-RV space on train+validation set.

        Parameters:
        
        X_train     : (n_train, 3)  — log(RVD, RVW, RVM) training features
        y_log_train : (n_train,)    — log(RV_{t+1}) training targets
        X_val       : (n_val, 3)    — log features for validation
        y_log_val   : (n_val,)      — log targets for validation
        """
        X_tv = np.column_stack([
            np.ones(len(X_train) + len(X_val)),
            np.vstack([X_train, X_val])
        ])
        y_tv = np.concatenate([y_log_train, y_log_val])
        self.beta_, _, _, _ = lstsq(X_tv, y_tv, rcond=None)
        fitted            = X_tv @ self.beta_
        self.resid_var_   = np.var(y_tv - fitted)
        return self
    
    # Generate bias-corrected one-day-ahead RV forecasts
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Parameters:
    
        X_test : (n_test, 3)  — log(RVD, RVW, RVM) test features

        Returns:
        
        np.ndarray  shape (n_test,)  — forecast in original RV space
        """
        if self.beta_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_te    = np.column_stack([np.ones(len(X_test)), X_test])
        log_hat = X_te @ self.beta_
        return np.exp(log_hat + 0.5 * self.resid_var_)
