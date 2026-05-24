import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestForecaster:
    
    def __init__(self,
                 n_estimators:     int = 500,
                 min_samples_leaf: int = 5,
                 random_state:     int = 42,
                 n_jobs:           int = -1):
        self.n_estimators     = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state     = random_state
        self.n_jobs           = n_jobs
        self.model_           = None
        
    # Fit random forest on the combined train+validation set
    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray):
        """
        Parameters:
        
        X_train : (n_train, J)
        y_train : (n_train,)
        X_val   : (n_val, J)    — included for combined fit
        y_val   : (n_val,)      — included for combined fit
        """
        X_tv = np.vstack([X_train, X_val])
        y_tv = np.concatenate([y_train, y_val])

        # max_features: floor(J/3), minimum 1
        max_feats = max(1, X_train.shape[1] // 3)

        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=max_feats,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.model_.fit(X_tv, y_tv)
        return self
    # Generate one-day-ahead RV forecasts
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Parameters:
        
        X_test : (n_test, J)

        Returns:
       
        np.ndarray  shape (n_test,)
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model_.predict(X_test)
