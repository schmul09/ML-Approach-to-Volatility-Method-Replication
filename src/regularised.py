import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Shared tuning grid 
ALPHA_GRID = np.logspace(-5, 2, 50)


def _tune_and_fit(ModelClass, model_kwargs,
                  X_train, y_train,
                  X_val,   y_val,
                  X_test,
                  alpha_grid=ALPHA_GRID):
    
    # standardise on train, tune on val
    sc1    = StandardScaler()
    Xtr_s  = sc1.fit_transform(X_train)
    Xva_s  = sc1.transform(X_val)

    best_mse, best_alpha = np.inf, None
    for alpha in alpha_grid:
        m = ModelClass(alpha=alpha, **model_kwargs)
        m.fit(Xtr_s, y_train)
        mse = mean_squared_error(y_val, m.predict(Xva_s))
        if mse < best_mse:
            best_mse, best_alpha = mse, alpha

    # refit on train+val with best alpha
    X_tv  = np.vstack([X_train, X_val])
    y_tv  = np.concatenate([y_train, y_val])
    sc2   = StandardScaler()
    Xtv_s = sc2.fit_transform(X_tv)
    Xte_s = sc2.transform(X_test)

    m_final = ModelClass(alpha=best_alpha, **model_kwargs)
    m_final.fit(Xtv_s, y_tv)

    return m_final.predict(Xte_s), best_alpha, sc2, m_final


class RidgeForecaster:

    def __init__(self):
        self.best_alpha_ = None
        self._scaler     = None
        self._model      = None
        self._preds      = None

    def fit(self, X_train, y_train, X_val, y_val):
        """Tune lambda on val set; refit on train+val."""
        self._preds, self.best_alpha_, self._scaler, self._model = \
            _tune_and_fit(Ridge, {}, X_train, y_train, X_val, y_val,
                          np.zeros((1, X_train.shape[1])))  # placeholder
        # We need to store X_test outside; restructure for proper predict
        self._X_train = X_train
        self._y_train = y_train
        self._X_val   = X_val
        self._y_val   = y_val
        return self

    def predict(self, X_test):
        """Generate forecasts on X_test."""
        X_tv  = np.vstack([self._X_train, self._X_val])
        y_tv  = np.concatenate([self._y_train, self._y_val])
        sc    = StandardScaler()
        Xtv_s = sc.fit_transform(X_tv)
        Xte_s = sc.transform(X_test)
        m     = Ridge(alpha=self.best_alpha_)
        m.fit(Xtv_s, y_tv)
        return m.predict(Xte_s)


class LassoForecaster:
    
    def __init__(self):
        self.best_alpha_ = None
        self._X_train = None
        self._y_train = None
        self._X_val   = None
        self._y_val   = None

    def fit(self, X_train, y_train, X_val, y_val):
        # Tune alpha
        sc1   = StandardScaler()
        Xtr_s = sc1.fit_transform(X_train)
        Xva_s = sc1.transform(X_val)
        best_mse, best_alpha = np.inf, None
        for alpha in ALPHA_GRID:
            m = Lasso(alpha=alpha, max_iter=10000)
            m.fit(Xtr_s, y_train)
            mse = mean_squared_error(y_val, m.predict(Xva_s))
            if mse < best_mse:
                best_mse, best_alpha = mse, alpha
        self.best_alpha_ = best_alpha
        self._X_train = X_train
        self._y_train = y_train
        self._X_val   = X_val
        self._y_val   = y_val
        return self

    def predict(self, X_test):
        X_tv  = np.vstack([self._X_train, self._X_val])
        y_tv  = np.concatenate([self._y_train, self._y_val])
        sc    = StandardScaler()
        Xtv_s = sc.fit_transform(X_tv)
        Xte_s = sc.transform(X_test)
        m     = Lasso(alpha=self.best_alpha_, max_iter=10000)
        m.fit(Xtv_s, y_tv)
        return m.predict(Xte_s)


class ElasticNetForecaster:
    
    def __init__(self, l1_ratio: float = 0.5):
        self.l1_ratio    = l1_ratio
        self.best_alpha_ = None
        self._X_train = None
        self._y_train = None
        self._X_val   = None
        self._y_val   = None

    def fit(self, X_train, y_train, X_val, y_val):
        sc1   = StandardScaler()
        Xtr_s = sc1.fit_transform(X_train)
        Xva_s = sc1.transform(X_val)
        best_mse, best_alpha = np.inf, None
        for alpha in ALPHA_GRID:
            m = ElasticNet(alpha=alpha, l1_ratio=self.l1_ratio,
                           max_iter=10000)
            m.fit(Xtr_s, y_train)
            mse = mean_squared_error(y_val, m.predict(Xva_s))
            if mse < best_mse:
                best_mse, best_alpha = mse, alpha
        self.best_alpha_ = best_alpha
        self._X_train = X_train
        self._y_train = y_train
        self._X_val   = X_val
        self._y_val   = y_val
        return self

    def predict(self, X_test):
        X_tv  = np.vstack([self._X_train, self._X_val])
        y_tv  = np.concatenate([self._y_train, self._y_val])
        sc    = StandardScaler()
        Xtv_s = sc.fit_transform(X_tv)
        Xte_s = sc.transform(X_test)
        m     = ElasticNet(alpha=self.best_alpha_, l1_ratio=self.l1_ratio,
                           max_iter=10000)
        m.fit(Xtv_s, y_tv)
        return m.predict(Xte_s)
