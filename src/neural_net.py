import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class _FeedForwardNN(nn.Module):
    
    def __init__(self,
                 input_dim:   int,
                 hidden_sizes: tuple = (4, 2),
                 dropout:     float  = 0.2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=dropout)
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralNetForecaster:
    """
    This FFN trains in log-RV space and back-transforms forecasts to RV space
    using a log-normal bias correction.

    Parameters:
    
    hidden_sizes : tuple   — neurons per hidden layer (default (4,2) = NN2)
    n_seeds      : int     — number of independently seeded networks to train
    n_ensemble   : int     — number of best networks (by val MSE) to average
    epochs       : int     — maximum training epochs
    patience     : int     — early stopping patience (epochs without improvement)
    lr           : float   — Adam learning rate
    dropout      : float   — dropout rate during training
    weight_decay : float   — L2 regularisation via Adam weight_decay
    """

    def __init__(self,
                 hidden_sizes: tuple = (4, 2),
                 n_seeds:      int   = 5,
                 n_ensemble:   int   = 3,
                 epochs:       int   = 300,
                 patience:     int   = 50,
                 lr:           float = 0.001,
                 dropout:      float = 0.2,
                 weight_decay: float = 1e-4):
        self.hidden_sizes = hidden_sizes
        self.n_seeds      = n_seeds
        self.n_ensemble   = n_ensemble
        self.epochs       = epochs
        self.patience     = patience
        self.lr           = lr
        self.dropout      = dropout
        self.weight_decay = weight_decay

        # Set after fit
        self._scaler      = None
        self._ensemble    = None   # list of (val_mse, model_state_dict)
        self._resid_var   = None
        self._min_rv      = None

    # set up the internal helpers 
    @staticmethod
    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32)

    def _train_one(self, seed: int,
                   Xtr_s: np.ndarray, y_log_tr: np.ndarray,
                   Xva_s: np.ndarray, y_log_va: np.ndarray):
        """Train one network; return (best_val_mse, best_state_dict)."""
        torch.manual_seed(seed)
        model   = _FeedForwardNN(Xtr_s.shape[1],
                                  self.hidden_sizes,
                                  self.dropout)
        opt     = torch.optim.Adam(model.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_val, best_state, wait = np.inf, None, 0

        Xtr_t = self._to_tensor(Xtr_s)
        ytr_t = self._to_tensor(y_log_tr)
        Xva_t = self._to_tensor(Xva_s)
        yva_t = self._to_tensor(y_log_va)

        for _ in range(self.epochs):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t), ytr_t)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xva_t), yva_t).item()

            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        return best_val, best_state

    # set up the interface 
    def fit(self,
            X_train:     np.ndarray, y_log_train: np.ndarray,
            y_train_raw: np.ndarray,
            X_val:       np.ndarray, y_log_val:   np.ndarray):
        """
        Parameters:
        
        X_train      : (n_train, J)  — log-RV feature matrix
        y_log_train  : (n_train,)    — log(RV_{t+1}) training targets
        y_train_raw  : (n_train,)    — RV_{t+1} in original scale
                                       (used for insanity filter floor)
        X_val        : (n_val, J)    — log-RV features for validation
        y_log_val    : (n_val,)      — log(RV_{t+1}) validation targets
        """
        # Standardise on training data
        self._scaler = StandardScaler()
        Xtr_s = self._scaler.fit_transform(X_train)
        Xva_s = self._scaler.transform(X_val)

        # Compute bias-correction variance on training residuals
        # this will be updated after ensemble selection below
        self._min_rv = float(y_train_raw.min())

        # Train n_seeds networks
        seed_results = []
        for seed in range(self.n_seeds):
            val_mse, state = self._train_one(
                seed, Xtr_s, y_log_train, Xva_s, y_log_val)
            seed_results.append((val_mse, state))

        # Keep best n_ensemble by validation MSE
        seed_results.sort(key=lambda x: x[0])
        self._ensemble = seed_results[:self.n_ensemble]

        # Compute residual variance for bias correction using best model
        best_state = self._ensemble[0][1]
        model      = _FeedForwardNN(X_train.shape[1],
                                     self.hidden_sizes,
                                     self.dropout)
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p_log_tr = model(self._to_tensor(Xtr_s)).numpy()
        self._resid_var = float(np.var(y_log_train - p_log_tr))

        return self
    
    # Generate ensemble RV forecasts
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Parameters:
        
        X_test : (n_test, J)

        Returns:
        
        np.ndarray  shape (n_test,)
        """
        if self._ensemble is None:
            raise RuntimeError("Call fit() before predict().")

        Xte_s = self._scaler.transform(X_test)
        Xte_t = self._to_tensor(Xte_s)

        log_preds = []
        for _, state in self._ensemble:
            model = _FeedForwardNN(X_test.shape[1],
                                    self.hidden_sizes,
                                    self.dropout)
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                log_preds.append(model(Xte_t).numpy())

        # Ensemble average in log space, then back transform
        avg_log = np.mean(log_preds, axis=0)
        rv_hat  = np.exp(avg_log + 0.5 * self._resid_var)

        # Set up insanity filter
        return np.maximum(rv_hat, self._min_rv)
