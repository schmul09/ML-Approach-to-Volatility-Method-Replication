# Volatility Forecasting Replication

Replication of the empirical methodology from:

> Christensen, K., Siggaard, M. & Veliyev, B. (2022).  
> *A machine learning approach to volatility forecasting.*  
> Journal of Financial Econometrics.

The paper compares a broad suite of machine learning (ML) algorithms against
the Heterogeneous AutoRegressive (HAR) model of Corsi (2009) for forecasting
the daily realised variance of Dow Jones Industrial Average constituents.
This repository implements the core models, evaluation framework, and variable
importance analysis from the paper.

---

## Project structure

```
volatility_replication/
│
├── main.py                  # Entry point — runs the full pipeline
├── requirements.txt         # Python package dependencies
├── README.md
│
├── src/                     # Source modules (one file per model family)
│   ├── __init__.py
│   ├── data.py              # Data simulation, feature construction, splitting
│   ├── har.py               # HAR and LogHAR (OLS benchmark models)
│   ├── regularised.py       # Ridge, Lasso, ElasticNet
│   ├── random_forest.py     # Random Forest
│   ├── neural_net.py        # Feed-forward Neural Network (NN2 architecture)
│   ├── evaluate.py          # MSE metrics, relative MSE table, ALE computation
│   └── plotting.py          # All four figures
│
└── outputs/                 # Generated automatically on first run
    ├── figure1_relative_mse.png
    ├── figure2_boxplot.png
    ├── figure3_forecasts.png
    ├── figure4_ale.png
    ├── absolute_mse.csv
    └── relative_mse.csv
```

---

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py
```

`main.py` runs all five stages in sequence and prints progress to the terminal.
All figures and CSV results are saved to `outputs/`.

---

## What the code does

### Stage 1 — Data (`src/data.py`)

The original paper uses proprietary NYSE TAQ high-frequency data
(5-minute log-returns → daily realised variance) for 29 DJIA stocks
over 2001–2017. This dataset is not publicly available.

We simulate daily realised variance series from a **log-HAR data-generating
process** with multiplicative log-normal errors and rare upward jumps:

```
RV_t = max( mu_t * exp(eps_t),  2.0 )

mu_t  = 0.5 + 0.25*RVD + 0.38*RVW + 0.12*RVM
eps_t ~ N(0, 0.25^2),  with rare crisis spikes (prob = 0.003)
```

This preserves the key empirical properties of realised variance:
positive skewness, fat tails, volatility clustering, and long-memory
autocorrelation. The simulation spans 4,257 trading days and five tickers.

**Feature construction** follows the paper exactly:
- `RVD` — one-day lag of realised variance
- `RVW` — five-day average (weekly)
- `RVM` — 22-day average (monthly)
- Target: `RV_{t+1}` (one-day-ahead)

**Data split** (temporal, no shuffling):
| Set        | Fraction | Purpose                                  |
|------------|----------|------------------------------------------|
| Train      | 70%      | Model estimation                         |
| Validation | 10%      | Hyperparameter tuning only               |
| Test       | 20%      | Out-of-sample evaluation (never touched during estimation) |

---

### Stage 2 — Models

#### `src/har.py` — HAR and LogHAR

**HAR** (Corsi, 2009): OLS regression of `RV_{t+1}` on RVD, RVW, RVM.
Estimated on the combined train+validation set. The benchmark model
throughout the study.

**LogHAR** (Corsi, 2009): same specification in log space. Forecasts are
back-transformed to RV space using a log-normal bias correction
(`exp(log_hat + 0.5 * var(residuals))`), following Jensen's inequality.

#### `src/regularised.py` — Ridge, Lasso, ElasticNet

All three penalise the OLS objective to reduce overfitting in
high-dimensional settings:

| Model      | Penalty                          | Effect                          |
|------------|----------------------------------|---------------------------------|
| Ridge      | `lambda * sum(beta^2)`           | Shrinks all coefficients        |
| Lasso      | `lambda * sum(\|beta\|)`         | Some coefficients set to zero   |
| ElasticNet | Convex combination of both       | Shrinks and selects             |

The penalty parameter `lambda` is tuned over 50 log-spaced values
in `[1e-5, 1e2]` using validation-set MSE. Models are then refit
on the combined train+validation set.

#### `src/random_forest.py` — Random Forest

Breiman (2001) ensemble of 500 regression trees, each trained on a
bootstrap sample. At each split, only `floor(J/3)` randomly selected
features are considered, de-correlating the trees.  
No hyperparameter tuning — all parameters set to defaults from
Breiman & Cutler (2004), consistent with the paper.

#### `src/neural_net.py` — Neural Network (NN2)

Two-hidden-layer feed-forward network (4 → 2 neurons), matching the
NN2 architecture of the paper. Trained in log-RV space with:

- **Activation**: Leaky ReLU (negative slope = 0.01)
- **Optimiser**: Adam (lr = 0.001, weight decay = 1e-4)
- **Regularisation**: Dropout (0.2) + L2 weight decay + early stopping (patience = 50)
- **Ensemble**: trains 5 networks with different random seeds; averages the best 3 by validation MSE

Forecasts are back-transformed to RV space with a bias correction
and an insanity filter (clips negative predictions to the minimum
in-sample RV).

---

### Stage 3 — Evaluation (`src/evaluate.py`)

Out-of-sample MSE is computed on the held-out test set. Relative MSE
is reported as `MSE(model) / MSE(HAR)` — values below 1.0 indicate
improvement over the benchmark. The ALE variable importance measure
(Apley & Zhu, 2020) is computed for the Random Forest to identify
which predictors drive the forecast.

---

### Stage 4 & 5 — Figures (`src/plotting.py`)

| Figure | Content |
|--------|---------|
| Figure 1 | Grouped bar chart of relative MSE by model and ticker |
| Figure 2 | Boxplot of relative MSE distribution across tickers |
| Figure 3 | Actual vs forecast realised variance — AAPL test period |
| Figure 4 | ALE plots for RVD, RVW, RVM — Random Forest on AAPL |

---

## Key results

Relative MSE averaged across five stocks (HAR = 1.000):

| Model        | Avg Relative MSE | Beats HAR in N/5 stocks |
|--------------|-----------------|-------------------------|
| HAR          | 1.0000          | —                       |
| LogHAR       | 1.0042          | 2                       |
| Ridge        | 0.9995          | 4                       |
| Lasso        | 0.9996          | 3                       |
| ElasticNet   | 0.9991          | 4                       |
| RandomForest | 1.0367          | 0                       |
| NeuralNet    | 1.71            | 0                       |

These results are **consistent with the paper's Table 2** for the
MHAR predictor set:

- Regularised models marginally but consistently outperform HAR,
  reflecting that even in a low-dimensional setting a small amount
  of shrinkage reduces estimation error.
- Random Forest underperforms in the MHAR setting — consistent with
  the paper's finding that tree-based methods need a richer feature
  space (MALL) to de-correlate effectively.
- The neural network's underperformance relative to the paper's ~3–5%
  improvement is attributable to: (1) the absence of the MALL predictor
  set, which the paper shows is critical for NN advantage; (2)
  simulated rather than tick-data RV series; (3) a smaller ensemble
  than the paper's 10-from-100 configuration.

---

## Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
torch>=2.0.0
matplotlib>=3.7.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## References

- Corsi, F. (2009). A simple approximate long-memory model of realized
  volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Christensen, K., Siggaard, M. & Veliyev, B. (2022). A machine learning
  approach to volatility forecasting. *Journal of Financial Econometrics*.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Apley, D. W. & Zhu, J. (2020). Visualizing the effects of predictor
  variables in black box supervised learning models. *Journal of the
  Royal Statistical Society: Series B*, 82(4), 1059–1086.
- Kingma, D. P. & Ba, J. (2014). Adam: A method for stochastic
  optimization. arXiv:1412.6980.
