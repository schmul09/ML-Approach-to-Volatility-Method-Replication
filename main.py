import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# imports from src
from src.data import (
    build_dataset, build_har_features, build_log_har_features, split_data,
    N_DAYS, TICKERS
)
from src.har          import HAR, LogHAR
from src.regularised  import RidgeForecaster, LassoForecaster, ElasticNetForecaster
from src.random_forest import RandomForestForecaster
from src.neural_net   import NeuralNetForecaster
from src.evaluate     import (
    compute_mse_table, compute_relative_mse, print_results_table
)
from src.plotting import (
    plot_relative_mse_bars,
    plot_relative_mse_boxplot,
    plot_actual_vs_forecast,
    plot_ale
)

# Config 
OUTPUT_DIR  = 'outputs'
MODELS      = ['HAR', 'LogHAR', 'Ridge', 'Lasso',
                'ElasticNet', 'RandomForest', 'NeuralNet']
FEAT_NAMES  = ['RVD (daily lag)', 'RVW (weekly avg)', 'RVM (monthly avg)']
ALE_TICKER  = 'AAPL'          # ticker used for ALE illustration

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Build dataset 
print("=" * 55)
print("  Volatility Forecasting Replication")
print("  Christensen, Siggaard & Veliyev (2022)")
print("=" * 55)
print(f"\n[1/5] Simulating data  ({N_DAYS} days × {len(TICKERS)} stocks)...")
RV_DATA = build_dataset(N_DAYS, TICKERS)
print(RV_DATA.describe().round(2).to_string())


# Estimation loop 
print("\n[2/5] Fitting models...")
results_store = {}
y_test_store  = {}

for ticker in TICKERS:
    print(f"\n  {'─'*40}")
    print(f"  Stock: {ticker}")
    print(f"  {'─'*40}")

    # Features
    X,  y              = build_har_features(RV_DATA[ticker])
    Xl, y_log, y_raw   = build_log_har_features(RV_DATA[ticker])

    # Splits
    X_tr,  y_tr,  X_va,  y_va,  X_te,  y_te   = split_data(X,  y)
    Xl_tr, yl_tr, Xl_va, yl_va, Xl_te, yl_te   = split_data(Xl, y_log)
    _,     yr_tr, _,     _,     _,     _        = split_data(X,  y_raw)

    y_test_store[ticker] = y_te
    preds = {}

    # HAR 
    har = HAR().fit(X_tr, y_tr, X_va, y_va)
    preds['HAR'] = har.predict(X_te)
    print(f"  HAR          done")

    # LogHAR 
    loghar = LogHAR().fit(Xl_tr, yl_tr, Xl_va, yl_va)
    preds['LogHAR'] = loghar.predict(Xl_te)
    print(f"  LogHAR       done")

    # Ridge 
    ridge = RidgeForecaster().fit(X_tr, y_tr, X_va, y_va)
    preds['Ridge'] = ridge.predict(X_te)
    print(f"  Ridge        done  (lambda={ridge.best_alpha_:.5f})")

    # Lasso 
    lasso = LassoForecaster().fit(X_tr, y_tr, X_va, y_va)
    preds['Lasso'] = lasso.predict(X_te)
    print(f"  Lasso        done  (lambda={lasso.best_alpha_:.5f})")

    # ElasticNet
    enet = ElasticNetForecaster().fit(X_tr, y_tr, X_va, y_va)
    preds['ElasticNet'] = enet.predict(X_te)
    print(f"  ElasticNet   done  (lambda={enet.best_alpha_:.5f})")

    # Random Forest 
    rf = RandomForestForecaster().fit(X_tr, y_tr, X_va, y_va)
    preds['RandomForest'] = rf.predict(X_te)
    print(f"  RandomForest done")

    # Neural Network 
    nn = NeuralNetForecaster(
        hidden_sizes=(4, 2),
        n_seeds=5,
        n_ensemble=3,
        epochs=300,
        patience=50
    ).fit(Xl_tr, yl_tr, yr_tr, Xl_va, yl_va)
    preds['NeuralNet'] = nn.predict(Xl_te)
    print(f"  NeuralNet    done")

    results_store[ticker] = preds


# Evaluate
print("\n[3/5] Computing relative MSE...")
mse_df      = compute_mse_table(results_store, y_test_store, MODELS, TICKERS)
rel_mse, rel_mse_avg = compute_relative_mse(mse_df, benchmark='HAR')
print_results_table(rel_mse, rel_mse_avg, MODELS, TICKERS)

# Save results to CSV
mse_path = os.path.join(OUTPUT_DIR, 'absolute_mse.csv')
rel_path  = os.path.join(OUTPUT_DIR, 'relative_mse.csv')
mse_df.round(6).to_csv(mse_path)
rel_mse.round(6).to_csv(rel_path)
print(f"  Results saved to {mse_path} and {rel_path}")


# Figures 
print("\n[4/5] Generating figures...")

plot_relative_mse_bars(
    rel_mse, MODELS, TICKERS,
    os.path.join(OUTPUT_DIR, 'figure1_relative_mse.png')
)

plot_relative_mse_boxplot(
    rel_mse, MODELS,
    os.path.join(OUTPUT_DIR, 'figure2_boxplot.png')
)

plot_actual_vs_forecast(
    y_test_store, results_store,
    ticker=ALE_TICKER,
    output_path=os.path.join(OUTPUT_DIR, 'figure3_forecasts.png')
)

# ALE (Random Forest on ALE_TICKER)
print("\n[5/5] Computing ALE for Random Forest...")
X_ale, y_ale       = build_har_features(RV_DATA[ALE_TICKER])
X_tr_a, y_tr_a, X_va_a, y_va_a, _, _ = split_data(X_ale, y_ale)
X_tv_a  = np.vstack([X_tr_a, X_va_a])
y_tv_a  = np.concatenate([y_tr_a, y_va_a])
sc_ale  = StandardScaler()
X_tv_sc = sc_ale.fit_transform(X_tv_a)

rf_ale  = RandomForestForecaster().fit(X_tr_a, y_tr_a, X_va_a, y_va_a)
plot_ale(
    rf_ale.model_, X_tv_sc, FEAT_NAMES,
    os.path.join(OUTPUT_DIR, 'figure4_ale.png')
)

print("\n" + "=" * 55)
print(f"  Done. All outputs saved to ./{OUTPUT_DIR}/")
print("=" * 55)
