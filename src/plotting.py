import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.evaluate import compute_ale


# Charts styles 
MODEL_COLOURS = {
    'HAR':          '#2c3e50',
    'LogHAR':       '#8e44ad',
    'Ridge':        '#2980b9',
    'Lasso':        '#27ae60',
    'ElasticNet':   '#16a085',
    'RandomForest': '#e67e22',
    'NeuralNet':    '#c0392b',
}

# Figure 1: Relative MSE bar chart 
def plot_relative_mse_bars(rel_mse,
                            models:  list,
                            tickers: list,
                            output_path: str) -> None:
    
    fig, ax = plt.subplots(figsize=(11, 5))
    x     = np.arange(len(models))
    width = 0.14

    for j, ticker in enumerate(tickers):
        vals = [rel_mse.loc[ticker, m] for m in models]
        ax.bar(
            x + j * width, vals, width,
            label=ticker,
            color=plt.cm.Set2(j / len(tickers)),
            edgecolor='white', linewidth=0.4
        )

    ax.axhline(1.0, color='black', linestyle='--',
               linewidth=1.3, label='HAR benchmark')
    ax.set_xticks(x + width * (len(tickers) - 1) / 2)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Relative MSE  (HAR = 1.0)', fontsize=11)
    ax.set_title(
        'Figure 1: Out-of-Sample Forecast MSE Relative to HAR\n'
        '(MHAR dataset, one-day-ahead horizon)',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# Figure 2: Boxplot 
def plot_relative_mse_boxplot(rel_mse,
                               models: list,
                               output_path: str) -> None:
    
    fig, ax = plt.subplots(figsize=(10, 5))
    data_box = [rel_mse[m].values for m in models]

    bp = ax.boxplot(
        data_box, patch_artist=True, notch=False,
        medianprops=dict(color='black', linewidth=2)
    )
    for patch, m in zip(bp['boxes'], models):
        patch.set_facecolor(MODEL_COLOURS.get(m, '#95a5a6'))
        patch.set_alpha(0.75)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.3)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Relative MSE vs HAR', fontsize=11)
    ax.set_title(
        'Figure 2: Cross-Sectional Distribution of Relative MSE\n'
        '(MHAR dataset, one-day-ahead horizon)',
        fontsize=11, fontweight='bold'
    )
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# Figure 3: Actual vs Forecast 
def plot_actual_vs_forecast(y_test_store:  dict,
                             results_store: dict,
                             ticker:        str,
                             output_path:   str) -> None:
    
    y_te = y_test_store[ticker]
    x_ax = np.arange(len(y_te))

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    panels = [
        ('HAR',          MODEL_COLOURS['HAR']),
        ('RandomForest', MODEL_COLOURS['RandomForest']),
    ]

    for ax, (m, col) in zip(axes, panels):
        ax.fill_between(x_ax, y_te, alpha=0.20,
                        color='#7f8c8d', label='Actual RV')
        ax.plot(x_ax, y_te, color='#7f8c8d',
                linewidth=0.5, alpha=0.8)
        ax.plot(x_ax, results_store[ticker][m],
                color=col, linewidth=0.9,
                label=f'{m} Forecast', alpha=0.95)
        ax.set_ylabel('Realised Variance', fontsize=9)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.2)

    axes[0].set_title(
        f'Figure 3: Actual vs Forecast Realised Variance – '
        f'{ticker} (Test Set)',
        fontsize=11, fontweight='bold'
    )
    axes[1].set_xlabel('Test-set trading days', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


#  Figure 4: ALE plots 
def plot_ale(rf_model,
             X_tv_sc:    np.ndarray,
             feat_names: list,
             output_path: str) -> None:
    
    colours = ['#2980b9', '#e67e22', '#27ae60']
    fig, axes = plt.subplots(1, len(feat_names), figsize=(13, 4))

    for idx, (ax, fname, col) in enumerate(
            zip(axes, feat_names, colours)):
        z_mid, ale_vals = compute_ale(rf_model, X_tv_sc, idx, K=20)

        ax.plot(z_mid, ale_vals, color=col,
                linewidth=2.2, marker='o', markersize=3.5, zorder=3)
        ax.axhline(0, color='black', linestyle='--',
                   linewidth=0.8, alpha=0.5)
        ax.fill_between(z_mid, ale_vals, 0,
                        where=(ale_vals >= 0),
                        alpha=0.15, color=col)
        ax.fill_between(z_mid, ale_vals, 0,
                        where=(ale_vals < 0),
                        alpha=0.15, color='red')
        ax.set_xlabel(f'{fname}\n(standardised units)', fontsize=9)
        ax.set_ylabel('ALE  (impact on RV forecast)', fontsize=9)
        ax.set_title(fname, fontsize=10, fontweight='bold')
        ax.grid(alpha=0.2)

    fig.suptitle(
        'Figure 4: Accumulated Local Effects (ALE) – '
        'Random Forest\n'
        '(Positive values → higher predicted RV; '
        'each panel shows the isolated marginal effect)',
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
