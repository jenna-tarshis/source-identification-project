# gev_utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme
from analysis_utils import compute_block_maxima

def fit_gev_to_block_maxima(block_maxima):
    if len(block_maxima) < 5:
        return None, None, None
    try:
        shape, loc, scale = genextreme.fit(block_maxima)
        gamma = -shape
        return gamma, loc, scale
    except Exception as e:
        print(f"GEV fit failed: {e}")
        return None, None, None

def compute_normalized_params(mu_j, sigma_j, gamma_j, j):
    if gamma_j == 0:
        mu_star = mu_j - sigma_j * np.log(j)
        sigma_star = sigma_j
    else:
        mu_star = mu_j - sigma_j * j**(-gamma_j) * ((j**gamma_j - 1) / gamma_j)
        sigma_star = sigma_j * j**(-gamma_j)
    return mu_star, sigma_star

def evaluate_stability_across_blocks(df, col, block_sizes):
    results = []
    for i, block_size in enumerate(block_sizes):
        maxima, _ = compute_block_maxima(df, col, block_size)
        gamma, mu, sigma = fit_gev_to_block_maxima(maxima)
        if None in (gamma, mu, sigma):
            continue
        mu_star, sigma_star = compute_normalized_params(mu, sigma, gamma, i + 1)
        try:
            nll = -np.sum(genextreme.logpdf(maxima, -gamma, loc=mu, scale=sigma))
        except:
            nll = np.nan
        results.append({
            "block_size": block_size,
            "block_hours": block_size / 3600,
            "j": i + 1,
            "gamma": gamma,
            "mu": mu,
            "sigma": sigma,
            "mu_star": mu_star,
            "sigma_star": sigma_star,
            "gamma_ci_low": np.nan,
            "gamma_ci_high": np.nan,
            "num_maxima": len(maxima),
            "block_maxima": maxima,
            "nll": nll
        })
    return pd.DataFrame(results)

def select_optimal_block_size(results_df, alpha=50):
    if results_df.empty:
        return None
    df = results_df.dropna(subset=["nll"])
    df = df[df['num_maxima'] >= 25]
    if df.empty:
        return None
    df['ci_width'] = 0  # no bootstrapped CI, so neutralize its influence
    df['score'] = df['nll']
    best_idx = df['score'].idxmin()
    return df.loc[best_idx, 'block_size']

def plot_gev_parameter_stability(results_df):
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    x_vals = results_df["block_hours"]
    axs[0].plot(x_vals, results_df["gamma"], 'o-', color='tab:blue')
    axs[0].set_ylabel("Shape Parameter γ")
    axs[0].set_title("GEV Shape Parameter vs Block Size")
    axs[0].grid(True)

    axs[1].plot(x_vals, results_df["mu_star"], marker='o', color='tab:orange')
    axs[1].set_ylabel("Normalized Location μ*")
    axs[1].set_title("Normalized Location Parameter vs Block Size")
    axs[1].grid(True)

    axs[2].plot(x_vals, results_df["sigma_star"], marker='o', color='tab:green')
    axs[2].set_ylabel("Normalized Scale σ*")
    axs[2].set_title("Normalized Scale Parameter vs Block Size")
    axs[2].grid(True)

    axs[3].plot(x_vals, results_df["nll"], marker='o', color='tab:red')
    axs[3].set_ylabel("Negative Log-Likelihood")
    axs[3].set_title("NLL vs Block Size")
    axs[3].set_xlabel("Block Size (hours)")
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

def plot_gev_fits(results_df, selected_block_size):
    sorted_df = results_df.sort_values("block_size").reset_index(drop=True)
    idx = sorted_df.index[sorted_df['block_size'] == selected_block_size].tolist()
    if not idx:
        print("Selected block size not found in results.")
        return
    i = idx[0]
    lower = max(0, i - 1)
    upper = min(len(sorted_df), i + 2)
    selected = sorted_df.iloc[lower:upper]

    fig, axs = plt.subplots(len(selected), 1, figsize=(8, 4 * len(selected)))
    if len(selected) == 1:
        axs = [axs]
    for ax, (_, row) in zip(axs, selected.iterrows()):
        maxima = row['block_maxima']
        gamma, mu, sigma = row['gamma'], row['mu'], row['sigma']
        x = np.linspace(min(maxima), max(maxima), 200)
        shape = -gamma
        pdf = genextreme.pdf(x, shape, loc=mu, scale=sigma)
        ax.hist(maxima, bins=200, density=True, alpha=0.6, color='skyblue', label='Block Maxima')
        ax.plot(x, pdf, 'r-', label='GEV Fit')
        ax.set_title(f"GEV Fit (Block Size = {row['block_hours']:.1f} hr)")
        ax.set_xlabel("Block Maxima Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True)
        textstr = f"γ = {gamma:.3f}\nμ = {mu:.2f}\nσ = {sigma:.2f}"
        ax.text(0.98, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout(h_pad=2.0)
    plt.show()

def plot_gev_cdf_fits(results_df, selected_block_size):    
    
    sorted_df = results_df.sort_values("block_size").reset_index(drop=True)
    idx = sorted_df.index[sorted_df['block_size'] == selected_block_size].tolist()
    if not idx:
        print("Selected block size not found in results.")
        return
    i = idx[0]
    lower = max(0, i - 1)
    upper = min(len(sorted_df), i + 2)
    selected = sorted_df.iloc[lower:upper]

    fig, axs = plt.subplots(len(selected), 1, figsize=(8, 4 * len(selected)))
    if len(selected) == 1:
        axs = [axs]
    for ax, (_, row) in zip(axs, selected.iterrows()):
        maxima = np.sort(row['block_maxima'])
        gamma, mu, sigma = row['gamma'], row['mu'], row['sigma']
        shape = -gamma
        x_vals = np.linspace(min(maxima), max(maxima), 200)
        gev_cdf = genextreme.cdf(x_vals, shape, loc=mu, scale=sigma)

        # Empirical CDF
        y_empirical = np.arange(1, len(maxima) + 1) / len(maxima)

        ax.plot(maxima, y_empirical, 'o', label='Empirical CDF', color='tab:blue')
        ax.plot(x_vals, gev_cdf, '-', label='GEV CDF', color='tab:red')
        ax.set_title(f"GEV CDF Fit (Block Size = {row['block_hours']:.1f} hr)")
        ax.set_xlabel("Block Maxima Value")
        ax.set_ylabel("Cumulative Probability")
        ax.legend()
        ax.grid(True)
    plt.tight_layout(h_pad=2.0)
    plt.show()