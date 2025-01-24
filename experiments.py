import _pickle as cPickle
import torch, pdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

with open('Dataset.pkl', 'rb') as f: Dataset = cPickle.load(f)
assets = list(Dataset.columns.values)
valid_data = np.array(Dataset).T
valid_data = torch.from_numpy(valid_data).float()

def torch_cdf(x):
    neg_ones = x < 0
    x = x.abs()
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    k = 1.0 / (1.0 + 0.2316419 * x)
    k2 = k * k
    k3 = k2 * k
    k4 = k3 * k
    k5 = k4 * k
    c = (a1 * k + a2 * k2 + a3 * k3 + a4 * k4 + a5 * k5)
    phi = 1.0 - c * (-x*x/2.0).exp() * 0.3989422804014327
    phi[neg_ones] = 1.0 - phi[neg_ones]
    return phi

def calculate_psr(rewards):
    mean, std = rewards.mean(dim=0), rewards.std(dim=0)
    rdiff = rewards - mean
    zscore = rdiff / (std + 1e-8)
    skew = (zscore**3).mean(dim=0)
    kurto = ((zscore**4).mean(dim=0) - 4) / 4
    sharpe = mean / (std + 1e-8)
    #sharpe[sharpe.isnan()] = 0.0
    psr_in  = (1 - skew * sharpe + kurto * sharpe**2) / (len(rewards)-1)
    psr_out = torch_cdf(sharpe / psr_in.sqrt())
    psr_out[psr_out.isnan()] = 0.0
    return sharpe, psr_out  

"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    mean_return = log_returns.mean(dim=-1).exp()
    std_dev = (log_returns.var(dim=-1) + 5e-3) * (log_returns.std(dim=-1) + 5e-3)
    return (mean_return - risk_free_rate) / std_dev.sqrt()

def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0, decay_factor=0.94) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    mean_return = log_returns.mean(dim=1)
    excess_return = mean_return - risk_free_rate
    std_dev = log_returns.std(dim=1).clamp_min(1e-8)

    residuals = log_returns - mean_return.unsqueeze(1)
    skewness = (residuals.pow(3).mean(dim=1) / std_dev.pow(3)).clamp(min=-1.0, max=1.0)
    kurtosis = (residuals.pow(4).mean(dim=1) / std_dev.pow(4) - 3).clamp(min=-1.0, max=3.0)

    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 12))
    adjusted_sharpe = tail_adjusted_excess_return / (std_dev * (1 + skewness / 6).clamp(min=1e-8))

    weights = (1 - decay_factor) * decay_factor ** torch.arange(n_assets, dtype=log_returns.dtype, device=log_returns.device)
    weighting_factor = weights / weights.sum()
    weighted_adjusted_sharpe = (weighting_factor * adjusted_sharpe).sum()

    scaling_factor = torch.rsqrt(torch.tensor(n_periods, dtype=log_returns.dtype, device=log_returns.device)) * (1 + skewness.abs() / 8)
    penalty_factor = (std_dev / (std_dev + 1e-8)).clamp(max=1.0)

    out_of_sample_adjustment = (1 + (kurtosis.abs().clamp(max=3.0) - 3) / 12).sqrt()
    return (weighted_adjusted_sharpe * scaling_factor * out_of_sample_adjustment * (1 - penalty_factor)) / std_dev * (1 + (skewness.abs().clamp(max=3.0) / 8)) * (1 - torch.exp(-kurtosis.abs() / 4))

def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    mean_return = log_returns.mean(dim=1)
    excess_return = mean_return - risk_free_rate
    std_dev = log_returns.std(dim=1, unbiased=False).clamp(min=1e-8)

    residuals = log_returns - mean_return.unsqueeze(1)
    skewness = (residuals.pow(3).mean(dim=1) / std_dev.pow(3)).clamp(min=-1.0, max=1.0)
    kurtosis = (residuals.pow(4).mean(dim=1) / std_dev.pow(4)).clamp(max=6.0) - 3

    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 12).clamp(max=1.0))
    adjusted_sharpe = tail_adjusted_excess_return / (std_dev * (1 + skewness / 6).clamp(min=1e-8))

    weights = torch.softmax(torch.arange(n_assets, dtype=log_returns.dtype, device=log_returns.device) / n_assets, dim=0)
    weighted_adjusted_sharpe = (weights * adjusted_sharpe).sum()

    out_of_sample_adjustment = (1 + (kurtosis.abs().clamp(max=3.0) - 3) / 12).sqrt()
    penalty_factor = (std_dev / (std_dev + 1e-8)).clamp(max=1.0)
    
    scaling_factor = torch.rsqrt(torch.tensor(n_periods, dtype=log_returns.dtype, device=log_returns.device)) * (1 + skewness.abs() / 8)

    adjustment_factor = (1 + (skewness.abs().clamp(max=3.0) / 8)) * (1 - torch.exp(-0.5 * kurtosis.abs() / 4))
    
    return (weighted_adjusted_sharpe * scaling_factor * out_of_sample_adjustment * (1 - penalty_factor)) * adjustment_factor * (1 + torch.log1p(std_dev + 1e-8) / 2) / std_dev
"""

def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0, decay_factor = 0.94) -> torch.Tensor:
    # Shape variables
    n_assets, n_periods = log_returns.shape
    # Compute mean, standard deviation, and residuals
    mean_return = log_returns.mean(dim=1)
    std_dev = log_returns.std(dim=1) + 1e-8  # Avoid division by zero
    residuals = log_returns - mean_return.unsqueeze(1)
    # Compute higher-order statistics
    residual_std = residuals.std(dim=1) + 1e-8
    adjusted_std = std_dev * (1 + residual_std / std_dev)
    skewness = residuals.pow(3).mean(dim=1) / adjusted_std.pow(3)
    kurtosis = residuals.pow(4).mean(dim=1) / adjusted_std.pow(4) - 3
    # Tail-adjusted excess return
    excess_return = mean_return - risk_free_rate
    tail_adjusted_excess_return = excess_return * (1 - (kurtosis - 3) / 24)
    # Adjusted Sharpe ratio
    adjusted_sharpe = tail_adjusted_excess_return / adjusted_std * (1 + skewness / 6)
    # Exponential decay weights
    weights = (1 - decay_factor) * decay_factor ** torch.arange(
        n_assets, dtype=log_returns.dtype, device=log_returns.device
    )
    # Compute weighted adjusted Sharpe ratio
    cumulative_weights = weights.cumsum(dim=0) + 1e-8
    weighted_adjusted_sharpe = (weights * adjusted_sharpe).cumsum(dim=0) / cumulative_weights
    # Out-of-sample adjustment factor
    oos_adjustment = (1 + (kurtosis - 3) / 24).sqrt() * (1 - skewness / 12)
    # Final scaling with the number of periods
    scaling_factor = torch.sqrt(torch.tensor(n_periods, dtype=log_returns.dtype, device=log_returns.device))
    return (weighted_adjusted_sharpe * scaling_factor) / adjusted_std * oos_adjustment

def ndcg(scores: torch.Tensor, labels: torch.Tensor, log_base: int = 2) -> torch.Tensor:
    """Compute normalized discounted cumulative gain (NDCG) without cutoffs.

    Args:
        scores (torch.Tensor): Predicted item scores.
        labels (torch.Tensor): True item labels.

    Returns:
        torch.Tensor: Normalized discounted cumulative gain.
    """
    # Sort scores and labels in descending order of predicted scores
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices]

    # Compute DCG (Discounted Cumulative Gain)
    discounts = 1 / torch.log2(torch.arange(2, scores.size(0) + 2, dtype=torch.float32))
    dcg = torch.sum(sorted_labels * discounts)

    # Compute IDCG (Ideal DCG)
    ideal_sorted_labels = torch.sort(labels, descending=True).values
    ideal_dcg = torch.sum(ideal_sorted_labels * discounts)

    # Handle edge cases where ideal DCG is zero
    if ideal_dcg == 0:
        return torch.tensor(0.0)

    # Compute NDCG
    return dcg / ideal_dcg

# Assuming calculate_psr and robust_sharpe are defined elsewhere
def calculate_correlations(log_returns):
    cutoff_index = log_returns.size(1) // 5
    train = log_returns[:, :cutoff_index]
    test = log_returns[:, cutoff_index:]

    sharpe, psr = calculate_psr(train.T)
    sharpe_test = calculate_psr(test.T)[0]

    metrics = {
        "Sharpe": sharpe,
        "PSR": psr,  
        "AlphaSharpe": robust_sharpe(train),
    }

    results = []
    for metric_name, score_first_half in metrics.items():
        score_first_half_np = score_first_half.numpy()

        # Calculate correlations
        spearman_corr, _ = spearmanr(score_first_half_np, sharpe_test.numpy())
        kendall_corr, _ = kendalltau(score_first_half_np, sharpe_test.numpy())
        ndcg_corr = ndcg(score_first_half, sharpe_test).item()

        results.append({
            "Metric": metric_name,
            "Spearman": spearman_corr,
            "Kendall": kendall_corr,
            "NDCG": ndcg_corr
        })

    # Create a DataFrame for the results
    results_df = pd.DataFrame(results)
    return results_df


correlations_df = calculate_correlations(valid_data)
print(correlations_df)

def evaluate_asset_selection_with_percentage_increase(log_returns):
    """
    Evaluate portfolio performances based on asset selection using Sharpe, PSR, and AlphaSharpe metrics.
    Calculate percentage increases of AlphaSharpe compared to Sharpe and PSR.
    
    Args:
        log_returns (torch.Tensor): Log returns (n_assets x n_periods).
    
    Returns:
        pd.DataFrame: Test performance and percentage increases for each metric and asset selection percentage.
    """
    cutoff_index = log_returns.size(1) // 5
    train = log_returns[:, :cutoff_index]
    test = log_returns[:, cutoff_index:]
    
    metrics = {
        "Sharpe": lambda x: calculate_psr(x.T)[0],  # Use Sharpe ratio
        "PSR": lambda x: calculate_psr(x.T)[1],    # Use PSR
        "AlphaSharpe": lambda x: robust_sharpe(x), # Use robust Sharpe
    }
    
    percentages = [10, 15, 20, 25]  # Percentages of assets to select
    results = {metric_name: [] for metric_name in metrics}

    for metric_name, metric_fn in metrics.items():
        # Calculate the metric for all assets in the training data
        metric_values = metric_fn(train).detach().numpy()
        sorted_indices = np.argsort(metric_values)[::-1]  # Sort assets in descending order

        for pct in percentages:
            num_assets = max(1, len(sorted_indices) * pct // 100)  # Number of assets to select
            selected_indices = sorted_indices[:num_assets].copy()  # Copy to avoid negative strides
            
            # Create equal-weighted portfolio
            selected_weights = torch.zeros(log_returns.size(0))
            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long)
            selected_weights.index_fill_(0, selected_indices_tensor, 1.0 / num_assets)

            # Evaluate portfolio performance on test data
            portfolio_test_returns = (selected_weights @ test).unsqueeze(0)
            test_sharpe = calculate_psr(portfolio_test_returns.T)[0].item()
            
            results[metric_name].append(test_sharpe)

    # Calculate percentage increases for AlphaSharpe compared to Sharpe and PSR
    percentage_differences = []
    for i, pct in enumerate(percentages):
        alpha_sharpe = results["AlphaSharpe"][i]
        sharpe = results["Sharpe"][i]
        psr = results["PSR"][i]
        
        increase_vs_sharpe = 100 * (alpha_sharpe - sharpe) / sharpe if sharpe != 0 else np.nan
        increase_vs_psr = 100 * (alpha_sharpe - psr) / psr if psr != 0 else np.nan
        
        percentage_differences.append({
            "Percentage": pct,
            "AlphaSharpe vs Sharpe (%)": increase_vs_sharpe,
            "AlphaSharpe vs PSR (%)": increase_vs_psr
        })
    
    # Convert results to DataFrame
    percentage_differences_df = pd.DataFrame(percentage_differences)
    return percentage_differences_df

# Evaluate and print the results
percentage_increases = evaluate_asset_selection_with_percentage_increase(valid_data)
print(percentage_increases)

import matplotlib.pyplot as plt

# Extract data from the DataFrame
percentages = percentage_increases["Percentage"].tolist()
alpha_vs_sharpe = percentage_increases["AlphaSharpe vs Sharpe (%)"].tolist()
alpha_vs_psr = percentage_increases["AlphaSharpe vs PSR (%)"].tolist()

# Plotting
plt.figure(figsize=(10, 6))

# Plot AlphaSharpe vs Sharpe
plt.plot(percentages, alpha_vs_sharpe, marker='o', label='AlphaSharpe vs Sharpe (%)')

# Plot AlphaSharpe vs PSR
plt.plot(percentages, alpha_vs_psr, marker='s', label='AlphaSharpe vs PSR (%)')

# Add labels, title, and legend
plt.xlabel('Percentage of Assets Selected', fontsize=12)
plt.ylabel('Percentage Increase (%)', fontsize=12)
plt.title('Performance Improvement of AlphaSharpe Portfolio vs Traditional Metrics', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Highlight key points
for i, pct in enumerate(percentages):
    plt.text(pct, alpha_vs_sharpe[i] + 1, f"{alpha_vs_sharpe[i]:.1f}%", ha='center', fontsize=10)
    plt.text(pct, alpha_vs_psr[i] + 1, f"{alpha_vs_psr[i]:.1f}%", ha='center', fontsize=10)

# Save the plot to a file
plt.tight_layout()
plt.savefig("performance_improvement_plot.png", dpi=300)  # Save as PNG with 300 dpi resolution

# Show plot
plt.show()
