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

def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    excess_return = log_returns.mean(dim=1) - risk_free_rate
    std_dev = log_returns.std(dim=1, unbiased=False).clamp(min=1e-8)

    skewness = (log_returns - log_returns.mean(dim=1, keepdim=True)).pow(3).mean(dim=1) / std_dev.pow(3).clamp(min=1e-8)
    skewness = skewness.clamp(min=-1.0, max=1.0)

    kurtosis = (log_returns - log_returns.mean(dim=1, keepdim=True)).pow(4).mean(dim=1) / std_dev.pow(4).clamp(min=1e-8) - 3
    kurtosis = kurtosis.clamp(max=6.0)

    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 10).clamp(max=1.0))

    drawdown = (log_returns.cumsum(dim=1) - log_returns.cumsum(dim=1).max(dim=1, keepdim=True)[0]).min(dim=1)[0]
    max_drawdown = drawdown.clamp(min=0)
    drawdown_penalty = (1 / (1 + max_drawdown.abs())).clamp(max=1.0)

    entropy_measure = -torch.sum(torch.exp(log_returns - log_returns.mean(dim=1, keepdim=True)) * (log_returns - log_returns.mean(dim=1, keepdim=True)), dim=1) / n_periods
    entropy_penalty = torch.clamp(entropy_measure, min=1e-8)

    recent_returns = log_returns[:, -n_periods // 4:]
    recent_mean = recent_returns.mean(dim=1).clamp(min=1e-8)
    recent_std = recent_returns.std(dim=1).clamp(min=1e-8)

    volatility_scaling = (1 + std_dev / n_periods).clamp(min=1e-8)

    weights = torch.softmax(torch.arange(n_assets, dtype=log_returns.dtype, device=log_returns.device) / n_assets, dim=0)
    weighted_adjusted_sharpe = (weights * tail_adjusted_excess_return).sum()

    dynamic_scaling = (1 + recent_mean / (std_dev + 1e-8)).clamp(max=2.0)

    regime_indicator = (recent_std - std_dev).clamp(min=1e-8)
    regime_scaling = torch.exp(-regime_indicator)

    high_order_term = (1 + skewness.pow(2) * 0.5 + kurtosis / 10).clamp(min=1e-8)

    penalty_adjustment = (1 - drawdown_penalty) * (1 - (entropy_penalty / std_dev)).clamp(min=0, max=1)
    adjusted_sharpe = (weighted_adjusted_sharpe * dynamic_scaling * penalty_adjustment) / (std_dev * (1 + drawdown_penalty))

    final_metric = adjusted_sharpe * high_order_term * regime_scaling

    # Incorporate a time-varying entropy adjustment based on recent performance
    entropy_adjustment = torch.std(entropy_measure.view(-1, 1), dim=0).view(-1) / (std_dev + 1e-8)
    final_metric = final_metric * (1 - entropy_adjustment)

    return final_metric

def ndcg(scores: torch.Tensor, labels: torch.Tensor, percent: float = 0.25, log_base: int = 2) -> torch.Tensor:
    """Compute normalized discounted cumulative gain (NDCG) at a specific percentage cutoff.

    Args:
        scores (torch.Tensor): Predicted item scores (1D tensor).
        labels (torch.Tensor): True item labels (1D tensor).
        percent (float, optional): The percentage of the ranked list to consider (e.g., 0.25 for top 25%). Default is 0.25.
        log_base (int, optional): Base of the logarithm used for computing discounts. Default is 2.

    Returns:
        torch.Tensor: Normalized discounted cumulative gain at the specified percentage (scalar).
    """
    # Ensure percent is between 0 and 1
    if not (0.0 < percent <= 1.0):
        raise ValueError("percent must be a float between 0 and 1 (exclusive).")

    # Determine cutoff rank (top percent of the list)
    k = max(1, int(scores.size(0) * percent))  # Ensure at least 1 item is selected

    # Sort scores and labels in descending order of predicted scores
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices][:k]  # Take top-k labels

    # Compute DCG (Discounted Cumulative Gain) at k
    discounts = 1 / torch.log(torch.arange(2, k + 2, dtype=torch.float32)) / torch.log(torch.tensor(log_base, dtype=torch.float32))
    dcg = torch.sum(sorted_labels * discounts)

    # Compute IDCG (Ideal DCG) at k
    ideal_sorted_labels = torch.sort(labels, descending=True).values[:k]  # Top-k ideal labels
    ideal_dcg = torch.sum(ideal_sorted_labels * discounts)

    # Handle edge cases where ideal DCG is zero
    if ideal_dcg < 1e-8:  # Small epsilon for numerical stability
        return torch.tensor(0.0)

    # Compute NDCG@top percent
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
        "AlphaSharpe": robust_sharpe(train).squeeze(),
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
            "NDCG@25%": ndcg_corr
        })

    # Create a DataFrame for the results
    results_df = pd.DataFrame(results)
    return results_df

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

variants = [
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    excess_return = (log_returns.mean(dim=-1) - risk_free_rate).exp()
    std_dev = (log_returns.var(dim=-1) + 5e-3) * (log_returns.std(dim=-1) + 5e-3)
    return excess_return / std_dev.sqrt()
""",
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    adjusted_returns = log_returns - risk_free_rate
    excess_return = adjusted_returns.mean(dim=-1).exp()
    adjusted_std_dev = (log_returns.std(dim=-1) ** 2 + 5e-3).sqrt()
    downside_risk = (log_returns[log_returns < 0].std(dim=-1, unbiased=False) + (log_returns < 0).sum(dim=-1).float().clamp(min=1e-5).sqrt() * log_returns.std(dim=-1, unbiased=False)) / ((log_returns < 0).sum(dim=-1).float().clamp(min=1e-5) + 1e-5)
    risk_adjusted_std_dev = adjusted_std_dev + downside_risk
    return excess_return / risk_adjusted_std_dev
""",
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    dynamic_risk_free_rate = risk_free_rate + log_returns.mean(dim=-1).detach().std(dim=-1).exp() * 0.01
    adjusted_returns = log_returns - dynamic_risk_free_rate
    mean_excess_return = adjusted_returns.mean(dim=-1).exp()
    
    time_weighted_volatility = log_returns.unfold(1, 5, 1).std(dim=2).mean(dim=1)
    adjusted_std_dev = (log_returns.var(dim=-1, unbiased=False) + time_weighted_volatility).sqrt()

    downside_returns = log_returns[log_returns < 0]
    downside_risk = downside_returns.std(dim=-1, unbiased=False).nan_to_num(0)

    downside_count = (log_returns < 0).sum(dim=-1).float().clamp(min=1)
    safety_coefficient = (downside_count.sqrt() * adjusted_std_dev).nan_to_num(0)
    total_downside_risk = (downside_risk + safety_coefficient) / downside_count

    risk_adjusted_std_dev = adjusted_std_dev + total_downside_risk

    skewness = ((adjusted_returns - adjusted_returns.mean(dim=-1, keepdim=True)).pow(3).mean(dim=-1) /
                 (adjusted_std_dev.pow(3).clamp(min=1e-5)))
    kurtosis = ((adjusted_returns - adjusted_returns.mean(dim=-1, keepdim=True)).pow(4).mean(dim=-1) /
                 (adjusted_std_dev.pow(4).clamp(min=1e-5)))

    adjusted_excess_return = mean_excess_return * (1 + 0.5 * skewness - 0.25 * kurtosis)

    conditional_adjustment = 1 + (downside_count - 1).div(downside_count).clamp(min=0)

    tail_risk_adjustment = (log_returns < -0.02).float().mean(dim=-1)
    return adjusted_excess_return.mul(conditional_adjustment).div(risk_adjusted_std_dev) * (1 + skewness.abs() * 0.1 * tail_risk_adjustment)
""",
"""
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
""",
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    mean_return = log_returns.mean(dim=1)
    excess_return = mean_return - risk_free_rate
    std_dev = log_returns.std(dim=1, unbiased=False).clamp(min=1e-8)

    residuals = log_returns - mean_return.unsqueeze(1)
    skewness = (residuals.pow(3).mean(dim=1) / std_dev.pow(3)).clamp(min=-1.0, max=1.0)
    kurtosis = (residuals.pow(4).mean(dim=1) / std_dev.pow(4)).clamp(max=6.0) - 3
    
    drawdown = (log_returns.cumsum(dim=1) - log_returns.cumsum(dim=1).max(dim=1, keepdim=True)[0]).min(dim=1)[0]
    max_drawdown = drawdown.mean() 
    drawdown_penalty = torch.sigmoid(max_drawdown.abs()) 

    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 12).clamp(max=1.0)) * (1 - drawdown_penalty)
    adjusted_sharpe = tail_adjusted_excess_return / (std_dev * (1 + skewness / 6).clamp(min=1e-8))
    
    weights = torch.softmax(torch.arange(n_assets, dtype=log_returns.dtype, device=log_returns.device) / n_assets, dim=0)
    weighted_adjusted_sharpe = (weights * adjusted_sharpe).sum()

    out_of_sample_adjustment = (1 + (kurtosis.abs().clamp(max=3.0) - 3) / 12).sqrt()
    penalty_factor = (std_dev / (std_dev + 1e-8)).clamp(max=1.0)
    
    volatility_scaling = torch.rsqrt(torch.tensor(n_periods, dtype=log_returns.dtype, device=log_returns.device))

    adjustment_factor = (1 + (skewness.abs().clamp(max=3.0) / 8)) * (1 - torch.exp(-0.5 * kurtosis.abs() / 4))
    
    adjusted_metric = (weighted_adjusted_sharpe * volatility_scaling * out_of_sample_adjustment * (1 - penalty_factor) * adjustment_factor) / std_dev

    temporal_decay = log_returns[:, -n_periods//2:].mean(dim=1).clamp(min=1e-8)
    final_adjusted_metric = adjusted_metric * (1 + temporal_decay / std_dev)

    return final_adjusted_metric
""",
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    excess_return = log_returns.mean(dim=1) - risk_free_rate
    std_dev = log_returns.std(dim=1, unbiased=False).clamp(min=1e-8)

    residuals = log_returns - log_returns.mean(dim=1, keepdim=True)
    sk = (residuals.pow(3).mean(dim=1) / std_dev.pow(3)).clamp(min=-1.0, max=1.0)
    kt = (residuals.pow(4).mean(dim=1) / std_dev.pow(4)).clamp(max=6.0) - 3

    drawdown = (log_returns.cumsum(dim=1) - log_returns.cumsum(dim=1).max(dim=1, keepdim=True)[0]).min(dim=1)[0]
    max_drawdown = drawdown.mean()
    drawdown_penalty = torch.sigmoid(max_drawdown.abs())

    tail_adjusted_excess_return = excess_return * (1 - (kt.abs() / 12).clamp(max=1.0)) * (1 - drawdown_penalty)
    adjusted_sharpe = tail_adjusted_excess_return / (std_dev * (1 + sk / 6).clamp(min=1e-8))

    weights = torch.softmax(torch.arange(n_assets, dtype=log_returns.dtype, device=log_returns.device) / n_assets, dim=0)
    weighted_adjusted_sharpe = (weights * adjusted_sharpe).sum()

    vol_scaling = torch.rsqrt(torch.tensor(n_periods, dtype=log_returns.dtype, device=log_returns.device))

    decay_factors = log_returns[:, -n_periods // 2:].mean(dim=1).clamp(min=1e-8)
    dynamic_entropy = -torch.sum(torch.exp(log_returns) * log_returns, dim=1) / n_periods
    entropy_penalty = dynamic_entropy.clamp(min=1e-8)

    regime_factor = torch.where(excess_return > 0, torch.tensor(1.1, device=log_returns.device), torch.tensor(0.9, device=log_returns.device))

    future_volatility = log_returns[:,-n_periods//4:].std(dim=1, unbiased=False).clamp(min=1e-8)    
    final_adjusted_metric = (weighted_adjusted_sharpe * vol_scaling * (1 + decay_factors / std_dev) * regime_factor) / (entropy_penalty * future_volatility)

    return final_adjusted_metric
""",
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    excess_return = log_returns.mean(dim=1) - risk_free_rate
    std_dev = log_returns.std(dim=1, unbiased=False).clamp(min=1e-8)

    # Higher-order moments: skewness and kurtosis
    residuals = log_returns - log_returns.mean(dim=1, keepdim=True)
    skewness = (residuals.pow(3).mean(dim=1) / std_dev.pow(3)).clamp(min=-1.0, max=1.0)
    kurtosis = (residuals.pow(4).mean(dim=1) / std_dev.pow(4)).clamp(max=6.0) - 3
    
    # Tail-adjusted calculation
    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 12).clamp(max=1.0))

    # Enhanced drawdown penalties
    drawdown = (log_returns.cumsum(dim=1) - log_returns.cumsum(dim=1).max(dim=1, keepdim=True)[0]).min(dim=1)[0]
    max_drawdown = drawdown.mean()
    drawdown_penalty = (1 / (1 + max_drawdown.abs())).clamp(max=1.0)

    # Calculation of the dynamic metric
    adjusted_sharpe = tail_adjusted_excess_return * (1 - drawdown_penalty) / (std_dev * (1 + skewness / 6).clamp(min=1e-8))
    weights = torch.softmax(torch.arange(n_assets, dtype=log_returns.dtype, device=log_returns.device) / n_assets, dim=0)
    
    weighted_adjusted_sharpe = (weights * adjusted_sharpe).sum()

    # Temporal decay normalized to avoid overfitting
    ols_estimate = (log_returns[:, -n_periods // 4:].mean(dim=1) / (log_returns[:, -n_periods // 4:].std(dim=1) + 1e-8)).clamp(min=0.0)
    temporal_decay = log_returns[:, -n_periods // 2:].mean(dim=1).clamp(min=1e-8)

    # Incorporating entropy
    dynamic_entropy = -torch.sum(torch.exp(log_returns) * log_returns, dim=1) / n_periods
    entropy_penalty = dynamic_entropy.clamp(min=1e-8)

    # Regime factor with dynamic adjustment
    regime_factor = torch.where(excess_return > 0, torch.tensor(1.1, device=log_returns.device), torch.tensor(0.9, device=log_returns.device))

    # Future volatility adjustment
    future_volatility = log_returns[:, -n_periods // 4:].std(dim=1, unbiased=False).clamp(min=1e-8)    
    final_adjusted_metric = (weighted_adjusted_sharpe * (1 + ols_estimate * temporal_decay) * regime_factor) / (entropy_penalty * future_volatility)

    return final_adjusted_metric
""",
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    excess_return = log_returns.mean(dim=1) - risk_free_rate
    std_dev = log_returns.std(dim=1, unbiased=False).clamp(min=1e-8)

    residuals = log_returns - log_returns.mean(dim=1, keepdim=True)
    skewness = (residuals.pow(3).mean(dim=1) / std_dev.pow(3)).clamp(min=-1.0, max=1.0)
    kurtosis = (residuals.pow(4).mean(dim=1) / std_dev.pow(4)).clamp(max=6.0) - 3

    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 12).clamp(max=1.0))

    drawdown = (log_returns.cumsum(dim=1) - log_returns.cumsum(dim=1).max(dim=1, keepdim=True)[0]).min(dim=1)[0]
    max_drawdown = drawdown.mean()
    drawdown_penalty = torch.sigmoid(max_drawdown.abs())

    entropy_measure = -torch.sum(torch.exp(log_returns) * log_returns, dim=1) / n_periods
    entropy_penalty = entropy_measure.clamp(min=1e-8)

    adjusted_sharpe = tail_adjusted_excess_return * (1 - drawdown_penalty) / (std_dev * (1 + skewness / 6).clamp(min=1e-8))

    weights = torch.softmax(torch.arange(n_assets, dtype=log_returns.dtype, device=log_returns.device) / n_assets, dim=0)
    weighted_adjusted_sharpe = (weights * adjusted_sharpe).sum()

    ols_estimate = (log_returns[:, -n_periods // 4:].mean(dim=1) / (log_returns[:, -n_periods // 4:].std(dim=1) + 1e-8)).clamp(min=0.0)
    
    regime_factor = torch.where(excess_return > 0, torch.tensor(1.1, device=log_returns.device), torch.tensor(0.9, device=log_returns.device))

    future_volatility = log_returns[:, -n_periods // 4:].std(dim=1, unbiased=False).clamp(min=1e-8)    
    final_adjusted_metric = (weighted_adjusted_sharpe * (1 + ols_estimate) * regime_factor) / (entropy_penalty * future_volatility)

    adjusted_for_high_order_moments = final_adjusted_metric * (1 + (skewness.pow(2) + kurtosis) / 8).clamp(min=1e-8)

    return adjusted_for_high_order_moments
"""
]

for i, variant in enumerate(variants):

    print(f"--- Results of AlphaSharpe Variant {i+1}")

    exec(variant, globals())

    correlations_df = calculate_correlations(valid_data)
    print(correlations_df)

    # Evaluate and print the results
    percentage_increases = evaluate_asset_selection_with_percentage_increase(valid_data)
    print(percentage_increases)
