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

def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0, epsilon: float = 1e-5) -> torch.Tensor:
    n_periods = log_returns.shape[-1]
    log_returns = log_returns.unsqueeze(0) if log_returns.ndim == 1 else log_returns

    # Calculate mean log excess return (expected log excess return)
    mean_log_excess_return = log_returns.mean(dim=-1) - risk_free_rate

    # Calculate standard deviation of log returns
    std_log_returns = log_returns.std(dim=-1, unbiased=False)

    # Alpha S1 calculation
    numerator = mean_log_excess_return.exp()
    denominator_s1 = torch.sqrt((std_log_returns.pow(2) + epsilon) * (std_log_returns + epsilon))
    alpha_s1 = numerator / denominator_s1

    # Downside Risk (DR) calculation
    negative_returns = log_returns[log_returns < 0]
    downside_risk = (
        negative_returns.std(dim=-1, unbiased=False) +
        (negative_returns.numel() ** 0.5) * std_log_returns
    ) / (negative_returns.numel() + epsilon)

    # Forecasted Volatility (V) calculation
    forecasted_volatility = log_returns[:, -n_periods // 4:].std(dim=-1, unbiased=False).sqrt()

    # Alpha S2 calculation
    denominator_s2 = torch.sqrt(std_log_returns.pow(2) + epsilon) + downside_risk + forecasted_volatility
    return numerator / denominator_s2

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
    n_assets, n_periods = log_returns.shape
    adjusted_rf_rate = risk_free_rate * (1 + log_returns.std(dim=1) / log_returns.mean(dim=1).clamp(min=1e-8))
    excess_return = log_returns.mean(dim=1) - adjusted_rf_rate

    rolling_std_dev = log_returns.unfold(1, n_periods // 4, 1).std(dim=2, unbiased=False).mean(dim=1).clamp(min=1e-8)

    residuals = log_returns - log_returns.mean(dim=1, keepdim=True)
    skewness = (residuals.pow(3).mean(dim=1) / rolling_std_dev.pow(3)).clamp(min=-1.0, max=1.0)
    kurtosis = (residuals.pow(4).mean(dim=1) / rolling_std_dev.pow(4)).clamp(max=6.0) - 3

    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 12).clamp(max=1.0))

    drawdown = (log_returns.cumsum(dim=1) - log_returns.cumsum(dim=1).max(dim=1, keepdim=True)[0]).min(dim=1)[0]
    max_drawdown = drawdown.mean()
    drawdown_penalty = (1 / (1 + max_drawdown.abs())).clamp(max=1.0)

    adjusted_sharpe = tail_adjusted_excess_return * (1 - drawdown_penalty) / (rolling_std_dev * (1 + skewness / 6).clamp(min=1e-8))

    volatility_forecast = log_returns[:, -n_periods // 4:].std(dim=1, unbiased=False).clamp(min=1e-8)
    regime_factor = torch.where(excess_return > 0, 1.1, 0.9)

    combined_metric = (adjusted_sharpe.mean() * regime_factor) / (rolling_std_dev + volatility_forecast)

    high_order_adjustment = (1 + (skewness.pow(2) + kurtosis) / 8).clamp(min=1e-8)
    temporal_decay_adjustment = log_returns.mean(dim=1).clamp(min=1e-8)

    final_metric = combined_metric * high_order_adjustment * (1 + temporal_decay_adjustment / rolling_std_dev)

    adaptive_penalty = (drawdown_penalty * volatility_forecast).clamp(min=1e-8)

    return final_metric - adaptive_penalty * drawdown_penalty
""",
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    excess_return = log_returns.mean(dim=1) - risk_free_rate

    rolling_std_dev = log_returns.unfold(1, n_periods // 4, 1).std(dim=2, unbiased=False).mean(dim=1).clamp(min=1e-8)

    residuals = log_returns - log_returns.mean(dim=1, keepdim=True)
    skewness = (residuals.pow(3).mean(dim=1) / rolling_std_dev.pow(3)).clamp(min=-1.0, max=1.0)
    kurtosis = (residuals.pow(4).mean(dim=1) / rolling_std_dev.pow(4)).clamp(max=6.0) - 3

    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 12).clamp(max=1.0))

    drawdown = (log_returns.cumsum(dim=1) - log_returns.cumsum(dim=1).max(dim=1, keepdim=True)[0]).min(dim=1)[0]
    max_drawdown = drawdown.mean()
    drawdown_penalty = (1 / (1 + max_drawdown.abs())).clamp(max=1.0)

    adjusted_sharpe = tail_adjusted_excess_return * (1 - drawdown_penalty) / (rolling_std_dev * (1 + skewness / 6).clamp(min=1e-8))

    volatility_forecast = log_returns[:, -n_periods // 4:].std(dim=1, unbiased=False).clamp(min=1e-8)
    regime_factor = (1 + (excess_return > 0).to(log_returns.dtype) * 0.1).clamp(max=1.2)

    combined_metric = (adjusted_sharpe.mean() * regime_factor) / (rolling_std_dev + volatility_forecast)

    high_order_adjustment = (1 + (skewness.pow(2) + kurtosis) / 8).clamp(min=1e-8)
    temporal_decay_adjustment = log_returns.mean(dim=1).clamp(min=1e-8)

    final_metric = combined_metric * high_order_adjustment * (1 + temporal_decay_adjustment / rolling_std_dev)

    return final_metric
""",
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    excess_return = log_returns.mean(dim=1) - risk_free_rate
    
    rolling_std_dev = log_returns.unfold(1, n_periods // 4, 1).std(dim=2, unbiased=False).mean(dim=1).clamp(min=1e-8)

    residuals = log_returns - log_returns.mean(dim=1, keepdim=True)
    skewness = (residuals.pow(3).mean(dim=1) / rolling_std_dev.pow(3)).clamp(min=-1.0, max=1.0)
    kurtosis = (residuals.pow(4).mean(dim=1) / rolling_std_dev.pow(4)).clamp(max=6.0) - 3

    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 12).clamp(max=1.0))

    drawdown = (log_returns.cumsum(dim=1) - log_returns.cumsum(dim=1).max(dim=1, keepdim=True)[0]).min(dim=1)[0]
    max_drawdown = drawdown.mean()
    drawdown_penalty = (1 / (1 + max_drawdown.abs())).clamp(max=1.0)

    adjusted_sharpe = tail_adjusted_excess_return * (1 - drawdown_penalty) / (rolling_std_dev * (1 + skewness / 6).clamp(min=1e-8))

    volatility_forecast = log_returns[:, -n_periods // 4:].std(dim=1, unbiased=False).clamp(min=1e-8)
    regime_factor = torch.where(excess_return > 0, 1.1, 0.9)

    combined_metric = (adjusted_sharpe.mean() * regime_factor) / (rolling_std_dev + volatility_forecast)

    high_order_adjustment = (1 + (skewness.pow(2) + kurtosis) / 8).clamp(min=1e-8)
    temporal_decay_adjustment = log_returns.mean(dim=1).clamp(min=1e-8)

    stochastic_drawdown_penalty = (1 + torch.exp(-drawdown_penalty)).clamp(max=1.0)
    final_metric = combined_metric * high_order_adjustment * (1 + temporal_decay_adjustment / rolling_std_dev) * stochastic_drawdown_penalty

    return final_metric
""",
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    excess_return = log_returns.mean(dim=1) - risk_free_rate

    rolling_std_dev = log_returns.unfold(1, n_periods // 4, 1).std(dim=2, unbiased=False).mean(dim=1).clamp(min=1e-8)

    weighted_returns = (log_returns - log_returns.mean(dim=1, keepdim=True)) / rolling_std_dev.unsqueeze(1)
    high_order_moments = torch.stack([weighted_returns.pow(3).mean(dim=1), weighted_returns.pow(4).mean(dim=1)], dim=1)

    skewness = high_order_moments[:, 0].clamp(min=-1.0, max=1.0)
    kurtosis = (high_order_moments[:, 1] / 3).clamp(max=6.0) - 3

    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 12).clamp(max=1.0))

    drawdown = (log_returns.cumsum(dim=1) - log_returns.cumsum(dim=1).max(dim=1, keepdim=True)[0]).min(dim=1)[0]
    max_drawdown = drawdown.mean()
    drawdown_penalty = (1 / (1 + max_drawdown.abs())).clamp(max=1.0)

    adjusted_sharpe = tail_adjusted_excess_return * (1 - drawdown_penalty) / (rolling_std_dev * (1 + skewness / 6).clamp(min=1e-8))

    volatility_forecast = log_returns[:, -n_periods // 4:].std(dim=1, unbiased=False).clamp(min=1e-8)
    regime_factor = torch.where(excess_return > 0, 1.1, 0.9)

    combined_metric = (adjusted_sharpe.mean() * regime_factor) / (rolling_std_dev + volatility_forecast)

    high_order_adjustment = (1 + (skewness.pow(2) + kurtosis) / 8).clamp(min=1e-8)
    decay_factor = (log_returns.mean(dim=1).exp() + 1).clamp(min=1e-8)

    final_metric = combined_metric * high_order_adjustment * decay_factor / (1 + volatility_forecast)

    return final_metric
""",
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    excess_return = log_returns.mean(dim=1) - risk_free_rate
    
    rolling_std_dev = log_returns.unfold(1, n_periods // 4, 1).std(dim=2, unbiased=False).mean(dim=1).clamp(min=1e-8)
    
    residuals = log_returns - log_returns.mean(dim=1, keepdim=True)
    skewness = (residuals.pow(3).mean(dim=1) / rolling_std_dev.pow(3)).clamp(min=-1.0, max=1.0)
    kurtosis = (residuals.pow(4).mean(dim=1) / rolling_std_dev.pow(4)).clamp(max=6.0) - 3

    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 12).clamp(max=1.0))

    drawdown = (log_returns.cumsum(dim=1) - log_returns.cumsum(dim=1).max(dim=1, keepdim=True)[0]).min(dim=1)[0]
    max_drawdown = drawdown.mean()
    drawdown_penalty = (1 / (1 + max_drawdown.abs())).clamp(max=1.0)

    adjusted_sharpe = tail_adjusted_excess_return * (1 - drawdown_penalty) / (rolling_std_dev * (1 + skewness / 10).clamp(min=1e-8))

    volatility_forecast = log_returns[:, -n_periods // 4:].std(dim=1, unbiased=False).clamp(min=1e-8)
    regime_factor = torch.where(excess_return > 0, 1.05, 0.95)

    combined_metric = (adjusted_sharpe.mean() * regime_factor) / (rolling_std_dev + volatility_forecast)

    high_order_adjustment = (1 + (skewness.pow(2) + kurtosis) / 12).clamp(min=1e-8)
    temporal_decay_adjustment = log_returns.mean(dim=1).clamp(min=1e-8)

    final_metric = combined_metric * high_order_adjustment * (1 + temporal_decay_adjustment / rolling_std_dev)

    # Reflecting transaction costs with a simple assumption
    transaction_cost = 0.001 * log_returns.std(dim=1).clamp(min=1e-8)
    final_metric = final_metric - transaction_cost

    return final_metric
""",
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    excess_return = log_returns.mean(dim=1) - risk_free_rate

    rolling_std_dev = log_returns.unfold(1, n_periods // 4, 1).std(dim=2, unbiased=False).mean(dim=1).clamp(min=1e-8)

    residuals = log_returns - log_returns.mean(dim=1, keepdim=True)
    skewness = (residuals.pow(3).mean(dim=1) / rolling_std_dev.pow(3)).clamp(min=-1.0, max=1.0)
    kurtosis = (residuals.pow(4).mean(dim=1) / rolling_std_dev.pow(4)).clamp(max=6.0) - 3

    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 8).clamp(max=1.0))

    drawdown = (log_returns.cumsum(dim=1) - log_returns.cumsum(dim=1).max(dim=1, keepdim=True)[0]).min(dim=1)[0]
    max_drawdown = drawdown.mean()
    drawdown_penalty = (1 / (1 + max_drawdown.abs())).clamp(max=1.0)

    adjusted_sharpe = tail_adjusted_excess_return * (1 - drawdown_penalty) / (rolling_std_dev * (1 + skewness / 4).clamp(min=1e-8))

    volatility_forecast = log_returns[:, -n_periods // 4:].std(dim=1, unbiased=False).clamp(min=1e-8)
    regime_factor = 1.0 + (torch.where(excess_return > 0, 0.1, -0.1)) 

    combined_metric = (adjusted_sharpe.mean() * regime_factor) / (rolling_std_dev + volatility_forecast)

    high_order_adjustment = (1 + (skewness.pow(2) + kurtosis) / 4).clamp(min=1e-8)
    temporal_decay_adjustment = log_returns.mean(dim=1).clamp(min=1e-8)

    final_metric = combined_metric * high_order_adjustment * (1 + temporal_decay_adjustment / rolling_std_dev)

    regime_adjusted_penalty = (1 + torch.abs(drawdown_penalty) * regime_factor).clamp(min=1.0)
    final_metric *= regime_adjusted_penalty

    return final_metric
""",
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    adjusted_rf_rate = risk_free_rate * (1 + log_returns.std(dim=1) / log_returns.mean(dim=1).clamp(min=1e-8))
    excess_return = log_returns.mean(dim=1) - adjusted_rf_rate

    rolling_std_dev = log_returns.unfold(1, n_periods // 4, 1).std(dim=2, unbiased=False).mean(dim=1).clamp(min=1e-8)

    residuals = log_returns - log_returns.mean(dim=1, keepdim=True)
    skewness = (residuals.pow(3).mean(dim=1) / rolling_std_dev.pow(3)).clamp(min=-1.0, max=1.0)
    kurtosis = (residuals.pow(4).mean(dim=1) / rolling_std_dev.pow(4)).clamp(max=6.0) - 3

    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 10).clamp(max=1.0))

    drawdown = (log_returns.cumsum(dim=1) - log_returns.cumsum(dim=1).max(dim=1, keepdim=True)[0]).min(dim=1)[0]
    max_drawdown = drawdown.mean()
    drawdown_penalty = (1 / (1 + max_drawdown.abs())).clamp(max=1.0)

    adjusted_sharpe = tail_adjusted_excess_return * (1 - drawdown_penalty) / (rolling_std_dev * (1 + skewness / 6).clamp(min=1e-8))

    volatility_forecast = log_returns[:, -n_periods // 4:].std(dim=1, unbiased=False).clamp(min=1e-8)
    regime_factor = torch.where(excess_return > 0, 1.1, 0.9)

    combined_metric = (adjusted_sharpe.mean() * regime_factor) / (rolling_std_dev + volatility_forecast)

    high_order_adjustment = (1 + (skewness.pow(2) + kurtosis) / 12).clamp(min=1e-8)

    entropy = -torch.sum(log_returns.softmax(dim=1) * log_returns.softmax(dim=1).log(), dim=1).clamp(min=1e-8)

    transaction_cost = 0.001 * log_returns.std(dim=1).clamp(min=1e-8)

    final_metric = combined_metric * high_order_adjustment * (1 + 0.5 * entropy) - transaction_cost
    
    temporal_decay_adjustment = log_returns.mean(dim=1).clamp(min=1e-8)
    
    high_order_adjustment = high_order_adjustment * (1 + temporal_decay_adjustment / rolling_std_dev)

    return final_metric * high_order_adjustment * (1 + volatility_forecast / rolling_std_dev)
""",
"""
def robust_sharpe(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    n_assets, n_periods = log_returns.shape
    decay_factor = torch.exp(-torch.arange(n_periods, dtype=log_returns.dtype, device=log_returns.device) / (n_periods / 2))
    weighted_log_returns = log_returns * decay_factor[None, :]

    adjusted_rf_rate = risk_free_rate * (1 + weighted_log_returns.std(dim=1) / weighted_log_returns.mean(dim=1).clamp(min=1e-8))
    excess_return = weighted_log_returns.mean(dim=1) - adjusted_rf_rate

    rolling_std_dev = weighted_log_returns.unfold(1, n_periods // 4, 1).std(dim=2, unbiased=False).mean(dim=1).clamp(min=1e-8)

    residuals = weighted_log_returns - weighted_log_returns.mean(dim=1, keepdim=True)
    skewness = (residuals.pow(3).mean(dim=1) / rolling_std_dev.pow(3)).clamp(min=-1.0, max=1.0)
    kurtosis = (residuals.pow(4).mean(dim=1) / rolling_std_dev.pow(4)).clamp(max=6.0) - 3

    tail_adjusted_excess_return = excess_return * (1 - (kurtosis.abs() / 12).clamp(max=1.0))

    drawdown = (weighted_log_returns.cumsum(dim=1) - weighted_log_returns.cumsum(dim=1).max(dim=1, keepdim=True)[0]).min(dim=1)[0]
    max_drawdown = drawdown.mean()
    drawdown_penalty = (1 / (1 + max_drawdown.abs())).clamp(max=1.0)

    adjustment_factor = (1 + skewness.abs() / 4).clamp(min=1e-8)
    adjusted_sharpe = tail_adjusted_excess_return * (1 - drawdown_penalty) / (rolling_std_dev * adjustment_factor)

    volatility_forecast = weighted_log_returns[:, -n_periods // 4:].std(dim=1, unbiased=False).clamp(min=1e-8)
    regime_factor = torch.where(excess_return > 0, 1.1, 0.9)

    combined_metric = (adjusted_sharpe.mean() * regime_factor) / (rolling_std_dev + volatility_forecast)

    high_order_adjustment = (1 + (skewness.pow(2) + kurtosis) / 8).clamp(min=1e-8)
    temporal_decay_adjustment = (1 + weighted_log_returns.mean(dim=1)).clamp(min=1e-8)

    final_metric = combined_metric * high_order_adjustment * temporal_decay_adjustment

    entropy = -torch.sum(weighted_log_returns.softmax(dim=1) * weighted_log_returns.softmax(dim=1).log(), dim=1).clamp(min=1e-8)
    final_metric *= (1 + 0.5 * entropy)

    transaction_cost = 0.001 * weighted_log_returns.std(dim=1).clamp(min=1e-8)
    final_metric = final_metric - transaction_cost

    return final_metric
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


from flaml import AutoML
import matplotlib.pyplot as plt

cutoff_index = valid_data.size(1) // 5

# Step 1: Extract Features from Robust Sharpe Variants
features = []
for i, variant in enumerate(variants):  # Iterate through all variants
    exec(variant, globals())  # Execute each variant definition
    features.append(robust_sharpe(valid_data[:, :cutoff_index]).numpy())  # Extract features using the robust_sharpe function

# Combine features into a DataFrame
features_df = pd.DataFrame(features).T  # Shape: (n_assets, n_variants)
features_df.columns = [f"Variant_{i+1}" for i in range(len(variants))]

# Step 2: Calculate Sharpe Ratios for the Test Period
test_data = valid_data[:, cutoff_index:]
sharpe_test = calculate_psr(test_data.T)[0].numpy()  # Sharpe ratios for the test period

# Step 3: Define Groups
groups = np.arange(len(features_df))  # Each asset gets its unique group ID

# Step 4: Train FLAML Model for Regression with Groups
automl = AutoML()
automl.fit(
    X_train=features_df, 
    y_train=sharpe_test,  # Use actual Sharpe ratios as targets
    groups=groups,  # Use groups for grouped cross-validation
    task="regression",  # Regression task
    metric="mae",  # Optimize for R² score
    eval_method="cv",  # Use cross-validation
    time_budget=60,  # Time budget in seconds (5 minutes)
    estimator_list=["xgboost"],  # Use LightGBM for regression
    verbose=1,  # Verbose output
)

predicted_sharpe = automl.predict(features_df)

spearman_corr, _ = spearmanr(predicted_sharpe, sharpe_test)
kendall_corr, _ = kendalltau(predicted_sharpe, sharpe_test)

print(f"Ensemble Spearman Correlation: {spearman_corr:.6f}")
print(f"Ensemble Kendall Correlation: {kendall_corr:.6f}")

# Calculate NDCG correlation
ndcg_corr = ndcg(torch.tensor(predicted_sharpe, dtype=torch.float32), 
    torch.tensor(sharpe_test, dtype=torch.float32)).item()

print(f"Ensemble NDCG@25% Score: {ndcg_corr:.6f}")

# Step 5: Extract Feature Importance
feature_importances = automl.model.estimator.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": features_df.columns,
    "Importance": feature_importances
})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

# Print feature importances in a readable format
print("Ensemble Variant Importances:")
print(feature_importance_df.to_string(index=False, float_format="%.2f"))

# Plot the ranked feature importances
plt.figure(figsize=(12, 6))
plt.bar(feature_importance_df["Feature"], feature_importance_df["Importance"], color="skyblue")
plt.xlabel("Feature (Variant)", fontsize=12)
plt.ylabel("Percentage Importance", fontsize=12)
plt.title("Ensemble Variant Importances", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.show()
