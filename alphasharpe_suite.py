import _pickle as cPickle
import torch
import numpy as np

def alphasharpe_metric(
    excess_log_returns: torch.Tensor,
    epsilon: float = 1.5e-5,
    downside_risk_factor: float = 2.0,
    forecast_volatility_factor: float = 1.33,
    forecast_window: int = 3
) -> torch.Tensor:
    n_periods = excess_log_returns.shape[-1]
    excess_log_returns = excess_log_returns.unsqueeze(0) if excess_log_returns.ndim == 1 else excess_log_returns
    
    # Calculate mean log excess return (expected log excess return)
    mean_log_excess_return = excess_log_returns.mean(dim=-1)

    # Calculate standard deviation of log returns
    std_excess_log_returns = excess_log_returns.std(dim=-1, unbiased=False)

    # Downside Risk (DR) calculation
    negative_returns = excess_log_returns[excess_log_returns < 0]
    downside_risk = downside_risk_factor * (
        negative_returns.std(dim=-1, unbiased=False) +
        (negative_returns.numel() ** 0.5) * std_excess_log_returns
    ) / (negative_returns.numel() + epsilon)

    # Forecasted Volatility (V) calculation
    forecasted_volatility = forecast_volatility_factor * excess_log_returns[:, -n_periods // forecast_window:].std(dim=-1, unbiased=False).sqrt()
    return mean_log_excess_return.exp() / (std_excess_log_returns + downside_risk + forecasted_volatility)

def alphasharpe_portfolio(excess_log_returns: torch.Tensor) -> torch.Tensor:
    # Compute the covariance matrix of excess log returns and add a small diagonal component for numerical stability
    cov_matrix = excess_log_returns.cov() + 1e-6 * torch.eye(excess_log_returns.shape[0], device=excess_log_returns.device)

    # Compute risk-adjusted returns using the inverse covariance matrix and clamp negative ones to zero (ensuring positive weights)
    risk_adjusted_returns = (torch.linalg.inv(cov_matrix) @ excess_log_returns.mean(dim=1)).clamp(min=0.0)

    # Adjust returns by incorporating the stability factor and volatility adjustment (sqrt of the covariance matrix diagonal)
    enhanced_returns = risk_adjusted_returns * (1 + risk_adjusted_returns.std()) 
    enhanced_returns /= ((torch.diagonal(cov_matrix) + 1e-6).sqrt() + 1e-8)

    # Compute portfolio weights using the softmax function for normalization
    weights = enhanced_returns.softmax(dim=0)

    # Apply entropy-based regularization to encourage diversification by penalizing concentrated allocations
    final_weights = (weights * (weights.mean() * torch.log(weights + 1e-8)).exp()).clamp(min=0.0)

    return final_weights / final_weights.sum()

with open('Dataset.pkl', 'rb') as f: Dataset = cPickle.load(f)
valid_data = np.array(Dataset).T
valid_data = torch.from_numpy(valid_data).float()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
valid_data = valid_data.to(device)
cutoff_index = valid_data.size(1) // 5
train = valid_data[:, :cutoff_index]
test = valid_data[:, cutoff_index:]

# Step 1: Extract only top 25% of assets based on AlphaSharpe metric for portfolio allocation
top_indices = torch.argsort(alphasharpe_metric(train), descending=True)[:train.shape[0] // 4]  

# Step 2: Compute portfolio weights using AlphaSharpe Portfolio function
portfolio_weights = alphasharpe_portfolio(train[top_indices, :])

# Step 3: Compute portfolio return on the test set
portfolio_returns = (portfolio_weights.unsqueeze(1) * test[top_indices, :]).sum(dim=0)

# Step 4: Compute Sharpe Ratio on test set
sharpe_ratio = portfolio_returns.mean() / (portfolio_returns.std(unbiased=False) + 1e-8) 
print(f"AlphaSharpe Out-of-Sample Sharpe Ratio: {sharpe_ratio.item():.6f}")
