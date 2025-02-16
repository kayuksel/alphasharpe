import _pickle as cPickle
import torch
import numpy as np

def sharpe_ratio_metric(excess_log_returns: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Compute the traditional Sharpe ratio (ignoring risk-free rate)."""
    mean_return = excess_log_returns.mean(dim=-1)
    std_return = excess_log_returns.std(dim=-1, unbiased=False) + epsilon
    return mean_return / std_return

def alphasharpe_metric(excess_log_returns: torch.Tensor, epsilon: float = 1.5e-5,
    downside_risk_factor: float = 2.0, forecast_volatility_factor: float = 1.33, forecast_window: int = 3) -> torch.Tensor:

    n_periods = excess_log_returns.shape[-1]
    excess_log_returns = excess_log_returns.unsqueeze(0) if excess_log_returns.ndim == 1 else excess_log_returns
    
    mean_log_excess_return = excess_log_returns.mean(dim=-1)
    std_excess_log_returns = excess_log_returns.std(dim=-1, unbiased=False)

    # Downside Risk Calculation
    negative_returns = excess_log_returns[excess_log_returns < 0]
    downside_risk = downside_risk_factor * (
        negative_returns.std(dim=-1, unbiased=False) +
        (negative_returns.numel() ** 0.5) * std_excess_log_returns
    ) / (negative_returns.numel() + epsilon)

    # Forecasted Volatility Calculation
    forecasted_volatility = forecast_volatility_factor * excess_log_returns[:, -n_periods // forecast_window:].std(dim=-1, unbiased=False).sqrt()
    
    return mean_log_excess_return.exp() / (std_excess_log_returns + downside_risk + forecasted_volatility)

def alphasharpe_portfolio(excess_log_returns: torch.Tensor) -> torch.Tensor:
    # Compute risk-adjusted returns
    cov_matrix = excess_log_returns.cov() + 1e-6 * torch.eye(excess_log_returns.shape[0], device=excess_log_returns.device)
    risk_adjusted_returns = (torch.linalg.inv(cov_matrix) @ excess_log_returns.mean(dim=1)).clamp(min=0.0)
    
    # Adjust returns using stability factor
    enhanced_returns = risk_adjusted_returns * (1 + risk_adjusted_returns.std()) 
    enhanced_returns /= ((torch.diagonal(cov_matrix) + 1e-6).sqrt() + 1e-8)

    # Normalize using softmax
    weights = enhanced_returns.softmax(dim=0)

    # Apply entropy-based regularization
    final_weights = (weights * (weights.mean() * torch.log(weights + 1e-8)).exp()).clamp(min=0.0)
    return final_weights / final_weights.sum()

# Load dataset
with open('Dataset.pkl', 'rb') as f: 
    Dataset = cPickle.load(f)

valid_data = torch.from_numpy(np.array(Dataset).T).float()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
valid_data = valid_data.to(device)

# Train-test split
cutoff_index = valid_data.size(1) // 5
train = valid_data[:, :cutoff_index]
test = valid_data[:, cutoff_index:]

# Select top 25% based on AlphaSharpe and Sharpe Ratio
top_indices_alphasharpe = torch.argsort(alphasharpe_metric(train), descending=True)[:train.shape[0] // 4]
top_indices_sharpe = torch.argsort(sharpe_ratio_metric(train), descending=True)[:train.shape[0] // 4]

# Compute portfolio weights using AlphaSharpe Portfolio and Equal Weighting
def compute_portfolio_returns(selection_indices, optimization_method, test_data):
    """Computes portfolio returns for a given selection and optimization strategy."""
    if optimization_method == "alphasharpe":
        portfolio_weights = alphasharpe_portfolio(train[selection_indices, :])
    else:  # "equal_weight"
        portfolio_weights = torch.ones_like(selection_indices, dtype=torch.float, device=device) / len(selection_indices)

    # Compute portfolio returns
    portfolio_returns = (portfolio_weights.unsqueeze(1) * test_data[selection_indices, :]).sum(dim=0)
    
    # Compute out-of-sample Sharpe Ratio
    sharpe_ratio = portfolio_returns.mean() / (portfolio_returns.std(unbiased=False) + 1e-8)
    return sharpe_ratio.item()

# Compute Sharpe Ratios for all combinations
sharpe_results = {
    "Sharpe Selection → Equal Weight Portfolio": compute_portfolio_returns(top_indices_sharpe, "equal_weight", test),
    "Sharpe Selection → AlphaSharpe Portfolio": compute_portfolio_returns(top_indices_sharpe, "alphasharpe", test),
    "AlphaSharpe Selection → Equal Weight Portfolio": compute_portfolio_returns(top_indices_alphasharpe, "equal_weight", test)
}

# Print results
for strategy, sharpe in sharpe_results.items():
    print(f"{strategy}: Out-of-Sample Sharpe Ratio = {sharpe:.6f}")
