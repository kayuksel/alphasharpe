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
    return mean_log_excess_return.exp() / (torch.sqrt(std_excess_log_returns.pow(2) + epsilon) + downside_risk + forecasted_volatility)

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

# Step 1: Compute AlphaSharpe metric for each asset in the training set
alpha_sharpe_values = alphasharpe_metric(train)

# Step 2: Select the top 25% of assets based on AlphaSharpe metric
top_indices = torch.argsort(alpha_sharpe_values, descending=True)[:train.shape[0] // 4]  

# Step 3: Extract only top 25% of assets for computing portfolio weights
top_train_returns = train[top_indices, :]  # Training set with only selected assets
top_test_returns = test[top_indices, :]  # Test set with only selected assets

# Step 4: Compute portfolio weights using AlphaSharpe Portfolio function
portfolio_weights = alphasharpe_portfolio(top_train_returns)

# Step 5: Compute portfolio return on the test set
portfolio_returns = (portfolio_weights.unsqueeze(1) * top_test_returns).sum(dim=0)

# Step 6: Compute Sharpe Ratio on test set
mean_portfolio_return = portfolio_returns.mean()
std_portfolio_return = portfolio_returns.std(unbiased=False)
sharpe_ratio = mean_portfolio_return / (std_portfolio_return + 1e-8) 

print(f"Mean Portfolio Return: {mean_portfolio_return.item():.6f}")
print(f"Portfolio Sharpe Ratio: {sharpe_ratio.item():.6f}")
