import _pickle as cPickle
import torch
import numpy as np

def portfolio_calmar(log_returns: torch.Tensor, weights: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    """
    Computes the Calmar Ratio of a portfolio given log returns and asset weights.
    """
    # Ensure weights are properly shaped: (assets,)
    weights = weights.abs() / weights.abs().sum()  # Normalize weights to sum to 1

    # Compute portfolio cumulative log returns
    portfolio_returns = (weights @ log_returns).cumsum(dim=0).exp()  # Convert log returns to price series

    # Compute drawdowns
    peak = torch.cummax(portfolio_returns, dim=0)[0]  # Track the highest value up to each time step
    drawdown = (portfolio_returns - peak) / peak  # Compute percentage drawdown
    max_drawdown = drawdown.min().abs() + 1e-8  # Max drawdown (add small value to avoid division by zero)

    # Compute mean log return and annualized return
    mean_log_return = log_returns.mean()  # Scalar
    annualized_return = (torch.exp(mean_log_return) - 1) * 252  # Assuming 252 trading days

    # Compute Calmar Ratio
    calmar_ratio = annualized_return / max_drawdown
    return calmar_ratio

def portfolio_sharpe(log_returns: torch.Tensor, weights: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    """
    Computes the Sharpe Ratio of a portfolio given log returns and asset weights.
    """
    # Ensure weights are properly shaped: (assets,)
    weights = weights / weights.abs().sum()  # Normalize weights to sum to 1

    # Compute portfolio returns over time
    portfolio_returns = (weights @ log_returns)  # (time,)

    # Compute mean return and volatility over time
    mean_return = portfolio_returns.mean()  # Scalar
    portfolio_volatility = portfolio_returns.std() + 1e-8  # Scalar (prevent division by zero)

    # Compute Sharpe Ratio
    sharpe_ratio = (mean_return - risk_free_rate) / portfolio_volatility
    return sharpe_ratio

# List of function definitions as strings
function_definitions = [
"""
def optimize_portfolio(log_returns):
    log_returns = log_returns.cuda()
    mean_returns = log_returns.mean(dim=1)
    cov_matrix = log_returns.cov() + 1e-6 * torch.eye(log_returns.shape[0], device='cuda')
    inv_cov_matrix = torch.linalg.pinv(cov_matrix)
    adjusted_returns = inv_cov_matrix @ mean_returns
    adjusted_returns = torch.clamp(adjusted_returns, min=0.0)
    robustness_factor = torch.std(adjusted_returns) + 1e-4
    risk_adjusted_returns = adjusted_returns / (cov_matrix.diag().sqrt() + 1e-4)
    normalized_returns = risk_adjusted_returns / torch.linalg.norm(risk_adjusted_returns, ord=1)
    weights = normalized_returns * robustness_factor
    weights = weights / torch.sum(weights)
    return weights
""",
"""
def optimize_portfolio(log_returns):
    log_returns = log_returns.cuda()
    mean_returns = log_returns.mean(dim=1)
    cov_matrix = log_returns.cov() + 1e-6 * torch.eye(log_returns.shape[0], device='cuda')
    inv_cov_matrix = torch.linalg.pinv(cov_matrix)
    risk_adjusted_returns = inv_cov_matrix @ mean_returns
    risk_adjusted_returns = torch.clamp(risk_adjusted_returns, min=0.0)
    stability_factor = 1 + torch.std(risk_adjusted_returns)
    enhanced_risk_adjusted_returns = risk_adjusted_returns * stability_factor
    volatility_adjustment = torch.sqrt(torch.diag(cov_matrix))    
    normalized_returns = enhanced_risk_adjusted_returns / volatility_adjustment
    normalized_returns = torch.clamp(normalized_returns, min=0.0)
    weights = normalized_returns / torch.sum(normalized_returns)
    return weights
""",
"""
def optimize_portfolio(log_returns):
    n_assets = log_returns.shape[0]
    return torch.ones(n_assets, device='cuda') / n_assets
""",
"""
def optimize_portfolio(log_returns):
    log_returns = log_returns.cuda()
    mean_returns = log_returns.mean(dim=1)
    cov_matrix = log_returns.cov() + 1e-6 * torch.eye(log_returns.shape[0], device='cuda')
    inv_cov_matrix = torch.linalg.inv(cov_matrix)
    risk_adjusted_returns = inv_cov_matrix @ mean_returns
    risk_adjusted_returns = torch.clamp(risk_adjusted_returns, min=0.0)
    stability_factor = 1 + torch.std(risk_adjusted_returns)
    volatility_adjustment = torch.sqrt(torch.diagonal(cov_matrix) + 1e-6)
    adjusted_returns = risk_adjusted_returns * stability_factor
    enhanced_returns = adjusted_returns / (volatility_adjustment + 1e-8)
    weights = torch.softmax(enhanced_returns, dim=0)
    entropy_penalty = -weights * torch.log(weights + 1e-8)
    entropy_regularization = entropy_penalty.mean()
    penalized_weights = weights * torch.exp(-entropy_regularization)
    final_weights = torch.clamp(penalized_weights, min=0.0)
    return final_weights / final_weights.sum() if final_weights.sum() > 0 else final_weights
"""
]

function_names = ["Risk Parity Portfolio", "Equal Risk Contribution Portfolio", "Equal Weighted Portfolio",
"AlphaSharpe Portfolio"]

def compare_optimization_methods(train, test, function_definitions, risk_free_rate=0.0):
    """
    Compiles and compares different portfolio optimization methods given as strings.
    """
    results = {}
    for i, func_str in enumerate(function_definitions):
        local_namespace = {}
        try:
            exec(func_str, globals(), local_namespace)  # Compile and execute function
            portfolio_function = local_namespace["optimize_portfolio"]  # Retrieve function
            weights = portfolio_function(train)
            sharpe_ratio = portfolio_sharpe(test, weights, risk_free_rate)
            calmar_ratio = portfolio_calmar(test, weights, risk_free_rate)
            results[f"Method_{i+1}"] = {"Function Name": function_names[i], "Sharpe Ratio": sharpe_ratio.item(), "Calmar Ratio": calmar_ratio.item()}
        except Exception as e:
            results[f"Method_{i+1}"] = {"Error": str(e)}
    
    return results

with open('Dataset.pkl', 'rb') as f: Dataset = cPickle.load(f)
valid_data = np.array(Dataset).T
valid_data = torch.from_numpy(valid_data).float().cuda()
cutoff_index = valid_data.size(1) // 5
train = valid_data[:, :cutoff_index]
test = valid_data[:, cutoff_index:]

import pandas as pd
# Compare optimization methods
optimized_results = compare_optimization_methods(train, test, function_definitions)
print(pd.DataFrame(optimized_results))
