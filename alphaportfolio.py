import torch
import numpy as np
import pandas as pd
import _pickle as cPickle

def portfolio_statistics(log_returns: torch.Tensor, weights: torch.Tensor, risk_free_rate: float = 0.0) -> dict:
    """
    Computes various portfolio performance metrics given log returns and asset weights.
    """
    # Normalize weights to sum to 1
    weights = weights.abs() / weights.abs().sum()  

    # Compute portfolio returns over time
    portfolio_returns = (weights @ log_returns).cumsum(dim=0).exp()  # Convert log returns to price series

    # Compute drawdowns
    peak = torch.cummax(portfolio_returns, dim=0)[0]  # Track the highest value up to each time step
    drawdown = (portfolio_returns - peak) / peak  # Compute percentage drawdown
    max_drawdown = drawdown.min().abs() + 1e-8  # Max drawdown (add small value to avoid division by zero)

    # Compute mean log return and annualized return
    mean_log_return = log_returns.mean()  # Scalar
    annualized_return = (torch.exp(mean_log_return) - 1) * 252  # Assuming 252 trading days

    # Compute Calmar Ratio (identical to portfolio_calmar function)
    calmar_ratio = annualized_return / max_drawdown

    # Compute annualized volatility
    std_dev = (weights @ log_returns).std() + 1e-8  # Avoid division by zero
    annualized_volatility = std_dev * (252 ** 0.5)  # Annualized volatility

    # Compute Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # Compute Sortino Ratio (only downside deviation)
    downside_returns = torch.where(portfolio_returns < risk_free_rate, portfolio_returns, torch.tensor(0.0, device=portfolio_returns.device))
    downside_std = downside_returns.std() + 1e-8  # Avoid division by zero
    sortino_ratio = (annualized_return - risk_free_rate) / downside_std

    return {
        "Annual Return": annualized_return.item(),
        "Volatility": annualized_volatility.item(),
        "Max Drawdown": max_drawdown.item(),
        "Sharpe Ratio": sharpe_ratio.item(),
        "Sortino Ratio": sortino_ratio.item(),
        "Calmar Ratio": calmar_ratio.item(),  # Now identical to the standalone portfolio_calmar function
    }


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

function_names = ["Risk Parity Portfolio", "Equal Risk Contribution Portfolio", "Equal Weighted Portfolio", "AlphaSharpe Portfolio"]

def compare_optimization_methods_with_improvement(train, test, function_definitions, risk_free_rate=0.0):
    """
    Compiles and compares different portfolio optimization methods given as strings.
    Also calculates percentage improvement over the Equal Weighted Portfolio (Method_3).
    """
    results = {}
    baseline_method = None  # Will hold the equal-weighted portfolio results
    
    for i, func_str in enumerate(function_definitions):
        local_namespace = {}
        try:
            exec(func_str, globals(), local_namespace)  # Compile and execute function
            portfolio_function = local_namespace["optimize_portfolio"]  # Retrieve function
            weights = portfolio_function(train)
            stats = portfolio_statistics(test, weights, risk_free_rate)
            stats["Function Name"] = function_names[i]
            results[f"Method_{i+1}"] = stats

            # Identify the baseline (Equal Weighted Portfolio - Method_3)
            if function_names[i] == "Equal Weighted Portfolio":
                baseline_method = stats.copy()
        except Exception as e:
            results[f"Method_{i+1}"] = {"Function Name": function_names[i], "Error": str(e)}

    # Convert to DataFrame
    df_results = pd.DataFrame.from_dict(results, orient="index")

    # Compute percentage improvements over Equal Weighted Portfolio
    if baseline_method:
        baseline_values = {key: value for key, value in baseline_method.items() if isinstance(value, (int, float))}
        for metric in baseline_values.keys():
            df_results[f"% Improvement ({metric})"] = df_results[metric].apply(
                lambda x: ((x - baseline_values[metric]) / abs(baseline_values[metric]) * 100) if isinstance(x, (int, float)) else None
            )

    return df_results



# Load dataset
with open('Dataset.pkl', 'rb') as f: 
    Dataset = cPickle.load(f)

valid_data = np.array(Dataset).T
valid_data = torch.from_numpy(valid_data).float().cuda()

# Split data into training and testing
cutoff_index = valid_data.size(1) // 5
train = valid_data[:, :-cutoff_index]
test = valid_data[:, -cutoff_index:]

# Run the enhanced comparison function
optimized_results_with_improvement = compare_optimization_methods_with_improvement(train, test, function_definitions)
print(optimized_results_with_improvement)
