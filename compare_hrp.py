import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.stats import skew, kurtosis, norm

# ---------------------------
# 1. Portfolio Construction Functions (HRP)
# ---------------------------

def get_cluster_variance(cov: np.ndarray, cluster: np.ndarray) -> float:
    """
    Compute the variance for a given cluster using inverse variance portfolio (IVP) weights.
    """
    sub_cov = cov[np.ix_(cluster, cluster)]
    ivp = 1. / np.diag(sub_cov)
    ivp /= ivp.sum()
    return np.dot(ivp, np.dot(sub_cov, ivp))

def recursive_bisection(cov: np.ndarray, sort_ix: np.ndarray) -> np.ndarray:
    """
    Recursively allocates weights following de Prado's HRP algorithm.
    """
    n = len(sort_ix)
    weights = np.ones(n)
    clusters = [np.arange(n)]
    
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        
        split = len(cluster) // 2
        left_cluster = cluster[:split]
        right_cluster = cluster[split:]
        
        # Map sorted indices to original asset indices
        left_indices = sort_ix[left_cluster]
        right_indices = sort_ix[right_cluster]
        
        var_left = get_cluster_variance(cov, left_indices)
        var_right = get_cluster_variance(cov, right_indices)
        
        # Allocate risk inversely proportional to the cluster variance
        alpha = 1 - var_left / (var_left + var_right)
        weights[left_cluster] *= alpha
        weights[right_cluster] *= (1 - alpha)
        
        clusters.append(left_cluster)
        clusters.append(right_cluster)
        
    return weights

def hrp_portfolio(returns: np.ndarray) -> np.ndarray:
    """
    Constructs portfolio weights using Hierarchical Risk Parity (HRP).
    """
    # Compute covariance matrix (adding a small term for numerical stability)
    cov_matrix = np.cov(returns, ddof=1) + 1e-6 * np.eye(returns.shape[0])
    std_dev = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_dev, std_dev)
    corr_matrix = np.clip(corr_matrix, -1, 1)
    
    # Convert correlation matrix to distance matrix
    distance = np.sqrt(0.5 * (1 - corr_matrix))
    distance = (distance + distance.T) / 2.0

    # Convert to condensed distance matrix for clustering
    dist_condensed = squareform(distance, checks=False)
    link = sch.linkage(dist_condensed, method='single')
    sort_ix = sch.leaves_list(link)
    
    sorted_weights = recursive_bisection(cov_matrix, sort_ix)
    hrp_weights = np.empty_like(sorted_weights)
    hrp_weights[sort_ix] = sorted_weights
    hrp_weights /= hrp_weights.sum()
    
    return hrp_weights

# ---------------------------
# 2. Ranking Metrics
# ---------------------------

def alphasharpe_metric(raw_log_returns: np.ndarray, r_min: float, epsilon: float = 1.5e-5,
                        downside_risk_factor: float = 2.0, forecast_volatility_factor: float = 1.33, 
                        forecast_window: int = 3) -> np.ndarray:
    """
    Compute a modified risk-adjusted metric (AlphaSharpe) for each asset.
    
    This metric takes into account the exponential of the mean excess log-return
    relative to a combination of the standard deviation, downside risk, and forecasted volatility.
    """
    # Convert raw returns to excess returns by subtracting r_min
    excess_log_returns = raw_log_returns - r_min
    n_periods = excess_log_returns.shape[1]
    
    mean_log_excess_return = np.mean(excess_log_returns, axis=1)
    std_excess_log_returns = np.std(excess_log_returns, axis=1, ddof=0)
    
    # Compute downside risk per asset
    downside_risk = np.zeros(excess_log_returns.shape[0])
    for i in range(excess_log_returns.shape[0]):
        asset_returns = excess_log_returns[i]
        negative_returns = asset_returns[asset_returns < 0]
        if negative_returns.size > 0:
            downside_std = np.std(negative_returns, ddof=0)
            downside_risk[i] = downside_risk_factor * (
                downside_std + (negative_returns.size ** 0.5) * std_excess_log_returns[i]
            ) / (negative_returns.size + 1e-8)
        else:
            downside_risk[i] = 0.0
    
    # Compute forecasted volatility using the most recent periods.
    recent_periods = max(1, n_periods // forecast_window)
    recent_std = np.std(excess_log_returns[:, -recent_periods:], axis=1, ddof=0)
    forecasted_volatility = forecast_volatility_factor * recent_std
    
    # Combine components into a single metric.
    return np.exp(mean_log_excess_return) / (std_excess_log_returns + downside_risk + forecasted_volatility + epsilon)

def betasharpe_metric(log_returns: np.ndarray, r_min: float) -> np.ndarray:
    """
    Compute an alternative risk-adjusted metric (BetaSharpe) that incorporates 
    decayed returns, rolling volatility, skewness, kurtosis, and drawdown.
    
    Parameters:
        log_returns (np.ndarray): 2D array where each row is an asset's log returns.
        r_min (float): Minimum acceptable return.
    
    Returns:
        np.ndarray: The BetaSharpe metric for each asset.
    """
    n_assets, n_periods = log_returns.shape
    decay = np.exp(-np.arange(n_periods, dtype=log_returns.dtype) / (n_periods / 2))
    wlr = (log_returns - r_min) * decay
    mean_wlr = np.mean(wlr, axis=1)
    
    rolling_windows = np.lib.stride_tricks.sliding_window_view(wlr, window_shape=n_periods // 4, axis=1)
    win_std = np.mean(np.std(rolling_windows, axis=2, ddof=0), axis=1)
    
    res = wlr - mean_wlr[:, None]
    skew_val = np.clip(np.mean(res**3, axis=1) / (win_std**3), -1, 1)
    kurt = np.minimum(np.mean(res**4, axis=1) / (win_std**4), 6) - 3
    
    cumsum = np.cumsum(wlr, axis=1)
    dd = np.min(cumsum - np.max(cumsum, axis=1, keepdims=True), axis=1)
    
    adj_sharpe = mean_wlr * (1 - np.minimum(np.abs(kurt) / 12, 1))
    adj_sharpe *= ((1 - 1 / (1 + np.abs(np.mean(dd)))) / win_std) * (1 + np.abs(skew_val) / 4)
    adj_sharpe = np.mean(adj_sharpe) * np.where(mean_wlr > 0, 1.1, 0.9)
    adj_sharpe /= (win_std + np.std(wlr[:, -n_periods // 4:], axis=1, ddof=0))
    return adj_sharpe * (1 + ((skew_val**2 + kurt) / 8)) * (1 + np.mean(wlr, axis=1))

def probabilistic_sharpe(raw_returns: np.ndarray, target: float = 0.0) -> np.ndarray:
    """
    Compute the Probabilistic Sharpe Ratio (PSR) for each asset.
    
    Parameters:
        raw_returns (np.ndarray): 2D array where each row is an asset's returns.
        target (float): The minimum acceptable Sharpe ratio (SR*). Default is 0.
    
    Returns:
        np.ndarray: PSR for each asset, representing the probability that the true Sharpe exceeds the target.
    """
    n_assets = raw_returns.shape[0]
    psr_values = np.zeros(n_assets)
    
    for i in range(n_assets):
        asset_returns = raw_returns[i]
        T = len(asset_returns)
        # Compute sample mean and standard deviation
        mean_ret = np.mean(asset_returns)
        std_ret = np.std(asset_returns, ddof=1) + 1e-8  # avoid division by zero
        hat_SR = mean_ret / std_ret
        
        # Compute sample skewness and kurtosis (using Fisher’s definition, so excess kurtosis)
        hat_skew = skew(asset_returns)
        # Using fisher=True gives excess kurtosis; add 3 to get the full kurtosis.
        hat_kurt = kurtosis(asset_returns, fisher=True) + 3
        
        # Adjustment term for estimation error and non-normality:
        denom = np.sqrt(1 - hat_skew * hat_SR + ((hat_kurt - 1) / 4.) * hat_SR**2)
        numerator = hat_SR * np.sqrt(T - 1) - target * np.sqrt(T)
        psr = norm.cdf(numerator / denom)
        psr_values[i] = psr
        
    return psr_values

# ---------------------------
# 3. Data Loading and Preparation
# ---------------------------
# Load the dataset (assumed to be a pickled array-like object)
with open('Dataset.pkl', 'rb') as f: 
    Dataset = cPickle.load(f)

# Assume the dataset has rows as observations and columns as assets.
# Transpose so that rows correspond to assets and columns to time periods.
data = np.array(Dataset).T.astype(np.float64)

# Train-test split (for example, 80% training, 20% testing)
cutoff_index = data.shape[1] // 5  # using 20% for testing
train = data[:, :-cutoff_index]
test = data[:, -cutoff_index:]

# ---------------------------
# 4. Asset Ranking and Portfolio Evaluation
# ---------------------------
n_assets = train.shape[0]
# Define selection ratios (from 20% up to 100% of assets)
selection_ratios = np.linspace(0.2, 1.0, 20)

# Compute ranking scores for each method on training data:
# (a) AlphaSharpe ranking
alpha_scores = alphasharpe_metric(train, r_min=0.0)
ranking_alpha = np.argsort(alpha_scores)[::-1]  # descending order

# (b) Probabilistic Sharpe Ratio ranking
psr_scores = probabilistic_sharpe(train, target=0.0)
ranking_psr = np.argsort(psr_scores)[::-1]  # descending order (higher probability is better)

# (c) BetaSharpe ranking
beta_scores = betasharpe_metric(train, r_min=0.0)
ranking_beta = np.argsort(beta_scores)[::-1]  # descending order

# Containers for out-of-sample Sharpe ratios for each ranking & weighting method
results = {
    'alpha_equal': [],
    'alpha_hrp': [],
    'psr_equal': [],
    'psr_hrp': [],
    'beta_equal': [],
    'beta_hrp': []
}

def calc_sharpe(log_returns, periods_per_year=252):
    """
    Compute the annualized Sharpe ratio for log returns (risk-free rate assumed 0).
    
    Parameters:
        log_returns (np.ndarray): Array of log returns.
        periods_per_year (int): Number of periods per year (default is 252 for daily data).
    
    Returns:
        float: The annualized Sharpe ratio based on log returns.
    """
    mean_log_return = np.mean(log_returns)
    std_log_return = np.std(log_returns, ddof=1)
    sharpe_ratio_annualized = (mean_log_return / (std_log_return + 1e-8)) * np.sqrt(periods_per_year)
    return sharpe_ratio_annualized

# Loop over selection ratios to simulate portfolio performance
for ratio in selection_ratios:
    n_select = int(np.ceil(ratio * n_assets))
    
    # --- AlphaSharpe-based Portfolio Selection ---
    selected_alpha = ranking_alpha[:n_select]
    train_alpha = train[selected_alpha]
    test_alpha = test[selected_alpha]
    
    # Equal weights for AlphaSharpe selection
    eq_weights_alpha = np.ones(n_select) / n_select
    eq_returns_alpha = eq_weights_alpha.dot(test_alpha)
    
    # --- PSR-based Portfolio Selection ---
    selected_psr = ranking_psr[:n_select]
    train_psr = train[selected_psr]
    test_psr = test[selected_psr]
    
    # Equal weights for PSR selection
    eq_weights_psr = np.ones(n_select) / n_select
    eq_returns_psr = eq_weights_psr.dot(test_psr)
    
    # HRP weights for PSR selection
    hrp_weights_psr = hrp_portfolio(train_psr)
    hrp_returns_psr = hrp_weights_psr.dot(test_psr)
    
    # --- BetaSharpe-based Portfolio Selection ---
    selected_beta = ranking_beta[:n_select]
    train_beta = train[selected_beta]
    test_beta = test[selected_beta]
    
    # Equal weights for BetaSharpe selection
    eq_weights_beta = np.ones(n_select) / n_select
    eq_returns_beta = eq_weights_beta.dot(test_beta)
    
    # Compute and store out-of-sample Sharpe ratios for each method
    results['alpha_equal'].append(calc_sharpe(eq_returns_alpha))
    results['psr_equal'].append(calc_sharpe(eq_returns_psr))
    results['psr_hrp'].append(calc_sharpe(hrp_returns_psr))
    results['beta_equal'].append(calc_sharpe(eq_returns_beta))

# ---------------------------
# 5. Visualization
# ---------------------------
plt.figure(figsize=(12, 7))
plt.plot(selection_ratios, results['alpha_equal'], marker='o', label='AlphaSharpe + Equal Weights')
plt.plot(selection_ratios, results['beta_equal'], marker='v', label='BetaSharpe + Equal Weights')
plt.plot(selection_ratios, results['psr_equal'], marker='s', label='PSR + Equal Weights (1/N)')
plt.plot(selection_ratios, results['psr_hrp'], marker='d', label='PSR + Hierarchical Risk Parity')

plt.xlabel('Selection Ratio (Fraction of Assets Selected)')
plt.ylabel('Out-of-Sample Sharpe Ratio')
plt.title('Out-of-Sample Performance: Ranking vs. Weighting Schemes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
