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

def alphacalmar_metric(log_returns: np.ndarray, r_min: float = 0.0) -> np.ndarray:
    # Remove NaNs and subtract risk free rate
    log_returns = np.nan_to_num(log_returns - r_min)
    n = log_returns.shape[1]
    
    # Cumulative returns and drawdowns
    cumulative_returns = np.exp(np.cumsum(log_returns, axis=1))
    running_max = np.maximum.accumulate(cumulative_returns, axis=1)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = np.abs(np.min(drawdowns, axis=1)) + 1e-8

    # Quantile-based trimming
    q_lower, q_upper = np.quantile(log_returns, [0.1, 0.9], axis=1, keepdims=True)
    trimming_weight = np.clip((log_returns - q_lower) / (q_upper - q_lower + 1e-8), 0.0, 1.0)
    trimmed_log_returns = log_returns * trimming_weight

    # Time decay weighting: using linspace to guarantee n elements
    time_weights = np.exp(np.linspace(-1.0, 0.0, n, endpoint=False))[None, :]
    time_decayed_mean = np.sum(trimmed_log_returns * time_weights, axis=1) / (np.sum(time_weights, axis=1) + 1e-8)

    # Downside measures
    downside_returns = np.minimum(log_returns, 0)
    var95 = -np.quantile(downside_returns, 0.95, axis=1, keepdims=True)
    cvar = -np.mean(downside_returns * (downside_returns < var95).astype(float), axis=1) / (np.sum((downside_returns < var95).astype(float), axis=1) + 1e-8)
    downside_std = np.sqrt(np.mean(np.square(downside_returns), axis=1) + 1e-8)

    # GARCH-like volatility estimate
    mean_lr = np.mean(log_returns, axis=1, keepdims=True)
    garch_variance = np.mean(np.square(log_returns - mean_lr), axis=1) + 0.1 * np.square(np.std(log_returns, axis=1))
    garch_volatility = np.sqrt(garch_variance + 1e-8)

    # Combined risk measure (L2 norm of several risk metrics)
    risk_components = np.stack([
        var95.squeeze(), 
        max_drawdown, 
        downside_std, 
        garch_volatility
    ], axis=0)
    risk_measure = np.linalg.norm(risk_components, axis=0) + 1e-8

    # Geometric mean and skew/kurtosis calculations
    geometric_mean_return = np.exp(np.mean(trimmed_log_returns, axis=1)) - 1
    centered_returns = trimmed_log_returns - geometric_mean_return[:, None]
    skewness = np.mean(np.power(centered_returns, 3), axis=1) / (np.power(downside_std, 3) + 1e-8)
    kurtosis = np.clip((np.mean(np.power(centered_returns, 4), axis=1) / (np.power(downside_std, 4) + 1e-8) - 3), -3.0, 3.0)

    # Windowed entropy using a sliding window view
    window_size = min(5, n)
    # Create sliding windows along the time axis (axis=1)
    shape = (log_returns.shape[0], n - window_size + 1, window_size)
    strides = log_returns.strides[:1] + (log_returns.strides[1], log_returns.strides[1])
    windowed = np.lib.stride_tricks.as_strided(log_returns, shape=shape, strides=strides)
    
    def softmax(x, axis):
        # subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
    windowed_sm = softmax(windowed, axis=2)
    windowed_entropy = -np.sum(windowed_sm * np.log(np.clip(windowed_sm, 1e-8, 1.0)), axis=2).mean(axis=1)
    
    realized_vol = np.std(log_returns, axis=1, ddof=0) + 1e-8

    # Concatenate features
    features = np.concatenate([
        trimmed_log_returns,
        skewness[:, None],
        kurtosis[:, None],
        windowed_entropy[:, None],
        realized_vol[:, None],
        garch_volatility[:, None]
    ], axis=1)
    features_mean = np.mean(features, axis=1, keepdims=True)
    features_std = np.std(features, axis=1, keepdims=True)
    normalized_features = (features - features_mean) / (features_std + 1e-8)
    
    # PCA via SVD
    u, s, vh = np.linalg.svd(normalized_features, full_matrices=False)
    num_components = min(normalized_features.shape[1], u.shape[1])
    features_pca = u[:, :num_components]

    # Adaptive risk aversion and uncertainty
    adaptive_risk_aversion = (np.mean(drawdowns, axis=1) + 1e-8) / (downside_std + garch_volatility + 1e-8)
    uncertainty = np.var(features, axis=1) / (np.mean(features, axis=1) + 1e-8)

    # Hurst exponent estimate (negated ratio)
    hurst_exponent = - (np.std(log_returns, axis=1) / (np.mean(np.abs(log_returns), axis=1) + 1e-8))

    # Tail excess risk
    tail_excess_risk = (max_drawdown * (downside_std + garch_volatility) * np.mean(np.abs(drawdowns), axis=1)) / (max_drawdown + downside_std + 1e-8)

    # Final score: combine multiple components
    # Calculate the incremental return term from recent log_returns differences
    recent_diff = (log_returns[:, -1] - log_returns[:, -min(10, n)])
    final_score = (time_decayed_mean * (1 + skewness - kurtosis) / risk_measure) \
                  + (1 + np.clip(recent_diff, 0, None) / (downside_std + garch_volatility + 1e-8))
    final_score = final_score * adaptive_risk_aversion + np.mean(features_pca, axis=1) - uncertainty + hurst_exponent

    diversification_ratio = np.std(trimmed_log_returns, axis=1) / (np.mean(trimmed_log_returns, axis=1) + 1e-8)
    final_score += diversification_ratio

    # Entropy over trimmed log returns
    trimmed_sm = softmax(trimmed_log_returns, axis=1)
    entropies = -np.sum(trimmed_sm * np.log(np.clip(trimmed_sm, 1e-8, 1.0)), axis=1)
    final_score += entropies

    final_score -= tail_excess_risk

    # Renyi entropy calculation
    renyi_entropy = np.log(np.mean(np.exp(0.5 * trimmed_log_returns), axis=1) + 1e-8) / (0.5 - 1 + 1e-8)
    final_score += renyi_entropy

    # Multi-resolution entropy from low frequency component of FFT
    fft_vals = np.fft.fft(trimmed_log_returns, axis=1)
    low_freq_filter = (np.abs(np.arange(n).astype(float) - (n // 2)) < (n // 10)).astype(float)
    low_freq_component = np.abs(fft_vals) * low_freq_filter
    low_freq_sm = softmax(low_freq_component, axis=1)
    multi_resolution_entropy = -np.sum(low_freq_sm * np.log(np.clip(low_freq_sm, 1e-8, 1.0)), axis=1)
    final_score += multi_resolution_entropy

    # Denoised returns and their volatility
    denoised_returns = log_returns - np.mean(log_returns, axis=1, keepdims=True)
    final_score += np.std(denoised_returns, axis=1)

    final_score -= 0.05 * (downside_std + garch_volatility)

    # Implicit volatility
    implicit_volatility = np.mean(np.exp(np.std(log_returns, axis=1))) + 1e-8
    final_score += implicit_volatility

    # Hyperbolic geometric mean
    hyperbolic_geom_mean = (np.mean(np.exp(trimmed_log_returns), axis=1) - 1 + 1e-8) * np.cosh(np.clip(np.mean(trimmed_log_returns, axis=1), 1e-8, None))
    final_score += hyperbolic_geom_mean

    return final_score

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
selection_ratios = np.linspace(0.2, 0.6, 20)

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

# (d) AlphaCalmar ranking
calmar_scores = alphacalmar_metric(train, r_min=0.0)
ranking_calmar = np.argsort(calmar_scores)[::-1]  # descending order


# ---------------------------
# 5. Define Containers for Metrics
# ---------------------------
results_sharpe = {
    'alpha_equal': [],
    'psr_equal': [],
    'psr_hrp': [],
    'beta_equal': [],
    'calmar_equal': [],
    'calmar_hrp': []
}

results_mean = {
    'alpha_equal': [],
    'psr_equal': [],
    'psr_hrp': [],
    'beta_equal': [],
    'calmar_equal': [],
    'calmar_hrp': []
}

# New container for Calmar ratios
results_calmar = {
    'alpha_equal': [],
    'psr_equal': [],
    'psr_hrp': [],
    'beta_equal': [],
    'calmar_equal': [],
    'calmar_hrp': []
}

# ---------------------------
# 6. Helper Functions for Performance Metrics
# ---------------------------
def calc_sharpe(log_returns, periods_per_year=252):
    """
    Compute the annualized Sharpe ratio for log returns (risk-free rate assumed 0).
    """
    mean_log_return = np.mean(log_returns)
    std_log_return = np.std(log_returns, ddof=1)
    sharpe_ratio_annualized = (mean_log_return / (std_log_return + 1e-8)) * np.sqrt(periods_per_year)
    return sharpe_ratio_annualized

def calc_calmar(log_returns, periods_per_year=252):
    """
    Compute the annualized Calmar ratio for log returns.
    
    The Calmar ratio is defined as the annualized return divided by the maximum drawdown.
    """
    cum_returns = np.exp(np.cumsum(log_returns))  # cumulative wealth
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (running_max - cum_returns) / (running_max + 1e-8)
    max_dd = np.max(drawdowns)
    mean_log_return = np.mean(log_returns)
    annualized_return = mean_log_return * periods_per_year
    return annualized_return / (max_dd + 1e-8)

# ---------------------------
# 7. Loop over Selection Ratios to Simulate Portfolio Performance
# ---------------------------
for ratio in selection_ratios:
    n_select = int(np.ceil(ratio * n_assets))
    
    # --- AlphaSharpe-based Portfolio Selection (Equal Weights) ---
    selected_alpha = ranking_alpha[:n_select]
    train_alpha = train[selected_alpha]
    test_alpha = test[selected_alpha]
    eq_weights_alpha = np.ones(n_select) / n_select
    eq_returns_alpha = eq_weights_alpha.dot(test_alpha)
    
    # --- PSR-based Portfolio Selection (Equal Weights & HRP) ---
    selected_psr = ranking_psr[:n_select]
    train_psr = train[selected_psr]
    test_psr = test[selected_psr]
    eq_weights_psr = np.ones(n_select) / n_select
    eq_returns_psr = eq_weights_psr.dot(test_psr)
    
    hrp_weights_psr = hrp_portfolio(train_psr)
    hrp_returns_psr = hrp_weights_psr.dot(test_psr)
    
    # --- BetaSharpe-based Portfolio Selection (Equal Weights) ---
    selected_beta = ranking_beta[:n_select]
    train_beta = train[selected_beta]
    test_beta = test[selected_beta]
    eq_weights_beta = np.ones(n_select) / n_select
    eq_returns_beta = eq_weights_beta.dot(test_beta)
    
    # --- AlphaCalmar-based Portfolio Selection (Equal Weights & HRP) ---
    selected_calmar = ranking_calmar[:n_select]
    train_calmar = train[selected_calmar]
    test_calmar = test[selected_calmar]
    eq_weights_calmar = np.ones(n_select) / n_select
    eq_returns_calmar = eq_weights_calmar.dot(test_calmar)
    
    hrp_weights_calmar = hrp_portfolio(train_calmar)
    hrp_returns_calmar = hrp_weights_calmar.dot(test_calmar)
    
    # Compute and store out-of-sample Sharpe ratios for each method
    results_sharpe['alpha_equal'].append(calc_sharpe(eq_returns_alpha))
    results_sharpe['psr_equal'].append(calc_sharpe(eq_returns_psr))
    results_sharpe['psr_hrp'].append(calc_sharpe(hrp_returns_psr))
    results_sharpe['beta_equal'].append(calc_sharpe(eq_returns_beta))
    results_sharpe['calmar_equal'].append(calc_sharpe(eq_returns_calmar))
    results_sharpe['calmar_hrp'].append(calc_sharpe(hrp_returns_calmar))
    
    # Compute and store mean log returns for each method
    results_mean['alpha_equal'].append(np.mean(eq_returns_alpha))
    results_mean['psr_equal'].append(np.mean(eq_returns_psr))
    results_mean['psr_hrp'].append(np.mean(hrp_returns_psr))
    results_mean['beta_equal'].append(np.mean(eq_returns_beta))
    results_mean['calmar_equal'].append(np.mean(eq_returns_calmar))
    results_mean['calmar_hrp'].append(np.mean(hrp_returns_calmar))
    
    # Compute and store annualized Calmar ratios for each method
    results_calmar['alpha_equal'].append(calc_calmar(eq_returns_alpha))
    results_calmar['psr_equal'].append(calc_calmar(eq_returns_psr))
    results_calmar['psr_hrp'].append(calc_calmar(hrp_returns_psr))
    results_calmar['beta_equal'].append(calc_calmar(eq_returns_beta))
    results_calmar['calmar_equal'].append(calc_calmar(eq_returns_calmar))
    results_calmar['calmar_hrp'].append(calc_calmar(hrp_returns_calmar))

# ---------------------------
# 8. Visualization: Three-Panel Plot for Sharpe, Mean Return, and Calmar Ratios
# ---------------------------
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 18), sharex=True)

# Panel 1: Annualized Sharpe Ratios
ax1.plot(selection_ratios, results_sharpe['calmar_equal'], marker='o', label='AlphaCalmar + Equal Weights')
ax1.plot(selection_ratios, results_sharpe['alpha_equal'], marker='o', label='AlphaSharpe + Equal Weights')
ax1.plot(selection_ratios, results_sharpe['beta_equal'], marker='v', label='BetaSharpe + Equal Weights')
ax1.plot(selection_ratios, results_sharpe['psr_equal'], marker='s', label='PSR + Equal Weights (1/N)')
ax1.plot(selection_ratios, results_sharpe['psr_hrp'], marker='d', label='PSR + Hierarchical Risk Parity')
ax1.plot(selection_ratios, results_sharpe['calmar_hrp'], marker='^', label='AlphaCalmar + HRP')
ax1.set_ylabel('Annualized Sharpe Ratio')
ax1.set_title('Out-of-Sample Performance: Sharpe Ratios')
ax1.legend()
ax1.grid(True)

# Panel 2: Annualized Calmar Ratios
ax2.plot(selection_ratios, results_calmar['calmar_equal'], marker='o', label='AlphaCalmar + Equal Weights')
ax2.plot(selection_ratios, results_calmar['alpha_equal'], marker='o', label='AlphaSharpe + Equal Weights')
ax2.plot(selection_ratios, results_calmar['beta_equal'], marker='v', label='BetaSharpe + Equal Weights')
ax2.plot(selection_ratios, results_calmar['psr_equal'], marker='s', label='PSR + Equal Weights (1/N)')
ax2.plot(selection_ratios, results_calmar['psr_hrp'], marker='d', label='PSR + Hierarchical Risk Parity')
ax2.plot(selection_ratios, results_calmar['calmar_hrp'], marker='^', label='AlphaCalmar + HRP')
ax2.set_ylabel('Annualized Calmar Ratio')
ax2.set_title('Out-of-Sample Performance: Calmar Ratios')
ax2.legend()
ax2.grid(True)

# Panel 3: Mean Log Returns
ax3.plot(selection_ratios, results_mean['calmar_equal'], marker='o', label='AlphaCalmar + Equal Weights')
ax3.plot(selection_ratios, results_mean['alpha_equal'], marker='o', label='AlphaSharpe + Equal Weights')
ax3.plot(selection_ratios, results_mean['beta_equal'], marker='v', label='BetaSharpe + Equal Weights')
ax3.plot(selection_ratios, results_mean['psr_equal'], marker='s', label='PSR + Equal Weights (1/N)')
#ax3.plot(selection_ratios, results_mean['psr_hrp'], marker='d', label='PSR + Hierarchical Risk Parity')
#ax3.plot(selection_ratios, results_mean['calmar_hrp'], marker='^', label='AlphaCalmar + HRP')
ax3.set_xlabel('Selection Ratio (Fraction of Assets Selected)')
ax3.set_ylabel('Mean Log Return')
ax3.set_title('Out-of-Sample Performance: Mean Returns')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()
