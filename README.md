# The Repository of AlphaSharpe & AlphaPortfolio

AlphaSharpe: LLM-Driven "Discovery" of Robust Risk-Adjusted Metrics

```python
def alpha_sharpe(
    log_returns: torch.Tensor, risk_free: 0.0, eps: 1.5e-5, dr: 2.0, fv: 1.33, window: int = 3):
    log_returns = log_returns.unsqueeze(0) if log_returns.ndim == 1 else log_returns
    # Calculate mean log excess return (expected log excess return) and standard deviation of log returns
    mean_log_excess_return = log_returns.mean(dim=-1) - risk_free
    std_log_returns = log_returns.std(dim=-1, unbiased=False)
    # Downside Risk (DR) calculation
    negative_returns = log_returns[log_returns < 0]
    downside = dr * (
        negative_returns.std(dim=-1, unbiased=False) +
        (negative_returns.numel() ** 0.5) * std_log_returns
    ) / (negative_returns.numel() + eps)
    # Forecasted Volatility (V) calculation
    volatility = fv * log_returns[:, -log_returns.shape[-1] // window:].std(dim=-1, unbiased=False).sqrt()
    return mean_log_excess_return.exp() / ((std_log_returns.pow(2) + eps).sqrt() + downside + volatility)
```

AlphaPortfolio: Discovery of Portfolio Allocation Methods Using LLMs
```python
def train_loss(weights, log_returns, alpha=0.02, gamma=0.1, theta=0.0001, window_size=200):
    weights = torch.softmax(weights, dim=0)
    rets = log_returns.matmul(weights)
    discounted_rets = gamma * rets
    window_size = min(window_size, len(log_returns))
    rets_rolled = discounted_rets.unsqueeze(1).unfold(0, window_size, 1)
    lower_idx = int(theta * window_size)
    (trimmed_rets, _) = torch.topk(rets_rolled, k=window_size - lower_idx, dim=2, largest=False)
    trimmed_mean = torch.mean(trimmed_rets, dim=2)
    portfolio_ret = trimmed_mean.mean()
    lower_quartile = torch.quantile(rets_rolled, 0.25, dim=2, keepdim=True)
    median_quartile = torch.quantile(rets_rolled, 0.5, dim=2, keepdim=True)
    semi_std = (median_quartile - lower_quartile).mean()
    cov_matrix = torch.cov(log_returns.T)
    portfolio_variance = (weights @ cov_matrix @ weights).sum()
    diversification_penalty = alpha * portfolio_variance
    cvar_penalty = -rets_rolled.min(dim=2).values.mean()
    l_penalty = alpha * ((weights ** 2).mean() + torch.mean(torch.abs(weights)) * 0.1)
    loss_components = [-portfolio_ret, semi_std, cvar_penalty, l_penalty, diversification_penalty]
    loss_weights = torch.sigmoid(torch.abs(torch.stack(loss_components)))
    loss_weights = loss_weights / loss_weights.sum()
    return (torch.stack(loss_components) * loss_weights).sum()
```
