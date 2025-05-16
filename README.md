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
def alpha_portfolio(lr: torch.Tensor, eps = 1e-8):
    # Compute covariance matrix of log returns and add eps for numerical stability
    cov = lr.cov() + eps * torch.eye(len(lr), device=lr.device)
    # Compute risk-adjusted returns (RAR) via inverse covariance weighted mean returns
    rar = (torch.linalg.inv(cov) @ lr.mean(dim=1)).clamp(min=0.0)
    # Enhance RAR by scaling with (1 + std deviation of RAR)
    enhanced = rar * (1 + rar.std())
    # Adjust enhanced returns by each asset’s volatility (sqrt of covariance diagonal)
    enhanced /= (cov.diagonal() + eps).sqrt() + eps
    # Apply softmax to get initial portfolio weights (positive and sum to 1)
    w = enhanced.softmax(dim=0)
    # Compute entropy-based scaling factor and apply to weights
    w = w * ((w * (w + eps).log()).exp().mean()).clamp(min=0.0)
    return w / w.sum()
```
