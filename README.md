# The Repository of AlphaSharpe & AlphaPortfolio

AlphaSharpe: LLM-Driven "Discovery" of Robust Risk-Adjusted Metrics

```python
def alpha_sharpe(
    log_returns: torch.Tensor, risk_free: float = 0.0, eps: float = 1.5e-5, dr: float = 2.0, fv: float = 1.33, window: int = 3
) -> torch.Tensor:
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
def optimize_portfolio(lr: torch.Tensor, eps = 1e-8):
    cov = lr.cov() + eps * torch.eye(len(lr), device=lr.device)
    rar = (torch.linalg.inv(cov) @ lr.mean(dim=1)).clamp(min=0.0)
    enhanced = rar * (1 + rar.std())
    enhanced /= (cov.diagonal() + eps).sqrt() + eps
    w = enhanced.softmax(dim=0)
    w = w * ((w * (w + eps).log()).exp().mean()).clamp(min=0.0)
    return w / w.sum()
```
