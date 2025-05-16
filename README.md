# The Repository of AlphaSharpe & AlphaPortfolio

AlphaSharpe: LLM-Driven "Discovery" of Robust Risk-Adjusted Metrics

AlphaPortfolio: Discovery of Portfolio Allocation Methods Using LLMs

def optimize_portfolio(lr: torch.Tensor, eps = 1e-8):
    cov = lr.cov() + eps * torch.eye(len(lr), device=lr.device)
    rar = (torch.linalg.inv(cov) @ lr.mean(dim=1)).clamp(min=0.0)
    enhanced = rar * (1 + rar.std())
    enhanced /= (cov.diagonal() + eps).sqrt() + eps
    w = enhanced.softmax(dim=0)
    w = w * ((w * (w + eps).log()).exp().mean()).clamp(min=0.0)
    return w / w.sum()
