# The Repository of AlphaSharpe & AlphaPortfolio

AlphaSharpe: LLM-Driven "Discovery" of Robust Risk-Adjusted Metrics

AlphaPortfolio: Discovery of Portfolio Allocation Methods Using LLMs

## AlphaSharpe Formula

The **AlphaSharpe** ratio is calculated as:

\[
\text{AlphaSharpe} = \frac{e^{\frac{1}{T} \sum_{t=1}^T R_t - r_f}}{\sqrt{\sigma_R^2 + \epsilon} + DR + V}
\]

### Components:

1. **Downside Risk (DR):**
   \[
   DR = k_d \cdot \frac{\sigma_{R^-} + \sqrt{N^-} \cdot \sigma_R}{N^- + \epsilon}
   \]
   - \( R^- \): Negative returns (\( R_t < 0 \)).
   - \( N^- \): Number of negative returns.
   - \( \sigma_{R^-} \): Standard deviation of \( R^- \).
   - \( \sigma_R \): Standard deviation of all returns.

2. **Forecasted Volatility (V):**
   \[
   V = k_v \cdot \sqrt{\sigma_{R_\text{recent}}^2}
   \]
   - \( R_\text{recent} \): Most recent \( \lfloor T / w \rfloor \) returns.

3. **Additional Parameters:**
   - \( r_f \): Risk-free rate.
   - \( \epsilon \): Small constant for numerical stability.
   - \( k_d \): Downside risk factor.
   - \( k_v \): Forecast volatility factor.
   - \( w \): Forecast window (number of recent periods).

---

### Summary:

Integrates **excess returns**, **downside risk**, and **forecasted volatility** into a single metric, balancing risk and expected performance.
