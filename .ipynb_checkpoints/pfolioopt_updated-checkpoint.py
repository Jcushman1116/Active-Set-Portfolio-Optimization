# %%
# # Active Set QP Methods for Single-Factor Covariance Model
# **Author:** Jonathan Cushman  
# **Date:** October 2026  

# %%
# ## Libraries

# %%
import numpy as np
import requests, pandas as pd
import yfinance as yf
from io import StringIO
import matplotlib.pyplot as plt

# %%
# ## Feasible Starting Portfolio
# Constructs an initial feasible portfolio $w_0 \ge 0$:
# - When `target=None` (Long-Only GMV problem with only $e^\top w = 1$), returns the strictly positive equal-weight portfolio $w_0 = e / p$.
# - When `target` is given (Target-Return problem with $e^\top w = 1$ and $\mu^\top w = f$), pairs two assets $i, j$ with $\mu_i \le f \le \mu_j$ and sets $w_i = \frac{\mu_j - f}{\mu_j - \mu_i}, w_j = \frac{f - \mu_i}{\mu_j - \mu_i}$.

# %%
def feasible_start(mu, target=None, tol=1e-10):
    p = len(mu)
    if target is None:
        # Long-Only GMV only requires e^T w = 1, w >= 0
        return np.ones(p) / p

    # Boundary handlers when target equals min(mu) or max(mu)
    if abs(target - np.min(mu)) <= tol:
        w = np.zeros(p)
        idx = np.flatnonzero(np.abs(mu - np.min(mu)) <= tol)
        w[idx] = 1.0 / len(idx)
        return w
    if abs(target - np.max(mu)) <= tol:
        w = np.zeros(p)
        idx = np.flatnonzero(np.abs(mu - np.max(mu)) <= tol)
        w[idx] = 1.0 / len(idx)
        return w

    best_w = None
    best_min_weight = -np.inf

    for i in range(p):
        for j in range(i + 1, p):
            if abs(mu[j] - mu[i]) < tol:
                continue

            wi = (mu[j] - target) / (mu[j] - mu[i])
            wj = (target - mu[i]) / (mu[j] - mu[i])

            if wi >= -tol and wj >= -tol:
                w = np.zeros(p)
                w[i] = wi
                w[j] = wj
                w[np.abs(w) < tol] = 0.0

                if min(w[i], w[j]) > best_min_weight:
                    best_w = w
                    best_min_weight = min(w[i], w[j])

    if best_w is None:
        raise ValueError(
            "Could not construct a feasible starting portfolio."
        )

    return best_w

# %%
# ## Active Set Algorithm & Analytical $R_i$ Benchmark
# - `active_set_qp(V, mu=None, target=None)`: Solves both the **Long-Only GMV** problem (`target=None`, 1 equality constraint $e^\top w = 1$) and the **Target-Return Long-Only** problem (`target=f`, 2 equality constraints $e^\top w = 1, \mu^\top w = f$) via Nocedal & Wright Algorithm 16.3.
# - `analytical_long_only_gmv(sigma_f_sq, beta, delta_sq)`: Computes the exact closed-form Long-Only GMV portfolio using the $R_i$ sequence characterization from Gunther, Kercheval, & Sowunmi (2025) for validation.

# %%
def active_set_qp(V, mu=None, target=None, tol=1e-9, max_iter=500, w_init=None):
    V = np.asarray(V, dtype=float)
    p = V.shape[0]

    if V.shape != (p, p):
        raise ValueError("V must be a square matrix matching p.")

    if not np.allclose(V, V.T, atol=tol):
        raise ValueError("V must be symmetric.")

    if np.min(np.linalg.eigvalsh(V)) <= 0:
        raise ValueError("V must be positive definite.")

    # Configure equality constraints E^T w = b
    if target is None:
        # Long-Only GMV mode: only 1 equality constraint (e^T w = 1)
        E = np.ones((p, 1))
        b = np.array([1.0])
        m_eq = 1
        if mu is None:
            mu = np.zeros(p)
    else:
        # Target-Return mode: 2 equality constraints (e^T w = 1, mu^T w = target)
        if mu is None:
            raise ValueError("mu must be provided when target is specified.")
        mu = np.asarray(mu, dtype=float).flatten()
        if len(mu) != p:
            raise ValueError("Length of mu must match V.")
        if target < np.min(mu) - tol or target > np.max(mu) + tol:
            raise ValueError("Target return is outside the feasible range.")
        E = np.column_stack((np.ones(p), mu))
        b = np.array([1.0, float(target)])
        m_eq = 2

    w = feasible_start(mu, target, tol) if w_init is None else np.asarray(w_init, dtype=float).copy()
    active = set(np.flatnonzero(w <= tol))

    for iteration in range(max_iter):
        free = np.array([i for i in range(p) if i not in active], dtype=int)
        active_idx = np.array(sorted(active), dtype=int)

        # Handle boundary/corner working sets where free set has fewer than m_eq independent columns
        if len(free) < m_eq or np.linalg.matrix_rank(E[free].T) < m_eq:
            grad = 2.0 * (V @ w)
            eta, *_ = np.linalg.lstsq(E[free], grad[free], rcond=None)
            nu_active = grad[active_idx] - E[active_idx] @ eta if len(active_idx) > 0 else np.array([])
            if len(active_idx) == 0 or np.min(nu_active) >= -tol:
                return {
                    "weights": w,
                    "variance": float(w @ V @ w),
                    "iterations": iteration + 1,
                    "active": sorted(active),
                    "lambda": float(eta[0]),
                    "gamma": float(eta[1]) if m_eq == 2 else 0.0,
                    "multipliers": dict(zip(active_idx.tolist(), nu_active.tolist())),
                }
            remove_pos = int(np.argmin(nu_active))
            active.remove(int(active_idx[remove_pos]))
            continue

        H = 2.0 * V[np.ix_(free, free)]
        E_free = E[free, :]
        grad_free = 2.0 * (V @ w)[free]

        KKT = np.block([
            [H, -E_free],
            [E_free.T, np.zeros((m_eq, m_eq))]
        ])

        rhs = np.concatenate((-grad_free, np.zeros(m_eq)))
        solution = np.linalg.solve(KKT, rhs)

        direction_free = solution[:len(free)]
        eta = solution[len(free):]
        direction = np.zeros(p)
        direction[free] = direction_free

        if np.linalg.norm(direction, ord=np.inf) <= tol:
            grad = 2.0 * V @ w
            nu_active = grad[active_idx] - E[active_idx] @ eta if len(active_idx) > 0 else np.array([])

            if len(active_idx) == 0 or np.min(nu_active) >= -tol:
                return {
                    "weights": w,
                    "variance": float(w @ V @ w),
                    "iterations": iteration + 1,
                    "active": sorted(active),
                    "lambda": float(eta[0]),
                    "gamma": float(eta[1]) if m_eq == 2 else 0.0,
                    "multipliers": dict(
                        zip(active_idx.tolist(), nu_active.tolist())
                    ),
                }

            remove_pos = int(np.argmin(nu_active))
            active.remove(int(active_idx[remove_pos]))
            continue

        alpha = 1.0
        blocker = None

        for i in free:
            if direction[i] < -tol:
                candidate = -w[i] / direction[i]

                if candidate < alpha:
                    alpha = candidate
                    blocker = int(i)

        w = w + alpha * direction
        w[np.abs(w) < tol] = 0.0

        if np.min(w) < -10 * tol:
            raise RuntimeError("Numerical loss of feasibility.")

        w = np.maximum(w, 0.0)

        if not np.allclose(E.T @ w, b, atol=100 * tol):
            raise RuntimeError("Equality constraints lost feasibility.")

        if blocker is not None and alpha < 1.0 - tol:
            active.add(blocker)

    raise RuntimeError("Maximum number of iterations reached.")


def analytical_long_only_gmv(sigma_f_sq, beta, delta_sq):
    """
    Closed-form R_i characterization of the Long-Only GMV portfolio
    for V = sigma_f_sq * beta @ beta^T + diag(delta_sq) (Gunther et al., 2025).
    """
    beta = np.asarray(beta, dtype=float).flatten()
    delta_sq = np.asarray(delta_sq, dtype=float).flatten()
    p = len(beta)

    # Ensure sum(beta_i / delta_i^2) > 0
    if np.sum(beta / delta_sq) < 0:
        beta = -beta

    # Sort betas in increasing order: beta_1 <= beta_2 <= ... <= beta_p
    sort_idx = np.argsort(beta)
    beta_s = beta[sort_idx]
    delta_sq_s = delta_sq[sort_idx]

    # Compute R_i sequence
    R = np.zeros(p)
    R[0] = 1.0 / sigma_f_sq
    for i in range(1, p):
        j = np.arange(i)
        R[i] = R[0] + np.sum((beta_s[j] / delta_sq_s[j]) * (beta_s[j] - beta_s[i]))

    # Cutoff k = max{i <= p : R_i > 0}
    k = int(np.flatnonzero(R > 0)[-1]) + 1
    K = np.arange(k)

    # Solve on the k lowest-beta assets
    beta_K = beta_s[K].reshape(-1, 1)
    V_K = sigma_f_sq * (beta_K @ beta_K.T) + np.diag(delta_sq_s[K])
    e_k = np.ones(k)
    V_K_inv_e = np.linalg.solve(V_K, e_k)

    w_sorted = np.zeros(p)
    w_sorted[K] = V_K_inv_e / (e_k @ V_K_inv_e)

    # Restore original asset ordering
    w = np.zeros(p)
    w[sort_idx] = w_sorted
    V_full = sigma_f_sq * np.outer(beta, beta) + np.diag(delta_sq)

    return {
        "weights": w,
        "variance": float(w @ V_full @ w),
        "k_positive": k,
        "R_sequence": R,
        "active": sorted(np.flatnonzero(w <= 1e-12).tolist())
    }

# %%
# ## Part 1: Basic Testing & $100 \times 100$ Simple Factor Model Validation
# First, we test the 3-asset toy example, and then we construct a controlled $100 \times 100$ Single-Factor Covariance Matrix:
# $$V_{\text{simple}} = \sigma_f^2 \beta \beta^\top + \Delta, \quad \Delta = \text{diag}(\delta_1^2, \dots, \delta_{100}^2)$$
# We validate that `active_set_qp(V_simple, target=None)` recovers the exact same weights, active set, and minimum variance as the analytical $R_i$ sequence formula from Section 2 of the proposal.

# %%
# 1. Toy 3-Asset Target-Return Test
mu_toy = np.array([0.05, 0.10, 0.15])
V_toy = np.eye(3)
target_toy = 0.10

result_toy = active_set_qp(V_toy, mu_toy, target_toy)
print("--- 3-Asset Toy Target-Return Test ---")
print("Optimal weights:", result_toy["weights"])
print("Portfolio variance:", result_toy["variance"])
print("Iterations:", result_toy["iterations"])

# 2. 100x100 Simple Single-Factor Model Validation (Active-Set QP vs Analytical R_i)
np.random.seed(42)
p_test = 100
sigma_f_sq_test = 0.0025
beta_test = np.linspace(0.4, 1.6, p_test)
delta_sq_test = np.random.uniform(0.0005, 0.0030, p_test)
V_simple_100 = sigma_f_sq_test * np.outer(beta_test, beta_test) + np.diag(delta_sq_test)

res_ri = analytical_long_only_gmv(sigma_f_sq_test, beta_test, delta_sq_test)
res_qp = active_set_qp(V_simple_100, target=None)

print("\n--- 100x100 Simple Single-Factor Model GMV Validation ---")
print(f"Analytical R_i cutoff k (positive weights): {res_ri['k_positive']} / {p_test}")
print(f"Active-Set QP positive weights:             {p_test - len(res_qp['active'])} / {p_test}")
print(f"Analytical Minimum Variance:                {res_ri['variance']:.10f}")
print(f"Active-Set QP Minimum Variance:             {res_qp['variance']:.10f}")
print(f"Max absolute weight difference ||w_QP - w_Ri||_inf: {np.max(np.abs(res_qp['weights'] - res_ri['weights'])):.2e}")
print(f"Active sets match exactly:                  {res_qp['active'] == res_ri['active']}")
print(f"Active-Set QP Iterations:                   {res_qp['iterations']}")

# %%
# ## Web Scrape for Stock Tickers

# %%
# Define wikipedia API endpoint
API = 'https://en.wikipedia.org/w/api.php'

# Parameters to request the 'List of S&P 500 companies' page parsed as JSON
params = {
    'action': 'parse',
    'page': 'List_of_S&P_500_companies',
    'prop': 'text',
    'format': 'json'
}

# Identity header to prevent the API from blocking the request
headers = {'User-Agent': 'sp500-ticker-tutorial/1.0'}
r = requests.get(API, params=params, headers=headers)
html = r.json()['parse']['text']['*']

# Read the HTML table into a DataFrame and extract the symbol column as a list
df = pd.read_html(StringIO(html))[0]
# Clean periods to hyphens directly (e.g. 'BRK.B' -> 'BRK-B', 'BF.B' -> 'BF-B')
tickers = df['Symbol'].astype(str).str.replace('.', '-', regex=False).tolist()

# %%
# ## Data Importation & Organization (Streamlined Single-Download)
# Instead of downloading `top_100` a second time, we slice `closing_prices[top_100_tickers]` directly from the initial download and use `fast_info` for faster share-count lookups.

# %%
# Define Date Range (26-week window)
start_date = "2025-05-26"
end_date = "2025-11-28"

# Download raw data for all S&P 500 tickers once
data = yf.download(tickers, start=start_date, end=end_date, interval="1wk")
closing_prices = data['Close']

# Download raw data for S&P 500 benchmark (SPY)
spy_data = yf.download("SPY", start=start_date, end=end_date, interval="1wk")
spy_prices = spy_data['Close']

# Drop any ticker columns that have missing values in the 26-week window first
valid_closing = closing_prices.dropna(axis=1)

# Compute Market Cap for valid tickers
results = []
for ticker in valid_closing.columns:
    try:
        last_price = valid_closing[ticker].iloc[-1]
        stock = yf.Ticker(ticker)
        # fast_info is significantly faster than stock.info; fall back to info if needed
        shares = stock.fast_info.get('shares') if hasattr(stock, 'fast_info') else None
        if shares is None:
            shares = stock.info.get('sharesOutstanding')

        if shares is not None:
            results.append({'Ticker': ticker, 'MarketCap': float(last_price * shares)})
    except Exception:
        continue

# Sort from largest to smallest and grab top 100 stocks by market cap
dataframe = pd.DataFrame(results)
top_100 = dataframe.sort_values(by='MarketCap', ascending=False).head(100)
top_100_tickers = top_100["Ticker"].tolist()

# Slice directly from valid_closing (no second yf.download needed!)
top_100_close = valid_closing[top_100_tickers]

# Check for missing values in data
missing_counts = top_100_close.isna().sum().sum()
if missing_counts == 0:
    print(f"Selected Top {len(top_100_tickers)} stocks. The Data contains no missing values.")
else:
    print(f"The Data contains {missing_counts} missing values.")

# %%
# ## Constructing the Sample Covariance Matrix

# %%
# Create the returns matrix and take the transpose for required shape (p x n)
returns = top_100_close.pct_change().dropna()
returns_matrix = returns.T.to_numpy()

# Dynamically set p (100 assets) and n (26 weeks) from returns_matrix
p, n = returns_matrix.shape
print(f"Returns matrix shape: p = {p} assets, n = {n} weeks")

# Create return vector for benchmark
spy_returns = spy_prices.pct_change().dropna()

# 6-month treasury yield rates
risk_free_rate = 0.0375
weekly_RFR = risk_free_rate / 52

# Excess Returns Formula
excess_returns_matrix = returns_matrix - weekly_RFR
spy_excess_returns = spy_returns - weekly_RFR

# axis=1 takes mean of each row and creates a column vector of each stock's mean
excess_returns_mean_vector = excess_returns_matrix.mean(axis=1, keepdims=True)
mu_vec = excess_returns_mean_vector.flatten()

# Creating the p x n de-meaned excess returns matrix Y
Y = excess_returns_matrix - excess_returns_mean_vector

# Calculate sample covariance matrix S (p x p)
S = (Y @ Y.T) / n

# %%
# ## Compute Single Factor Covariance Models ($100 \times 100$)
# We construct all three $100 \times 100$ single-factor covariance models of the form $V = \sigma_f^2 \beta \beta^\top + \Delta$:
# 1. **Basic Empirical Single-Factor Model ($\Sigma_{\text{Simple}}$):** Regresses each stock's excess returns on the `SPY` market factor ($r_{i,t} = \alpha_i + \beta_i r_{M,t} + \epsilon_{i,t}$), giving market variance $\sigma_M^2$, exposure vector $\beta$, and diagonal idiosyncratic variances $\Delta = \text{diag}(\delta_1^2, \dots, \delta_{100}^2)$.
# 2. **PCA Single-Factor Model ($\Sigma_{\text{PCA}}$):** Uses the leading eigenvalue $\lambda^2$ and unit eigenvector $h$ of $S$ with $\Delta = (n/p)\ell^2 I_p$.
# 3. **JSE Single-Factor Model ($\Sigma_{\text{JSE}}$):** Shrinks the leading eigenvector $h$ toward its mean $m(h)e$ via the James-Stein shrinkage constant $c^{\text{JSE}}$.

# %%
# --- 1. Basic Market Single-Factor Covariance Model (Sigma_simple) ---
spy_excess_vec = np.asarray(spy_excess_returns).flatten()
spy_centered = spy_excess_vec - spy_excess_vec.mean()
sigma_m_sq = float(np.dot(spy_centered, spy_centered) / n)

# Factor exposures beta_i = Cov(R_i, R_M) / Var(R_M)
beta_simple = (Y @ spy_centered) / (n * sigma_m_sq)
residuals_simple = Y - np.outer(beta_simple, spy_centered)
delta_sq_simple = np.sum(residuals_simple ** 2, axis=1) / (n - 2)

Sigma_simple = sigma_m_sq * np.outer(beta_simple, beta_simple) + np.diag(delta_sq_simple)

# --- 2 & 3. PCA and JSE Single-Factor Covariance Models (Sigma_pca, Sigma_jse) ---
# Compute eigen-pairs for S
eigenvalues_S, eigenvectors_S = np.linalg.eigh(S)

# Compute leading eigenvalue and corresponding unit vector
lambda_sq = eigenvalues_S[-1]
h = eigenvectors_S[:, -1].reshape(-1, 1)

# Compute ell_sq
trace_S = np.trace(S)
ell_sq = (trace_S - lambda_sq) / (n - 1)

# Compute m(h)
m_h = h.mean()
ones = np.ones((p, 1))

# Compute nu_sq and s_sq(h)
nu_sq = (trace_S - lambda_sq) / (p * (n - 1))
s_sq = (lambda_sq / p) * np.sum((h - m_h) ** 2)

# Compute c_jse - ensure does not go negative
c_jse = max(0.0, 1.0 - (nu_sq / s_sq))

# Compute h_jse = m(h)e + c_jse * (h - m(h)e)
h_jse = (m_h * ones) + c_jse * (h - (m_h * ones))
h_jse_norm_sq = float(np.linalg.norm(h_jse) ** 2)

# Assemble JSE Single-Factor Covariance Matrix (p x p)
Sigma_jse_term1 = (lambda_sq - ell_sq) * (h_jse @ h_jse.T) / h_jse_norm_sq
Sigma_jse_term2 = (n / p) * ell_sq * np.eye(p)
Sigma_jse = Sigma_jse_term1 + Sigma_jse_term2

# Assemble PCA Single-Factor Covariance Matrix (p x p) -- FIXED p=100 (replaced hardcoded 400)
Sigma_pca_term1 = (lambda_sq - ell_sq) * (h @ h.T)
Sigma_pca_term2 = (n / p) * ell_sq * np.eye(p)
Sigma_pca = Sigma_pca_term1 + Sigma_pca_term2

# Grab eigenvalues and eigenvectors from both PCA and JSE
eigenvalues_pca, eigenvectors_pca = np.linalg.eigh(Sigma_pca)
eigenvalues_jse, eigenvectors_jse = np.linalg.eigh(Sigma_jse)
h_pca = eigenvectors_pca[:, -1].reshape(-1, 1)
h_jse_eig = eigenvectors_jse[:, -1].reshape(-1, 1)

# Align signs with sample eigenvector h
if np.dot(h.T, h_pca) < 0:
    h_pca = -h_pca
if np.dot(h.T, h_jse_eig) < 0:
    h_jse_eig = -h_jse_eig

# Compute MSE between eigenvectors
mse_eigenvector_pca = float(np.mean((h - h_pca) ** 2))
mse_eigenvector_jse = float(np.mean((h - h_jse_eig) ** 2))

# %%
# ## Compute the Long-Only GMV Portfolios (Applying the Long-Only Constraints)
# Here is where the **long-only constraints ($w_i \ge 0$)** replace the unconstrained $\Sigma^{-1}e / (e^\top \Sigma^{-1} e)$ formula:
# - Instead of inverting $\Sigma$ directly (which produces negative short weights), we pass `Sigma_simple`, `Sigma_pca`, and `Sigma_jse` into `active_set_qp(Sigma, target=None)`.
# - We also verify that our numerical active-set solution matches the analytical $R_i$ formula on all three empirical models.

# %%
# 1. Solve Long-Only GMV Portfolios using Active-Set QP (w >= 0, e^T w = 1)
res_lo_simple = active_set_qp(Sigma_simple, target=None)
res_lo_pca = active_set_qp(Sigma_pca, target=None)
res_lo_jse = active_set_qp(Sigma_jse, target=None)

holding_LO_simple = res_lo_simple["weights"].reshape(-1, 1)
holding_LO_pca = res_lo_pca["weights"].reshape(-1, 1)
holding_LO_jse = res_lo_jse["weights"].reshape(-1, 1)

# Verify against Analytical R_i formula for each single-factor model
ri_simple = analytical_long_only_gmv(sigma_m_sq, beta_simple, delta_sq_simple)
ri_pca = analytical_long_only_gmv(lambda_sq - ell_sq, h.flatten(), np.full(p, (n / p) * ell_sq))
ri_jse = analytical_long_only_gmv(
    (lambda_sq - ell_sq) / h_jse_norm_sq,
    h_jse.flatten(),
    np.full(p, (n / p) * ell_sq)
)

print("--- Empirical Long-Only GMV Validation (Active-Set QP vs Analytical R_i) ---")
print(f"Simple Factor Max Weight Diff: {np.max(np.abs(res_lo_simple['weights'] - ri_simple['weights'])):.2e} | Positive Weights: {ri_simple['k_positive']}/{p}")
print(f"PCA Factor Max Weight Diff:    {np.max(np.abs(res_lo_pca['weights'] - ri_pca['weights'])):.2e} | Positive Weights: {ri_pca['k_positive']}/{p}")
print(f"JSE Factor Max Weight Diff:    {np.max(np.abs(res_lo_jse['weights'] - ri_jse['weights'])):.2e} | Positive Weights: {ri_jse['k_positive']}/{p}")

# 2. Helper to compute annualized portfolio metrics
def compute_portfolio_stats(w_col, Sigma, mu_col):
    var_wk = float((w_col.T @ Sigma @ w_col).item())
    ann_var = var_wk * 52
    ann_std = float(np.sqrt(ann_var))
    ann_ret = float((w_col.T @ mu_col).item() * 52)
    sr = ann_ret / ann_std
    beta_vec = (Sigma @ w_col) / var_wk
    pos_count = int(np.sum(w_col.flatten() > 1e-7))
    return {
        "ann_var": ann_var,
        "ann_std": ann_std,
        "ann_ret": ann_ret,
        "sr": sr,
        "beta_mean": float(beta_vec.mean()),
        "pos_count": pos_count
    }

stats_simple = compute_portfolio_stats(holding_LO_simple, Sigma_simple, excess_returns_mean_vector)
stats_pca = compute_portfolio_stats(holding_LO_pca, Sigma_pca, excess_returns_mean_vector)
stats_jse = compute_portfolio_stats(holding_LO_jse, Sigma_jse, excess_returns_mean_vector)

# Compute Individual Stock Variances
annual_individual_variance_simple = np.diag(Sigma_simple) * 52
annual_individual_variance_pca = np.diag(Sigma_pca) * 52
annual_individual_variance_jse = np.diag(Sigma_jse) * 52

# Benchmark (SPY) metrics
spy_annual_variance = float(np.asarray(spy_excess_returns.var()).item() * 52)
spy_annual_std_dev = float(np.sqrt(spy_annual_variance))
spy_annual_expected_excess_returns = float(np.asarray(spy_excess_returns.mean()).item() * 52)
SR_spy = spy_annual_expected_excess_returns / spy_annual_std_dev

# %%
# ## Long-Only GMV Portfolio Metrics Comparison

# %%
print("\n------ JSE REGULARIZATION STATS (p = 100, n = 26) ------")
print(f"Shrinkage Constant (c_jse): {c_jse:.4f}")
print(f"MSE (Sample h vs JSE h):    {mse_eigenvector_jse:.8f}")
print(f"MSE (Sample h vs PCA h):    {mse_eigenvector_pca:.8f}")

print("\n------ LONG-ONLY GMV PORTFOLIO PERFORMANCE COMPARISON ------")
print(f"{'Metric':<25} | {'Simple Model':<12} | {'PCA Model':<12} | {'JSE Model':<12} | {'SPY (Bench)'}")
print("-" * 82)
print(f"{'Annualized Variance':<25} | {stats_simple['ann_var']:<12.6f} | {stats_pca['ann_var']:<12.6f} | {stats_jse['ann_var']:<12.6f} | {spy_annual_variance:.6f}")
print(f"{'Annualized Std Dev':<25} | {stats_simple['ann_std']:<12.6f} | {stats_pca['ann_std']:<12.6f} | {stats_jse['ann_std']:<12.6f} | {spy_annual_std_dev:.6f}")
print(f"{'Expected Excess Return':<25} | {stats_simple['ann_ret']:<12.6f} | {stats_pca['ann_ret']:<12.6f} | {stats_jse['ann_ret']:<12.6f} | {spy_annual_expected_excess_returns:.6f}")
print(f"{'Sharpe Ratio':<25} | {stats_simple['sr']:<12.4f} | {stats_pca['sr']:<12.4f} | {stats_jse['sr']:<12.4f} | {SR_spy:.4f}")
print(f"{'Positive Weights (k/100)':<25} | {stats_simple['pos_count']:<12d} | {stats_pca['pos_count']:<12d} | {stats_jse['pos_count']:<12d} | 500")
print(f"{'Average Portfolio Beta':<25} | {stats_simple['beta_mean']:<12.4f} | {stats_pca['beta_mean']:<12.4f} | {stats_jse['beta_mean']:<12.4f} | 1.0000")

print("\n------ INDIVIDUAL ASSET BENCHMARK ------")
print(f"{'Mean Stock Variance':<25} | {annual_individual_variance_simple.mean():<12.6f} | {annual_individual_variance_pca.mean():<12.6f} | {annual_individual_variance_jse.mean():<12.6f}")
print(f"{'Min. Stock Variance':<25} | {annual_individual_variance_simple.min():<12.6f} | {annual_individual_variance_pca.min():<12.6f} | {annual_individual_variance_jse.min():<12.6f}")

# %%
# ## Long-Only Efficient Frontier & Tangency Portfolio
# Using the 2-constraint `active_set_qp(Sigma, mu_vec, target=f)` solver across a grid of target returns $f \in [\mu_{\text{GMV}}, \max_i \mu_i]$, we trace out the long-only efficient frontier and identify the Sharpe-maximizing tangency portfolio for each covariance model.

# %%
def compute_efficient_frontier(Sigma, mu_vec, n_points=40):
    # First solve Long-Only GMV to get the starting return at the bottom of the efficient frontier
    gmv_res = active_set_qp(Sigma, target=None)
    mu_gmv = float(mu_vec @ gmv_res["weights"])
    mu_max = float(np.max(mu_vec))

    # Grid of weekly target excess returns from GMV return up to 99% of max single-stock return
    targets = np.linspace(mu_gmv, mu_max * 0.99, n_points)
    ann_stds = []
    ann_rets = []
    sharpes = []
    weights_list = []

    for f_target in targets:
        res = active_set_qp(Sigma, mu_vec, target=f_target)
        w = res["weights"]
        ann_std = float(np.sqrt(res["variance"] * 52))
        ann_ret = float(f_target * 52)
        sr = ann_ret / ann_std if ann_std > 0 else 0.0

        ann_stds.append(ann_std)
        ann_rets.append(ann_ret)
        sharpes.append(sr)
        weights_list.append(w)

    best_idx = int(np.argmax(sharpes))
    return {
        "ann_stds": np.array(ann_stds),
        "ann_rets": np.array(ann_rets),
        "sharpes": np.array(sharpes),
        "weights": weights_list,
        "tangency_idx": best_idx,
        "tangency_std": ann_stds[best_idx],
        "tangency_ret": ann_rets[best_idx],
        "tangency_sr": sharpes[best_idx],
        "tangency_weights": weights_list[best_idx]
    }

frontier_simple = compute_efficient_frontier(Sigma_simple, mu_vec)
frontier_pca = compute_efficient_frontier(Sigma_pca, mu_vec)
frontier_jse = compute_efficient_frontier(Sigma_jse, mu_vec)

print("--- LONG-ONLY TANGENCY (MAX SHARPE) PORTFOLIOS ---")
for name, fr in [("Simple Factor", frontier_simple), ("PCA Factor", frontier_pca), ("JSE Factor", frontier_jse)]:
    n_pos = int(np.sum(fr["tangency_weights"] > 1e-7))
    print(f"{name:<15} | Max Sharpe: {fr['tangency_sr']:.4f} | Ann Return: {fr['tangency_ret']:.4f} | Ann Std: {fr['tangency_std']:.4f} | Active Stocks: {n_pos}/100")

# Plot the Long-Only Efficient Frontiers
plt.figure(figsize=(10, 6))
plt.plot(frontier_simple["ann_stds"], frontier_simple["ann_rets"], label="Simple Factor Frontier", lw=2)
plt.plot(frontier_pca["ann_stds"], frontier_pca["ann_rets"], label="PCA Factor Frontier", lw=2)
plt.plot(frontier_jse["ann_stds"], frontier_jse["ann_rets"], label="JSE Factor Frontier", lw=2)

# Highlight Tangency Portfolios
plt.scatter(frontier_simple["tangency_std"], frontier_simple["tangency_ret"], marker="*", s=160, zorder=5, label=f"Simple Tangency (SR={frontier_simple['tangency_sr']:.2f})")
plt.scatter(frontier_pca["tangency_std"], frontier_pca["tangency_ret"], marker="*", s=160, zorder=5, label=f"PCA Tangency (SR={frontier_pca['tangency_sr']:.2f})")
plt.scatter(frontier_jse["tangency_std"], frontier_jse["tangency_ret"], marker="*", s=160, zorder=5, label=f"JSE Tangency (SR={frontier_jse['tangency_sr']:.2f})")

plt.title("Long-Only Efficient Frontiers (Top 100 S&P 500 Stocks)", loc="left", fontsize=13)
plt.xlabel("Annualized Volatility (Std Dev)", fontsize=11)
plt.ylabel("Annualized Expected Excess Return", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

