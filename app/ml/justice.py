"""
Justice Statistics — Poisson xG Soft Labels (Y_soft) + Justice Index (W).

Module 1 of the Post-Match Auditor (GDT directive).

Y_soft: Dixon-Coles corrected Poisson fair probabilities from xG.
    Used for evaluation/Brier decomposition. NOT as XGBoost labels.

W (Justice Index): Sample weight based on how "fair" the result was.
    Integrated as multiplicative sample_weight in XGBoost training.

GDT Overrides applied:
    1. Dixon-Coles ρ=-0.15 (low-scoring correlation)
    2. Variance-scaled error: σ = √(xG_h + xG_a + 1.0)
    4. max_goals=10 (covers xG > 5.0)
"""

import numpy as np

# GDT Override 4: max_goals=10 to cover xG > 5.0 without residual
DEFAULT_MAX_GOALS = 10

# GDT Override 1: Dixon-Coles rho (empirical low-score correlation)
DEFAULT_RHO = -0.15


def compute_y_soft_batch(
    xg_home: np.ndarray,
    xg_away: np.ndarray,
    max_goals: int = DEFAULT_MAX_GOALS,
    rho: float = DEFAULT_RHO,
) -> np.ndarray:
    """Vectorized Dixon-Coles soft-labels. Shape (N, 3) = [p_home, p_draw, p_away].

    Base: P(H=h, A=a) = Poisson(h|λ_h) × Poisson(a|λ_a)

    Dixon-Coles adjustment (GDT Override 1): Applies τ(h,a) correction to the
    4 low-scoring cells (0-0, 1-0, 0-1, 1-1) to fix the systematic
    underestimation of draws in independent Poisson.

    With ρ=-0.15:
    - τ(0,0) = 1 - λ_h × λ_a × ρ  → INFLATES P(0-0)
    - τ(1,0) = 1 + λ_a × ρ          → DEFLATES P(1-0)
    - τ(0,1) = 1 + λ_h × ρ          → DEFLATES P(0-1)
    - τ(1,1) = 1 - ρ                 → INFLATES P(1-1)

    Args:
        xg_home: Array of shape (N,) with home xG values.
        xg_away: Array of shape (N,) with away xG values.
        max_goals: Maximum goals to enumerate per side (GDT: 10).
        rho: Dixon-Coles correlation parameter (GDT: -0.15).

    Returns:
        Array of shape (N, 3) with [p_home, p_draw, p_away] per row.
        Each row sums to 1.0.
    """
    goals = np.arange(max_goals + 1)

    # Log-factorials: log(0!)=0, log(k!)=Σlog(1..k)
    log_fact = np.zeros(max_goals + 1)
    for k in range(1, max_goals + 1):
        log_fact[k] = log_fact[k - 1] + np.log(k)

    lam_h = np.clip(xg_home, 1e-10, None)[:, None]  # (N, 1)
    lam_a = np.clip(xg_away, 1e-10, None)[:, None]

    log_p_h = goals[None, :] * np.log(lam_h) - lam_h - log_fact[None, :]  # (N, G)
    log_p_a = goals[None, :] * np.log(lam_a) - lam_a - log_fact[None, :]

    joint = np.exp(log_p_h)[:, :, None] * np.exp(log_p_a)[:, None, :]  # (N, G, G)

    # === GDT Override 1: Dixon-Coles τ adjustment ===
    l_h = lam_h[:, 0]  # (N,)
    l_a = lam_a[:, 0]  # (N,)

    # τ(0,0) = 1 - (λ × μ × ρ)  — inflates 0-0
    tau_00 = np.clip(1 - (l_h * l_a * rho), 0, None)
    # τ(1,0) = 1 + (μ × ρ)       — deflates 1-0
    tau_10 = np.clip(1 + (l_a * rho), 0, None)
    # τ(0,1) = 1 + (λ × ρ)       — deflates 0-1
    tau_01 = np.clip(1 + (l_h * rho), 0, None)
    # τ(1,1) = 1 - ρ              — inflates 1-1
    tau_11 = np.clip(1 - rho, 0, None)

    joint[:, 0, 0] *= tau_00
    joint[:, 1, 0] *= tau_10
    joint[:, 0, 1] *= tau_01
    joint[:, 1, 1] *= tau_11

    # Re-normalize to guarantee sum = 1.0 after τ adjustment
    joint /= joint.sum(axis=(1, 2), keepdims=True)
    # === End Dixon-Coles ===

    h_grid, a_grid = np.meshgrid(goals, goals, indexing="ij")  # (G, G)

    p_home = (joint * (h_grid > a_grid)[None]).sum(axis=(1, 2))
    p_draw = (joint * (h_grid == a_grid)[None]).sum(axis=(1, 2))
    p_away = (joint * (h_grid < a_grid)[None]).sum(axis=(1, 2))

    result = np.column_stack([p_home, p_draw, p_away])
    return result / result.sum(axis=1, keepdims=True)  # final normalize


def compute_y_soft(
    xg_home: float,
    xg_away: float,
    max_goals: int = DEFAULT_MAX_GOALS,
    rho: float = DEFAULT_RHO,
) -> tuple[float, float, float]:
    """Single-match Dixon-Coles soft-label computation.

    Returns:
        (p_home, p_draw, p_away) summing to 1.0.
    """
    r = compute_y_soft_batch(
        np.array([xg_home]), np.array([xg_away]), max_goals, rho
    )
    return float(r[0, 0]), float(r[0, 1]), float(r[0, 2])


def compute_justice_weight(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    xg_home: np.ndarray,
    xg_away: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Justice Index: W = exp(-α × |GD - xGD| / σ). NaN xG → W = 1.0.

    GDT Override 2: Scale error by expected standard deviation.
    σ = √(xG_home + xG_away + 1.0)
    The +1.0 smooths low-xG matches and prevents division by zero.

    Interpretation: A 1-goal error in a 4+ total-goals match is LESS
    "unjust" than the same error in a tight 0-0. Poisson variance scales
    with the mean, so we normalize by σ.

    Distribution with α=0.5, xG_h=1.5, xG_a=1.0 (σ≈1.87):
      |GD-xGD|=0 → W=1.00, |GD-xGD|=1 → W=0.77,
      |GD-xGD|=2 → W=0.59, |GD-xGD|=3 → W=0.45

    For matches without xG: W = 1.0 (neutral, no information to judge).

    Args:
        home_goals: Actual home goals, shape (N,).
        away_goals: Actual away goals, shape (N,).
        xg_home: Expected goals home, shape (N,). May contain NaN.
        xg_away: Expected goals away, shape (N,). May contain NaN.
        alpha: Decay rate (GDT recommends starting at 0.5).

    Returns:
        Array of shape (N,) with justice weights in (0, 1].
    """
    gd = home_goals - away_goals
    xgd = xg_home - xg_away
    has_xg = np.isfinite(xg_home) & np.isfinite(xg_away)

    # GDT Pro-Tip: suppress cosmetic RuntimeWarning for sqrt/div with NaN
    with np.errstate(invalid="ignore"):
        # GDT Override 2: scale by expected standard deviation
        std_dev = np.sqrt(xg_home + xg_away + 1.0)
        scaled_error = np.abs(gd - xgd) / std_dev

    w = np.exp(-alpha * scaled_error)
    return np.where(has_xg, w, 1.0)
