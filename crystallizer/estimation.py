"""Model calibration: estimate growth kinetics from concentration data.

The 'experimental' data here are SYNTHETIC: generated from the model with
known parameters plus Gaussian measurement noise. The exercise shows that the
calibration workflow recovers the true values; swap in real concentration
measurements (e.g. from ATR-FTIR) to calibrate against a real system.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from .model import COOLING_PROFILES, Params
from .moments import simulate_moments


@dataclass
class FitResult:
    kg: float
    g: float
    ci95: np.ndarray  # half-widths for (ln kg, g)
    rmse: float
    cost: float


def make_synthetic_data(p: Params, profiles=("linear", "natural"),
                        n_samples: int = 25, noise: float = 5e-4, seed: int = 0):
    """Noisy concentration measurements for each cooling experiment."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, p.t_batch, n_samples)
    data = {}
    for name in profiles:
        res = simulate_moments(p, COOLING_PROFILES[name], t_eval=t)
        data[name] = (t, res.C + rng.normal(0.0, noise, t.size))
    return data


def fit_growth_kinetics(data, p_guess: Params):
    """Least-squares fit of kg and g to concentration data from all runs."""

    def residuals(theta):
        p = replace(p_guess, kg=float(np.exp(theta[0])), g=float(theta[1]))
        out = []
        for name, (t, C_meas) in data.items():
            res = simulate_moments(p, COOLING_PROFILES[name], t_eval=t)
            out.append(res.C - C_meas)
        return np.concatenate(out)

    theta0 = np.array([np.log(p_guess.kg), p_guess.g])
    sol = least_squares(residuals, theta0, x_scale=[1.0, 1.0],
                        bounds=([np.log(1e-8), 0.5], [np.log(1e-3), 3.0]))
    n, k = sol.fun.size, sol.x.size
    sigma2 = np.sum(sol.fun**2) / (n - k)
    cov = sigma2 * np.linalg.inv(sol.jac.T @ sol.jac)
    return FitResult(kg=float(np.exp(sol.x[0])), g=float(sol.x[1]),
                     ci95=1.96 * np.sqrt(np.diag(cov)),
                     rmse=float(np.sqrt(np.mean(sol.fun**2))),
                     cost=float(sol.cost))
