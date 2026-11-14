"""Method-of-moments solution of the batch population balance.

For size-independent growth and nucleation at L = 0 the PBE reduces to

    d(mu_0)/dt = B
    d(mu_j)/dt = j * G * mu_(j-1),   j >= 1
    dC/dt      = -3 * rho_c * kv * G * mu_2

Seed-grown and nucleated crystals are tracked as two separate populations so
the share of product mass that comes from nucleation (fines) can be reported.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .model import (Params, growth_rate, nucleation_rate, seed_moments,
                    solubility, supersaturation)

N_MOM = 5  # mu_0 .. mu_4


@dataclass
class MomentResult:
    t: np.ndarray
    T: np.ndarray
    C: np.ndarray
    S: np.ndarray
    seed: np.ndarray       # shape (5, nt)
    nucleated: np.ndarray  # shape (5, nt)
    params: Params

    @property
    def mu(self):
        return self.seed + self.nucleated

    @property
    def L43(self):
        """Volume-weighted mean size [m]."""
        return self.mu[4] / self.mu[3]

    @property
    def nucleated_mass_fraction(self):
        return self.nucleated[3] / self.mu[3]

    @property
    def crystal_mass(self):
        return self.params.rho_c * self.params.kv * self.mu[3]

    def mass_balance_error(self):
        """Relative drift of total solute (dissolved + solid)."""
        total = self.C + self.crystal_mass
        return np.max(np.abs(total - total[0])) / total[0]


def _rhs(t, y, p: Params, cooling):
    seed, nuc, C = y[:N_MOM], y[N_MOM:2 * N_MOM], y[-1]
    T = cooling(t, p)
    S = supersaturation(C, T)
    G = growth_rate(S, p)
    mu = seed + nuc
    B = nucleation_rate(S, mu[3], p)

    dseed = np.zeros(N_MOM)
    dnuc = np.zeros(N_MOM)
    dnuc[0] = B
    for j in range(1, N_MOM):
        dseed[j] = j * G * seed[j - 1]
        dnuc[j] = j * G * nuc[j - 1]
    dC = -3.0 * p.rho_c * p.kv * G * mu[2]
    return np.concatenate([dseed, dnuc, [dC]])


def simulate_moments(p: Params, cooling, t_eval=None, n_points: int = 241):
    """Integrate the moment equations over one batch."""
    if t_eval is None:
        t_eval = np.linspace(0.0, p.t_batch, n_points)
    seed0 = seed_moments(p, N_MOM)
    y0 = np.concatenate([seed0, np.zeros(N_MOM), [solubility(p.T0)]])
    # The moments span ~25 orders of magnitude, so scale absolute tolerances.
    atol = np.concatenate([1e-9 * seed0, 1e-9 * seed0, [1e-10]])
    sol = solve_ivp(_rhs, (0.0, p.t_batch), y0, args=(p, cooling),
                    method="LSODA", t_eval=t_eval, rtol=1e-8, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    T = cooling(sol.t, p)
    C = sol.y[-1]
    return MomentResult(t=sol.t, T=T, C=C, S=supersaturation(C, T),
                        seed=sol.y[:N_MOM], nucleated=sol.y[N_MOM:2 * N_MOM],
                        params=p)
