"""Discretized population balance equation (full crystal size distribution).

    dn/dt + G(t) * dn/dL = 0,      n(0, t) = B / G

Solved with a finite-volume scheme (van Leer flux limiter, second order in
smooth regions, no spurious oscillations) and SSP-RK2 time stepping with a
CFL-limited step. Solute concentration follows from the overall mass balance,
so total solute is conserved exactly.
"""
from dataclasses import dataclass

import numpy as np

from .model import (Params, growth_rate, nucleation_rate, seed_density,
                    solubility, supersaturation)


@dataclass
class PBEResult:
    L: np.ndarray        # cell centres [m]
    dL: float
    n: np.ndarray        # final number density [#/(m kg solvent)]
    t: np.ndarray
    C: np.ndarray
    L43: np.ndarray
    params: Params

    def moment(self, j):
        return np.sum(self.n * self.L**j) * self.dL

    def volume_density(self):
        """Normalised volume-weighted size distribution [1/m]."""
        w = self.n * self.L**3
        return w / (np.sum(w) * self.dL)

    def volume_cv(self):
        """Coefficient of variation of the volume-weighted distribution."""
        q = self.volume_density() * self.dL
        mean = np.sum(q * self.L)
        var = np.sum(q * (self.L - mean) ** 2)
        return np.sqrt(var) / mean


def _van_leer(r):
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def _dndt(n, G, B, dL):
    """Finite-volume right-hand side for growth with positive G."""
    dn = np.diff(n)                                  # n[i+1] - n[i]
    safe = np.where(np.abs(dn[1:]) > 1e-300, dn[1:], 1e-300)
    r = dn[:-1] / safe                               # smoothness ratio
    flux = np.empty(n.size + 1)
    flux[0] = B                                      # nucleation at L = 0
    flux[1] = G * n[0]                               # first-order next to inlet
    flux[2:-1] = G * (n[1:-1] + 0.5 * _van_leer(r) * dn[1:])
    flux[-1] = G * n[-1]                             # outflow
    return -(flux[1:] - flux[:-1]) / dL


def simulate_pbe(p: Params, cooling, L_max: float = 700e-6, n_cells: int = 700,
                 cfl: float = 0.4, dt_max: float = 5.0):
    dL = L_max / n_cells
    L = (np.arange(n_cells) + 0.5) * dL
    n = seed_density(L, p)
    L3 = L**3

    def solid_mass(n_):
        return p.rho_c * p.kv * np.sum(n_ * L3) * dL

    total = solubility(p.T0) + solid_mass(n)  # conserved solute inventory

    def rates(n_, t_):
        C_ = total - solid_mass(n_)
        S_ = supersaturation(C_, cooling(t_, p))
        G_ = growth_rate(S_, p)
        B_ = nucleation_rate(S_, np.sum(n_ * L3) * dL, p)
        return G_, B_, C_

    t = 0.0
    ts, Cs, L43s = [], [], []
    while True:
        G, B, C = rates(n, t)
        ts.append(t)
        Cs.append(C)
        L43s.append(np.sum(n * L**4) / np.sum(n * L3))
        if t >= p.t_batch - 1e-9:
            break
        dt = min(dt_max, p.t_batch - t)
        if G > 0:
            dt = min(dt, cfl * dL / G)
        # SSP-RK2 (Heun)
        n1 = n + dt * _dndt(n, G, B, dL)
        G1, B1, _ = rates(n1, t + dt)
        n = 0.5 * (n + n1 + dt * _dndt(n1, G1, B1, dL))
        t += dt

    return PBEResult(L=L, dL=dL, n=n, t=np.array(ts), C=np.array(Cs),
                     L43=np.array(L43s), params=p)
