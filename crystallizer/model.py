"""Physical model shared by the moment and population-balance solvers.

System: seeded batch cooling crystallization of a KNO3-like salt from water.
All quantities are per kg of solvent, SI units (m, s, kg), temperature in deg C.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    """Kinetic, physical and operating parameters.

    The kinetic constants are illustrative values chosen to give realistic
    behaviour; they are not fitted to a specific experimental data set.
    """

    # Growth: G = kg * S**g  [m/s], S = relative supersaturation
    kg: float = 6.0e-6
    g: float = 1.3
    # Secondary nucleation: B = kb * S**b * mu3  [#/(kg solvent s)]
    kb: float = 1.0e13
    b: float = 3.0
    # Crystal properties
    rho_c: float = 2110.0  # crystal density [kg/m3]
    kv: float = 1.0        # volume shape factor [-]
    # Operation
    T0: float = 32.0       # start temperature [C]
    Tf: float = 22.0       # end temperature [C]
    t_batch: float = 7200.0  # batch time [s]
    # Seed: Gaussian size distribution
    seed_mass: float = 0.005   # [kg seed / kg solvent]
    seed_mean: float = 100e-6  # [m]
    seed_std: float = 10e-6    # [m]


def solubility(T):
    """Saturation concentration [kg solute / kg water], T in deg C.

    Quadratic fit that approximates KNO3-water solubility between 20 and 40 C.
    """
    return 0.1286 + 5.88e-3 * T + 1.721e-4 * T**2


def supersaturation(C, T):
    """Relative supersaturation S = (C - C*) / C*, clipped at zero."""
    Cs = solubility(T)
    return np.maximum(C - Cs, 0.0) / Cs


def growth_rate(S, p: Params):
    return p.kg * S**p.g


def nucleation_rate(S, mu3, p: Params):
    return p.kb * S**p.b * mu3


# ---------------------------------------------------------------- cooling ---
def cooling_linear(t, p: Params):
    return p.T0 - (p.T0 - p.Tf) * (t / p.t_batch)


def cooling_natural(t, p: Params, n_tau: float = 4.0):
    """Exponential (uncontrolled jacket) cooling: fast at the start."""
    x = (1.0 - np.exp(-n_tau * t / p.t_batch)) / (1.0 - np.exp(-n_tau))
    return p.T0 - (p.T0 - p.Tf) * x


def cooling_cubic(t, p: Params):
    """Controlled cooling: slow at the start, fast at the end."""
    return p.T0 - (p.T0 - p.Tf) * (t / p.t_batch) ** 3


COOLING_PROFILES = {
    "natural": cooling_natural,
    "linear": cooling_linear,
    "cubic": cooling_cubic,
}


# ------------------------------------------------------------------- seed ---
def seed_number(p: Params):
    """Number of seed crystals per kg solvent for a Gaussian seed."""
    m, s = p.seed_mean, p.seed_std
    third_moment_per_crystal = m**3 + 3.0 * m * s**2
    return p.seed_mass / (p.rho_c * p.kv * third_moment_per_crystal)


def seed_moments(p: Params, n_moments: int = 5):
    """Raw moments mu_0..mu_4 of the Gaussian seed distribution."""
    m, s = p.seed_mean, p.seed_std
    raw = [
        1.0,
        m,
        m**2 + s**2,
        m**3 + 3 * m * s**2,
        m**4 + 6 * m**2 * s**2 + 3 * s**4,
    ]
    return seed_number(p) * np.array(raw[:n_moments])


def seed_density(L, p: Params):
    """Seed number density n(L) [#/(m kg solvent)]."""
    m, s = p.seed_mean, p.seed_std
    pdf = np.exp(-0.5 * ((L - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    return seed_number(p) * pdf
