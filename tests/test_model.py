"""Verification tests. Run with:  pytest"""
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crystallizer import (COOLING_PROFILES, Params, fit_growth_kinetics,
                          make_synthetic_data, simulate_moments, simulate_pbe)
from crystallizer.model import seed_density, seed_moments, solubility

P = Params()


def test_solubility_increases_with_temperature():
    T = np.linspace(20, 40, 21)
    assert np.all(np.diff(solubility(T)) > 0)


def test_seed_mass_is_consistent():
    mu = seed_moments(P)
    assert P.rho_c * P.kv * mu[3] == pytest.approx(P.seed_mass, rel=1e-12)
    L = np.linspace(0, 300e-6, 20001)
    numeric = np.trapezoid(seed_density(L, P) * L**3, L)
    assert numeric == pytest.approx(mu[3], rel=1e-6)


@pytest.mark.parametrize("name", list(COOLING_PROFILES))
def test_moment_model_conserves_mass(name):
    res = simulate_moments(P, COOLING_PROFILES[name])
    assert res.mass_balance_error() < 1e-8


@pytest.mark.parametrize("name", list(COOLING_PROFILES))
def test_profiles_hit_start_and_end_temperature(name):
    f = COOLING_PROFILES[name]
    assert f(0.0, P) == pytest.approx(P.T0)
    assert f(P.t_batch, P) == pytest.approx(P.Tf)


def test_no_supersaturation_means_no_change():
    """Holding at the saturation temperature must leave the seed untouched."""
    res = simulate_moments(P, lambda t, p: p.T0 + 0.0 * t)
    assert res.mu[3][-1] == pytest.approx(res.mu[3][0], rel=1e-10)
    assert res.nucleated[0][-1] == 0.0


@pytest.mark.parametrize("name", list(COOLING_PROFILES))
def test_pbe_agrees_with_moments(name):
    """Two independent numerical methods must give the same answer."""
    mom = simulate_moments(P, COOLING_PROFILES[name])
    pbe = simulate_pbe(P, COOLING_PROFILES[name])
    assert pbe.L43[-1] == pytest.approx(mom.L43[-1], rel=0.01)
    assert pbe.C[-1] == pytest.approx(mom.C[-1], rel=0.002)
    assert pbe.moment(3) == pytest.approx(mom.mu[3][-1], rel=0.005)


def test_controlled_cooling_gives_fewer_fines():
    frac = {k: simulate_moments(P, f).nucleated_mass_fraction[-1]
            for k, f in COOLING_PROFILES.items()}
    assert frac["cubic"] < frac["linear"] < frac["natural"]


def test_parameter_estimation_recovers_truth():
    data = make_synthetic_data(P)
    guess = replace(P, kg=2.0e-6, g=1.0)
    fit = fit_growth_kinetics(data, guess)
    assert fit.kg == pytest.approx(P.kg, rel=0.15)
    assert fit.g == pytest.approx(P.g, abs=0.05)
