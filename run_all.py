"""Run every study and regenerate the figures and the results table.

    python run_all.py
"""
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crystallizer import (COOLING_PROFILES, Params, fit_growth_kinetics,
                          make_synthetic_data, simulate_moments, simulate_pbe,
                          solubility)

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
COLORS = {"natural": "#c44e52", "linear": "#4c72b0", "cubic": "#2a9d8f"}
LABELS = {"natural": "Natural (uncontrolled)", "linear": "Linear",
          "cubic": "Cubic (controlled)"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 130})


def cooling_study(p):
    out = {}
    for name, f in COOLING_PROFILES.items():
        out[name] = (simulate_moments(p, f), simulate_pbe(p, f))
    return out


def print_table(p, study):
    max_yield = solubility(p.T0) - solubility(p.Tf)
    head = ("| Cooling profile | L43 (um) | Volume CV | Mass from nucleation "
            "| Peak S | Yield | MoM vs PBE (L43) |")
    print(head)
    print("|---|---|---|---|---|---|---|")
    for name, (mom, pbe) in study.items():
        yld = (mom.C[0] - mom.C[-1]) / max_yield
        diff = abs(pbe.L43[-1] - mom.L43[-1]) / mom.L43[-1]
        print(f"| {LABELS[name]} | {mom.L43[-1] * 1e6:.0f} | "
              f"{pbe.volume_cv():.2f} | "
              f"{100 * mom.nucleated_mass_fraction[-1]:.1f}% | "
              f"{mom.S.max():.3f} | {100 * yld:.1f}% | {100 * diff:.2f}% |")


def plot_cooling_study(study):
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True)
    for name, (mom, _) in study.items():
        t = mom.t / 60
        kw = dict(color=COLORS[name], lw=2, label=LABELS[name])
        ax[0, 0].plot(t, mom.T, **kw)
        ax[0, 1].plot(t, mom.S, **kw)
        ax[1, 0].plot(t, mom.L43 * 1e6, **kw)
        ax[1, 1].plot(t, 100 * mom.nucleated_mass_fraction, **kw)
    ax[0, 0].set(ylabel="Temperature (°C)", title="Cooling profile")
    ax[0, 1].set(ylabel="Relative supersaturation S", title="Supersaturation")
    ax[1, 0].set(ylabel="L43 (µm)", xlabel="Time (min)",
                 title="Volume-weighted mean size")
    ax[1, 1].set(ylabel="Crystal mass from nucleation (%)",
                 xlabel="Time (min)", title="Fines formation")
    ax[0, 0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("Seeded batch cooling crystallization: effect of cooling profile",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "cooling_study.png")
    plt.close(fig)


def plot_csd(p, study):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, (_, pbe) in study.items():
        ax.plot(pbe.L * 1e6, pbe.volume_density() * 1e-6, color=COLORS[name],
                lw=2, label=LABELS[name])
    ax.axvline(p.seed_mean * 1e6, color="grey", ls=":", lw=1)
    ax.text(p.seed_mean * 1e6 + 5, ax.get_ylim()[1] * 0.93, "seed mean size",
            color="grey", fontsize=8)
    ax.set(xlabel="Crystal size L (µm)", ylabel="Volume density (1/µm)",
           title="Final crystal size distribution (discretized PBE)",
           xlim=(0, 500))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "final_csd.png")
    plt.close(fig)


def estimation_study(p):
    data = make_synthetic_data(p)
    guess = replace(p, kg=2.0e-6, g=1.0)
    fit = fit_growth_kinetics(data, guess)
    fitted = replace(p, kg=fit.kg, g=fit.g)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, (t, C_meas) in data.items():
        tt = np.linspace(0, p.t_batch, 200)
        C_fit = simulate_moments(fitted, COOLING_PROFILES[name], t_eval=tt).C
        C_guess = simulate_moments(guess, COOLING_PROFILES[name], t_eval=tt).C
        ax.plot(t / 60, C_meas, "o", ms=4, color=COLORS[name],
                label=f"{LABELS[name]}: synthetic data")
        ax.plot(tt / 60, C_fit, color=COLORS[name], lw=2)
        ax.plot(tt / 60, C_guess, color=COLORS[name], lw=1, ls="--", alpha=0.6)
    ax.plot([], [], color="k", lw=2, label="Calibrated model")
    ax.plot([], [], color="k", lw=1, ls="--", alpha=0.6, label="Initial guess")
    ax.set(xlabel="Time (min)", ylabel="Concentration (kg solute / kg water)",
           title="Model calibration against concentration data")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png")
    plt.close(fig)

    lo, hi = np.exp(np.log(fit.kg) - fit.ci95[0]), np.exp(np.log(fit.kg) + fit.ci95[0])
    print("\n| Parameter | True | Initial guess | Estimated | 95% interval |")
    print("|---|---|---|---|---|")
    print(f"| kg (m/s) | {p.kg:.2e} | {guess.kg:.2e} | {fit.kg:.2e} | "
          f"{lo:.2e} to {hi:.2e} |")
    print(f"| g | {p.g:.2f} | {guess.g:.2f} | {fit.g:.3f} | "
          f"{fit.g - fit.ci95[1]:.3f} to {fit.g + fit.ci95[1]:.3f} |")
    print(f"\nFit RMSE: {fit.rmse:.2e} kg/kg (noise added: 5.0e-04 kg/kg)")


if __name__ == "__main__":
    p = Params()
    study = cooling_study(p)
    print_table(p, study)
    plot_cooling_study(study)
    plot_csd(p, study)
    estimation_study(p)
    print(f"\nFigures written to {FIG}")
