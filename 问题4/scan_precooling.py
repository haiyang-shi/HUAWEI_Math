"""Recompute the 10--100 min constant-strategy scan at 84-cell resolution."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynamic_model import PrecoolingModel
from optimize import OUT, cooling_scan, load_q3_policy


def main():
    model = PrecoolingModel()
    powers, duration = load_q3_policy()
    rows = cooling_scan(model, powers, duration, 10.0,
                        mesh_factor=3, dt_max=0.1)
    scan_t = np.array([r["cooling_minutes"] for r in rows])
    initial_t = np.array([r["initial_min_cell_temperature_c"] for r in rows])
    startup = np.array([np.nan if r["startup_s"] is None else r["startup_s"]
                        for r in rows])
    voltage = np.array([r["min_voltage_v"] for r in rows])
    ice = np.array([r["max_ice_fraction"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(scan_t, initial_t, marker="o")
    axes[0, 0].set_ylabel("Initial minimum cell T (degC)")
    axes[0, 1].plot(scan_t, startup, marker="o")
    axes[0, 1].set_ylabel("Startup time (s)")
    axes[1, 0].plot(scan_t, voltage, marker="o")
    axes[1, 0].axhline(0.30, color="black", ls="--", lw=0.8)
    axes[1, 0].set_ylabel("Minimum voltage (V)")
    axes[1, 1].plot(scan_t, ice, marker="o")
    axes[1, 1].axhline(0.99, color="black", ls="--", lw=0.8)
    axes[1, 1].set_ylabel("Maximum ice fraction")
    for ax in axes.flat:
        ax.set_xlabel("Cooling time (min)")
        ax.grid(alpha=0.25)
    fig.suptitle("Q3 constant-power strategy versus pre-cooling duration")
    fig.savefig(OUT / "constant_strategy_cooling_scan.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
