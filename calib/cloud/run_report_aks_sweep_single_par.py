"""
Overlay `cases_by_month` across trials while holding all params near-best
except one "sweep" parameter (e.g., seasonal_amplitude).

- Uses Optuna study (AKS/MySQL port-forward just like your existing script)
- Selects near-best trials for all params except the sweep param
- Splits the sweep param into low/med/high (tertiles by default, or explicit bins)
- Overlays `cases_by_month` for N trials from each bin on separate figures
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cloud_calib_config as cfg
import matplotlib.pyplot as plt
import numpy as np
import optuna

# ------------------- USER CONFIGS -------------------
# Override your study + selection logic here (CLI still works if you prefer).
USER = {
    "study_name": None,  # Which study to load. If none, use the one in cloud_calib_config.py
    "sweep_param": "seasonal_amplitude",  # Which parameters to sweep.
    "n_bins": 3,  # Number of bins to split the sweep param into
    "n_per_bin": 10,  # Number of trials to plotper bin
    "output_root": None,  # Output directory root; if None -> results/<study_name>/sweep_<param>/
}
# ----------------- END USER CONFIGS -----------------


def port_forward():
    print("🔌 Setting up port forwarding to MySQL...")
    pf = subprocess.Popen(["kubectl", "port-forward", "mysql-0", "3307:3306"])
    time.sleep(3)  # wait for port-forward to take effect
    return pf


def main():
    pf_process = port_forward()
    if USER["study_name"] is None:
        study_name = cfg.study_name
    else:
        study_name = USER["study_name"]
    print(f"📊 Loading study '{study_name}'...")
    study = optuna.load_study(study_name=study_name, storage=cfg.storage_url)
    study.storage_url = cfg.storage_url
    study.study_name = cfg.study_name
    sweep_par = USER["sweep_param"]
    results_path = Path("results") / cfg.study_name / f"sweep_{sweep_par}"
    results_path.mkdir(parents=True, exist_ok=True)

    # Fetch the trials data
    trials = study.trials_dataframe()
    trials = trials[trials["state"] == "COMPLETE"]

    # Split the sweep param into bins
    n_bins = USER["n_bins"]
    n_per_bin = USER["n_per_bin"]
    trials = trials.sort_values(by=f"params_{sweep_par}")
    values = trials[f"params_{sweep_par}"].values
    lo, hi = float(values.min()), float(values.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    # Filter to trials with param values within each bin
    for i in range(n_bins):
        bin_trials = trials[(trials[f"params_{sweep_par}"].values >= edges[i]) & (trials[f"params_{sweep_par}"].values < edges[i + 1])]
        # sample n_per_bin trials from each bin evenly spaced
        bin_values = bin_trials[f"params_{sweep_par}"].values
        samples_idx, samples_vals = sample_even_value_space(bin_values, k=n_per_bin, return_indices=True)
        bin_trials = bin_trials.iloc[samples_idx]
        plot_cases_overlay(
            study,
            bin_trials.index,
            sweep_par,
            title=f"Sweep over {sweep_par} - Bin {i}",
            output_path=results_path / f"sweep_{sweep_par}_bin_{i}.png",
        )

    print("🧹 Cleaning up port forwarding...")
    try:
        pf_process.terminate()
    except Exception:  # noqa: S110
        pass


def sample_even_value_space(values, k, include_first=True, include_last=True, return_indices=False):
    """
    Pick k samples spread evenly across the numeric range of `values`.
    Works even if values are highly concentrated in one region.
    """
    vals = np.asarray(values, dtype=float)
    lo, hi = float(vals.min()), float(vals.max())

    # Build k target points in VALUE space
    if include_first and include_last:  # include both ends
        targets = np.linspace(lo, hi, k)
    elif include_first and not include_last:  # include lo, exclude hi
        step = (hi - lo) / k
        targets = lo + step * np.arange(k)
    elif not include_first and include_last:  # exclude lo, include hi
        step = (hi - lo) / k
        targets = lo + step * (np.arange(1, k + 1))
    else:  # exclude both ends
        targets = np.linspace(lo, hi, k + 2)[1:-1]

    # Greedy nearest-neighbor pick with de-dup (remove chosen each time)
    avail_idx = np.arange(len(vals))
    avail_vals = vals.copy()
    chosen = []
    for t in targets:
        j = int(np.argmin(np.abs(avail_vals - t)))
        chosen.append(avail_idx[j])
        # remove chosen so we don't pick duplicates
        avail_vals = np.delete(avail_vals, j)
        avail_idx = np.delete(avail_idx, j)

    chosen = np.array(chosen, dtype=int)
    return (chosen, values[chosen]) if return_indices else values[chosen]


def plot_cases_overlay(study, trial_numbers, sweep_par, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    actual = study.trials[trial_numbers[0]].user_attrs.get("actual", None)
    y = actual["cases_by_month"]
    x = np.arange(len(y))
    ax.plot(x, y, alpha=0.6, label="actual", color="black", linewidth=3)
    for tn in trial_numbers:
        predicted = study.trials[tn].user_attrs.get("predicted", None)[0]
        y = predicted["cases_by_month"]
        ax.plot(x, y, alpha=0.6, label=f"trial {tn}, par={study.trials[tn].params[sweep_par]:.4g}", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Months since start")
    ax.set_ylabel("Cases")
    ax.legend(loc="best", fontsize=8)
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"💾 Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
