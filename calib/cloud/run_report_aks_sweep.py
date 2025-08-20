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
    # Which parameters to sweep. We'll produce one set of plots per param here.
    "sweep_params": ["seasonal_amplitude"],
    # Trials per bin (low / med / high)
    "n_per_bin": 5,
    # Near-best window:
    #   - abs_tol: per-parameter ±band around the best value (wins if provided)
    #   - rel_tol: fallback as a fraction of the best value for any param not listed in abs_tol
    "abs_tol": {
        "r0": 0.1,
        # "radiation_k_log10": 0.1,
        # add others as you wish...
    },
    "rel_tol": 0.1,  # ±1% of best for params not in abs_tol
    # Optional: custom bins per sweep param. If a param isn't listed here, we use terciles.
    # Each bin is (low_inclusive, high_exclusive, label). Use float("inf") or -float("inf") as needed.
    "custom_bins": {
        # Example:
        # "seasonal_amplitude": [(0.0, 0.1, "low"), (0.1, 0.3, "med"), (0.3, float("inf"), "high")]
    },
    # Output directory root; if None -> results/<study_name>/sweep_<param>/
    "output_root": None,
}
# ----------------- END USER CONFIGS -----------------


def port_forward():
    print("🔌 Setting up port forwarding to MySQL...")
    pf = subprocess.Popen(["kubectl", "port-forward", "mysql-0", "3307:3306"])
    time.sleep(3)  # wait for port-forward to take effect
    return pf


def main():
    pf_process = port_forward()
    print(f"📊 Loading study '{cfg.study_name}'...")
    study = optuna.load_study(study_name=cfg.study_name, storage=cfg.storage_url)
    study.storage_url = cfg.storage_url
    study.study_name = cfg.study_name
    results_path = Path("results") / cfg.study_name
    results_path.mkdir(parents=True, exist_ok=True)

    # Fetch the best pars
    best_params = study.best_trial.params

    # Fetch the trials data
    trials = study.trials_dataframe()

    # Filter trials so that each param is within 10% of the best value
    trials = trials[trials["state"] == "COMPLETE"]
    sweep_par = "seasonal_amplitude"
    for param in ["r0", "seasonal_peak_doy", "radiation_k_log10", "max_migr_frac"]:
        trials = trials[trials[f"params_{param}"] > best_params[param] * (1 - USER["rel_tol"])]
        trials = trials[trials[f"params_{param}"] < best_params[param] * (1 + USER["rel_tol"])]

    # Filter to low, med, high terciles of seasonal_amplitude
    trials = trials.sort_values(by=f"params_{sweep_par}")
    terciles = trials[f"params_{sweep_par}"].quantile([0.33, 0.66])
    low_trials = trials[trials[f"params_{sweep_par}"] < terciles[0.33]]
    med_trials = trials[(trials[f"params_{sweep_par}"] >= terciles[0.33]) & (trials[f"params_{sweep_par}"] < terciles[0.66])]
    high_trials = trials[trials[f"params_{sweep_par}"] >= terciles[0.66]]
    # for 5 of each of the low, med, high terciles, plot the cases by month
    plot_cases_overlay(
        study,
        [int(i) for i in low_trials.sample(n=5).index],
        sweep_par,
        title=f"Low {sweep_par}",
        output_path=results_path / f"low_{sweep_par}.png",
    )
    plot_cases_overlay(
        study,
        [int(i) for i in med_trials.sample(n=5).index],
        sweep_par,
        title=f"Med {sweep_par}",
        output_path=results_path / f"med_{sweep_par}.png",
    )
    plot_cases_overlay(
        study,
        [int(i) for i in high_trials.sample(n=5).index],
        sweep_par,
        title=f"High {sweep_par}",
        output_path=results_path / f"high_{sweep_par}.png",
    )

    print("🧹 Cleaning up port forwarding...")
    try:
        pf_process.terminate()
    except Exception:  # noqa: S110
        pass


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


# @dataclass
# class NearBestFilter:
#     """Absolute +/- tolerances per param win over relative tolerance."""

#     abs_tol: dict[str, float]
#     rel_tol: float = 0.01  # 1% fallback if abs not provided


# def trials_dataframe(study: optuna.study.Study):
#     """Return a tidy DataFrame of COMPLETE trials: number, value, and params."""
#     import pandas as pd

#     rows = []
#     for t in study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]):
#         row = {"number": t.number, "value": t.value}
#         for k, v in t.params.items():
#             row[k] = v
#         rows.append(row)
#     if not rows:
#         return pd.DataFrame(columns=["number", "value"])
#     return pd.DataFrame(rows)


# def near_best_mask(df, best_params: dict[str, float], sweep_param: str, filt: NearBestFilter):
#     import pandas as pd

#     mask = pd.Series(True, index=df.index)
#     for p, best in best_params.items():
#         if p == sweep_param or p not in df.columns:
#             continue
#         if p in filt.abs_tol:
#             tol = filt.abs_tol[p]
#         else:
#             base = abs(best) if best != 0 else (df[p].abs().median() or 1.0)
#             tol = filt.rel_tol * base
#         mask &= df[p].between(best - tol, best + tol)
#     return mask


# def sample_by_tertiles(df, param: str, n_per_bin: int, random_state: int | None = 42) -> dict[str, list[int]]:
#     import pandas as pd

#     s = df[param]
#     unique_vals = s.dropna().unique()
#     labels = ["low", "med", "high"]

#     if unique_vals.size < 3:
#         idx = df.sample(n=min(len(df), n_per_bin * 3), random_state=random_state).index
#         chunks = np.array_split(idx, 3)
#         return {lab: df.loc[list(chunk), "number"].tolist() for lab, chunk in zip(labels, chunks, strict=False)}

#     try:
#         bins = pd.qcut(s, q=3, labels=labels)
#     except Exception:
#         bins = pd.cut(s, bins=3, labels=labels, include_lowest=True)

#     out: dict[str, list[int]] = {}
#     for lab in labels:
#         cand = df[bins == lab]
#         if cand.empty:
#             out[lab] = []
#         else:
#             k = min(len(cand), n_per_bin)
#             out[lab] = cand.sample(n=k, random_state=random_state)["number"].tolist()
#     return out


# def select_trials_for_sweep(
#     study: optuna.study.Study,
#     sweep_param: str,
#     n_per_bin: int = 5,
#     filter_cfg: NearBestFilter | None = None,
#     custom_bins: list[tuple[float, float, str]] | None = None,  # [(low, high, label), ...]
#     random_state: int | None = 42,
# ) -> dict[str, list[int]]:
#     df = trials_dataframe(study)
#     if df.empty:
#         return {"low": [], "med": [], "high": []}

#     best = study.best_trial.params
#     if filter_cfg is None:
#         filter_cfg = NearBestFilter(abs_tol={})

#     mask = near_best_mask(df, best, sweep_param, filter_cfg)
#     base = df[mask].dropna(subset=[sweep_param]).copy()
#     if base.empty:
#         return {"low": [], "med": [], "high": []}

#     if custom_bins:
#         selections: dict[str, list[int]] = {}
#         for lo, hi, label in custom_bins:
#             cand = base[(base[sweep_param] >= lo) & (base[sweep_param] < hi)]
#             if cand.empty:
#                 selections[label] = []
#             else:
#                 k = min(len(cand), n_per_bin)
#                 selections[label] = cand.sample(n=k, random_state=random_state)["number"].tolist()
#         return selections

#     return sample_by_tertiles(base, sweep_param, n_per_bin, random_state)


# # ------------------- DATA ACCESS & PLOTTING -------------------


# def fetch_cases_by_month(study: optuna.study.Study, trial_number: int, actual: True):
#     """
#     Return a 1D array-like for cases_by_month for the given trial.
#     Adjust this if your outputs live elsewhere.
#     """
#     if actual:
#         t = study.trials[0]
#         targets = t.user_attrs.get("actual", None)
#     else:
#         t = study.trials[trial_number]
#         targets = t.user_attrs.get("predicted", None)[0]
#     if isinstance(targets, dict) and "cases_by_month" in targets:
#         return np.asarray(targets["cases_by_month"], dtype=float)

#     # Fallback: wire this to your own loader if needed, e.g.:
#     # from report import get_trial_targets
#     # targets = get_trial_targets(study, trial_number)
#     # if "cases_by_month" in targets: return np.asarray(targets["cases_by_month"], dtype=float)

#     return None


# def plot_cases_overlay(
#     study: optuna.study.Study,
#     trial_numbers: list[int],
#     sweep_param: str | None,
#     title: str | None,
#     output_path: Path | None,
# ):
#     fig, ax = plt.subplots(figsize=(10, 6))

#     plotted = 0
#     # Plot actual cases
#     y = fetch_cases_by_month(study, trial_numbers[0], actual=True)
#     if y is None or len(y) == 0:
#         print("⚠️  No actual cases found for provided trials; skipping plot.")
#         plt.close(fig)
#         return
#     x = np.arange(len(y))  # months since start (keeps it robust)
#     params = study.trials[trial_numbers[0]].params
#     label = "actual"
#     ax.plot(x, y, alpha=0.6, label=label)

#     for tn in trial_numbers:
#         # Plot predicted cases
#         y = fetch_cases_by_month(study, tn, actual=False)
#         if y is None or len(y) == 0:
#             continue

#         x = np.arange(len(y))  # months since start (keeps it robust)
#         params = study.trials[tn].params
#         if sweep_param and sweep_param in params:
#             label = f"trial {tn} ({sweep_param}={params[sweep_param]:.4g})"
#         else:
#             label = f"trial {tn}"

#         ax.plot(x, y, alpha=0.6, label=label)
#         plotted += 1

#     if plotted == 0:
#         print("⚠️  No cases_by_month found for provided trials; skipping plot.")
#         plt.close(fig)
#         return

#     ax.set_title(title or "Cases by Month")
#     ax.set_xlabel("Months since start")
#     ax.set_ylabel("Cases")
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc="best", fontsize=8)

#     if output_path:
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(output_path, dpi=200, bbox_inches="tight")
#         print(f"💾 Saved: {output_path}")

#     plt.close(fig)


# # ------------------- PORT FORWARD + DRIVER -------------------


# def port_forward():
#     print("🔌 Setting up port forwarding to MySQL...")
#     pf = subprocess.Popen(["kubectl", "port-forward", "mysql-0", "3307:3306"])
#     time.sleep(3)  # wait for port-forward to take effect
#     return pf


# def main():
#     # Resolve config: USER overrides cfg when provided
#     study_name = USER["study_name"] or cfg.study_name
#     storage_url = USER["storage_url"] or cfg.storage_url
#     model_config = USER["model_config"] or cfg.model_config

#     sweep_params = list(USER["sweep_params"])
#     n_per_bin = int(USER["n_per_bin"])
#     tol = NearBestFilter(abs_tol=dict(USER["abs_tol"]), rel_tol=float(USER["rel_tol"]))
#     custom_bins_by_param: dict[str, list[tuple[float, float, str]]] = USER.get("custom_bins", {}) or {}

#     pf_process = port_forward()
#     try:
#         print(f"📊 Loading study '{study_name}'...")
#         study = optuna.load_study(study_name=study_name, storage=storage_url)

#         with open(Path("calib/model_configs/") / model_config) as f:
#             model_cfg = yaml.safe_load(f)
#         # start_year available if you want calendar-based x; we stick to index for robustness
#         # start_year = int(model_cfg.get("start_year", 0))

#         # Output root
#         out_root_cfg = USER["output_root"]
#         base_out_root = Path(out_root_cfg) if out_root_cfg else Path("results") / study_name

#         for sweep_param in sweep_params:
#             print(f"\n🔎 Selecting trials near-best (except sweeping '{sweep_param}')...")
#             groups = select_trials_for_sweep(
#                 study,
#                 sweep_param=sweep_param,
#                 n_per_bin=n_per_bin,
#                 filter_cfg=tol,
#                 custom_bins=custom_bins_by_param.get(sweep_param),
#                 random_state=42,
#             )
#             print(f"Selected groups for {sweep_param}: {groups}")

#             out_root = base_out_root / f"sweep_{sweep_param}"
#             out_root.mkdir(parents=True, exist_ok=True)

#             for label, trial_ids in groups.items():
#                 if not trial_ids:
#                     print(f"⚠️  No trials in bin: {label}")
#                     continue
#                 title = f"{study_name} — {sweep_param} {label} (n={len(trial_ids)})"
#                 out_path = out_root / f"{sweep_param}_{label}_cases.png"
#                 print(f"📈 Plotting {sweep_param} {label} ({len(trial_ids)} trials) → {out_path}")
#                 plot_cases_overlay(study=study, trial_numbers=trial_ids, sweep_param=sweep_param, title=title, output_path=out_path)

#         print("\n✅ Done.")

#     finally:
#         print("🧹 Cleaning up port forwarding...")
#         try:
#             pf_process.terminate()
#         except Exception:
#             pass


if __name__ == "__main__":
    main()
