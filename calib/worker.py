import importlib
from functools import partial
from pathlib import Path

import calib_db
import optuna
import sciris as sc
import yaml
from objective import objective

import laser_polio as lp


def load_function(module_path: str, function_name: str):
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def run_worker_main(
    study_name=None,
    n_trials=None,
    calib_config=None,
    model_config=None,
    fit_function=None,
    results_path=None,
    actual_data_file=None,
    n_replicates=None,
    dry_run=False,
):
    """Run Optuna trials to calibrate the model via CLI or programmatically."""

    # 👇 Provide defaults for programmatic use
    n_trials = n_trials or 1
    calib_config = calib_config or lp.root / "calib/calib_configs/calib_pars_r0.yaml"
    model_config = model_config or lp.root / "calib/model_configs/config_zamfara.yaml"
    fit_function = fit_function or "mse"  # options are "log_likelihood" or "mse"
    results_path = results_path or lp.root / "calib/results" / study_name
    actual_data_file = actual_data_file or lp.root / "examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv"
    n_replicates = n_replicates or 1

    # We want to show users on console what values we ended up going with based on command line args and defaults.
    print(f"study_name: {study_name}")
    print(f"n_trials: {n_trials}")
    print(f"calib_config: {calib_config}")
    print(f"model_config: {model_config}")
    print(f"fit_function: {fit_function}")
    print(f"results_path: {results_path}")
    print(f"actual_data_file: {actual_data_file}")
    print(f"n_replicates: {n_replicates}")

    if dry_run:
        return

    sc.printcyan(f"[INFO] Running study: {study_name} with {n_trials} trials")
    storage_url = calib_db.get_storage()

    # sampler = optuna.samplers.RandomSampler(seed=42)  # seed is optional for reproducibility
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)  # , sampler=sampler)
    except Exception:
        print(f"[INFO] Creating new study: '{study_name}'")
        study = optuna.create_study(study_name=study_name, storage=storage_url)

    # Load the calibration config
    with open(calib_config) as f:
        calib_config_dict = yaml.safe_load(f)
    study.set_user_attr("parameter_spec", calib_config_dict.get("parameters", {}))
    with open(model_config) as f:
        model_config_dict = yaml.safe_load(f)
    study.set_user_attr("model_config", model_config_dict)
    for k, v in calib_config_dict.get("metadata", {}).items():
        study.set_user_attr(k, v)
    metadata = calib_config_dict.get("metadata", {})
    scoring_fn = load_function(
        module_path="scoring",
        function_name=metadata.get("scoring_fn", "compute_fit"),
    )
    target_fn = load_function(
        module_path="targets",
        function_name=metadata.get("target_fn", "calc_calib_targets"),
    )

    # Load the actual case data (moved from run_sim into calib code)
    regions = model_config_dict.pop("regions", ["ZAMFARA"])
    admin_level = model_config_dict.get("admin_level", None)
    dot_names = lp.find_matching_dot_names(
        regions,
        lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv",
        verbose=0,
        admin_level=admin_level,
    )
    node_lookup = lp.get_node_lookup(lp.root / "data/node_lookup.json", dot_names)
    actual_data = model_config_dict.pop("actual_data", lp.root / "data/epi_africa_20250421.h5")
    start_year = model_config_dict["start_year"]
    n_days = model_config_dict["n_days"]
    epi = lp.get_epi_data(actual_data, dot_names, node_lookup, start_year, n_days)
    epi.rename(columns={"cases": "P"}, inplace=True)
    # Apply temporal and regional groupings using the same config as simulation data
    summary_config = model_config_dict.get("summary_config", None)
    if summary_config is not None:
        # Ensure date column is datetime
        if "date" in epi.columns and not epi["date"].dtype.name.startswith("datetime"):
            epi["date"] = epi["date"].astype("datetime64[ns]")
        # Apply temporal groupings
        if "time_periods" in summary_config:
            epi = lp.add_temporal_groupings(epi, summary_config["time_periods"])
        # Apply regional groupings
        if "region_groupings" in summary_config and "grouping_level" in summary_config:
            epi = lp.add_regional_groupings(epi, summary_config["region_groupings"], grouping_level=summary_config["grouping_level"])
        elif "region_groupings" in summary_config:
            epi = lp.add_regional_groupings(epi, summary_config["region_groupings"], grouping_level="adm0")
        elif "grouping_level" in summary_config:
            epi = lp.add_regional_groupings(epi, grouping_level=summary_config["grouping_level"])
        else:
            epi = lp.add_regional_groupings(epi, grouping_level="dot_name")  # Default to dot_name

    # Save actual data with groupings applied
    epi.to_csv(results_path / "actual_data.csv", index=False)

    # Run the study
    wrapped_objective = partial(
        objective,
        calib_config=calib_config_dict,
        model_config_path=Path(model_config),
        fit_function=fit_function,
        results_path=Path(results_path),
        actual_data_file=Path(actual_data_file),
        n_replicates=n_replicates,
        scoring_fn=scoring_fn,
        target_fn=target_fn,
    )

    study.optimize(wrapped_objective, n_trials=n_trials)
