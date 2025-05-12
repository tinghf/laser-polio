import subprocess
import sys
import time
from pathlib import Path

import cloud_calib_config as cfg
import optuna

sys.path.append(str(Path(__file__).resolve().parent.parent))
from report import plot_likelihoods
from report import plot_runtimes
from report import plot_stuff
from report import plot_targets
from report import save_study_results


def port_forward():
    print("🔌 Setting up port forwarding to MySQL...")
    pf = subprocess.Popen(["kubectl", "port-forward", "mysql-0", "3306:3306"])
    time.sleep(3)  # wait for port-forward to take effect
    return pf


def main():
    pf_process = port_forward()
    try:
        print(f"📊 Loading study '{cfg.study_name}'...")
        study = optuna.load_study(study_name=cfg.study_name, storage=cfg.storage_url)
        study.storage_url = cfg.storage_url
        study.study_name = cfg.study_name

        results_path = Path("results") / cfg.study_name
        results_path.mkdir(parents=True, exist_ok=True)

        print("💾 Saving results...")
        save_study_results(study, output_dir=results_path)

        print("📈 Plotting results...")
        plot_stuff(cfg.study_name, study.storage_url, output_dir=results_path)

        # TODO: plot a timeseries with map below it
        # # Try loading shapefile for mapping
        # shp = None
        # try:
        #     # Geography
        #     regions = study.user_attrs["model_config"]["regions"]
        #     dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
        #     shp = gpd.read_file(filename="data/shp_africa_low_res.gpkg", layer="adm2")
        #     shp = shp[shp["dot_name"].isin(dot_names)]
        #     shp.set_index("dot_name", inplace=True)  # Sort the GeoDataFrame by the order of dot_names
        #     shp = shp.loc[dot_names].reset_index()
        # except Exception as e:
        #     print(f"⚠️ Warning: Could not load shapefile for mapping: {e}")
        print("📊 Plotting target comparisons...")
        plot_targets(study, output_dir=results_path)

        print("Plotting runtimes...")
        plot_runtimes(study, output_dir=results_path)

        print("Plotting likelihoods...")
        plot_likelihoods(study, output_dir=Path(results_path), use_log=True)

    finally:
        print("🧹 Cleaning up port forwarding...")
        pf_process.terminate()


if __name__ == "__main__":
    main()
