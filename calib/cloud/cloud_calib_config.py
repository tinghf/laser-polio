# calib_job_config.py
from pathlib import Path

import yaml

# ------------------- USER CONFIGS -------------------

# Goal: region calib after fixing the strain paralysis issue
job_name = "lpsk5"
study_name = "calib_nigeria_7y_2017_doy_bymonth_20250912"
model_config = "nigeria_7y_2017_regions_r0_radk_mmf_ssn_nozi_underwt.yaml"
calib_config = "doy_bymonth.yaml"

# # Goal: region calib after fixing the strain paralysis issue
# job_name = "lpsk6"
# study_name = "calib_nigeria_7y_2017_doy_bymonth_bytimeseries_20250912"
# model_config = "nigeria_7y_2017_regions_r0_radk_mmf_ssn_nozi_underwt.yaml"
# calib_config = "doy_bymonth_bytimeseries.yaml"

# # Goal: region calib after fixing the strain paralysis issue
# job_name = "lpsk7"
# study_name = "calib_nigeria_7y_2017_doy_amp_bymonth_20250912"
# model_config = "nigeria_7y_2017_regions_r0_radk_mmf_ssn_nozi_underwt.yaml"
# calib_config = "doy_amp_bymonth.yaml"

# # Goal: region calib after fixing the strain paralysis issue
# job_name = "lpsk8"
# study_name = "calib_nigeria_7y_2017_doy_amp_bymonth_bytimeseries_20250912"
# model_config = "nigeria_7y_2017_regions_r0_radk_mmf_ssn_nozi_underwt.yaml"
# calib_config = "doy_amp_bymonth_bytimeseries.yaml"

fit_function = "log_likelihood"
n_trials = 1  # Number of trials to run per pod
n_replicates = 1  # Number of replicates to run for each trial
parallelism = 50  # The number of pods (i.e., jobs) to run in parallel
completions = 5000  # The total number of pods (i.e., jobs) that need to successfully complete before the job is considered "done"

# ---------------------------------------------------

# Default settings
namespace = "default"
image = "idm-docker-staging.packages.idmod.org/laser/laser-polio:latest"

# Define the path to the YAML file with the storage URL from the docs
storage_path = Path("calib/cloud/local_storage.yaml")

# Try loading the storage URL from YAML, fallback to env var
storage_url = None
if storage_path.exists():
    storage = yaml.safe_load(storage_path.read_text())
    storage_url = storage.get("storage_url")
# Safety check
print(f"Storage URL: {storage_url}")
if storage_url is None:
    raise RuntimeError("Missing STORAGE_URL in local_storage.yaml")
