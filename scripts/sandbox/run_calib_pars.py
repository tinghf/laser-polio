from pathlib import Path

import sciris as sc
import yaml

import laser_polio as lp

# Record start time
start_time = sc.tic()

"""
This script runs a simulation using a similar configuration to the calibration approach in objective.py. 
It takes as input a model_config.yaml file & a dictionary of parameters. The latter is 
typically printed during calibration runs.
"""

###################################
######### USER PARAMETERS #########

# Load model configuration from YAML file
model_config_path = Path("calib/model_configs/config_nigeria_6y_2018_underwt_gravity_zinb_ipv_nwecnga_3periods.yaml")

# Calib pars
suggested_params = {
    "r0": 11.376375828469293,
    "seasonal_amplitude": 0.06618410394577416,
    "seasonal_peak_doy": 229,
    "gravity_k_exponent": -8.896082688911651,
    "gravity_a": 1.048239808810217,
    "gravity_b": 0.7554399511356332,
    "gravity_c": 0.459539396713007,
    "node_seeding_zero_inflation": 0.6910803759617196,
    "r0_scalar_wt_slope": 65.53324288590474,
    "r0_scalar_wt_intercept": 0.10098281242018137,
    "r0_scalar_wt_center": 0.40362946074304457,
    "sia_re_center": 0.04566830306726448,
    "sia_re_scale": 0.03337002867425776,
    "init_immun_scalar": 1.0961955322491634,
}

# Runtime parameters
verbose = 1
results_path = "results/nigeria_calib_20250812"

######### END OF USER PARS ########
###################################

# Merge parameters into config
with open(model_config_path) as f:
    model_config = yaml.safe_load(f)
config = {**model_config, **suggested_params}
config["results_path"] = results_path
config["save_plots"] = True
config["plot_pars"] = True

# Run simulation
sc.printcyan("🚀 Running simulation...")
sim = lp.run_sim(config, verbose=verbose)

# Print elapsed time
elapsed_time = sc.toc(start_time, output=True)
sc.printcyan("✅ Simulation completed successfully!")
sc.printcyan(f"🕒 Time elapsed: {elapsed_time}")
sc.printcyan(f"💾 Results saved to: {results_path}")
