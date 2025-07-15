import numpy as np
import pandas as pd
import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

# Base pars
regions = ["NIGERIA"]
start_year = 2018
n_days = 2190
pop_scale = 1 / 1
init_region = "BIRINIWA"
init_prev = 0
seed_schedule = [
    {"date": "2018-01-02", "dot_name": "AFRO:NIGERIA:JIGAWA:HADEJIA", "prevalence": 100},
    {"date": "2018-01-02", "dot_name": "AFRO:NIGERIA:JIGAWA:GARKI", "prevalence": 100},
    {"date": "2020-07-01", "dot_name": "AFRO:NIGERIA:ZAMFARA:TALATA_MAFARA", "prevalence": 100},
    {"date": "2020-10-01", "dot_name": "AFRO:NIGERIA:NIGER:SULEJA", "prevalence": 100},
]
migration_method = "gravity"
vx_prob_ri = 0.0
missed_frac = 0.0
node_seeding_dispersion = 1.0
use_pim_scalars = False
save_plots = True
save_data = True
stop_if_no_cases = False
results_path = "calib/synth_calib/results"

# Calib pars
seed = 117
r0 = 8.12
seasonal_amplitude = 0.12
seasonal_peak_doy = 148
gravity_k_exponent = -12.78
gravity_a = 1.99
gravity_b = 1.99
gravity_c = 0.99
node_seeding_zero_inflation = 0.83
r0_scalar_wt_slope = 87.0
r0_scalar_wt_intercept = 0.06
r0_scalar_wt_center = 0.22
sia_re_center = 0.05
sia_re_scale = 0.2
init_immun_scalar = 1.0


######### END OF USER PARS ########
###################################


sim = lp.run_sim(
    regions=regions,
    start_year=start_year,
    n_days=n_days,
    pop_scale=pop_scale,
    init_region=init_region,
    init_prev=init_prev,
    seed_schedule=seed_schedule,
    migration_method=migration_method,
    vx_prob_ri=vx_prob_ri,
    missed_frac=missed_frac,
    node_seeding_dispersion=node_seeding_dispersion,
    use_pim_scalars=use_pim_scalars,
    save_plots=save_plots,
    save_data=save_data,
    stop_if_no_cases=stop_if_no_cases,
    results_path=results_path,
    seed=seed,
    r0=r0,
    seasonal_amplitude=seasonal_amplitude,
    seasonal_peak_doy=seasonal_peak_doy,
    gravity_k_exponent=gravity_k_exponent,
    gravity_a=gravity_a,
    gravity_b=gravity_b,
    gravity_c=gravity_c,
    node_seeding_zero_inflation=node_seeding_zero_inflation,
    r0_scalar_wt_slope=r0_scalar_wt_slope,
    r0_scalar_wt_intercept=r0_scalar_wt_intercept,
    r0_scalar_wt_center=r0_scalar_wt_center,
    sia_re_center=sia_re_center,
    sia_re_scale=sia_re_scale,
    init_immun_scalar=init_immun_scalar,
)


def make_synth_df_from_results(sim):
    """
    Build a DataFrame from simulation results (S, E, I, R, P, new_exposed).

    :param sim: The sim object containing a results object with numpy arrays for S, E, I, R, etc.
    :return: pandas.DataFrame with columns: timestep, date, node, S, E, I, R, P, new_exposed
    """
    timesteps = sim.nt
    datevec = sim.datevec
    nodes = len(sim.nodes)
    results = sim.results
    node_lookup = sim.pars.node_lookup

    # Prepare list of records
    records = []
    for t in range(timesteps):
        for n in range(nodes):
            dot_name = node_lookup[n]["dot_name"] if n in node_lookup else "Unknown"
            records.append(
                {
                    "timestep": t,
                    "date": datevec[t],
                    "node": n,
                    "dot_name": dot_name,
                    "S": results.S[t, n],
                    "E": results.E[t, n],
                    "I": results.I[t, n],
                    "R": results.R[t, n],
                    "P": results.paralyzed[t, n],
                    "new_exposed": results.new_exposed[t, n],
                }
            )

    # Create DataFrame
    df = pd.DataFrame.from_records(records)

    # Ensure date column is datetime (if not already)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Create a month_start column (1st of the month)
    df["month_start"] = df["date"].values.astype("datetime64[M]")  # fast way

    # Group by dot_name and month_start, then sum the P column
    grouped = df.groupby(["dot_name", "month_start"])["new_exposed"].sum().reset_index()

    # Divide by 2000 & convert to integers
    grouped["new_exposed"] /= 2000

    cases = []
    for i in range(len(grouped["new_exposed"])):
        expected = grouped["new_exposed"][i]
        paralytic_cases = np.random.poisson(expected)
        cases.append(paralytic_cases)
    grouped["cases"] = cases

    return grouped


# Extract & summarize the results
df = make_synth_df_from_results(sim)
df["month_start"] = pd.to_datetime(df["month_start"]).astype("datetime64[ns]")
print(df.dtypes)
print(df.head())

# Save as h5
synth_filename = f"{results_path}/synth_data_nigeria_r0{r0}.h5"
df.to_hdf(synth_filename, key="epi", mode="w", format="table")

# Load the saved data to verify
loaded_df = pd.read_hdf(synth_filename, key="epi")
print(loaded_df.dtypes)
print(loaded_df.head())

sc.printcyan("Done.")
