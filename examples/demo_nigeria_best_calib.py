import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["NIGERIA"]
start_year = 2018
n_days = 2190  # 6 years
pop_scale = 1 / 1
init_region = "BIRINIWA"
init_prev = 0
migration_method = "radiation"
max_migr_frac = 1.0
results_path = "results/demo_nigeria_best_calib"

# From best calib
init_prev = 0
seed_schedule = [
    {"date": "2018-02-06", "dot_name": "AFRO:NIGERIA:JIGAWA:BIRINIWA", "prevalence": 200},
    {"date": "2020-11-24", "dot_name": "AFRO:NIGERIA:ZAMFARA:SHINKAFI", "prevalence": 200},
]
r0 = 18.42868241650029
radiation_k = 0.0956175096168661
seasonal_factor = 0.19102804223372347
seasonal_phase = 193

######### END OF USER PARS ########
###################################


sim = lp.run_sim(
    regions=regions,
    start_year=start_year,
    n_days=n_days,
    pop_scale=pop_scale,
    init_region=init_region,
    init_prev=init_prev,
    r0=r0,
    migration_method=migration_method,
    radiation_k=radiation_k,
    max_migr_frac=max_migr_frac,
    results_path=results_path,
    save_plots=True,
    save_data=True,
    verbose=1,
    seed=1,
    save_pop=True,
    seed_schedule=seed_schedule,
    seasonal_factor=seasonal_factor,
    seasonal_phase=seasonal_phase,
)

sc.printcyan("Done.")
