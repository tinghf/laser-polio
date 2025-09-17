import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["AFRO", "EMRO"]
start_year = 2018
n_days = 365 * 3
pop_scale = 1 / 1
init_region = "BIRINIWA"
init_prev = 200
r0 = 14
migration_method = "radiation"
radiation_k_log10 = -0.3
max_migr_frac = 1.0
results_path = "results/demo_africa"
seed_schedule = [
    {"date": "2018-02-06", "dot_name": "AFRO:NIGERIA:JIGAWA:BIRINIWA", "prevalence": 200},  # day 1
    {"date": "2020-11-24", "dot_name": "AFRO:NIGERIA:ZAMFARA:SHINKAFI", "prevalence": 200},  # day 2
]

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
    r0=r0,
    migration_method=migration_method,
    radiation_k_log10=radiation_k_log10,
    max_migr_frac=max_migr_frac,
    results_path=results_path,
    save_plots=True,
    save_data=True,
    verbose=1,
    seed=1,
    save_init_pop=True,
    plot_pars=True,
    use_pim_scalars=True,
)

sc.printcyan("Done.")
