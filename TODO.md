# Priorities
- Track down what happened to the priority ordering
    # Set the order in which components should be run during step()
    PRIORITY_ORDER = [
        "VitalDynamics_ABM",
        "DiseaseState_ABM",
        "RI_ABM",
        "SIA_ABM",
        "Transmission_ABM"
    ]
- Rename variables to distinguish between exposure and infection
- Testing
    - Transmission_abm
    - RI_abm
    - SIA_abm
- Add default pars, then replace if user specifies pars
- Set a random number seed
- Use KM's gravity model scaling approach
- Check the RI & SIA figures - they're plotting strange results
- Update the birth and death plot to summarize by country
- Calibration
- Add step size to components (e.g., like vital dynamics)
- Save results & specify frequency
- Reactive SIAs

# Refinement
- Add EMOD style seasonality
- Fork polio-immunity-mapping repo
- Double check that I'm using the ri_eff and sia_prob values correctly - do I need to multiply sia_prob by vx_eff?
- Get total pop data, not just <5
- Investigate extra dot_names in the pop dataset
- Look into age-specific death rates
- Setup EULAs - currently are only age based, needs to be immunity based
- Import/seed infections throughout the sim after OPV use?
- Write pars to disk
- Add partial susceptibility & paralysis protection
- Add distributions for duration of each state
- Add in default pars and allow user pars to overwrite them
- Add CBR by country-year
- Add age pyramid by country
- Calculate distance between gps coordinates
