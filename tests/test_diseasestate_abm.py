import numpy as np
import pytest
import laser_polio as lp
from laser_core.propertyset import PropertySet

def setup_sim():
    """Initialize a test simulation with DiseaseState_ABM component."""
    pars = PropertySet(dict(
        start_date      = lp.date('2020-01-01'),
        timesteps       = 30,
        n_ppl           = np.array([1000, 500]),  # Two nodes with populations
        cbr             = np.array([30, 25]),  # Birth rate per 1000/year
        beta_spatial    = np.array([0.5, 2.0]),  # Spatial transmission scalar (multiplied by global rate)
        age_pyramid_path= 'data/Nigeria_age_pyramid_2024.csv',  # From https://www.populationpyramid.net/nigeria/2024/
        init_immun      = 0.2,  # 20% initially immune
        init_prev       = 0.05,  # 5% initially infected
        dur_exp         = lp.normal(mean=3, std=1),  # Duration of the exposed state
        dur_inf         = lp.gamma(shape=4.51, scale=5.32),  # Duration of the infectious state
        p_paralysis     = 1 / 2000,  # 1% paralysis probability
    ))
    sim = lp.SEIR_ABM(pars)
    sim.add_component(lp.DiseaseState_ABM(sim))
    return sim

# Test Initialization
def test_disease_state_initialization():
    """Ensure disease state properties are correctly initialized."""
    sim = setup_sim()
    assert hasattr(sim.people, "disease_state")
    assert hasattr(sim.people, "paralyzed")
    assert hasattr(sim.people, "exposure_timer")
    assert hasattr(sim.people, "infection_timer")

# Test Initial Population Counts
def test_initial_population_counts():
    """Ensure the correct fraction of individuals start in each state."""
    sim = setup_sim()
    total_pop = np.sum(sim.pars.n_ppl)
    exp_rec = int(sim.pars.init_immun * total_pop)  # Note that some of the recovered can become infected during intitialization
    exp_inf = int(sim.pars.init_prev * total_pop)  
    exp_exp = 0
    assert (total_pop - exp_rec - exp_inf) <= np.sum(sim.people.disease_state == 0) <= (total_pop - exp_rec), 'Susceptible counts are incorrect'
    assert np.sum(sim.people.disease_state == 1) == exp_exp, 'Exposed counts are incorrect'
    assert np.sum(sim.people.disease_state == 2) == exp_inf, 'Infected counts are incorrect'
    assert (exp_rec - exp_inf) <= np.sum(sim.people.disease_state == 3) <= exp_rec, 'Recovered counts are incorrect'

# Test Disease Progression
def test_exposed_to_infected():
    # Setup sim with 0 infections
    pars = PropertySet(dict(
        start_date      = lp.date('2020-01-01'),
        timesteps       = 1,
        n_ppl           = np.array([1000, 500]),  # Two nodes with populations
        cbr             = np.array([30, 25]),  # Birth rate per 1000/year
        beta_spatial    = np.array([0.5, 2.0]),  # Spatial transmission scalar (multiplied by global rate)
        age_pyramid_path= 'data/Nigeria_age_pyramid_2024.csv',  # From https://www.populationpyramid.net/nigeria/2024/
        init_immun      = 0.0,  # 20% initially immune
        init_prev       = 0.0,  # 5% initially infected
        dur_exp         = lp.constant(value=1),  # Duration of the exposed state
        dur_inf         = lp.constant(value=1),  # Duration of the infectious state
        p_paralysis     = 1 / 2000,  # 1% paralysis probability
    ))
    sim = lp.SEIR_ABM(pars)
    sim.add_component(lp.DiseaseState_ABM(sim))
    sim.DiseaseState_ABM.step()
    sim.people.exposure_timer[:] = 1  # Fast-track exposure
    sim.run()
    assert np.sum(sim.people.disease_state == 1) == 0  # No one should be Exposed
    assert np.sum(sim.people.disease_state == 2) > 0  # Some should be Infected


def test_infected_to_recovered():
    """Ensure Infected individuals transition to Recovered after infection_timer expires."""
    sim = setup_sim()
    sim.people.disease_state[:] = 2  # Set all to Infected
    sim.people.infection_timer[:] = 1  # Fast-track infection duration
    sim.run()
    assert np.sum(sim.people.disease_state == 2) == 0  # No one should be Infected
    assert np.sum(sim.people.disease_state == 3) > 0  # Some should be Recovered

# Test Paralysis Probability
def test_paralysis_probability():
    """Ensure the correct fraction of infected individuals become paralyzed."""
    sim = setup_sim()
    sim.people.disease_state[:] = 2  # Set all to Infected
    sim.people.infection_timer[:] = 1
    sim.run()
    expected_paralyzed = int(0.01 * np.sum(sim.people.disease_state == 2))
    assert np.sum(sim.people.paralyzed == 1) == expected_paralyzed

# Test Edge Cases
def test_no_transmission_in_immune_population():
    """Ensure that if all individuals are immune, no infections occur."""
    pars = PropertySet(dict(
        start_date  = lp.date('2020-01-01'),
        timesteps   = 30,
        n_ppl       = np.array([1000, 500]),
        init_immun  = [1.0],  # 100% immune
        init_prev   = [0.0],  # No initial infections
    ))
    sim = lp.SEIR_ABM(pars)
    sim.add_component(lp.DiseaseState_ABM(sim))
    sim.run()
    assert np.sum(sim.people.disease_state == 2) == 0  # No one should be Infected

if __name__ == '__main__':
    # test_disease_state_initialization()
    # test_initial_population_counts()
    test_exposed_to_infected()
    # test_infected_to_recovered()
    # test_paralysis_probability()
    # test_no_transmission_in_immune_population()