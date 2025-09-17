import numpy as np
from laser_core.propertyset import PropertySet

import laser_polio as lp


# Fixture to set up the simulation environment
def setup_sim(dur=30, init_pop=None, vx_prob_ri=0.5, vx_prob_ipv=0.75, cbr=None, r0=14, new_pars=None, seed=123):
    if init_pop is None:
        init_pop = np.array([50000, 50000])
    if cbr is None:
        cbr = np.array([30, 25])
    strain_r0_scalars_zero = {0: 1.0, 1: 0.0, 2: 0.0}

    pars = PropertySet(
        {
            "dur": dur,
            "init_pop": init_pop,
            "cbr": cbr,  # Birth rate per 1000/year
            "init_immun": 0.0,  # initially immune
            "init_prev": 0.0,  # initially infected from any age
            "r0": r0,  # Basic reproduction number
            "dur_exp": lp.constant(value=2),  # Duration of the exposed state
            "dur_inf": lp.constant(value=1),  # Duration of the infectious state
            "vx_prob_ri": vx_prob_ri,  # Routine immunization probability
            "vx_prob_ipv": vx_prob_ipv,  # IPV probability
            "stop_if_no_cases": False,  # Stop simulation if no cases are present,
            "seed": seed,
            "strain_r0_scalars": strain_r0_scalars_zero,
        }
    )
    pars += new_pars if new_pars is not None else {}
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.VitalDynamics_ABM, lp.DiseaseState_ABM, lp.RI_ABM, lp.SIA_ABM, lp.Transmission_ABM]
    return sim


# --- RI_ABM Tests ---


def test_ri_initialization():
    """Ensure that RI_ABM initializes correctly."""
    sim = setup_sim()
    sim.run()
    assert hasattr(sim.people, "ri_timer")
    assert hasattr(sim.results, "ri_vaccinated")


def test_ri_manually_seeded():
    """Ensure that routine immunization occurs when manually seeded."""
    n_vx = 1000
    dur = 28
    sim = setup_sim(dur=dur, vx_prob_ri=1.0, vx_prob_ipv=1.0)
    sim.people.ri_timer[:n_vx] = np.random.randint(0, dur, n_vx)  # Set timers to trigger vaccination
    sim.run()
    assert sim.results.ri_vaccinated.sum() >= n_vx, "The number of RI OPV vaccinations was lower than the number manually seeded."
    assert sim.results.ipv_vaccinated.sum() >= n_vx, "The number of RI IPV vaccinations was lower than the number manually seeded."


def test_ri_on_births():
    dur = 365
    cbr = np.array([300, 250])
    sim = setup_sim(dur=dur, cbr=cbr, vx_prob_ri=1.0)
    sim.run()
    assert np.sum(sim.results.ri_vaccinated) > 0, "No routine OPV immunizations occurred on births."
    assert np.sum(sim.results.ipv_vaccinated) > 0, "No routine OPV immunizations occurred on births."


def test_ri_zero():
    dur = 365

    # Test RI when there are no births (there can still be some RI in existing population)
    cbr = np.array([0, 0])
    vx_prob_ri = 1.0
    vx_prob_ipv = 1.0
    sim_no_births = setup_sim(dur=dur, cbr=cbr, vx_prob_ri=vx_prob_ri, vx_prob_ipv=vx_prob_ipv)
    sim_no_births.run()
    assert np.sum(sim_no_births.results.ri_vaccinated[(98 + 14) :]) == 0, (
        "No RI vaccinations should've occurred after initial cohort aged out of RI (oldest 98 days + time_step)."
    )
    assert np.sum(sim_no_births.results.ipv_vaccinated[(98 + 14) :]) == 0, (
        "No RI IPV vaccinations should've occurred after initial cohort aged out of RI (oldest 98 days + time_step)."
    )

    # Zero routine immunization probability
    cbr = np.array([300, 250])
    sim_zero_ri_prob = setup_sim(dur=dur, cbr=cbr, vx_prob_ri=0.0, vx_prob_ipv=0.0)
    sim_zero_ri_prob.run()
    assert np.sum(sim_zero_ri_prob.results.ri_vaccinated) == 0, "RI OPV vaccinations occurred, but there should've been zero."
    assert np.sum(sim_zero_ri_prob.results.ipv_vaccinated) == 0, "RI IPV vaccinations occurred, but there should've been zero."


def test_ri_vx_prob(n_reps=10):
    """Ensure that the RI vaccination probability is respected in the absence of births."""
    init_pop = np.array([50, 50])
    total_agents = np.sum(init_pop)
    dur = 28
    vx_prob_ri = 0.65
    vx_prob_ipv = 0.85
    vx_counts = []
    vx_ipv_counts = []
    e_counts = []
    for _ in range(n_reps):
        sim = setup_sim(init_pop=init_pop, dur=dur, vx_prob_ri=vx_prob_ri, vx_prob_ipv=vx_prob_ipv, cbr=np.array([0, 0]), seed=_)
        sim.people.ri_timer[:total_agents] = np.random.randint(0, dur, total_agents)
        sim.run()
        vx_counts.append(np.sum(sim.results.ri_vaccinated))
        vx_ipv_counts.append(np.sum(sim.results.ipv_vaccinated))
        e_counts.append(np.sum(sim.results.new_exposed))

    vx_counts = np.array(vx_counts)
    vx_ipv_counts = np.array(vx_ipv_counts)
    e_counts = np.array(e_counts)

    # Binomial confidence interval around expected value
    expected_mean = total_agents * vx_prob_ri
    expected_std = np.sqrt(total_agents * vx_prob_ri * (1 - vx_prob_ri))
    mean_vx = vx_counts.mean()
    mean_vx_ipv = vx_ipv_counts.mean()
    expected_mean_ipv = total_agents * vx_prob_ipv
    expected_std_ipv = np.sqrt(total_agents * vx_prob_ipv * (1 - vx_prob_ipv))

    # Allow for 2 standard deviations (approx 95% CI)
    margin = 2 * expected_std
    assert abs(mean_vx - expected_mean) < margin, (
        f"Mean vaccinated ({mean_vx:.1f}) outside expected CI around {expected_mean:.1f} ± {margin:.1f}"
    )
    margin_ipv = 2 * expected_std_ipv
    assert abs(mean_vx_ipv - expected_mean_ipv) < margin_ipv, (
        f"Mean vaccinated ({mean_vx_ipv:.1f}) outside expected CI around {expected_mean_ipv:.1f} ± {margin_ipv:.1f}"
    )

    # Exposed should exactly match vaccinated in each replicate (vx efficacy = 100%)
    assert np.all(np.abs(vx_counts - e_counts) <= 1), "Each vaccinated individual should be exposed (within ±1) if efficacy is 100%."
    # IPV does not affect recovery


def test_ri_no_effect_on_non_susceptibles():
    """Ensure RI does not affect infected or recovered individuals."""
    init_pop = np.array([10, 10])
    r0 = 0
    vx_prob_ri = 1.0
    sim = setup_sim(init_pop=init_pop, r0=r0, vx_prob_ri=vx_prob_ri)
    sim.people.ri_timer[:20] = 0
    sim.people.disease_state[:5] = 1  # Exposed
    sim.people.disease_state[5:10] = 2  # Infected
    sim.people.disease_state[10:15] = 3  # Recovered
    sim.run()
    assert np.sum(sim.results.ri_vaccinated) == 20, "All individuals should've been vaccinated."
    assert np.sum(sim.results.new_exposed) == 5, "Only the 5 susceptible individuals should've become exposed via RI."


# --- SIA_ABM Tests ---


def test_sia_schedule():
    """Ensure that SIA occurs on the correct date."""
    sia_pars = {
        "sia_schedule": [{"date": "2019-01-10", "nodes": [0], "age_range": (0, 5 * 365), "vaccinetype": "nOPV2"}],
        "vx_prob_sia": [0.6, 0.8],  # SIA effectiveness per node
    }
    sim = setup_sim(vx_prob_ri=0, new_pars=sia_pars)
    sim.run()

    assert hasattr(sim.results, "sia_vaccinated"), "SIA component is missing results array"
    assert hasattr(sim.results, "sia_protected"), "SIA component is missing results array"

    n_vx_day10 = np.sum(sim.results.sia_vaccinated[9, :])
    n_vx_rest = np.sum(sim.results.sia_vaccinated) - n_vx_day10

    assert np.any(n_vx_day10 > 0), "SIA did not execute on the scheduled date."
    assert np.all(n_vx_rest == 0), "SIA should not run on any other dates."
    assert sim.results.sia_vaccinated[9, 1] == 0, "SIA should not have affected individuals in node 1."

    exp_protected = n_vx_day10 * 0.56  # 56% of vaccinated individuals should be protected since using nOPV2
    assert np.isclose(np.sum(sim.results.sia_protected), exp_protected, atol=100), (
        "SIA should protect 56% of vaccinated individuals since using nOPV2."
    )

    # Check if the number vaccinated is close to the expected value
    alive_in_node = (sim.people.node_id == 0) & (sim.people.disease_state >= 0)
    age = sim.t - sim.people.date_of_birth[alive_in_node]  # Age of individuals
    age_eligible = np.sum(age < 5 * 365)  # Filter to <5 years old
    exp_vx = age_eligible * sim.pars.vx_prob_sia[0]  # Expected number of vaccinated individuals
    assert np.isclose(n_vx_day10, exp_vx, atol=500), "Number of vaccinated individuals does not match expected value."

    # Check exposed count
    n_exposed = np.sum(sim.results.new_exposed)
    assert n_exposed > 0, "No individuals exposed after SIA."
    assert n_exposed == np.sum(sim.results.sia_protected), "Number of exposed individuals does not match expected value."

    # Check ages of exposeds
    exposeds = sim.people.disease_state == 1
    ages = sim.t - sim.people.date_of_birth[exposeds]
    assert np.all(ages <= (5 * 365 + 22)), "Exposed individuals should be <5 years old."


def test_chronically_missed():
    """Test that chronically missed individuals are not vaccinated."""

    sia_pars = {
        "sia_schedule": [{"date": "2019-01-10", "nodes": [0, 1], "age_range": (0, 5 * 365), "vaccinetype": "perfect"}],
        "vx_prob_sia": [1.0, 1.0],  # SIA effectiveness per node
        "missed_frac": 0.2,
    }
    sim = setup_sim(vx_prob_ri=0, new_pars=sia_pars)
    sim.run()

    # Check missed group
    missed = sim.people.chronically_missed[: sim.people.count]
    n_missed = np.sum(missed)
    assert np.isclose(n_missed, sim.pars.init_pop.sum() * 0.2, atol=100), "No missed individuals were created."

    # Assert none of the missed were exposed
    eir = np.isin(sim.people.disease_state[: sim.people.count], [1, 2, 3])
    assert not np.any(eir & missed), "Some chronically missed individuals were vaccinated!"

    # Assert that the number of vaccinated individuals is equal to the expected number of vaccinated individuals
    age = sim.t - sim.people.date_of_birth[: sim.people.count]  # Age of individuals
    age_eligible = np.sum(age < 5 * 365)  # Filter to <5 years old
    exp_vx = age_eligible * (1 - sim.pars.missed_frac)  # Expected number of vaccinated individuals
    n_vx = np.sum(eir)
    assert np.isclose(n_vx, exp_vx, atol=500), "Number of vaccinated individuals does not match expected value."


if __name__ == "__main__":
    test_ri_initialization()
    test_ri_manually_seeded()
    test_ri_on_births()
    test_ri_zero()
    test_ri_vx_prob()
    test_ri_no_effect_on_non_susceptibles()
    test_sia_schedule()
    test_chronically_missed()

    print("All initialization tests passed.")
