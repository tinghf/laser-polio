# laser_polio
This is a spatial polio transmission model built on the LASER framework.

## Installation 
The recommended approach is to use uv to setup your venv and install the package. 

Install uv & setup your venv:
```
pip install uv
uv venv --python 3.12
```

Download the repo, then install with uv:
```
uv pip install -e .
```

## Usage
You can run a simple demo with `scripts/demos/demo_nigeria.py`

## Repo organization
All core model code is located in the `src\laser_polio` subfolder. The majority of the model architecture resides in `seir_abm.py` which contains classes like `SEIR_ABM`, `DiseaseState_ABM`, `Transmission_ABM`, `VitalDynamics_ABM`, `RI_ABM`, and `SIA_ABM`. These classes contain methods for running the sim, tracking infection status, transmitting, managing births and deaths, and applying vaccination interventions. The `distributions.py` file contains a `Distributions` class which facilitates specification of distributions in the pars (see `scripts/demos/demo_nigeria.py`) so that in the model you can call the distribution from pars and the only input it requires is the number of draws. The  `utils.py` file has a variety of helper functions. The `seir_mpm.py` file contains an experimental meta-population model which will be developed at a future date.

The contents of the other folders is as follows:
- The data folder contains curated files needed for modeling along with the raw versions of the data and curation scripts.
- The docs folder contains information about model design and architecture.
- The scripts folder contains code for running and calibrating the model, profiling, and demos. 
- The tests folder contains code for testing model functionality and benchmarking. 

## Tests
Tests can be run `python -m pytest tests/`
