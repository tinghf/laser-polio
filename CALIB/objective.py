import json
import subprocess
import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

laser_script = Path("laser.py").resolve(strict=True)

def evaluate_something( filename ):
    """
    Load simulation results from a CSV file and calculate interesting stats based on the I column.
    """
    df = pd.read_csv(filename)
    
    # Calculate statistics for the I column
    max_I = df['I'].max()
    min_I = df['I'].min()
    mean_I = df['I'].mean()
    median_I = df['I'].median()
    total_infected = df['I'].sum()
    peak_infection_time = df.loc[df['I'].idxmax(), 'Time']
    
    print(f"Max I: {max_I}")
    print(f"Min I: {min_I}")
    print(f"Mean I: {mean_I}")
    print(f"Median I: {median_I}")
    print(f"Total Infected: {total_infected}")
    print(f"Peak Infection Time: {peak_infection_time}")

    score = 0.01 - mean_I
    return score

# Paths to input/output files
PARAMS_FILE = "params.json"
RESULTS_FILE = "simulation_results.csv"


def objective(trial):
    """Optuna objective function that runs laser.py with trial parameters and evaluates results."""

    Path(RESULTS_FILE).unlink(missing_ok=True)

    # Suggest values for calibration
    #migration_rate = trial.suggest_float("migration_rate", 0.0001, 0.01)
    r_nought = trial.suggest_float("r0", 4, 20)
    # migration_rate = trial.suggest_float("migration_rate", 0.004, 0.004)
    # transmission_rate = trial.suggest_float("transmission_rate", 0.145, 0.145)

    # Set up parameters
    params = {
        "r0": r_nought
    }

    # Write parameters to JSON file
    with Path(PARAMS_FILE).open("w") as f:
        json.dump(params, f, indent=4)

    def get_docker_runstring():
        # cmd = f"docker run --rm -v .:/app/shared docker.io/jbloedow/my-laser-app:latest"
        cmd = "docker run --rm -v .:/app/shared my-laser-app:latest"
        return cmd.split()

    def get_native_runstring():
        return [sys.executable, str(laser_script)]

    print(f"Will be looking for {RESULTS_FILE}")
    # Run laser.py as a subprocess
    try:
        # Run the model 4 times and collect results
        scores = []
        for _ in range(4):
            # subprocess.run(get_docker_runstring(), check=True)
            # subprocess.run(["python3", "laser.py"], check=True)
            subprocess.run(get_native_runstring(), check=True)

            print( f"Waiting for output file {RESULTS_FILE}" )
            # Wait until RESULTS_FILE is written
            while not Path(RESULTS_FILE).exists():
                print( os.listdir(".") )
                time.sleep(0.1)

            score = evaluate_something(RESULTS_FILE)  # Evaluate results
            scores.append(score)

        # Return the average score
        return np.mean(scores)

    except subprocess.CalledProcessError as e:
        print(f"Error running laser.py: {e}")
        return float("inf")  # Penalize failed runs
