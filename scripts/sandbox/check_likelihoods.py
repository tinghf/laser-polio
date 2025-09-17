import json
import os
import sys
from pathlib import Path

import pandas as pd

import laser_polio as lp

if os.getenv("POLIO_ROOT"):
    lp.root = Path(os.getenv("POLIO_ROOT"))
sys.path.append(str(lp.root))
from calib.scoring import compute_log_likelihood_fit

# Load results
results_path = Path("results/calib_nigeria_6y_recovered_fix_20250506")
df = pd.read_csv(results_path / "trials.csv")
print(df.head())

# Find the best trial with the lowest value of the fit function
best = df.loc[df["value"].idxmin()]
print("Best trial:")
print(best)

# Check total calculation
actual = best.user_attrs_actual.replace("'", '"')
predicted = best.user_attrs_predicted.replace("'", '"')
actual = json.loads(actual)
predicted = json.loads(predicted)
predicted = predicted[0]  # Extract the first replicate if multiple are present
ll = compute_log_likelihood_fit(actual, predicted, weights=None)

# Check piecewise calculations
ll_normalized = {}
ll_not_normalized = {}
for key in actual:
    if key not in predicted:
        print(f"[WARN] Key missing in predicted: {key}")
        continue

    # Create sub-dictionaries with a single key
    actual_subset = {key: actual[key]}
    predicted_subset = {key: predicted[key]}

    # Compute log-likelihood for this key
    llp_norm = compute_log_likelihood_fit(actual_subset, predicted_subset, weights=None)
    llp_not_norm = compute_log_likelihood_fit(actual_subset, predicted_subset, weights=None, norm_by_n=False)
    ll_normalized[key] = llp_norm
    ll_not_normalized[key] = llp_not_norm
    # print(f"Log-likelihood for {key} (normalized): {llp_norm:.2f}")
    # print(f"Log-likelihood for {key} (not normalized): {llp_not_norm:.2f}")

# Print normalized values
print("\n--- Log-likelihood (normalized) ---")
for key, val in ll_normalized.items():
    print(f"{key}: {val:.2f}")

# Print unnormalized values
print("\n--- Log-likelihood (not normalized) ---")
for key, val in ll_not_normalized.items():
    print(f"{key}: {val:.2f}")

print("Done.")
