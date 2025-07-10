from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Load actual data
actual_path = Path(
    "C:/github/laser-polio/results/calib_nigeria_6y_2018_underwt_gravity_zinb_ipv_vxtrans_nwecnga_pop0.01_20250708/actual_data.csv"
)
df = pd.read_csv(actual_path)
df.head()

# Load model_config with regional groups
config_file = "C:/github/laser-polio/calib/model_configs/config_nigeria_6y_2018_underwt_gravity_zinb_ipv_nwecnga_3periods.yaml"
config = yaml.safe_load(open(config_file))

# Get the regional groups
summary_config = config["summary_config"]
regions = summary_config["region_groups"]

# Read bin configuration from model config (same as targets.py)
if "case_bins" in summary_config:
    bin_config = summary_config["case_bins"]
    bin_edges = bin_config["bin_edges"]
    bin_labels = bin_config["bin_labels"]
else:
    # Fallback to default values (same as targets.py)
    bin_edges = [0, 1, 2, 3, 4, 5, 10, 20, np.inf]
    bin_labels = ["0", "1", "2", "3", "4", "5-9", "10-19", "20+"]

# Create a 'region' column: use matched region name or fall back to adm01
df["region"] = df["dot_name"]  # Default to dot_name

# Override with region group names where patterns match
for group_name, patterns in regions.items():
    # Create a boolean mask for rows matching any of the patterns
    mask = pd.Series(False, index=df.index)
    for pattern in patterns:
        mask |= df["dot_name"].str.contains(pattern, case=False, na=False)

    # Set region name for matching rows
    df.loc[mask, "region"] = group_name


# Sum cases by dot_name
cases_by_dot_name = df.groupby("dot_name")["P"].sum()
cases_by_dot_region = df.groupby(["region", "dot_name"])["P"].sum()

# Plot overall histogram
plt.figure()
plt.title("Number of districts with X cases")
plt.xlabel("Number of cases")
plt.ylabel("Number of districts")
plt.hist(cases_by_dot_name, bins=100)
plt.show()

# Create subplot for each region
region_names = list(regions.keys())
n_regions = len(region_names)

# Define specific bin edges for epidemiologically meaningful categories

# Create subplot grid (2x2 for 4 regions, adjust if different number)
n_cols = 2
n_rows = (n_regions + n_cols - 1) // n_cols  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
fig.suptitle("Number of Districts with X Cases by Region", fontsize=16)

# Flatten axes for easier indexing if multiple rows
if n_regions > 1:
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
else:
    axes = [axes]

# Plot histogram for each region using defined bins
for idx, region in enumerate(region_names):
    ax = axes[idx]

    # Get case data for this region
    region_cases = cases_by_dot_region[region] if region in cases_by_dot_region.index.get_level_values(0) else []

    # Count districts in each bin (same as targets.py)
    if len(region_cases) > 0:
        hist_counts, _ = np.histogram(region_cases, bins=bin_edges)
    else:
        hist_counts = [0] * (len(bin_edges) - 1)

    # Create bar plot with the binned data
    x_positions = range(len(bin_labels))
    bars = ax.bar(x_positions, hist_counts, edgecolor="black", alpha=0.7)

    ax.set_title(f"{region.replace('_', ' ').title()}")
    ax.set_xlabel("Number of cases")
    ax.set_ylabel("Number of districts")
    ax.grid(True, alpha=0.3)

    # Set x-tick labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bin_labels, rotation=0)

    # Add count annotations on top of bars
    for i, count in enumerate(hist_counts):
        if count > 0:
            ax.text(i, count + max(hist_counts) * 0.02, f"{int(count)}", ha="center", va="bottom", fontsize=9)

# Hide any unused subplots
for idx in range(n_regions, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.show()

# Output the binned data structure (same format as targets.py)
print("\nDistrict case bin counts by region:")
district_case_bin_counts = {}
for region in region_names:
    if region in cases_by_dot_region.index.get_level_values(0):
        region_cases = cases_by_dot_region[region].values
        hist_counts, _ = np.histogram(region_cases, bins=bin_edges)
        district_case_bin_counts[region] = hist_counts.tolist()
        print(f"{region}: {hist_counts.tolist()}")
    else:
        district_case_bin_counts[region] = [0] * (len(bin_edges) - 1)
        print(f"{region}: {[0] * (len(bin_edges) - 1)} (no cases)")

print(f"\nBin edges: {bin_edges}")
print(f"Bin labels: {bin_labels}")
print("Done.")
