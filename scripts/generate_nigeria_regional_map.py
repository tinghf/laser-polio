from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import yaml

import laser_polio as lp

# Load shapefile
shp = gpd.read_file(filename=lp.root / "data/shp_africa_low_res.gpkg", layer="adm1")
shp = shp[shp["adm0_name"] == "NIGERIA"].copy()

# Load model_config with regional groups
config_file = lp.root / "calib/model_configs/config_nigeria_6y_2018_underwt_gravity_zinb_ipv_nsnga_bins.yaml"
config = yaml.safe_load(open(config_file))

# Get the regional groups
summary_config = config["summary_config"]
regions = summary_config["region_groups"]

# Create a 'region' column: use matched region name or fall back to adm01
shp["region"] = shp["dot_name"]  # Default to dot_name

# Override with region group names where patterns match
for group_name, patterns in regions.items():
    # Create a boolean mask for rows matching any of the patterns
    mask = pd.Series(False, index=shp.index)
    for pattern in patterns:
        mask |= shp["dot_name"].str.contains(pattern, case=False, na=False)

    # Set region name for matching rows
    shp.loc[mask, "region"] = group_name

# Create the map
fig, ax = plt.subplots(figsize=(12, 10))

# Create colormap
region_names = list(regions.keys())
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red
region_colors = dict(zip(region_names, colors, strict=False))

# Plot each region with its color
for region in region_names:
    region_data = shp[shp["region"] == region]
    if not region_data.empty:
        region_data.plot(ax=ax, color=region_colors[region], edgecolor="black", linewidth=0.5, label=region.replace("_", " ").title())

# Plot unmapped areas in gray (states that didn't match any pattern)
unmapped_data = shp[~shp["region"].isin(region_names)]
if not unmapped_data.empty:
    unmapped_data.plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.5, label="Unmapped")

# Customize the plot
ax.set_title("Nigeria Regional Groupings", fontsize=16, fontweight="bold")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
ax.grid(True, alpha=0.3)

# Remove axes ticks for cleaner look
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()

# Save the map
output_path = Path("results/nigeria_regional_map.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Map saved to: {output_path}")

# Show summary
print("\nRegional groupings summary:")
for region in region_names:
    count = len(shp[shp["region"] == region])
    print(f"  {region.replace('_', ' ').title()}: {count} states")

# Show any unmapped states
unmapped_count = len(unmapped_data) if not unmapped_data.empty else 0
if unmapped_count > 0:
    print(f"  Unmapped: {unmapped_count} states")
    print(f"  Unmapped states: {unmapped_data['adm1_name'].tolist()}")

plt.show()
