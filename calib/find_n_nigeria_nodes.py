import pandas as pd

import laser_polio as lp

regions = ["NIGERIA"]  # Define your regions here
dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")


# Load NGA_NORTH patterns
nga_north_df = pd.read_csv(lp.root / "data/curation_scripts/shp/NGA_NORTH.csv")
patterns = nga_north_df["dot_name"].dropna().unique()  # base strings to match

# Find matching indices or values
matches = [name for name in dot_names if any(p in name for p in patterns)]

# Get indices:
matching_indices = [i for i, name in enumerate(dot_names) if any(p in name for p in patterns)]
print(f"Matching indices: {matching_indices}")

# Find which indices are not in the list of matches
not_matching_indices = [i for i in range(len(dot_names)) if i not in matching_indices]
print(f"Indices not matching: {not_matching_indices}")
