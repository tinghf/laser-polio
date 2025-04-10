import numpy as np
import pandas as pd

import laser_polio as lp

start_year = 2018
n_days = 365
regions = ["ZAMFARA"]
dot_names = lp.find_matching_dot_names(regions, "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")

# # Load data
# df = pd.read_hdf("data/epi_africa_20250408.h5", key="epi")

# # Filter by dot_names & dates
# df = df[df["dot_name"].isin(dot_names)]
# df["date"] = pd.to_datetime(df["month_start"])
# start_date = pd.to_datetime(f"{start_year}-01-01")
# end_date = pd.to_datetime(start_date + pd.DateOffset(days=n_days))
# df = df[(df["date"] >= start_date) & (df["date"] < end_date)]

# # Sort by date then node
# df = df.sort_values(by=["date", "dot_name"])
# df = df.reset_index(drop=True)
# # Ensure that the nodes are in the same order
# assert np.all(df["dot_name"][0 : len(dot_names)].values == dot_names), "The nodes are not in the same order as the dot_names."


def get_epi_data(filename, dot_names, start_year, n_days):
    """Curate the epi data for a specific set of dot names."""

    # Load data
    df = pd.read_hdf(filename, key="epi")

    # Filter by dot_names
    df = df[df["dot_name"].isin(dot_names)]

    # Filter by dates
    df["date"] = pd.to_datetime(df["month_start"])
    start_date = pd.to_datetime(f"{start_year}-01-01")
    end_date = pd.to_datetime(start_date + pd.DateOffset(days=n_days))
    df = df[(df["date"] >= start_date) & (df["date"] < end_date)]

    # Sort by date then node
    df = df.sort_values(by=["date", "dot_name"])
    df = df.reset_index(drop=True)

    # Ensure that the nodes are in the same order
    assert np.all(df["dot_name"][0 : len(dot_names)].values == dot_names), "The nodes are not in the same order as the dot_names."

    return df


epi = get_epi_data("data/epi_africa_20250408.h5", dot_names, start_year, n_days)

print("Done.")
