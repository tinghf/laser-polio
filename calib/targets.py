import numpy as np
import pandas as pd
import yaml

import laser_polio as lp


def calc_calib_targets_paralysis(filename, model_config_path=None, is_actual_data=True):
    """Load simulation results and extract features for comparison."""

    # Load the data & config
    df = pd.read_csv(filename)
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Parse dates to datetime object if needed
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Choose the column to summarize
    if is_actual_data:
        case_col = "P"
        scale_factor = 1.0
    else:
        case_col = "new_potentially_paralyzed"
        scale_factor = 1 / 2000.0
        # The actual data is in months & the sim has a tendency to rap into the next year (e.g., 2020-01-01) so we need to exclude and dates beyond the last month of the actual data
        max_date = lp.find_latest_end_of_month(df["date"])
        df = df[df["date"] <= max_date]

    targets = {}

    # 1. Total infected (scaled if simulated)
    targets["total_infected"] = np.array([df[case_col].sum() * scale_factor])

    # 2. Yearly cases
    targets["yearly_cases"] = df.groupby("year")[case_col].sum().values * scale_factor

    # 3. Monthly cases
    targets["monthly_cases"] = df.groupby("month")[case_col].sum().values * scale_factor

    # 4. Monthly timeseries (sorted full date x month time series)
    monthly_df = df.groupby([df["date"].dt.to_period("M")])[case_col].sum().sort_index().astype(float) * scale_factor
    targets["monthly_timeseries"] = monthly_df.values

    # 5. Regional group cases
    if model_config and "summary_config" in model_config:
        region_groups = model_config["summary_config"].get("region_groups", {})
        regional_cases = []
        for name in region_groups:
            node_list = region_groups[name]
            total = df[df["node"].isin(node_list)][case_col].sum() * scale_factor
            regional_cases.append(total)
        targets["regional_cases"] = np.array(regional_cases)

    print(f"{targets=}")
    return targets


def get_smoothed_node_case_presence(sim_df, case_column="new_potentially_paralyzed", cir=1 / 2000, n_draws=100, seed=42):
    np.random.seed(seed)
    sim_df = sim_df.copy()
    sim_df["date"] = pd.to_datetime(sim_df["date"])
    sim_df["month"] = sim_df["date"].dt.to_period("M")

    all_months = sim_df["month"].sort_values().unique()

    monthly_counts = np.zeros((n_draws, len(all_months)))
    total_counts = np.zeros(n_draws)

    for i in range(n_draws):
        expected = sim_df[case_column].values * cir
        paralytic_cases = np.random.poisson(expected)

        sim_df["has_case"] = paralytic_cases > 0

        total_counts[i] = sim_df.loc[sim_df["has_case"], "node"].nunique()

        monthly_group = sim_df[sim_df["has_case"]].groupby("month")["node"].nunique()
        monthly_group = monthly_group.reindex(all_months, fill_value=0)
        monthly_counts[i, :] = monthly_group.values

    return {"nodes_with_cases_total": total_counts.mean(), "nodes_with_cases_timeseries": monthly_counts.mean(axis=0)}


def calc_targets_temporal_regional_nodes(filename, model_config_path=None, is_actual_data=True):
    """Load simulation results and extract features for comparison."""

    # Load the data & config
    df = pd.read_csv(filename)
    model_config = {}
    if model_config_path is not None:
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)

    # Parse dates to datetime object if needed
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Choose the column to summarize
    if is_actual_data:
        case_col = "P"
        scale_factor = 1.0
    else:
        case_col = "new_potentially_paralyzed"
        scale_factor = 1 / 2000.0
        # The actual data is in months & the sim has a tendency to rap into the next year (e.g., 2020-01-01) so we need to exclude and dates beyond the last month of the actual data
        max_date = lp.find_latest_end_of_month(df["date"])
        df = df[df["date"] <= max_date]

    targets = {}

    # --- Temporal aggregation ---
    targets["total_infected"] = np.array([df[case_col].sum() * scale_factor])
    targets["yearly_cases"] = df.groupby("year")[case_col].sum().values * scale_factor
    targets["monthly_cases"] = df.groupby("month")[case_col].sum().values * scale_factor
    monthly_df = df.groupby([df["date"].dt.to_period("M")])[case_col].sum().sort_index().astype(float) * scale_factor
    targets["monthly_timeseries"] = monthly_df.values

    # --- Regional aggregation ---
    # Split dot_name into columns
    dot_parts = df["dot_name"].str.split(":", expand=True)
    df["adm0"] = dot_parts[1]
    df["adm1"] = dot_parts[2]
    df["adm01"] = df["adm0"] + ":" + df["adm1"]
    # Group and sum
    adm0_cases = df.groupby("adm0")[case_col].sum().to_dict()
    adm01_cases = df.groupby("adm01")[case_col].sum().to_dict()
    # Only return multi-region groups
    if len(adm0_cases) > 1:
        targets["adm0_cases"] = {k: v * scale_factor for k, v in adm0_cases.items()}
    if len(adm01_cases) > 1:
        targets["adm01_cases"] = {k: v * scale_factor for k, v in adm01_cases.items()}

    # Custom regional groups from model_config
    if model_config and "summary_config" in model_config:
        region_groups = model_config["summary_config"].get("region_groups", {})
        regional_cases = []
        for name in region_groups:
            node_list = region_groups[name]
            total = df[df["node"].isin(node_list)][case_col].sum() * scale_factor
            regional_cases.append(total)
        targets["regional_cases"] = np.array(regional_cases)

    # --- Number of nodes with cases ---
    if is_actual_data:
        has_case = df[case_col] > 0
        targets["nodes_with_cases_total"] = np.array([df.loc[has_case, "node"].nunique()])

        df["month_period"] = df["date"].dt.to_period("M")
        all_months = df["month_period"].sort_values().unique()
        monthly_node_counts = df.loc[has_case].groupby("month_period")["node"].nunique().sort_index()

        monthly_node_counts = monthly_node_counts.reindex(all_months, fill_value=0)
        targets["nodes_with_cases_timeseries"] = monthly_node_counts.values
    if not is_actual_data:
        node_case_metrics = get_smoothed_node_case_presence(df)
        targets["nodes_with_cases_total"] = np.array([node_case_metrics["nodes_with_cases_total"]])
        targets["nodes_with_cases_timeseries"] = node_case_metrics["nodes_with_cases_timeseries"]

    print(f"{targets=}")
    return targets


def calc_calib_targets(filename, model_config_path=None):
    """Load simulation results and extract features for comparison."""

    # Load the data & config
    df = pd.read_csv(filename)
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Parse dates to datetime object if needed
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    targets = {}

    # 1. Total infected
    targets["total_infected"] = df["I"].sum()

    # 2. Yearly cases

    # 3. Monthly cases
    targets["monthly_cases"] = df.groupby("month")["I"].sum().values

    # 4. Regional group cases as a single array
    if model_config and "summary_config" in model_config:
        region_groups = model_config["summary_config"].get("region_groups", {})
        regional_cases = []
        for name in region_groups:
            node_list = region_groups[name]
            total = df[df["node"].isin(node_list)]["I"].sum()
            regional_cases.append(total)
        targets["regional_cases"] = np.array(regional_cases)

    print(f"{targets=}")
    return targets


def process_data(filename):
    """Load simulation results and extract features for comparison."""
    df = pd.read_csv(filename)
    return {
        "total_infected": df["I"].sum(),
        "peak_infection_time": df.loc[df["I"].idxmax(), "Time"],
    }


def calc_targets_simplified_temporal(filename, model_config_path=None, is_actual_data=True):
    """Load simulation results and extract simplified features for comparison.

    Only uses total_infected, monthly_timeseries, and adm01_cases.
    Splits total_infected and adm01_cases by time period: 2018-2019, 2020-2021, and 2022-2023.
    """
    # Load the data & config
    df = pd.read_csv(filename)

    # Parse dates to datetime object if needed
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Choose the column to summarize
    if is_actual_data:
        case_col = "P"
        scale_factor = 1.0
    else:
        case_col = "new_potentially_paralyzed"
        scale_factor = 1 / 2000.0
        # The actual data is in months & the sim has a tendency to rap into the next year (e.g., 2020-01-01) so we need to exclude and dates beyond the last month of the actual data
        max_date = lp.find_latest_end_of_month(df["date"])
        df = df[df["date"] <= max_date]

    # Add time period column with three periods (fast version)
    bins = [pd.Timestamp.min, pd.Timestamp("2020-01-01"), pd.Timestamp("2022-01-01"), pd.Timestamp.max]
    labels = ["2018-2019", "2020-2021", "2022-2023"]
    df["time_period"] = pd.cut(df["date"], bins=bins, labels=labels, right=False)

    targets = {}

    # 1. Total infected (split by time period)
    total_by_period = df.groupby("time_period", observed=True)[case_col].sum() * scale_factor
    targets["total_by_period"] = total_by_period.to_dict()

    # 2. Monthly timeseries (full time series)
    monthly_df = df.groupby([df["date"].dt.to_period("M")])[case_col].sum().sort_index().astype(float) * scale_factor
    targets["monthly_timeseries"] = monthly_df.values

    # 3. Regional aggregation (adm01_cases split by time period)
    # Split dot_name into columns
    dot_parts = df["dot_name"].str.split(":", expand=True)
    df["adm0"] = dot_parts[1]
    df["adm1"] = dot_parts[2]
    df["adm01"] = df["adm0"] + ":" + df["adm1"]
    # Group by adm01 and time period
    adm01_by_period = df.groupby(["adm01", "time_period"], observed=True)[case_col].sum() * scale_factor
    targets["adm01_by_period"] = adm01_by_period.to_dict()

    print(f"{targets=}")
    return targets


def calc_targets_regional(filename, model_config_path=None, is_actual_data=True, verbose=0):
    """Load simulation results and extract target features for comparison.

    Targets are:
    - cases_total: Total infected
    - cases_by_period: Total infected by time period
    - cases_by_month: Monthly timeseries of total infected
    - cases_by_region: Regional cases
    - cases_by_region_period: Regional cases by time period
    - cases_by_region_month: Regional cases by month
    - case_bins_by_region: Number of districts with X cases by region
    """

    # Load the data & config
    if is_actual_data:
        df = pd.read_csv(filename)
    else:
        df = pd.read_hdf(filename)
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Parse dates to datetime object if needed
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Choose which column to summarize depending on the data type
    if is_actual_data:
        case_col = "P"
        scale_factor = 1.0
    else:
        case_col = "new_potentially_paralyzed"
        scale_factor = 1 / 2000.0
        # The actual data is in months & the sim has a tendency to rap into the next year (e.g., 2020-01-01) so we need to exclude and dates beyond the last month of the actual data
        max_date = lp.find_latest_end_of_month(df["date"])
        df = df[df["date"] <= max_date]

    targets = {}

    # Total cases
    targets["cases_total"] = df[case_col].sum() * scale_factor

    # Total cases by time period
    cases_by_period = df.groupby("time_period", observed=True)[case_col].sum() * scale_factor
    targets["cases_by_period"] = cases_by_period.to_dict()

    # Total cases by month and year (e.g., 2018-01, 2018-02, etc.)
    monthly_df = df.groupby([df["date"].dt.to_period("M")])[case_col].sum().sort_index().astype(float) * scale_factor
    targets["cases_by_month_timeseries"] = monthly_df.values

    # Total cases by month number (1-12) across all years
    cases_by_month_across_years = monthly_df.groupby(monthly_df.index.month).sum()
    targets["cases_by_month"] = cases_by_month_across_years.values

    # Total cases by region (across all time periods)
    cases_by_region = df.groupby("region", observed=True)[case_col].sum() * scale_factor
    targets["cases_by_region"] = cases_by_region.to_dict()

    # Cases by region and time_period
    cases_by_region_period = df.groupby(["region", "time_period"], observed=True)[case_col].sum() * scale_factor
    # Convert from Series with MultiIndex to nested dict
    cases_by_region_period_nested = {}
    for (region, period), value in cases_by_region_period.items():
        cases_by_region_period_nested.setdefault(region, {})[str(period)] = value  # convert Period object to string if needed
    targets["cases_by_region_period"] = cases_by_region_period_nested

    # Regional monthly timeseries
    regional_monthly_df = (
        df.groupby(["region", df["date"].dt.to_period("M")], observed=True)[case_col].sum().sort_index().astype(float) * scale_factor
    )
    # Convert to dictionary with arrays per region
    cases_by_region_month = {}
    for region in df["region"].unique():
        # Get data for this region and convert to array
        region_data = regional_monthly_df.loc[region] if region in regional_monthly_df.index.get_level_values(0) else pd.Series(dtype=float)
        cases_by_region_month[region] = region_data.values
    targets["cases_by_region_month"] = cases_by_region_month

    # Count the number of districts with X cases by region
    if "summary_config" in model_config and model_config["summary_config"] and "case_bins" in model_config["summary_config"]:
        # Get case bins from config or use defaults
        bin_config = model_config["summary_config"]["case_bins"]
        bins = bin_config["bin_edges"]
    else:
        # Fallback to default values
        bins = [0, 1, 2, 3, 4, 5, 10, 20, np.inf]
    # Count the number of unique dot_names by region
    dot_names_by_region = df.groupby("region", observed=True)["dot_name"].nunique().to_dict()
    if verbose > 0:
        print(f"{dot_names_by_region=}")
    case_bins_by_region = {}
    for region in df["region"].unique():
        region_mask = df["region"] == region
        region_district_cases = df[region_mask].groupby("dot_name", observed=True)[case_col].sum() * scale_factor
        # Count districts in each bin
        hist_counts, _ = np.histogram(region_district_cases.values, bins=bins)
        case_bins_by_region[region] = hist_counts.tolist()
        # Check that the number of districts with cases in the bins matches the number of unique dot_names in the region
        assert len(region_district_cases) == dot_names_by_region[region] == sum(hist_counts), (
            f"Number of districts with cases in region {region} does not match the number of unique dot_names or the number of districts with cases in the bins"
        )
    targets["case_bins_by_region"] = case_bins_by_region

    # print(f"{targets=}")
    return targets
