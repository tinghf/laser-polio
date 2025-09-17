import os
from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np

from laser_polio.utils import get_seasonality

results_path = "results/sweep_over_seasonality"


class MockSim:
    def __init__(self, seasonal_amplitude, seasonal_peak_doy):
        self.pars = {"seasonal_amplitude": seasonal_amplitude, "seasonal_peak_doy": seasonal_peak_doy}
        self.t = 0
        # Create a full year of dates starting from Jan 1, 2024
        start_date = datetime(2024, 1, 1)
        self.datevec = [start_date + timedelta(days=x) for x in range(365)]


def plot_seasonality_sweep(results_path=results_path):
    # Create parameter ranges
    amplitudes = np.linspace(0, 0.4, 5)  # 5 values from 0 to 1
    peak_doys = np.linspace(120, 300, 5)  # 5 values spread across the year

    # Create figure with subplots
    fig, axes = plt.subplots(len(amplitudes), 1, figsize=(12, 15))
    fig.suptitle("Seasonality Patterns for Different Parameter Values", fontsize=16)

    # Create x-axis dates for plotting
    dates = [datetime(2024, 1, 1) + timedelta(days=x) for x in range(365)]

    # Calculate overall y-axis limits
    all_seasonality = []
    for amplitude in amplitudes:
        for peak_doy in peak_doys:
            sim = MockSim(amplitude, peak_doy)
            seasonality = []
            for day in range(365):
                sim.t = day
                seasonality.append(get_seasonality(sim))
            all_seasonality.extend(seasonality)

    y_min = min(all_seasonality)
    y_max = max(all_seasonality)

    # Plot each combination
    for i, amplitude in enumerate(amplitudes):
        ax = axes[i]
        for peak_doy in peak_doys:
            # Create mock sim object with parameters
            sim = MockSim(amplitude, peak_doy)

            # Calculate seasonality for each day
            seasonality = []
            for day in range(365):
                sim.t = day
                seasonality.append(get_seasonality(sim))

            # Plot the seasonality curve
            label = f"Peak DoY: {int(peak_doy)}"
            ax.plot(dates, seasonality, label=label)

        ax.set_title(f"Seasonal Amplitude: {amplitude:.2f}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Seasonality Factor")
        ax.grid(True)
        ax.legend()

        # Set consistent axis limits
        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(os.path.join(results_path, "seasonality_sweep.png"))
    plt.close()


def plot_seasonality_monthly_peaks(results_path=results_path):
    """Plot 12 seasonal patterns with peaks every 30 days (30, 60, 90, ..., 360)."""

    # Create peak DOYs every 30 days
    peak_doys = np.arange(30, 361, 30)  # 30, 60, 90, ..., 360

    # Create different amplitudes to show variation
    amplitudes = [0.1, 0.3, 0.5]

    # Create figure with 3x4 grid of subplots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Seasonal Patterns with Monthly Shifted Peaks", fontsize=16)

    # Create x-axis dates for plotting
    dates = [datetime(2024, 1, 1) + timedelta(days=x) for x in range(365)]

    # Calculate month names for peak DOYs
    month_names = []
    for peak_doy in peak_doys:
        peak_date = datetime(2024, 1, 1) + timedelta(days=int(peak_doy) - 1)
        month_names.append(peak_date.strftime("%B"))

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Plot each peak DOY
    for i, peak_doy in enumerate(peak_doys):
        ax = axes_flat[i]

        for amplitude in amplitudes:
            # Create mock sim object with parameters
            sim = MockSim(amplitude, peak_doy)

            # Calculate seasonality for each day
            seasonality = []
            for day in range(365):
                sim.t = day
                seasonality.append(get_seasonality(sim))

            # Plot the seasonality curve
            label = f"Amp: {amplitude:.1f}"
            ax.plot(dates, seasonality, label=label, linewidth=2)

        # Add vertical line at peak
        peak_date = datetime(2024, 1, 1) + timedelta(days=int(peak_doy) - 1)
        ax.axvline(peak_date, color="red", linestyle="--", alpha=0.7, linewidth=1)

        ax.set_title(f"Peak: Day {int(peak_doy)} ({month_names[i]})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Seasonality Factor")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Set consistent axis limits
        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylim(0.5, 1.5)  # Fixed range based on typical amplitude values

        # Rotate x-axis labels for better readability
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(os.path.join(results_path, "seasonality_monthly_peaks.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_seasonality_sweep()
    plot_seasonality_monthly_peaks()
