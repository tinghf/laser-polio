import subprocess
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path

import requests


def main(azure: bool = False):
    platform = detect_os()
    display_cpu_info(platform)
    display_ram_info(platform)
    if (azure) and (platform != OS.MAC):
        display_sku_info()

    return


def display_cpu_info(platform):
    if platform == OS.MAC:
        cpu_info = subprocess.run(["system_profiler", "SPHardwareDataType"], text=True)
    elif (platform == OS.LINUX) or (platform == OS.WSL):
        cpu_info = subprocess.run(["lscpu"], text=True)
    else:
        cpu_info = "Unknown platform"

    print("CPU Info:\n", cpu_info)

    return


def display_ram_info(platform):
    if platform == OS.MAC:
        ram_info = "See above."
    elif (platform == OS.LINUX) or (platform == OS.WSL):
        ram_info = subprocess.run(["free", "-h"], text=True)
    else:
        ram_info = "Unknown platform"

    print("RAM Info:\n", ram_info)

    return


def display_sku_info():
    # Azure SKU via IMDS
    headers = {"Metadata": "true"}
    resp = requests.get(
        "http://169.254.169.254/metadata/instance/compute/vmSize?api-version=2021-02-01&format=text",
        headers=headers,
        timeout=5,  # Timeout in seconds
    )
    vm_size = resp.text

    print("Azure VM Size (SKU):", vm_size)

    return


class OS(Enum):
    MAC = "mac"
    LINUX = "linux"
    WSL = "wsl"
    OTHER = "other"


def detect_os() -> OS:
    plat = sys.platform

    if plat == "darwin":
        return OS.MAC

    if plat.startswith("linux"):
        # Distinguish WSL from native Linux
        try:
            if "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower():
                return OS.WSL
        except Exception:  # noqa: S110
            pass
        return OS.LINUX

    return OS.OTHER


# Usage:
# if detect_os() == OS.MAC: ...


if __name__ == "__main__":
    parser = ArgumentParser(description="System Information Script")
    parser.add_argument("--azure", action="store_true", help="Show Azure VM information")
    args = parser.parse_args()

    main(args.azure)
