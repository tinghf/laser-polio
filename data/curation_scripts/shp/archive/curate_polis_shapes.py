# import geopandas as gpd
# import numpy as np
# import pandas as pd
# from unidecode import unidecode
# import re
# from laser_polio.utils import clean_strings
# from alive_progress import alive_bar


# # List each of the unique iso3 codes in the africa_polis_adm0.shp file
# shapefile_path = 'data/curation_scripts/shapes/polis/polis_adm0_africa.shp'
# gdf = gpd.read_file(shapefile_path)
# print(gdf.columns)
# # List each of the unique ISO3 codes in the shapefile
# isos = gdf['is_3_cd'].unique()

# # Load an example shapefile
# shp_path = 'data/curation_scripts/shapes/polis/polis_adm2_COD.shp'
# shp = gpd.read_file(shp_path)

# # Curate the admin names & create a dot_name for each shape
# columns_to_clean = ['who_rgn', 'adm0_nm', 'adm1_nm', 'adm2_nm']
# shp[columns_to_clean] = shp[columns_to_clean].map(clean_strings)
# shp['dot_name'] = shp[columns_to_clean].agg(':'.join, axis=1)


# print('Done.')


from pathlib import Path

import geopandas as gpd
from alive_progress import alive_bar

from laser_polio.utils import clean_strings

# Paths
# Folder where input shapefiles are stored
input_dir = "data/curation_scripts/shapes/polis"
# Output folder for cleaned shapefiles
output_dir = "data/curation_scripts/shapes/curated"
# Ensure output directory exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Load the Africa-wide ADM0 shapefile to get unique ISO3 country codes
shapefile_path = Path(input_dir) / "polis_adm0_africa.shp"
gdf = gpd.read_file(shapefile_path)
# List unique ISO3 codes
isos = gdf["is_3_cd"].unique()
print(f"Processing {len(isos)} countries...")

# Columns to clean
columns_to_clean = ["who_rgn", "adm0_nm", "adm1_nm", "adm2_nm"]

# Progress bar
with alive_bar(len(isos)) as bar:
    for iso in isos:
        try:
            shp_path = Path(input_dir) / f"polis_adm2_{iso}.shp"

            # Skip if the input file does not exist
            if not shp_path.exists():
                print(f"Skipping {iso} - File not found")
                bar()  # Update progress bar
                continue

            # Load the shapefile
            shp = gpd.read_file(shp_path)

            # Clean the specified columns
            shp[columns_to_clean] = shp[columns_to_clean].map(clean_strings)

            # Generate dot-separated 'dot_name'
            shp["dot_name"] = shp[columns_to_clean].agg(":".join, axis=1)

            # Save the cleaned shapefile
            shp.to_file(shp_path, driver="ESRI Shapefile")
            print(f"Processed {iso} -> Saved: {shp_path}")

        except Exception as e:
            print(f"Error processing {iso}: {e}")

        bar()  # Update progress bar

print("All shapefiles processed successfully!")
