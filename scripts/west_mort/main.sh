#!/bin/bash

pip install xarray-spatial

# Download input data
./scripts/west_mort/download_data.sh

# Make damage, FAM, forest cover rasters
./merge_ads_polygons.sh
python scripts/west_mort/burn_ads_polygons.py

# Make genus basal area rasters
python scripts/west_mort/coarsen_genus_ba.py

# Make fire area rasters
python scripts/west_mort/burn_mtbs_polygons.py

# Make datasets from earthaccess
python scripts/west_mort/download_daymet.py
python scripts/west_mort/download_terrain.py
python scripts/west_mort/download_vodca.py