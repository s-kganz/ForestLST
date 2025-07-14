#!/bin/bash

mkdir -p $1/nlcd

# Download NLCD canopy cover
echo "Downloading NLCD canopy cover..."
NLCD_TCC_LINK="https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/nlcd_tcc_CONUS_2021_v2021-4.zip"
curl -sS $NLCD_TCC_LINK \
    --output "$1/$(basename $NLCD_TCC_LINK)"

unzip -qq "$1/$(basename $NLCD_TCC_LINK)" -d "$1/nlcd"
rm "$1/$(basename $NLCD_TCC_LINK)"