#!/bin/bash

mkdir -p $1/vodca

# Download VODCA rasters
VODCA_LINK="https://zenodo.org/records/2575599/files/VODCA_X-band_1997-2018_v01.0.0.zip"
echo "Downloading VOXCA X-band archive..."
curl -sS $VODCA_LINK --output "$1/vodca/$(basename $VODCA_LINK)"