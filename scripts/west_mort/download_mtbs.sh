#!/bin/bash

mkdir -p $1/mtbs

# Download MTBS polygons
echo "Downloading MTBS polygons..."
MTBS_LINK="https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip"
curl -sS $MTBS_LINK --output "$1/$(basename $MTBS_LINK)"
unzip -qq -o $1/$(basename $MTBS_LINK) -d $1/mtbs
rm "$1/$(basename $MTBS_LINK)"