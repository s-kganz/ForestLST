#!/bin/bash

mkdir -p $1
mkdir -p $1/ads

declare -a ADS_LINKS=(
    "https://www.fs.usda.gov/foresthealth/docs/IDS_Data_for_Download/CONUS_Region3_AllYears.gdb.zip"
    "https://www.fs.usda.gov/foresthealth/docs/IDS_Data_for_Download/CONUS_Region4_AllYears.gdb.zip"
    "https://www.fs.usda.gov/foresthealth/docs/IDS_Data_for_Download/CONUS_Region5_AllYears.gdb.zip"
    "https://www.fs.usda.gov/foresthealth/docs/IDS_Data_for_Download/CONUS_Region6_AllYears.gdb.zip"
)

for ads_link in "${ADS_LINKS[@]}"; do
    echo "Downloading" $(basename $ads_link)"..."
    curl -sS $ads_link \
      --output $1/$(basename $ads_link)

    unzip -qq -o $1/$(basename $ads_link) -d $1/ads

    rm "$1/$(basename $ads_link)"
done