#!/bin/bash

OUTPUT_DIR=data_in

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/ads
mkdir -p $OUTPUT_DIR/mtbs
mkdir -p $OUTPUT_DIR/nidrm
mkdir -p $OUTPUT_DIR/vodca

# Download ADS polygons
declare -a ADS_LINKS=(
    "https://www.fs.usda.gov/foresthealth/docs/IDS_Data_for_Download/CONUS_Region3_AllYears.gdb.zip"
    "https://www.fs.usda.gov/foresthealth/docs/IDS_Data_for_Download/CONUS_Region4_AllYears.gdb.zip"
    "https://www.fs.usda.gov/foresthealth/docs/IDS_Data_for_Download/CONUS_Region5_AllYears.gdb.zip"
    "https://www.fs.usda.gov/foresthealth/docs/IDS_Data_for_Download/CONUS_Region6_AllYears.gdb.zip"
)


for ads_link in "${ADS_LINKS[@]}"; do
    echo "Downloading" $(basename $ads_link)"..."
    curl -sS $ads_link \
      -H "User-Agent: Firefox/131.0" \
      -H "Accept-Language: en-US,en;q=0.5" \
      -H "Accept-Encoding: gzip, deflate, br, zstd" \
      --output $OUTPUT_DIR/$(basename $ads_link)

    unzip -qq -o $OUTPUT_DIR/$(basename $ads_link) -d $OUTPUT_DIR/ads

    rm "$OUTPUT_DIR/$(basename $ads_link)"
done


# Download VODCA rasters
VODCA_LINK="https://zenodo.org/records/2575599/files/VODCA_X-band_1997-2018_v01.0.0.zip"
echo "Downloading VOXCA X-band archive..."
curl -sS $VODCA_LINK --output "$OUTPUT_DIR/vodca/$(basename $VODCA_LINK)"

# Download MTBS polygons
echo "Downloading MTBS polygons..."
MTBS_LINK="https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip"

curl -sS $MTBS_LINK --output "$OUTPUT_DIR/$(basename $MTBS_LINK)"
unzip -qq -o $OUTPUT_DIR/$(basename $MTBS_LINK) -d $OUTPUT_DIR/mtbs
rm "$OUTPUT_DIR/$(basename $MTBS_LINK)"

# Download NIDRM basal area rasters
echo "Downloading NIDRM basal area rasters..."
NIDRM_TOTALS_LINK="https://www.fs.usda.gov/foresthealth/docs/L48_Totals.gdb.zip"
NIDRM_BA_LINK="https://www.fs.usda.gov/foresthealth/docs/L48_BA_by_spp.gdb.zip"

curl -sS $NIDRM_TOTALS_LINK \
    -H "User-Agent: Firefox/131.0" \
    -H "Accept-Language: en-US,en;q=0.5" \
    -H "Accept-Encoding: gzip, deflate, br, zstd" \
    --output "$OUTPUT_DIR/$(basename $NIDRM_TOTALS_LINK)"
    
unzip -qq -o $OUTPUT_DIR/$(basename $NIDRM_TOTALS_LINK) -d $OUTPUT_DIR/nidrm
rm "$OUTPUT_DIR/$(basename $NIDRM_TOTALS_LINK)"

curl -sS $NIDRM_BA_LINK \
    -H "User-Agent: Firefox/131.0" \
    -H "Accept-Language: en-US,en;q=0.5" \
    -H "Accept-Encoding: gzip, deflate, br, zstd" \
    --output "$OUTPUT_DIR/$(basename $NIDRM_BA_LINK)"

unzip -qq -o "$OUTPUT_DIR/$(basename $NIDRM_BA_LINK)" -d $OUTPUT_DIR/nidrm
rm "$OUTPUT_DIR/$(basename $NIDRM_BA_LINK)"

# Download NLCD canopy cover
echo "Downloading NLCD canopy cover..."
NLCD_TCC_LINK="https://s3-us-west-2.amazonaws.com/mrlc/nlcd_tcc_CONUS_2021_v2021-4.zip"
curl -sS $NLCD_TCC_LINK \
    --output "$OUTPUT_DIR/$(basename $NIDRM_TOTALS_LINK)"

unzip -qq -o "$OUTPUT_DIR/$(basename $NLCD_TCC_LINK)" -d $OUTPUT_DIR/nlcd
rm "$OUTPUT_DIR/$(basename $NLCD_TCC_LINK)"
 
