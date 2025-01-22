#!/bin/bash

OUTPUT_DIR=data_in

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/ads
mkdir -p $OUTPUT_DIR/mtbs
mkdir -p $OUTPUT_DIR/nidrm
mkdir -p $OUTPUT_DIR/vodca


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
 
