#!/bin/bash

OUTPUT_DIR=$1

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/nidrm


# Download NIDRM basal area rasters
echo "Downloading NIDRM basal area rasters..."
NIDRM_TOTALS_LINK="https://www.fs.usda.gov/foresthealth/docs/L48_Totals.gdb.zip"
NIDRM_BA_LINK="https://www.fs.usda.gov/foresthealth/docs/L48_BA_by_spp.gdb.zip"

curl -sS $NIDRM_TOTALS_LINK \
    --output "$OUTPUT_DIR/$(basename $NIDRM_TOTALS_LINK)"
    
unzip -qq -o $OUTPUT_DIR/$(basename $NIDRM_TOTALS_LINK) -d $OUTPUT_DIR/nidrm
rm "$OUTPUT_DIR/$(basename $NIDRM_TOTALS_LINK)"

curl -sS $NIDRM_BA_LINK \
    --output "$OUTPUT_DIR/$(basename $NIDRM_BA_LINK)"

unzip -qq -o "$OUTPUT_DIR/$(basename $NIDRM_BA_LINK)" -d $OUTPUT_DIR/nidrm
rm "$OUTPUT_DIR/$(basename $NIDRM_BA_LINK)"
 
