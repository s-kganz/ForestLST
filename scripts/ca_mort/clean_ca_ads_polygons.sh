#!/bin/bash

INPUT_GDB=data_in/ads/CONUS_Region5_AllYears.gdb
ADS_LINK="https://www.fs.usda.gov/foresthealth/docs/IDS_Data_for_Download/CONUS_Region5_AllYears.gdb.zip"
# Damage is either from insects (11XXX and 15XXX) or drought (50003)
DAMAGE_SQL="DCA_CODE/1000 = 11 OR DCA_CODE/1000 = 15 OR DCA_CODE=50003"
SURVEY_FIELDS="SURVEY_YEAR,IDS_DATA_SOURCE"
DAMAGE_FIELDS="DAMAGE_AREA_ID,REGION_ID,HOST_CODE,DCA_CODE,SURVEY_YEAR"

mkdir -p data_working

if [ ! -e $INPUT_GDB ]; then
    echo "Input ADS polygons not found! Downloading R5 polygons..."
    curl -sS $ADS_LINK \
      -H "User-Agent: Firefox/131.0" \
      -H "Accept-Language: en-US,en;q=0.5" \
      -H "Accept-Encoding: gzip, deflate, br, zstd" \
      --output data_in/$(basename $ADS_LINK)

    unzip -qq -o data_in/$(basename $ADS_LINK) -d data_in/ads
    rm data_in/$(basename $ADS_LINK)
fi

echo "Writing damage polygons..."

ogr2ogr data_working/damage.shp $INPUT_GDB DAMAGE_AREAS_FLAT_AllYears_CONUS_Rgn5 \
    -select $DAMAGE_FIELDS \
    -where "$DAMAGE_SQL" \
    -simplify 100 \
    -makevalid \
    > /dev/null

echo "Writing survey polygons..."

ogr2ogr data_working/survey.shp $INPUT_GDB SURVEYED_AREAS_FLAT_AllYears_CONUS_Rgn5 \
    -select $SURVEY_FIELDS \
    -simplify 100 \
    -makevalid \
    > /dev/null

echo "Now upload the shapefiles to your earth engine account"
echo "and see notebooks/make_ads_images.ipynb for the next steps."
exit 0
