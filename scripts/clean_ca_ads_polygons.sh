#!/bin/bash

INPUT_GDB=data_in/ads/CONUS_Region5_AllYears.gdb
DAMAGE_SQL="DCA_CODE/1000 = 11 OR DCA_CODE/1000 = 15 OR DCA_CODE=50003"
SURVEY_FIELDS="SURVEY_YEAR,IDS_DATA_SOURCE"
DAMAGE_FIELDS="DAMAGE_AREA_ID,REGION_ID,HOST_CODE,DCA_CODE,SURVEY_YEAR"

mkdir -p data_working

if [ ! -e $INPUT_GDB ]; then
    echo "Input ADS polygons not found! Download region 5 all years from:"
    echo "https://www.fs.usda.gov/science-technology/data-tools-products/fhp-mapping-reporting/detection-surveys"
    echo "For some reason wget and curl can't download from the USFS website."
    echo "Sorry about that :("
    exit 1
fi

echo "Writing damage polygons..."

ogr2ogr data_working/damage.shp $INPUT_GDB DAMAGE_AREAS_FLAT_AllYears_CONUS_Rgn5 \
    -select $DAMAGE_FIELDS \
    -where "$DAMAGE_SQL" \
    -simplify 100 \
    -makevalid

echo "Writing survey polygons..."

ogr2ogr data_working/survey.shp $INPUT_GDB SURVEYED_AREAS_FLAT_AllYears_CONUS_Rgn5 \
    -select $SURVEY_FIELDS \
    -simplify 100 \
    -makevalid

echo "Now upload the shapefiles to your earth engine account"
echo "and see notebooks/make_ads_images.ipynb for the next step."
exit 0
