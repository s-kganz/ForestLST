#!/bin/bash

SURVEY_FIELDS="SURVEY_YEAR,IDS_DATA_SOURCE"
INPUT_DIRECTORY=data_in/ads
OUTPUT_DIRECTORY=data_working
DAMAGE_TEMP=$INPUT_DIRECTORY/damage_temp
SURVEY_TEMP=$INPUT_DIRECTORY/survey_temp

mkdir -p $DAMAGE_TEMP
mkdir -p $SURVEY_TEMP

for f in $INPUT_DIRECTORY/*.gdb; do
    bname=$(basename -- $f)
    region=${bname//[^0-9]/}
    damage_layer=DAMAGE_AREAS_FLAT_AllYears_CONUS_Rgn${region}
    survey_layer=SURVEYED_AREAS_FLAT_AllYears_CONUS_Rgn${region}

    damage_sql="SELECT SHAPE,SURVEY_YEAR,PERCENT_MID,ifnull(PERCENT_MID, 100) AS SEVERITY FROM $damage_layer WHERE DCA_CODE/1000 = 11 OR DCA_CODE/1000 = 15 OR DCA_CODE=50003"
    
    ogr2ogr $DAMAGE_TEMP/$bname $f -dialect SQLite -sql "$damage_sql" -nln damage
    ogr2ogr $SURVEY_TEMP/$bname $f $survey_layer -select $SURVEY_FIELDS
done

damage_files=$(find $DAMAGE_TEMP -maxdepth 1 -mindepth 1)
survey_files=$(find $SURVEY_TEMP -maxdepth 1 -mindepth 1)

echo $damage_files
echo $survey_files

ogrmerge -o $OUTPUT_DIRECTORY/damage_merged.gdb $damage_files -single -overwrite_ds
ogrmerge -o $OUTPUT_DIRECTORY/survey_merged.gdb $survey_files -single -overwrite_ds

rm -rf $DAMAGE_TEMP
rm -rf $SURVEY_TEMP