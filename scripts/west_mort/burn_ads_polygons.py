from osgeo import gdal, ogr, osr
from osgeo_utils import gdal_calc
import numpy as np
ogr.UseExceptions()

from pyproj import Transformer

import os

def get_authority_code(layer: ogr.Layer):
    sref = layer.GetSpatialRef()
    auth = sref.GetAuthorityName(None)
    code = sref.GetAuthorityCode(None)
    return auth + ":" + code

# Read data
SURVEY_PATH  = "data_working/survey_merged.gdb"
DAMAGE_PATH  = "data_working/damage_merged.gdb"
OUTPUT_DIR   = "data_working/damage_rasters"
FAM_DIR      = "data_working/fam_rasters"
FOREST_COVER = "data_working/forest_cover.tif"

if not os.path.exists(FAM_DIR):
    os.makedirs(FAM_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

survey_ds    = gdal.OpenEx(SURVEY_PATH, 0)
damage_ds    = gdal.OpenEx(DAMAGE_PATH, 0)

survey_layer = survey_ds.GetLayerByIndex(0)
damage_layer = damage_ds.GetLayerByIndex(0)

# GDAL likes to silently fail so spam a bunch of 
# asserts to make sure we are ok.
assert(survey_layer is not None)
assert(damage_layer is not None)
assert(get_authority_code(survey_layer) == get_authority_code(damage_layer))

print("Finished reading data")

# Get extent of survey layer. Damage layer will have 
# smaller extent by definition.
xmin, xmax, ymin, ymax = survey_layer.GetExtent()
print("Projected Extent:", xmin, xmax, ymin, ymax)

# Get list of unique years
sql = 'SELECT DISTINCT SURVEY_YEAR FROM merged'
query = survey_ds.ExecuteSQL(sql)
years = list(int(feat.GetField(0)) for i,feat in enumerate(query))

for y in years:
    print("Now burning", y)
    sql = f'SELECT * FROM merged WHERE SURVEY_YEAR={y}'
    damage_subset = damage_ds.ExecuteSQL(sql)
    survey_subset = survey_ds.ExecuteSQL(sql)
    d_count = damage_subset.GetFeatureCount()
    s_count = survey_subset.GetFeatureCount()
    if d_count == 0 or s_count == 0:
        print(f"Survey or damage polygons are empty for year {y}! "
              f"Skipping this year.")
        continue
    
    fine_burn = gdal.Rasterize(
        f"temp_fine_burn_{y}.tif",
        survey_ds,
        xRes=100,
        yRes=100,
        noData=-1,
        allTouched=True,
        where=f"SURVEY_YEAR={y}",
        burnValues=[0],
        outputBounds=[xmin, ymin, xmax, ymax],
        outputSRS=survey_ds.GetLayerByIndex(0).GetSpatialRef(),
        creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
        outputType=gdal.GDT_Int8
    )

    fine_burn = gdal.Rasterize(
        fine_burn,
        damage_ds,
        allTouched=True,
        where=f"SURVEY_YEAR={y}",
        burnValues=[100]
    )

    output_ref = osr.SpatialReference().SetFromUserInput("EPSG:3857")

    gdal.Warp(
        os.path.join(OUTPUT_DIR, f"{y}.tif"),
        f"temp_fine_burn_{y}.tif",
        format="GTiff",
        xRes=1000,
        yRes=1000,
        dstSRS="EPSG:3857",
        creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
        outputType=gdal.GDT_Int8,
        resampleAlg=gdal.GRA_Average
    )

    gdal_calc.Calc(
        calc="100*np.clip(a / (b+1), a_min=0, a_max=1)",
        a=os.path.join(OUTPUT_DIR, f"{y}.tif"),
        b=FOREST_COVER,
        user_namespace={"np": np},
        outfile=os.path.join(FAM_DIR, f"{y}.tif"),
        type="Int8",
        overwrite=True,
        NoDataValue=-1,
        creation_options=["BIGTIFF=YES", "COMPRESS=DEFLATE"]
    )
    
    # Delete the temporary raster
    os.remove(f"temp_fine_burn_{y}.tif")