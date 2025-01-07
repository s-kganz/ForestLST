from osgeo import gdal, ogr, osr
from osgeo_utils import gdal_calc
import numpy as np
ogr.UseExceptions()

from pyproj import Transformer
from pyproj.crs import CRS

import os

def get_authority_code(layer: ogr.Layer):
    sref = layer.GetSpatialRef()
    auth = sref.GetAuthorityName(None)
    code = sref.GetAuthorityCode(None)
    return auth + ":" + code

# Input data
SURVEY_PATH   = "data_working/survey_merged.gdb"
DAMAGE_PATH   = "data_working/damage_merged.gdb"
TCC_PATH      = "data_in/nlcd/nlcd_tcc_conus_2021_v2021-4.tif"

# Outputs
OUTPUT_DIR    = "data_working"
DAMAGE_DIR    = os.path.join(OUTPUT_DIR, "damage_rasters")
FAM_DIR       = os.path.join(OUTPUT_DIR, "fam_rasters")

# Processing settings
TCC_THRESHOLD = 30 # pixel percent cover to count as forested
FINE_RES      = 100 # m, initial rasterization resolution
COARSE_RES    = 1000 # m, final rasterization resolution
OUT_SREF      = "EPSG:3857"

for d in (OUTPUT_DIR, DAMAGE_DIR, FAM_DIR):
    if not os.path.exists(d):
        os.makedirs(d)

# Open ADS datasets
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

# Get extent of survey layer. This defines the modeling area.
# N.b. the damage layer has a smaller extent.
# !! These coordinates are passed to later functions in a different order !!
xmin, xmax, ymin, ymax = survey_layer.GetExtent()

# First we make the forest cover raster so we can calculate
# FAM along with the damage rasters.
print("Making forest cover raster")
gdal.Warp(
    "temp_forest_cover.tif",
    TCC_PATH,
    format="GTiff",
    outputBounds=[xmin, ymin, xmax, ymax],
    xRes=COARSE_RES,
    yRes=COARSE_RES,
    dstSRS=OUT_SREF,
    srcNodata=255,
    creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
    outputType=gdal.GDT_Byte,
    # Use median resampling so edge pixels don't get weird
    # values for cover.
    resampleAlg=gdal.GRA_Med,
)

# There's still a few weird edge pixels, reclassify them as zero.
print("Reclassify edge pixels")
gdal_calc.Calc(
    calc="np.where(a > 100, 0, a)",
    user_namespace={"np": np},
    a="temp_forest_cover.tif",
    outfile=os.path.join(OUTPUT_DIR, "forest_cover.tif"),
    creation_options=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
    overwrite=True
)

print("Thresholding forest cover")
gdal_calc.Calc(
    calc=f"np.logical_and(a > {TCC_THRESHOLD}, a <= 100)",
    user_namespace={"np": np},
    a=os.path.join(OUTPUT_DIR, "forest_cover.tif"),
    outfile=os.path.join(OUTPUT_DIR, "forest_mask.tif"),
    type="Byte",
    overwrite=True,
    creation_options=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
    NoDataValue=255
)

print("Deleting temporary forest cover")
os.remove("temp_forest_cover.tif")

print("Burning damage rasters")
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
        xRes=FINE_RES,
        yRes=FINE_RES,
        noData=-1,
        allTouched=True,
        where=f"SURVEY_YEAR={y}",
        burnValues=[0],
        outputBounds=[xmin, ymin, xmax, ymax],
        outputSRS=OUT_SREF,
        creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
        outputType=gdal.GDT_Int8,
    )

    fine_burn = gdal.Rasterize(
        fine_burn,
        damage_ds,
        allTouched=True,
        where=f"SURVEY_YEAR={y}",
        attribute="SEVERITY"
    )

    # Reduce resolution
    gdal.Warp(
        os.path.join(DAMAGE_DIR, f"{y}.tif"),
        f"temp_fine_burn_{y}.tif",
        format="GTiff",
        xRes=COARSE_RES,
        yRes=COARSE_RES,
        dstSRS=OUT_SREF,
        creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
        outputType=gdal.GDT_Int8,
        resampleAlg=gdal.GRA_Average,
    )

    # Calculate FAM
    gdal_calc.Calc(
        calc="100*np.clip(a / (b+1), a_min=0, a_max=1)",
        a=os.path.join(DAMAGE_DIR, f"{y}.tif"),
        b=os.path.join(OUTPUT_DIR, "forest_cover.tif"),
        user_namespace={"np": np},
        outfile=os.path.join(FAM_DIR, f"{y}.tif"),
        type="Int8",
        overwrite=True,
        NoDataValue=-1,
        creation_options=["BIGTIFF=YES", "COMPRESS=DEFLATE"]
    )
    
    # Delete the temporary raster
    os.remove(f"temp_fine_burn_{y}.tif")