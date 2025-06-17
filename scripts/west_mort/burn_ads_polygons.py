from osgeo import gdal, ogr, osr
from osgeo_utils import gdal_calc
import numpy as np
ogr.UseExceptions()

from pyproj import Transformer
from pyproj.crs import CRS

import os

if 'snakemake' in globals():
    from snakemake.script import snakemake
    # Inputs
    SURVEY_PATH = os.path.join(snakemake.config["data_working"], "survey_merged.gdb")
    DAMAGE_PATH = os.path.join(snakemake.config["data_working"], "damage_merged.gdb")
    TCC_PATH = os.path.join(snakemake.config["data_in"], "nlcd", "nlcd_tcc_conus_2021_v2021-4.tif")
    # Outputs
    OUTPUT_DIR = snakemake.config["data_working"]
    DAMAGE_DIR = os.path.join(OUTPUT_DIR, "damage_rasters")
    FAM_DIR = os.path.join(OUTPUT_DIR, "fam_rasters")
    # Resolution/extent/srs
    COARSE_RES = int(snakemake.config["resolution"])
    OUT_SREF = snakemake.config["srs"]
    xmin = float(snakemake.config["xmin"])
    ymin = float(snakemake.config["ymin"])
    xmax = float(snakemake.config["xmax"])
    ymax = float(snakemake.config["ymax"])
    years = list(range(int(snakemake.config["year_start"]), int(snakemake.config["year_end"])+1))
else:
    raise RuntimeError("Not running in snakemake pipeline!")

def get_authority_code(layer: ogr.Layer):
    sref = layer.GetSpatialRef()
    auth = sref.GetAuthorityName(None)
    code = sref.GetAuthorityCode(None)
    return auth + ":" + code

# Processing settings
TCC_THRESHOLD = 0 # pixel percent cover to count as forested
FINE_RES      = 25 # m, initial rasterization resolution

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
    # Nodata is also not forest
    dstNodata=-1,
    creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE", "PIXELTYPE=SIGNEDBYTE"],
    outputType=gdal.GDT_Int8,
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
    calc=f"np.logical_and(a >= {TCC_THRESHOLD}, a <= 100)",
    user_namespace={"np": np},
    a=os.path.join(OUTPUT_DIR, "forest_cover.tif"),
    outfile=os.path.join(OUTPUT_DIR, "forest_mask.tif"),
    type="Byte",
    overwrite=True,
    creation_options=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
    NoDataValue=0
)

print("Deleting temporary forest cover")
os.remove("temp_forest_cover.tif")

print("Burning damage rasters")
# Get list of unique years
sql = 'SELECT DISTINCT SURVEY_YEAR FROM merged'
query = survey_ds.ExecuteSQL(sql)

for y in years:
    print("Now burning", y)
    sql = f'SELECT * FROM merged WHERE SURVEY_YEAR={y}'
    damage_subset = damage_ds.ExecuteSQL(sql)
    survey_subset = survey_ds.ExecuteSQL(sql)
    d_count = damage_subset.GetFeatureCount()
    s_count = survey_subset.GetFeatureCount()
    print("# Damage:", d_count)
    print("# Survey:", s_count)
    if d_count == 0 or s_count == 0:
        print(f"Survey or damage polygons are empty for year {y}! "
              f"Skipping this year.")
        continue

    # Initial fine burn
    gdal.Rasterize(
        f"temp_damage_burn_fine_{y}.tif",
        damage_ds,
        xRes=FINE_RES,
        yRes=FINE_RES,
        allTouched=True,
        where=f"SURVEY_YEAR={y}",
        attribute="SEVERITY",
        initValues=[0],
        outputBounds=[xmin, ymin, xmax, ymax],
        outputSRS=OUT_SREF,
        creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
        outputType=gdal.GDT_Int8,
    )

    # Coarsen
    gdal.Warp(
        f"temp_damage_burn_coarse_{y}.tif",
        f"temp_damage_burn_fine_{y}.tif",
        format="GTiff",
        xRes=COARSE_RES,
        yRes=COARSE_RES,
        dstSRS=OUT_SREF,
        creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
        outputType=gdal.GDT_Int8,
        resampleAlg=gdal.GRA_Average,
    )

    gdal.Rasterize(
        f"temp_survey_burn_{y}.tif",
        survey_ds,
        xRes=COARSE_RES,
        yRes=COARSE_RES,
        allTouched=True,
        where=f"SURVEY_YEAR={y}",
        burnValues=[1],
        initValues=[0],
        outputBounds=[xmin, ymin, xmax, ymax],
        outputSRS=OUT_SREF,
        creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
        outputType=gdal.GDT_Int8,
    )

    # If mortality was > 0, write the value. If survey was > 0, write zero. Otherwise, write nodata.
    gdal_calc.Calc(
        calc="np.where(a > 0, a, np.where(b > 0, 0, -1))",
        a=f"temp_damage_burn_coarse_{y}.tif",
        b=f"temp_survey_burn_{y}.tif",
        user_namespace={"np": np},
        outfile=f"temp_damage_burn_coarse_survey_mask{y}.tif",
        type="Int8",
        overwrite=True,
        creation_options=["BIGTIFF=YES", "COMPRESS=DEFLATE"]
    )

    # Mask out nonforest pixels
    damage = gdal_calc.Calc(
        calc="np.where(a > 0, b, -1)",
        a=os.path.join(OUTPUT_DIR, "forest_mask.tif"),
        b=f"temp_damage_burn_coarse_survey_mask{y}.tif",
        user_namespace={"np": np},
        outfile=os.path.join(DAMAGE_DIR, f"{y}.tif"),
        type="Int8",
        overwrite=True,
        creation_options=["BIGTIFF=YES", "COMPRESS=DEFLATE"]
    )

    # NoData isn't properly set so we have to do that ourselves
    rb = damage.GetRasterBand(1)
    rb.SetNoDataValue(-1)
    
    damage, rb = None, None

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
    
    # Delete the temporary rasters
    os.remove(f"temp_damage_burn_fine_{y}.tif")
    os.remove(f"temp_damage_burn_coarse_{y}.tif")
    os.remove(f"temp_survey_burn_{y}.tif")
    os.remove(f"temp_damage_burn_coarse_survey_mask{y}.tif")
