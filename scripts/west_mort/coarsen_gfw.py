from osgeo import gdal, ogr, osr
from osgeo_utils import gdal_calc
import numpy as np
ogr.UseExceptions()

from pyproj import Transformer
from pyproj.crs import CRS

import os
import warnings
import subprocess
from glob import glob

if 'snakemake' in globals():
    from snakemake.script import snakemake
    GFW_DIR = os.path.join(snakemake.config["data_in"], "gfw")
    DATA_WORKING = snakemake.config["data_working"]
    DATA_IN = snakemake.config["data_in"]
    RES = int(snakemake.config["resolution"])
    SRS = snakemake.config["srs"]
    xmin = float(snakemake.config["xmin"])
    ymin = float(snakemake.config["ymin"])
    xmax = float(snakemake.config["xmax"])
    ymax = float(snakemake.config["ymax"])
    years = list(range(int(snakemake.config["year_start"]), int(snakemake.config["year_end"])+1))
else:
    warnings.warn("Not running in snakemake pipeline, using default parameters")
    GFW_DIR = os.path.join("data_in", "gfw")
    DATA_WORKING = "data_working"
    DATA_IN = "data_in"
    RES = 1000
    SRS = "EPSG:3857"
    xmin = -13896215.609
    xmax = -11536215.609
    ymin = 3672302.419
    ymax = 6280302.419
    years = list(range(2000, 2022))

print("Downloading global forest watch loss year")
GFW_LINKS = [
    "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_lossyear_50N_130W.tif",
    "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_lossyear_50N_120W.tif",
    "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_lossyear_40N_130W.tif",
    "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_lossyear_40N_120W.tif"
]
GFW_FILES = [os.path.join(GFW_DIR, os.path.basename(link)) for link in GFW_LINKS]
MAX_FIRE_DELAY = 2

if not all(os.path.exists(f) for f in GFW_FILES):
    for link, file in zip(GFW_LINKS, GFW_FILES):
        subprocess.run(["wget", link, "-O", file])

print("Merging tiles")
if not os.path.exists("temp_gfw_lossyear_merge.tif"):
    subprocess.run([
        "gdal_merge",
        "-o", "temp_gfw_lossyear_merge.tif",   
        "-co", "BIGTIFF=YES", 
        "-co", "COMPRESS=DEFLATE"
    ] + GFW_FILES)

# Open MTBS perimeter dataset
mtbs_ds = gdal.OpenEx(os.path.join(DATA_IN, "mtbs", "mtbs_perims_DD.shp"), 0)

print("Annual loss")
for y in years:
    print(y)
    value = y % 2000

    # GFW encodes non-lost pixels as zero. Return a constant zero 
    # for the year 2000.
    expression = f"100 * (a == {value})" if y != 2000 else "0 * a"

    print("Get loss pixels")
    calc = gdal_calc.Calc(
        calc=expression,
        a="temp_gfw_lossyear_merge.tif",
        outfile=f"temp_gfw_loss_{y}.tif",
        type="Int8",
        overwrite=True,
        NoDataValue=-1,
        creation_options=["BIGTIFF=YES", "COMPRESS=DEFLATE"]
    )

    print("Mask out fire")

    # New post-fire canopy losses are substantial 2 years post-fire
    # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025EF006373
    sql = f"""
    SELECT *, CAST(strftime('%Y', Ig_Date) as INT) AS YEAR 
    FROM mtbs_perims_DD 
    WHERE YEAR >= {y-MAX_FIRE_DELAY} AND YEAR <= {y}
    """
    mtbs_subset = mtbs_ds.ExecuteSQL(sql, dialect="SQLITE")

    # Here we don't need to specify any creation options because the 
    # raster already exists.
    burn = gdal.Rasterize(
        calc,
        mtbs_ds,
        SQLStatement=sql,
        SQLDialect="SQLITE",
        burnValues=[0],
    )
    burn = None
    
    print("Coarsen")
    warp = gdal.Warp(
        os.path.join(DATA_WORKING, "gfw_damage", f"{y}.tif"),
        f"temp_gfw_loss_{y}.tif",
        format="GTiff",
        xRes=RES,
        yRes=RES,
        outputBounds=[xmin, ymin, xmax, ymax],
        dstSRS=SRS,
        resampleAlg=gdal.GRA_Average
    )
    warp = None

    os.remove(f"temp_gfw_loss_{y}.tif")

os.remove("temp_gfw_lossyear_merge.tif")




    