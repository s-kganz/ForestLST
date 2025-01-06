from osgeo import gdal, ogr, osr
from osgeo_utils import gdal_calc
import numpy as np
import os
ogr.UseExceptions()

input_tcc = "data_in/nlcd/nlcd_tcc_conus_2021_v2021-4.tif"
output_dir = "data_working"
template_raster = "data_working/damage_rasters/2000.tif"
template_srs = "EPSG:3857"
threshold = 30 # percent cover

# calculate output extent
src = gdal.Open(template_raster)
xmin, xres, xskew, ymax, yskew, yres  = src.GetGeoTransform()
xmax = xmin + (src.RasterXSize * xres)
ymin = ymax + (src.RasterYSize * yres)

print("Coarsening")
gdal.Warp(
    "temp_forest_cover.tif",
    input_tcc,
    format="GTiff",
    outputBounds=(xmin, ymin, xmax, ymax),
    xRes=1000,
    yRes=1000,
    dstSRS=template_srs,
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
    outfile=os.path.join(output_dir, "forest_cover.tif"),
    creation_options=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
    overwrite=True
)

print("Thresholding")
gdal_calc.Calc(
    calc=f"np.logical_and(a > {threshold}, a <= 100)",
    user_namespace={"np": np},
    a=os.path.join(output_dir, "forest_cover.tif"),
    outfile=os.path.join(output_dir, "forest_mask.tif"),
    type="Byte",
    overwrite=True,
    creation_options=["BIGTIFF=YES", "COMPRESS=DEFLATE", "NBITS=1"]
)

print("Deleting temporary data")
os.remove("temp_forest_cover.tif")