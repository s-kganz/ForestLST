from osgeo import gdal, ogr, osr
from osgeo_utils import gdal_calc
import numpy as np
import os
ogr.UseExceptions()

genus = {
    "abies": "b10",
    "picea": "b90",
    #"pinus": "b152",
    "populus": "b750",
    "pseudotsuga": "b202",
    "tsuga": "b260"
}

input_gdb = "data_in/nidrm/L48_BA.gdb"
input_totals_gdb = "data_in/nidrm/L48_Totals.gdb"
output_dir = "data_working/genus_basal_area/"
forest_mask = "data_working/forest_mask.tif"
template_raster = "data_working/damage_rasters/2000.tif"
template_srs = "EPSG:3857"

# calculate output extent
src = gdal.Open(template_raster)
xmin, xres, xskew, ymax, yskew, yres  = src.GetGeoTransform()
xmax = xmin + (src.RasterXSize * xres)
ymin = ymax + (src.RasterYSize * yres)

for genus, raster in genus.items():
    break
    print("Now coarsening", genus)
    print("Setting nodata")
    # Set nan values correctly
    gdal_calc.Calc(
        calc="np.where(a > 0, b, -32768)",
        user_namespace={"np": np},
        NoDataValue=-32768,
        a=forest_mask,
        b=f'OpenFileGDB:{input_gdb}:{raster}',
        outfile="temp_nanset.tif",
        type="Int16",
        overwrite=True
    )
    print("Warping")
    gdal.Warp(
        os.path.join(output_dir, f"{genus}.tif"),
        "temp_nanset.tif",
        format="GTiff",
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=1000,
        yRes=1000,
        dstSRS=template_srs,
        srcNodata=-32768,
        dstNodata=-32768,
        creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
        outputType=gdal.GDT_Int16,
        resampleAlg=gdal.GRA_Average
    )

# os.remove("temp_nanset.tif")

print("Making forest mask")
gdal.Warp(
    f'OpenFileGDB:{input_totals_gdb}:tf',
    forest_mask,
    format="GTiff",
    outputBounds=(xmin, ymin, xmax, ymax),
    xRes=1000,
    yRes=1000,
    dstSRS=template_srs,
    srcNodata=-32768,
    dstNodata=-32768,
    creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
    outputType=gdal.GDT_Int8,
    resampleAlg=gdal.GRA_Average
)

        