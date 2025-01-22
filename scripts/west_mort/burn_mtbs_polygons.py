from osgeo import gdal, ogr, osr
ogr.UseExceptions()
import os

import xarray as xr
import rioxarray

# Read data
MTBS_PATH   = "data_in/mtbs/mtbs_perims_DD.shp"
OUTPUT_DIR  = "data_working/mtbs_rasters"
TEMPLATE    = "data_working/damage_rasters/2010.tif"
TEMPLATE_CRS = "EPSG:3857"

mtbs_ds     = gdal.OpenEx(MTBS_PATH, 0)
mtbs_layer  = mtbs_ds.GetLayerByIndex(0)

# Determine output extent in the input CRS
src_xmin, src_xmax, src_ymin, src_ymax = mtbs_layer.GetExtent()
print("Input extent in input CRS:", src_xmin, src_ymin, src_xmax, src_ymax)

# Determine output extent in the output CRS
out_xmin, out_ymin, out_xmax, out_ymax = xr.open_dataset(TEMPLATE).rio.bounds()
print("Output extent in output CRS:", out_xmin, out_xmax, out_ymin, out_ymax)

# GDAL likes to silently fail so spam a bunch of 
# asserts to make sure we are ok.
assert(mtbs_ds is not None)
assert(mtbs_layer is not None)

print("Finished reading data")

years = list(range(1997, 2024))

for y in years:
    print("Now burning", y)
    sql = f"SELECT * FROM mtbs_perims_DD WHERE CAST(strftime('%Y', Ig_Date) as INT) = {y}"
    mtbs_subset = mtbs_ds.ExecuteSQL(sql, dialect="SQLITE")
    mtbs_count = mtbs_subset.GetFeatureCount()
    if mtbs_count == 0:
        print(f"Survey or damage polygons are empty for year {y}%"
              f"Skipping this year.")
        continue

    print("Fine burn...")
    fine_burn = gdal.Rasterize(
        f"temp_fine_burn_{y}.tif",
        mtbs_ds,
        SQLStatement=sql,
        SQLDialect="SQLITE",
        # Note we are thinking in degrees here
        xRes=1.0/1000,
        yRes=1.0/1000,
        #noData=-1,
        #allTouched=True,
        initValues=[0],
        burnValues=[1],
        outputBounds=[src_xmin, src_ymin, src_xmax, src_ymax],
        outputSRS=mtbs_ds.GetLayerByIndex(0).GetSpatialRef(),
        creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
        outputType=gdal.GDT_Int8
    )

    fine_burn = None

    print("Reproject...")
    warped = gdal.Warp(
        os.path.join(OUTPUT_DIR, f"{y}.tif"),
        f"temp_fine_burn_{y}.tif",
        format="GTiff",
        xRes=1000,
        yRes=1000,
        outputBounds=[out_xmin, out_ymin, out_xmax, out_ymax],
        outputBoundsSRS=TEMPLATE_CRS,
        dstSRS=TEMPLATE_CRS,
        creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
        outputType=gdal.GDT_Int8,
        resampleAlg=gdal.GRA_Average
    )
    warped = None
    
    # Delete the temporary raster
    os.remove(f"temp_fine_burn_{y}.tif")
