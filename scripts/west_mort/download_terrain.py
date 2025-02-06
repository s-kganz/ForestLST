'''
This file downloads ASTER DTM data through NASA Earthdata. rasterio throws
a read error when using earthaccess.open, so the tiles are downloaded
to a temporary directory and merged with GDAL.
'''

from osgeo import gdal
import os
import sys
import glob
import tempfile
import xarray as xr
import numpy as np
import rioxarray
import rasterio
import xrspatial

# Set up earthaccess
import earthaccess
assert(earthaccess.login(strategy="netrc").authenticated)

# Other parameters
OUTPUT = "data_working/terrain.nc"
TEMPLATE = xr.open_dataset("data_working/damage_rasters/2010.tif", engine="rasterio")

dem_granules = earthaccess.search_data(
    short_name="ASTGTM",
    bounding_box=TEMPLATE.rio.transform_bounds(4326)
)
print("Found", len(dem_granules), "granules")

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tempdir:
    print("Downloading tiles to", tempdir)
    # Download all the tiles
    earthaccess.download(dem_granules, local_path=tempdir)
    # Merge them together
    files_to_merge = glob.glob(os.path.join(tempdir, "*dem.tif"))
    print("Merging", len(files_to_merge), "files")
    merged = "data_working/elev_merged.tif"
    # Have to coarsen here otherwise the file will be way too big
    g = gdal.Warp(merged, files_to_merge, 
                  xRes=4000, yRes=4000, format="GTiff",
                  dstSRS="EPSG:3857")
    g = None
    # Reproject to match the target
    print("Reprojecting")
    dem = xr.open_dataset(merged, engine="rasterio").squeeze().rename(band_data="elev")
    dem_reproj = dem.rio.reproject_match(TEMPLATE, resampling=rasterio.enums.Resampling.bilinear)
    # Calculate slope, northness, and eastness
    print("Calculating slope, northness, and eastness")
    dem_reproj["slope"] = xrspatial.slope(dem_reproj.elev)
    aspect = xrspatial.aspect(dem_reproj.elev)
    dem_reproj["northness"] = np.sin(aspect * np.pi / 180).fillna(0) * 100 # range 0-100
    dem_reproj["eastness"] = np.cos(aspect * np.pi / 180).fillna(0) * 100 # range 0-100
    dem_reproj["slope"] = dem_reproj.slope.fillna(0) # range 0-90

    # Set fill values and save
    print("Saving output to", OUTPUT)
    dem_reproj.elev.attrs["_FillValue"] = 65535
    dem_reproj.slope.attrs["_FillValue"] = 0
    dem_reproj.northness.attrs["_FillValue"] = 0
    dem_reproj.eastness.attrs["_FillValue"] = 0
    
    dem_reproj.to_netcdf(
        OUTPUT,
        encoding = {
            "elev": {"dtype": "int16"},
            "slope": {"dtype": "int8"},
            "northness": {"dtype": "int8"},
            "eastness": {"dtype": "int8"}
        }
    )
