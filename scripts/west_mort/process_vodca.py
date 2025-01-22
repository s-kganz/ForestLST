import xarray as xr
import rasterio
import rioxarray
import numpy as np
import tempfile

import os
from glob import glob

input_zip = "data_in/vodca/VODCA_X-band_1997-2018_v01.0.0.zip"
temp_dir  = "data_working/vodca_summer_vod/"
template  = xr.open_dataset("data_working/damage_rasters/2010.tif")
output    = "data_working/summer_vod.nc"

# Make sure the input zip archive is there
if not os.path.isfile(input_zip):
    print(input_zip, "not found.")
    yn = input("Download VODCA X-band archive? (y/N) ")
    if yn.lower() == 'y':
        os.system(f"wget https://zenodo.org/records/2575599/files/VODCA_X-band_1997-2018_v01.0.0.zip -o {input_zip}")
        assert(os.path.isfile(input_zip))
    else:
        print("Aborting")
        import sys; sys.exit()
else:
    print("Found zip archive")

# Only extract the summer images
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
    os.system(f"/bin/bash -c 'unzip {input_zip} *-{{07,08,09}}-*.nc -d {temp_dir}'")

    # Read everything into a dataset
    print("Reading mfdataset")
    files = glob(os.path.join(temp_dir, "**", "*.nc"), recursive=True)
    assert(len(files) > 0)
    
    all_summer = xr.open_mfdataset(files, compat="override", coords="minimal")
    
    print("Clipping")
    xmin, ymin, xmax, ymax = template.rio.transform_bounds(4326)
    all_summer_clip = all_summer.sel(lon=slice(xmin, xmax), lat=slice(ymax, ymin))
    
    print("Calculating annual median")
    # We call .compute() on this array because we want it all in memory before we do the reproject call
    ann_median = all_summer_clip.resample(time="1YE").median().compute()
    
    print("Calulating delta vod")
    ann_median["vod_shift"] = ann_median.vod.shift(time=1)
    ann_median["delta_vod"] = (ann_median["vod_shift"] - ann_median["vod"]) / (ann_median["vod_shift"])
    
    print("Reprojecting")
    # Only grab the stuff we care about, cast as int16 so we can reproject in memory. Multiplying
    # by 1000 reduces the precision lost in the cast.
    ann_median_reproj = (ann_median[["vod", "delta_vod"]]*1000).fillna(-32768)\
        .astype(np.int16)\
        .rename(lon="x", lat="y")\
        .rio.write_crs(4326)\
        .rio.reproject_match(template)
    
    print("Saving output")
    ann_median_reproj.to_netcdf(output)
