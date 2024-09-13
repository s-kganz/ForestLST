import os
import xarray as xr
import rioxarray
import rasterio
import warnings

try: 
    import util
except ImportError:
    import sys; sys.path.append("..")
    import util

import dask
dask.config.set(scheduler='synchronous')

template = xr.open_dataset("data_working/damage_rasters/2020.tif")
output_dir = "data_working/daymet"
start_year = 2000
end_year   = 2023

def make_annual_ds(y):
    print(y)
    print("Prcp")
    prcp = util.data_source.daymet_water_year_ppt(y, template)
    print("Vp")
    vp = util.data_source.daymet_summer_mean_vp(y, template)
    print("Tmin")
    tmin = util.data_source.daymet_minimum_winter_air_temperature(y, template)
    print("Annual dataset")
    return xr.Dataset({
        "prcp": prcp,
        "vp": vp,
        "tmin": tmin
    }).expand_dims(time=[y])

if __name__ == "__main__":
    with warnings.catch_warnings(action="ignore"):
        for y in range(start_year, end_year+1):
            annual_ds = make_annual_ds(y)
            print("Writing output")
            annual_ds.to_netcdf(f"{output_dir}/{y}.nc")