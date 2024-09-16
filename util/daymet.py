import earthaccess
import xarray as xr
import rioxarray
import rasterio
import numpy as np

from util.const import LCC_PROJ
from typing import Callable

def _do_daymet_function(
    start: str,
    end: str,
    op_func: Callable, 
    template: xr.Dataset,
    filter_func: Callable=lambda x: True,
    access_args=dict(),
    xr_args=dict()) -> xr.Dataset:
    '''
    This function implements a common workflow with earthaccess
    calls. Specifically it does the following steps:
    
     - Calls earthaccess.search_data with access_args
     - Filters the resulting granules with filter_func
     - Opens the remaining granules with xr.open_mfdataset(**xr_args)
     - Clips the dataset to the (start, end) and the bounds of the template
     - Passes the clipped dataset to op_func
     - Reprojects the final dataset to match the template. Note that
         op_func must return a dataset with Dataset.rio.crs set.

    Note that the final reprojection step will load *all* the data
    into memory, so it is important that op_func is designed
    to reduce the memory overhead in the projection step.
    '''
    granules = earthaccess.search_data(
        short_name="Daymet_Monthly_V4R1_2131",
        temporal=(start, end),
        bounding_box=template.rio.transform_bounds(4326),
        **access_args
    )
    granules_filter = list(filter(filter_func, granules))
    # print("After filtering:", len(granules_filter))

    # Set coords="minimal" for safety that we don't pull the
    # whole array into memory. And compat="override" to avoid
    # numerical comparison nonsense.
    ds = xr.open_mfdataset(
        earthaccess.open(granules_filter), 
        coords="minimal",
        compat="override",
        chunks=dict(time=12, x="auto", y="auto"),
        **xr_args
    )

    ds_clip = _clip_daymet_dataset(ds, template).sel(time=slice(start, end))

    ds_op = op_func(ds_clip)
    if ds_op.rio.crs is None:
        ds_op.rio.write_crs(LCC_PROJ, inplace=True)

    return ds_op.rio.reproject_match(template)
    
def _clip_daymet_dataset(ds: xr.Dataset, template: xr.Dataset) -> xr.Dataset:
    '''
    Clip a Daymet dataset in the lambert conformal conic projection to the
    bounds of another dataset.
    '''
    lcc_xmin, lcc_ymin, lcc_xmax, lcc_ymax = template.rio.transform_bounds(LCC_PROJ)
    return ds.sel(x=slice(lcc_xmin, lcc_xmax), y=slice(lcc_ymax, lcc_ymin))

def summer_mean_vp(year: int, template: xr.Dataset) -> xr.DataArray:
    '''
    Calculate summer mean vapor pressure.
    '''
    start = f"{year}-06-01"
    end = f"{year}-09-01"

    def op(x):
        return x.vp.mean(dim="time").astype(np.int16)\
            .drop_vars(["lat", "lon"])

    def granule_filter(g):
        return (
            "vp" in g["meta"]["native-id"] and 
            g["meta"]["native-id"].endswith("nc")
        )
        
    return _do_daymet_function(
        start, end, op, template, granule_filter
    )

def minimum_winter_air_temperature(year: int, template: xr.Dataset) -> xr.DataArray:
    '''
    Calculate minimum monthly mean winter air temperature.
    '''
    start = f"{year-1}-12-01"
    end = f"{year}-03-01"
    
    def op(x):
        return x.tmin.min(dim="time").astype(np.int16)\
            .drop_vars(["lat", "lon"])

    def granule_filter(g):
        return (
            "tmin" in g["meta"]["native-id"] and 
            g["meta"]["native-id"].endswith("nc")
        )
        
    return _do_daymet_function(
        start, end, op, template, granule_filter
    )

def water_year_ppt(year: int, template: xr.Dataset) -> xr.DataArray:
    '''
    Calculate total water year precipitation.
    '''
    start = f"{year-1}-10-01"
    end = f"{year}-10-01"

    def op(x):
        return x.prcp.sum(dim="time").astype(np.int16)\
            .drop_vars(["lat", "lon"])

    def granule_filter(g):
        return (
            "prcp" in g["meta"]["native-id"] and 
            g["meta"]["native-id"].endswith("nc")
        )
        
    return _do_daymet_function(
        start, end, op, template, granule_filter
    )
