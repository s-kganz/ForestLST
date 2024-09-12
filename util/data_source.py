import earthaccess
import xarray as xr
import rioxarray
import rasterio
import numpy as np

from util.const import LCC_PROJ
from typing import Callable

def _filter_earthaccess_granules(granules: list, func: Callable):
    '''
    Subset a list of granules according to a user-defined function.
    Returns granules for which func returns True.
    '''
    return list(filter(func, granules))

def _filter_op_reproj(
    op_func: Callable, template: xr.Dataset,
    source_proj: str=None,
    filter_func: Callable=lambda x: True,
    access_args=dict(),
    xr_args=dict()) -> xr.Dataset:
    '''
    This function implements a common workflow with earthaccess
    calls. Specifically it does the following steps:
    
     - Calls earthaccess.search_data with access_args
     - Filters the resulting granules with filter_func
     - Opens the remaining granules with xr.open_mfdataset(**xr_args)
     - Passes the multifile dataset to op_func
     - Reprojects the final dataset to match the template. If the dataset
         does not have a CRS from earthaccess, source_proj must be set.

    Note that the final reprojection step will load *all* the data
    into memory, so it is important that op_func is designed
    to reduce the memory overhead in the projection step.
    '''
    granules = earthaccess.search_data(**access_args)
    granules_filter = _filter_earthaccess_granules(granules, filter_func)
    print("After filtering:", len(granules_filter))

    ds = xr.open_mfdataset(earthaccess.open(granules_filter), **xr_args)
    ds_op = op_func(ds)

    if ds_op.rio.crs is None:
        ds_op = ds_op.rio.write_crs(source_proj)

    return ds_op.rio.reproject_match(template)
    
def _clip_daymet_dataset(ds: xr.Dataset, template: xr.Dataset) -> xr.Dataset:
    '''
    Clip a Daymet dataset in the lambert conformal conic projection to the
    bounds of another dataset.
    '''
    lcc_xmin, lcc_ymin, lcc_xmax, lcc_ymax = template.rio.transform_bounds(LCC_PROJ)
    return ds.sel(x=slice(lcc_xmin, lcc_xmax), y=slice(lcc_ymax, lcc_ymin))


def daymet_summer_mean_vp(year: int, template: xr.Dataset) -> xr.DataArray:
    '''
    Calculate summer mean vapor pressure.
    '''
    start = f"{year}-06-01"
    end = f"{year}-09-01"

    query = dict(
        short_name="Daymet_Monthly_V4R1_2131",
        temporal=(start, end),
        bounding_box=template.rio.transform_bounds(4326)
    )

    def op(x):
        return _clip_daymet_dataset(x, template)\
            .sel(time=slice(start, end))\
            .vp.mean(dim="time").astype(np.int16)\
            .drop_vars(["lat", "lon"])

    def granule_filter(g):
        return (
            "vp" in g["meta"]["native-id"] and 
            g["meta"]["native-id"].endswith("nc")
        )
        
    return _filter_op_reproj(
        op, template, 
        source_proj=LCC_PROJ, filter_func=granule_filter, 
        access_args=query,
        xr_args=dict(chunks=dict(time=12, x="auto", y="auto"))
    )

def daymet_minimum_winter_air_temperature(year: int, template: xr.Dataset) -> xr.DataArray:
    '''
    Calculate minimum monthly mean winter air temperature.
    '''
    start = f"{year-1}-12-01"
    end = f"{year}-03-01"

    query = dict(
        short_name="Daymet_Monthly_V4R1_2131",
        temporal=(start, end),
        bounding_box=template.rio.transform_bounds(4326)
    )

    def op(x):
        return _clip_daymet_dataset(x, template)\
            .sel(time=slice(start, end))\
            .tmin.min(dim="time").astype(np.int16)\
            .drop_vars(["lat", "lon"])

    def granule_filter(g):
        return (
            "tmin" in g["meta"]["native-id"] and 
            g["meta"]["native-id"].endswith("nc")
        )
        
    return _filter_op_reproj(
        op, template, 
        source_proj=LCC_PROJ, filter_func=granule_filter, 
        access_args=query,
        xr_args=dict(chunks=dict(time=12, x="auto", y="auto"))
    )

def daymet_water_year_ppt(year: int, template: xr.Dataset) -> xr.DataArray:
    '''
    Calculate total water year precipitation.
    '''
    start = f"{year-1}-10-01"
    end = f"{year}-10-01"

    query = dict(
        short_name="Daymet_Monthly_V4R1_2131",
        temporal=(start, end),
        bounding_box=template.rio.transform_bounds(4326)
    )

    def op(x):
        return _clip_daymet_dataset(x, template)\
            .sel(time=slice(start, end))\
            .prcp.sum(dim="time").astype(np.int16)\
            .drop_vars(["lat", "lon"])

    def granule_filter(g):
        return (
            "prcp" in g["meta"]["native-id"] and 
            g["meta"]["native-id"].endswith("nc")
        )
        
    return _filter_op_reproj(
        op, template, 
        source_proj=LCC_PROJ, filter_func=granule_filter, 
        access_args=query,
        xr_args=dict(chunks=dict(time=12, x="auto", y="auto"))
    )
