import earthaccess
import xarray as xr
import rioxarray
import rasterio
import numpy as np

from util.const import LCC_PROJ

def daymet_summer_mean_vp(year: int, template: xr.Dataset) -> xr.DataArray:
    '''
    Calculate summer mean vapor pressure.
    '''
    start = f"{year}-06-01"
    end = f"{year}-09-01"

    lon_min, lat_min, lon_max, lat_max = template.rio.transform_bounds(4326)
    lcc_xmin, lcc_ymin, lcc_xmax, lcc_ymax = template.rio.transform_bounds(LCC_PROJ)
    
    granules = earthaccess.search_data(
        short_name="Daymet_Monthly_V4R1_2131",
        temporal=(start, end),
        bounding_box=(lon_min, lat_min, lon_max, lat_max)
    )

    vp_granules = [
        g for g in granules if
        "vp" in g["meta"]["native-id"] and g["meta"]["native-id"].endswith("nc")
    ]

    # We have to reproject this to match the template raster, but rio.reproject_match
    # will load the whole array into memory. We can mitigate this problem by waiting
    # until the prcp sum and clipping the data before the reproject.
    # https://github.com/corteva/rioxarray/discussions/222
    vp_ds = xr.open_mfdataset(earthaccess.open(vp_granules))

    vp_mean = vp_ds.sel(
        time=slice(start, end),
        x=slice(lcc_xmin, lcc_xmax),
        y=slice(lcc_ymax, lcc_ymin)
    ).vp.mean(dim="time").astype(np.int16)

    return vp_mean.drop_vars(["lat", "lon"])\
        .rio.write_crs(LCC_PROJ)\
        .rio.reproject_match(template, resampling=rasterio.enums.Resampling.nearest)

def daymet_minimum_winter_air_temperature(year: int, template: xr.Dataset) -> xr.DataArray:
    '''
    Calculate minimum winter air temperature.
    '''
    start = f"{year-1}-12-01"
    end = f"{year}-03-01"

    lon_min, lat_min, lon_max, lat_max = template.rio.transform_bounds(4326)
    lcc_xmin, lcc_ymin, lcc_xmax, lcc_ymax = template.rio.transform_bounds(LCC_PROJ)
    
    granules = earthaccess.search_data(
        short_name="Daymet_Monthly_V4R1_2131",
        temporal=(start, end),
        bounding_box=(lon_min, lat_min, lon_max, lat_max)
    )

    tmin_granules = [
        g for g in granules if
        "tmin" in g["meta"]["native-id"] and g["meta"]["native-id"].endswith("nc")
    ]

    # We have to reproject this to match the template raster, but rio.reproject_match
    # will load the whole array into memory. We can mitigate this problem by waiting
    # until the prcp sum and clipping the data before the reproject.
    # https://github.com/corteva/rioxarray/discussions/222
    tmin_ds = xr.open_mfdataset(earthaccess.open(tmin_granules))

    tmin_min = tmin_ds.sel(
        time=slice(start, end),
        x=slice(lcc_xmin, lcc_xmax),
        y=slice(lcc_ymax, lcc_ymin)
    ).tmin.min(dim="time").astype(np.int16)

    return tmin_min.drop_vars(["lat", "lon"])\
        .rio.write_crs(LCC_PROJ)\
        .rio.reproject_match(template, resampling=rasterio.enums.Resampling.nearest)

def daymet_water_year_ppt(year: int, template: xr.Dataset) -> xr.DataArray:
    '''
    Calculate water year precipitation.
    '''
    start = f"{year-1}-10-01"
    end = f"{year}-10-01"

    lon_min, lat_min, lon_max, lat_max = template.rio.transform_bounds(4326)
    lcc_xmin, lcc_ymin, lcc_xmax, lcc_ymax = template.rio.transform_bounds(LCC_PROJ)
    
    granules = earthaccess.search_data(
        short_name="Daymet_Monthly_V4R1_2131",
        temporal=(start, end),
        bounding_box=(lon_min, lat_min, lon_max, lat_max)
    )

    prcp_granules = [
        g for g in granules if
        "prcp" in g["meta"]["native-id"] and g["meta"]["native-id"].endswith("nc")
    ]

    # We have to reproject this to match the template raster, but rio.reproject_match
    # will load the whole array into memory. We can mitigate this problem by waiting
    # until the prcp sum and clipping the data before the reproject.
    # https://github.com/corteva/rioxarray/discussions/222
    prcp_ds = xr.open_mfdataset(earthaccess.open(prcp_granules))

    prcp_sum = prcp_ds.sel(
        time=slice(start, end),
        x=slice(lcc_xmin, lcc_xmax),
        y=slice(lcc_ymax, lcc_ymin)
    ).prcp.sum(dim="time").astype(np.int16)

    return prcp_sum.drop_vars(["lat", "lon"])\
        .rio.write_crs(LCC_PROJ)\
        .rio.reproject_match(template, resampling=rasterio.enums.Resampling.nearest)