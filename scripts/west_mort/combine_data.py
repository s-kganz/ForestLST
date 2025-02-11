import xarray as xr
import rioxarray
import rasterio
import numpy as np
import dask
import os
from dask.distributed import Client
#client = Client()
#client
dask.config.set(scheduler="synchronous")

CHUNK_SCHEMA = dict(time=27, x=512, y=512)
CA_BOUNDS_3857 = (-13846000.0, 3842000.0, -12950000.0, 5162000.0)

# Define several functions to parse intermediate data
def preprocess_mortality(ds):
    year = os.path.basename(ds.encoding["source"])
    year = int(year.replace(".tif", ""))
    return ds.squeeze(drop=True)\
        .drop_vars("spatial_ref")\
        .rename(band_data="mortality")\
        .assign_coords(time=year)

def preprocess_fam(ds):
    year = os.path.basename(ds.encoding["source"])
    year = int(year.replace(".tif", ""))
    return ds.squeeze(drop=True)\
        .drop_vars("spatial_ref")\
        .rename(band_data="fam")\
        .assign_coords(time=year)

def preprocess_fire(ds):
    year = os.path.basename(ds.encoding["source"])
    year = int(year.replace(".tif", ""))
    return ds.squeeze(drop=True)\
        .drop_vars("spatial_ref")\
        .rename(band_data="fire")\
        .assign_coords(time=year)

if __name__ == "__main__":
    # Read in all the data
    damage = xr.open_mfdataset(
        "data_working/damage_rasters/*.tif",
        preprocess=preprocess_mortality,
        concat_dim="time", 
        combine="nested"
    ).chunk(**CHUNK_SCHEMA)
    
    fam = xr.open_mfdataset(
        "data_working/fam_rasters/*.tif",
        preprocess=preprocess_fam,
        concat_dim="time", 
        combine="nested"
    ).chunk(**CHUNK_SCHEMA)

    treecover = xr.open_dataset("data_working/forest_cover.tif")\
        .squeeze(drop=True)\
        .drop_vars(["spatial_ref", "band"])\
        .rename(band_data="treecover")\
        .expand_dims(time=damage.time.shape[0])\
        .assign_coords(time=damage.time)\
        .chunk(**CHUNK_SCHEMA)

    daymet = xr.open_mfdataset('data_working/daymet/*.nc', concat_dim="time", combine="nested")\
        .drop_vars("spatial_ref")\
        .chunk(**CHUNK_SCHEMA)

    fire = xr.open_mfdataset(
        "data_working/mtbs_rasters/*.tif",
        preprocess=preprocess_fire,
        concat_dim="time", combine="nested"
    ).chunk(**CHUNK_SCHEMA)

    terrain = xr.open_dataset("data_working/terrain.nc")\
        .drop_vars(["spatial_ref", "band"])\
        .expand_dims(time=damage.time.shape[0])\
        .assign_coords(time=damage.time)\
        .chunk(**CHUNK_SCHEMA)

    vod = xr.open_dataset("data_working/summer_vod.nc")\
        .drop_vars("spatial_ref")\
        .chunk(**CHUNK_SCHEMA)
    vod = vod.assign_coords(time=vod.time.dt.year)

    # Merge it
    all_data = xr.combine_by_coords(
        [vod, terrain, fire, daymet, damage, treecover],
        coords="minimal",
        compat="override",
        combine_attrs="drop"
    ).drop_vars(["delta_vod"]).rio.write_crs(3857)
    
    # Define encoding
    int_encoding  = {"dtype": "int16", "_FillValue": -9999}
    byte_encoding = {"dtype": "int8", "_FillValue": -128}
    encoding={
        #"abies": int_encoding,
        #"picea": int_encoding,
        #"populus": int_encoding,
        #"pseudotsuga": int_encoding,
        #"tsuga": int_encoding,
        "vod": int_encoding,
        "elev": int_encoding,
        "slope": int_encoding,
        "northness": int_encoding,
        "eastness": int_encoding,
        "fire": byte_encoding,
        "mortality": byte_encoding,
        #"fam": byte_encoding,
        "prcp": int_encoding,
        "vp": int_encoding,
        "tmin": byte_encoding,
        "treecover": byte_encoding
    }

    # Write output
    all_data.to_netcdf(
        "data_working/westmort.nc",
        encoding=encoding
    )

    # Crop to CA and save
    ca_data = all_data.rio.clip_box(*CA_BOUNDS_3857)
    ca_data.to_netcdf(
        "data_working/camort.nc",
        encoding=encoding
    )
    