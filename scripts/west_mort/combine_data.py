import xarray as xr
import rioxarray
import rasterio
import numpy as np
import dask
import os
dask.config.set(scheduler="synchronous")
#client = Client()
#client

if 'snakemake' in globals():
    from snakemake.script import snakemake
    out_nc = snakemake.config["final_output"]
else:
    out_nc = "mortality_dataset.nc"

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

def preprocess_ba(ds):
    genus = os.path.basename(ds.encoding["source"]).replace(".tif", "")
    return ds.squeeze(drop=True).rename(band_data=genus)

if __name__ == "__main__":
    # Read in all the data.
    # Floating point imprecision leads to *very* slight differences
    # in x and y coordinates among the intermediate data. BUT they all
    # have the same width/eheight. So we assign x/y coords based on mortality
    # data.
    damage = xr.open_mfdataset(
        "data_working/damage_rasters/*.tif",
        preprocess=preprocess_mortality,
        concat_dim="time", 
        combine="nested"
    )

    '''
    fam = xr.open_mfdataset(
        "data_working/fam_rasters/*.tif",
        preprocess=preprocess_fam,
        concat_dim="time", 
        combine="nested"
    ).assign_coords(
        x=damage.x,
        y=damage.y
    )
    '''

    treecover = xr.open_dataset("data_working/forest_cover.tif")\
        .squeeze(drop=True)\
        .drop_vars(["spatial_ref"])\
        .rename(band_data="treecover")\
        .assign_coords(
            x=damage.x,
            y=damage.y
        )

    daymet = xr.open_mfdataset('data_working/daymet/*.nc', concat_dim="time", combine="nested")\
        .drop_vars("spatial_ref")\
        .assign_coords(
            x=damage.x,
            y=damage.y
        )

    fire = xr.open_mfdataset(
        "data_working/mtbs_rasters/*.tif",
        preprocess=preprocess_fire,
        concat_dim="time", combine="nested"
    ).assign_coords(
        x=damage.x,
        y=damage.y
    )

    terrain = xr.open_dataset("data_working/terrain.nc")\
        .drop_vars(["spatial_ref", "band"])\
        .assign_coords(
            x=damage.x,
            y=damage.y
        )

    '''
    vod = xr.open_dataset("data_working/summer_vod.nc")\
        .drop_vars("spatial_ref")
    
    vod = vod.assign_coords(
        time=vod.time.dt.year,
        x=damage.x,
        y=damage.y
    )
    '''

    genus_ba = xr.open_mfdataset(
        "data_working/genus_basal_area/*.tif",
        preprocess=preprocess_ba
    )\
        .assign_coords(
            x=damage.x,
            y=damage.y
        )

    gfw = xr.open_mfdataset(
        "data_working/gfw_damage/*.tif",
        preprocess=preprocess_mortality,
        concat_dim="time", 
        combine="nested"
    )\
        .assign_coords(
            x=damage.x,
            y=damage.y
        )\
        .rename(mortality="gfw_damage")

    # Merge it
    all_data = xr.combine_by_coords(
        [terrain, fire, daymet, damage, treecover, genus_ba, gfw],
        coords="minimal",
        compat="override",
        combine_attrs="drop"
    ).rio.write_crs(3857)
    
    # Define encoding
    int_encoding  = {"dtype": "int16", "_FillValue": -9999}
    byte_encoding = {"dtype": "int8", "_FillValue": -128}
    encoding={
        "abies": int_encoding,
        #"picea": int_encoding,
        "populus": int_encoding,
        "pseudotsuga": int_encoding,
        "tsuga": int_encoding,
        #"vod": int_encoding,
        "elev": int_encoding,
        "slope": int_encoding,
        "northness": int_encoding,
        "eastness": int_encoding,
        "fire": byte_encoding,
        "mortality": byte_encoding,
        "gfw_damage": byte_encoding,
        #"fam": byte_encoding,
        "prcp": int_encoding,
        "vp": int_encoding,
        "tmin": byte_encoding,
        "treecover": byte_encoding
    }

    # Write output
    all_data.to_netcdf(
        out_nc,
        encoding=encoding
    )
    