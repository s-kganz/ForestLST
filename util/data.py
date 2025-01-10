import pandas as pd
import xarray as xr
from google.cloud.storage import Client
from .gcs import read_gcs_csv

def df_to_xr(df: pd.DataFrame, drop_cols: list[str]=[], index_cols: list[str]=[], sparse=False) -> xr.Dataset:
    '''
    Convert a pd.DataFrame to an xr.Dataset. Columns in drop_cols are removed, and index_cols
    are interpreted as coordinates in the resulting Dataset. The sparse argument is passed
    to xr.Dataset.from_dataframe as is. Note that when sparse=True, the Dataset cannot be converted
    back to a dataframe easily.
    '''
    if not set(drop_cols).issubset(df.columns):
        raise ValueError(f"Input dataframe is missing a column to be dropped ({drop_cols})."
                         f"Columns present: {df.columns}")
    
    # Verify that the target columns is present.
    if not set(index_cols).issubset(df.columns):
        raise ValueError(f"Input datframe is missing at least one of the index columns ({index_cols})"
                         f"not in input dataframe. Columns present: {df.columns}")
        
    return xr.Dataset.from_dataframe(
        df.drop(drop_cols, axis=1).set_index(index_cols),
        sparse=sparse
    )

def check_for_coordinate(ds: xr.Dataset, coord: str) -> None:
    assert (coord in ds.coords), f"Expected coordinate {coord} to be in ds.coords"

def make_preisler_dataset(ds: xr.Dataset, prec="prcp", prism_prec="ppt_sum", near="near", fire="fire", rhost="rhost",
                          mtemp="win_tmin", mort="pct_mortality", latitude="latitude", 
                          longitude="longitude", year_coordinate="year") -> xr.Dataset:
    '''
    Remake the mortality dataset described in Preisler et al. (2017). Their paper calls for various
    lags which we accomplish with xr.DataArray.shift. We alter the paper's nomenclature to make the lags
    as explicit as possible. The number after each variable name indicates how many years prior to the
    mortality event this observation was made.

    The mortality variable is unshifted, while all other variables are. This means that the year coordinate
    corresponds to the year of the mortality event.

    It is assumed that the variables in ds are indexed by latitude and longitude coordinates, so these are
    not explicitly copied to the new Dataset.
    '''
    check_for_coordinate(ds, latitude)
    check_for_coordinate(ds, longitude)
    check_for_coordinate(ds, year_coordinate)
    
    return xr.combine_by_coords([
        # This year's mortality
        ds[mort].rename("mort"),
        # PRISM precip is a 36-year average so we don't have to shift it
        ds[prism_prec].rename("prism_prec"),
        # Precip, lagged 1-4 years. Lag is relative to next year's mort, so
        # no shift is needed for the first year.
        ds[prec ].shift({year_coordinate: 1}).rename("prec1"),
        ds[prec ].shift({year_coordinate: 2}).rename("prec2"),
        ds[prec ].shift({year_coordinate: 3}).rename("prec3"),
        ds[prec ].shift({year_coordinate: 4}).rename("prec4"),
        ds[mtemp].shift({year_coordinate: 1}).rename("mtemp1"),
        ds[near ].shift({year_coordinate: 1}).rename("near1"),
        # Fire is the sum of burned area 2-4 years prior, calculate that separately
        (
            ds[fire].shift({year_coordinate: 2}) +
            ds[fire].shift({year_coordinate: 3}) +
            ds[fire].shift({year_coordinate: 4})
        ).rename("fire"),
        ds[mort ].shift({year_coordinate: 1}).rename("mort1"),
        ds[rhost].shift({year_coordinate: 1}).rename("rhost1")
    ])

def make_preisler_dataframe():
    '''
    Utility function to load the Preisler dataset we use as a benchmark. See make_preisler_dataset
    for details on the data transformations involved.
    '''
    client = Client(project="forest-lst")
    df = read_gcs_csv(client, "preisler_tfdata", "preisler-rectangular")
    df_xr = df_to_xr(df, ["system:index", ".geo"], ["latitude", "longitude", "year"], sparse=False)
    return make_preisler_dataset(df_xr).to_dataframe().dropna()