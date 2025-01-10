import pandas as pd
import xarray as xr
from google.cloud.storage import Client
from .gcs import read_gcs_csv

def df_to_timeseries_dataset(df: pd.DataFrame, target: str="mort", drop_cols: list[str]=["system:index", ".geo"],
                                target_shift: int=-1, timeseries_length: int=5) -> xr.Dataset:
    '''
    Convert a pandas dataframe with latitude, longitude, and time columns to an xarray dataset. 
    
    The dataset has two variables: input and the name of the target column. Both are indexed by sample, which
    uniquely identifies combinations of latitude, longitude, and year corresponding to the end of the 
    input time series. The input variable has two additional coordinates: rolling_year and band. 
    These index into the time series and acrossthe predictor variables, respectively. 

    The target is shifted according to the target_shift parameter. When target_shift is -1, mortality
    is shifted one year backward in time. This means that for a tensor whose time series ends at 
    2012, the target will be mortality in 2013.

    If any NaNs are present in the input, that sample and its corresponding mortality datapoint are discarded.
    '''
    # Discard unwanted columns
    if drop_cols is not None:
        df = df.drop(drop_cols, axis=1)

    # Verify that latitude, longitude, year columns are present.
    if not set(["latitude", "longitude", "year"]).issubset(df.columns):
        raise ValueError(f"Input dataframe is missing one of (latitude, longitude, year). Columns present: {df.columns}")
    
    # Verify that the target columns is present.
    if target not in df.columns:
        raise ValueError(f"Target column {target} not in input dataframe. Columns present: {df.columns}")

    # Convert to an xarray
    df_xr = df.set_index(["latitude", "longitude", "year"]).to_xarray()

    # Shift the target and drop from windowing
    target_shift = df_xr[target].shift(year=target_shift)
    df_xr = df_xr.drop_vars([target])

    # Apply windowing to the input data
    df_windowed = df_xr.rolling(latitude=1, longitude=1, year=timeseries_length)\
        .construct(latitude="rolling_lat", longitude="rolling_lon", year="rolling_year")\
        .squeeze(["rolling_lat", "rolling_lon"])\
        .to_stacked_array("band", sample_dims=["latitude", "longitude", "year", "rolling_year"], name="input")
    
    # Merge the windowed data together. We don't stack latitude and longitude in the previous
    # step so that xr.combine_by_coords can figure out how the target and input data
    # align.
    combined = xr.combine_by_coords([target_shift, df_windowed])\
        .stack(sample=["latitude", "longitude", "year"])\
        .dropna(dim="sample", how="any")\
        .transpose("sample", "rolling_year", "band")

    return combined

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