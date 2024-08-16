from google.cloud.storage import Client
import pandas as pd
import tensorflow as tf
import xarray as xr

def read_gcs_csv(client: Client, bucket: str, prefix: str) -> pd.DataFrame:
    '''
    Read CSVs hosted on GCS to a pandas dataframe.
    '''
    files = [
        "/".join(["gs://{}".format(bucket), f.name])
        for f in client.list_blobs(bucket, prefix=prefix)
        if f.name.endswith("csv")
    ]

    ds = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    return ds

def csv_to_timeseries_dataset(df: pd.DataFrame, target: str="mort", drop_cols: list[str]=["system:index", ".geo"],
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
    if not target in df.columns:
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
        
