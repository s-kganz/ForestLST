import xarray as xr
import numpy as np
import pandas as pd
import psutil
import threading
import time

def make_windowed_data_roll_con(arr: xr.DataArray, window: dict) -> xr.DataArray:
    '''
    Return indices in arr where a window of the provided size contains no
    NAs.

    Uses rolling(...).construct(...), which can be extremely memory intensive
    depending on the size of the window.
    '''
    window_dims = {k: k+"_window" for k in window.keys()}
    arr_roll = arr.rolling(**window).construct(**window_dims)
    return np.where(arr_roll.notnull().all(dim=list(window_dims.values())))

def make_windowed_data_roll(arr: xr.DataArray, window: dict):
    '''
    As above, but avoids the .construct() call by assuming that nan means
    within a window result from missing data.
    '''
    return np.where(arr.rolling(**window).mean().notnull())


class MemoryMonitor:
    def __init__(self, interval=0.1):
        self.keep_monitoring = True
        self.peak_memory = 0
        self.interval = interval

    def monitor(self):
        while self.keep_monitoring:
            current = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)  # GB
            self.peak_memory = max(self.peak_memory, current)
            time.sleep(self.interval)  # Check every 100ms

def run_with_mem_monitor(f, ds, window):
    monitor = MemoryMonitor()
    monitor_thread = threading.Thread(target=monitor.monitor)
    monitor_thread.start()

    t0 = time.time()
    valid_indices = f(ds, window)
    t1 = time.time()

    monitor.keep_monitoring = False

    print(f"Shape of result: {valid_indices[0].shape[0]}")
    print(f"Elapsed time: {t1 - t0:.2f}")
    print(f"Peak memory usage: {monitor.peak_memory:.2f} GB")

to_test = {
    "Roll + construct": make_windowed_data_roll_con,
    "Roll": make_windowed_data_roll
}

if __name__ == "__main__":
    print("Reading data")
    ds_csv = pd.read_csv("https://zenodo.org/records/14606048/files/preisler_dim_allyears.csv?download=1", index_col=0)
    ds = xr.Dataset.from_dataframe(ds_csv.set_index(["latitude", "longitude", "year"]))
    print("Done reading")
    
    toyds = ds.mort
    
    print("Input data size (MB): {:.2f}".format(toyds.nbytes / (1024 * 1024)))
    
    window = dict(longitude=5, latitude=5, year=5)
    print("Window size", window)
    print()

    for name, f in to_test.items():
        print(name)
        run_with_mem_monitor(f, toyds, window)
        print()

    
    