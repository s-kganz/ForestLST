This dataset encodes tree mortality in California from 2002 - 2022, designed for use with a LSTM network. Each example describes a 1 km pixel in the MODIS projection for some number of years (depending on the config). The other features are as follows:

 - `year`: Year of observation.    
 - `EVI_p[5, 50, 95]`: MODIS-derived Enhanced Vegetation Index. Suffix indicates percentile for the year of observation, e.g. `EVI_p50` is the median EVI for the given year.
 - `dT_p[5, 50, 95]`: Difference between MODIS-derived land surface temperature and Daymet maximum air temperature (K).
 - `spei30d_p[5, 50, 95]`: Standardized precipitation index with 30-day aggregation period.
 - `winter_tmin`: Minimum air temperature from December of the previous year through February of the current year (deg C).
 - `prcp`: Water-year precipitation (mm). Water-years run from October - October. If `year == 2020`, then `prcp` is the sum of precipitation from Oct 1 2019 through Sep 30 2020.
 - `latitude`: Latitude of this pixel in decimal degrees (constant through time).
 - `longitude`: Longitude of this pixel in decimal degrees (constant through time).
 - `altitude`: Altitude of the pixel derived from SRTM (m).
 - `pct_mortality`: Approximate proportion of this pixel where tree die-off was observed. Derived from aerial detection surveys by the US Forest Service. See `notebooks/make_ads_images.ipynb` in the repo for more information.

Data are unstandardized. For full details on how data were generated, see `notebooks/make_tensors.ipynb`.