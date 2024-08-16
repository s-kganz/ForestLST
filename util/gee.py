# This file contains functions to generate Earth Engine objects for datasets.
from functools import wraps
from typing import Callable
import warnings
import datetime

from googleapiclient.errors import HttpError
from ee.ee_exception import EEException

import ee
ee.Initialize()

# Project assets
HOST = ee.Image.constant(0).blend(ee.Image("projects/forest-lst/assets/nidrms_host_present"))
DAMAGE = ee.ImageCollection("projects/forest-lst/assets/damage_img")
CALIF = ee.FeatureCollection("TIGER/2018/States")\
    .filter(ee.Filter.eq("NAME", "California"))\
    .first()

# Output projection
DEFAULT_PROJ = DAMAGE.first().projection()
DEFAULT_CELLSIZE = 4000 # m

def _aggregate_image(proj: ee.Projection=DEFAULT_PROJ, 
                     reducer: ee.Reducer=ee.Reducer.mean(), 
                     cellsize: int=DEFAULT_CELLSIZE) -> Callable:
    '''
    Decorator factory that calls reduceResolution() on output images, applying
    a projection, scale, and setting an aggregation method. For example, outputs
    expressed a pixel proportion should be reduced by ee.Reducer.mean(), while
    total precipitation in a cell should be reduced by ee.Reducer.sum().
    '''
    def reproj_reduce_decorator(func: Callable):
        @wraps(func)
        def reproject_reduce(*args, **kwargs):
            return func(*args, **kwargs)\
                .reproject(proj, None, cellsize)\
                .reduceResolution(reducer, maxPixels=1024, bestEffort=True)
        return reproject_reduce
    
    return reproj_reduce_decorator

@_aggregate_image(reducer=ee.Reducer.sum())
def daymet_water_year_ppt(year: int) -> ee.Image:
    '''
    Calculate annual precipitation for a given water-year. Water-years go from
    October to October. For example, water-year 2010 goes from Oct 2009 - Oct 2010.
    '''
    d = ee.Date.fromYMD(year, 1, 1)
    d_start = d.advance(-3, "month")
    d_end   = d.advance(9, "month")

    return ee.ImageCollection("NASA/ORNL/DAYMET_V4")\
        .filterDate(d_start, d_end)\
        .reduce(ee.Reducer.sum())\
        .select("prcp_sum").rename("prcp")

@_aggregate_image(reducer=ee.Reducer.min())
def daymet_minimum_winter_air_temperature(year: int) -> ee.Image:
    '''
    Minimum winter air temperature within a given year. "Winter" goes from Dec of (year-1)
    to Feb of (year). For example, winter 2010 is from Dec 2009 to Feb 2010.
    '''
    d = ee.Date.fromYMD(year, 1, 1)
    d_start = d.advance(-1, "month")
    d_end   = d.advance(2, "month")

    return ee.ImageCollection("NASA/ORNL/DAYMET_V4")\
        .filterDate(d_start, d_end)\
        .reduce(ee.Reducer.percentile([5]))\
        .select("tmin_p5")\
        .rename("winter_tmin")

@_aggregate_image()
def remaining_host(year: int) -> ee.Image:
    '''
    Pixel area containing trees susceptible to beetles minus the pixel area damaged
    by beetles in the given year.
    '''
    prior_damage = DAMAGE.filter(ee.Filter.calendarRange(year, year, "year")).first()
    return HOST.subtract(prior_damage).clamp(0, 1).rename("rhost")

@_aggregate_image()
def max_damage_to_neighbors(year: int) -> ee.Image:
    '''
    Highest area damaged by beetles among neighboring pixels.
    '''
    prior_damage = DAMAGE.filter(ee.Filter.calendarRange(year, year, "year")).first()
    kernel = ee.Kernel.fixed(
        width=3, height=3,
        weights=[
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
    )

    return prior_damage.focalMax(kernel=kernel).rename("near")

@_aggregate_image()
def burned_area(year: int) -> ee.Image:
    '''
    Percent of pixel area that burned in a given year.
    '''
    year_start = ee.Date.fromYMD(year, 1, 1).millis()
    year_end   = ee.Date.fromYMD(year, 12, 31).millis()

    year_filter = ee.Filter.rangeContains("Ig_Date", year_start, year_end)

    mtbs_filter = ee.FeatureCollection('USFS/GTAC/MTBS/burned_area_boundaries/v1')\
        .filterBounds(CALIF.geometry()).filter(year_filter)

    n_fires = mtbs_filter.size().getInfo()
    
    if n_fires == 0:
        warnings.warn(f"burned_area() found no fires in year {year}")
        return ee.Image.constant(0)

    # Rasterize
    mtbs_raster = mtbs_filter.map(lambda x: x.set("const", 1))\
        .reduceToImage(["const"], ee.Reducer.max())\
        .rename("burn_pct")

    return ee.Image.constant(0).blend(mtbs_raster)

@_aggregate_image()
def summer_median_rwc(year: int) -> ee.Image:
    '''
    Calculate median relative water content from SAR after Rao et al. (2019). Summer is
    defined as July - Sep.
    '''
    # Calculate 5th and 95th percentile VOD over all summers
    vod_summer = ee.ImageCollection("projects/sat-io/open-datasets/VODCA/X-BAND")\
        .select("b1")\
        .filter(ee.Filter.calendarRange(7, 9, "month"))
    
    percentiles = vod_summer.reduce(ee.Reducer.percentile([5, 95]))
    p5  = percentiles.select("b1_p5")
    p95 = percentiles.select("b1_p95")

    # Get median VOD for this summer
    this_vod = vod_summer.filter(ee.Filter.calendarRange(year, year, "year")).median()

    # Calculate RWC (Eq. 4 in Rao et al. 2019)
    return this_vod.subtract(p5).divide(p95.subtract(p5))

@_aggregate_image()
def elevation(img: str="CGIAR/SRTM90_V4") -> ee.Image:
    '''
    Just grabs your elevation raster of choice. This function is here
    for convenient use with the _aggregate_image.
    '''
    return ee.Image(img)

def annual_predictor_image(year: int):
    '''
    Collects all of the predictors that vary in time into one image.
    '''
    mort = DAMAGE.filter(ee.Filter.calendarRange(year, year, "year")).first().rename("mort")
    all_bands = ee.Image([
        # Water year precipitation
        daymet_water_year_ppt(year).rename("prcp"),
        # Minimum winter air temperature
        daymet_minimum_winter_air_temperature(year).rename("tmin"),
        # Remaining host after this year's mortality
        remaining_host(year).rename("rhost"),
        # Maximum damage this year among neighboring cells
        max_damage_to_neighbors(year).rename("near"),
        # Pixel area affected by fire
        burned_area(year).rename("fire"),
        # Median summer relative water content
        summer_median_rwc(year).rename("rwc"),
        # Elevation
        elevation().rename("elev"),
        # Pixel coordinates
        ee.Image.pixelLonLat(),
        # Year
        # Note that this shouldn't get passed to a model, but we need it to do windowing.
        ee.Image.constant(year).rename("year"),
        # Target: tree mortality
        mort
    ]).updateMask(mort.mask())

    # Set timekeeping properties
    epoch_start = datetime.datetime(year, 1, 1, 0, 0, 0, 
                                    tzinfo=datetime.timezone.utc)
    epoch_end   = datetime.datetime(year+1, 1, 1, 0, 0, 0, 
                                    tzinfo=datetime.timezone.utc) - datetime.timedelta(milliseconds=1)

    all_bands = all_bands.set({
        "system:time_start": epoch_start.timestamp() * 1000,
        "system:time_end": epoch_end.timestamp() * 1000
    })

    return all_bands

def make_rectangular_export_task(img):
    year = ee.Date(img.get("system:time_start")).get("year").getInfo()
    
    sample = img.sample(
        region=CALIF.geometry(),
        scale=DEFAULT_CELLSIZE,
        projection=DEFAULT_PROJ
    ).map(lambda x: x.setGeometry(None))

    return ee.batch.Export.table.toCloudStorage(
        description="yr{}".format(year),
        fileNamePrefix="preisler-rectangular-v2/yr{}".format(year),
        collection=sample,
        bucket="preisler_tfdata"
    )

def get_available_years():
    '''
    Determine for what years annual_predictor_image returns a valid
    image for export.
    '''
    first_year = ee.Date(DAMAGE.aggregate_min("system:time_start")).get("year").getInfo()
    last_year  = ee.Date(DAMAGE.aggregate_max("system:time_start")).get("year").getInfo()

    def succeeds(year):
        try:
            img = annual_predictor_image(year)
            img.getInfo()
            return True
        except (HttpError, EEException) as e:
            return False
    
    return [x for x in range(first_year, last_year+1) if succeeds(x)]

def export_annual_images(prompt=True):
    available_years = get_available_years()
    print("Available years:", available_years)

    tasks = [
        make_rectangular_export_task(annual_predictor_image(y))
        for y in available_years
    ]

    if prompt:
        i = input(f"About to start {len(tasks)} tasks. Proceed? ")
        if i.lower() != "y":
            print("Aborting")
            return
    
    for t in tasks: t.start()

if __name__ == "__main__":
    export_annual_images()