REGIONS = [3, 4, 5, 6]
DATA_IN = "data_in"
DATA_WORKING = "data_working"
FIGURES = "figures"
FINAL_OUTPUT = ""
SRS = "EPSG:5071"
YEAR_START = 1999
YEAR_END = 2023
OUTPUT = "data_working/westmort.zarr/.zmetadata"

YEARS=list(range(YEAR_START, YEAR_END+1))

import os
os.makedirs(DATA_IN, exist_ok=True)
os.makedirs(DATA_WORKING, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

rule all:
    input:
        # Final training dataset
        os.path.join(DATA_WORKING, "westmort.zarr/.zmetadata"),
        # Model artifacts
        os.path.join(DATA_WORKING, "mort_convnet", "mort_convnet_shap.nc"),
        os.path.join(DATA_WORKING, "gbm_temporal_cv.csv"),
        os.path.join(DATA_WORKING, "gbm_temporal_cv_predictions.parquet"),
        os.path.join(DATA_WORKING, "mort_convnet", "mort_convnet_shap.nc"),
        # Figures
        os.path.join(FIGURES, "nsurvey_map_damage_area.png"),
        os.path.join(FIGURES, "validation_performance.png"),
        os.path.join(FIGURES, "mort_area_validation_stats.png"),
        os.path.join(FIGURES, "insect_weather_site_over_time.png"),
        os.path.join(FIGURES, "driver_map_triangle.png"),
        os.path.join(FIGURES, "cnn_shap.png"),
        os.path.join(FIGURES, "hist2d_one_to_one.png")
        

rule figures:
    input: 
        os.path.join(DATA_IN, "usfs_region_boundaries", "usfs_regions_simple.shp"),
        os.path.join(DATA_WORKING, "westmort.zarr/.zmetadata"),
        os.path.join(DATA_WORKING, "gbm_temporal_cv.csv"),
        os.path.join(DATA_WORKING, "gbm_shap.zarr/.zmetadata"),
        os.path.join(DATA_WORKING, "mort_convnet", "mort_convnet_shap.nc"),
        os.path.join(DATA_WORKING, "gbm_temporal_cv_predictions.parquet")
    output:
        os.path.join(FIGURES, "nsurvey_map_damage_area.png"),
        os.path.join(FIGURES, "validation_performance.png"),
        os.path.join(FIGURES, "mort_area_validation_stats.png"),
        os.path.join(FIGURES, "insect_weather_site_over_time.png"),
        os.path.join(FIGURES, "driver_map_triangle.png"),
        os.path.join(FIGURES, "cnn_shap.png"),
        os.path.join(FIGURES, "hist2d_one_to_one.png")
    notebook:
        "notebooks/plots.ipynb"

rule gbm_temporal_cv:
    input:
        os.path.join(DATA_WORKING, "westmort.zarr/.zmetadata")
    output:
        os.path.join(DATA_WORKING, "gbm_temporal_cv.csv"),
        os.path.join(DATA_WORKING, "gbm_temporal_cv_predictions.csv")
    notebook:
        "notebooks/gbm_temporal_cv.ipynb"

rule gbm_shap:
    input:
        os.path.join(DATA_WORKING, "westmort.zarr/.zmetadata")
    output:
        os.path.join(DATA_WORKING, "gbm_shap.zarr/.zmetadata")
    notebook:
        "notebooks/gbm_shap.ipynb"

rule convnet:
    input:
        os.path.join(DATA_WORKING, "westmort.zarr/.zmetadata")
    output:
        os.path.join(DATA_WORKING, "mort_convnet", "mort_convnet_shap.nc")
    notebook:
        "notebooks/mort_convnet.ipynb"

rule combine:
    input:
        os.path.join(DATA_WORKING, "ads_damage.zarr"),
        os.path.join(DATA_WORKING, "terraclimate.zarr"),
        os.path.join(DATA_WORKING, "topo.zarr"),
        os.path.join(DATA_WORKING, "treemap2016_hostba_hydro.zarr")
    output:
        OUTPUT
    notebook:
        "notebooks/combine_data.ipynb"

rule download_ads:
    input:
    output:
        expand(os.path.join(DATA_IN, "ads/CONUS_Region{n}_AllYears.gdb/timestamps"), n=REGIONS)
    shell:
        "./scripts/download_ads.sh {DATA_IN}"

rule merge_ads:
    input:
        expand(os.path.join(DATA_IN, "ads/CONUS_Region{n}_AllYears.gdb/timestamps"), n=REGIONS)
    output:
        os.path.join(DATA_WORKING, "damage_merged.gdb/timestamps"),
        os.path.join(DATA_WORKING, "survey_merged.gdb/timestamps")
    run:
        # Snakemake automatically creates the .gdb folders, but this prevents gdal from making the dataset.
        # So we have to prepend the script with a rm -r to delete the directories.
        shell("rm -r {DATA_WORKING}/damage_merged.gdb {DATA_WORKING}/survey_merged.gdb") 
        shell("./scripts/merge_ads_polygons.sh")

rule burn_ads:
    input:
        os.path.join(DATA_WORKING, "damage_merged.gdb/timestamps"),
        os.path.join(DATA_WORKING, "survey_merged.gdb/timestamps")
    output:
        os.path.join(DATA_WORKING, "ads_damage.zarr/.zmetadata")
    notebook:
        "notebooks/burn_ads.ipynb"

rule treemap:
    input:
    output:
        os.path.join(DATA_WORKING, "treemap2016_hostba_hydro.zarr/.zmetadata")
    notebook:
        "notebooks/coarsen_treemap.ipynb"

rule terraclimate:
    input:
        os.path.join(DATA_WORKING, "treemap2016_hostba_hydro.zarr/.zmetadata")
    output:
        os.path.join(DATA_WORKING, "terraclimate.zarr/.zmetadata")
    notebook:
        "notebooks/download_terraclimate.ipynb"

rule topo:
    input:
        os.path.join(DATA_WORKING, "treemap2016_hostba_hydro.zarr/.zmetadata")
    output:
        os.path.join(DATA_WORKING, "topo.zarr/.zmetadata")
    notebook:
        "notebooks/download_dem.ipynb"


