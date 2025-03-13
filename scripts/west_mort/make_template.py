'''
Make a blank template raster for convenience in later processing steps
'''
from osgeo import gdal
import os

if 'snakemake' in globals():
    from snakemake.script import snakemake
    out_dir = snakemake.config["data_working"]
    xmin = float(snakemake.config["xmin"])
    ymin = float(snakemake.config["ymin"])
    xmax = float(snakemake.config["xmax"])
    ymax = float(snakemake.config["ymax"])
    srs = snakemake.config["srs"]
    res = int(snakemake.config["resolution"])
else:
    raise RuntimeError("Not running in snakemake pipeline!")

width  = round((xmax - xmin) / res)
height = round((ymax - ymin) / res)

os.system(
    f"gdal_create -ot Byte -outsize {width} {height} -bands 1 -burn 0 -a_srs {srs} -a_ullr {xmin} {ymax} {xmax} {ymin} {os.path.join(out_dir, 'template.tif')}"
)