from osgeo import gdal, ogr, osr
from osgeo_utils import gdal_calc
import numpy as np
import os
ogr.UseExceptions()

# Get parameters
if 'snakemake' in globals():
    from snakemake.script import snakemake
    input_gdb = os.path.join(snakemake.config["data_in"], "nidrm", "L48_BA.gdb")
    input_totals_gdb = os.path.join(snakemake.config["data_in"], "nidrm", "L48_totals.gdb")
    output_dir = os.path.join(snakemake.config["data_working"], genus_basal_area)
    out_xmin = float(snakemake.config["xmin"])
    out_ymin = float(snakemake.config["ymin"])
    out_xmax = float(snakemake.config["xmax"])
    out_ymax = float(snakemake.config["ymax"])
    template_srs = snakemake.config["srs"]
    res = int(snakemake.config["resolution"])
    forest_mask = os.path.join(snakemake.config["data_working"], "forest_mask.tif")
else:
    raise RuntimeError("Not running in snakemake pipeline!")

genus = {
    "abies": "b10",
    "picea": "b90",
    #"pinus": "b152",
    "populus": "b750",
    "pseudotsuga": "b202",
    "tsuga": "b260"
}

for genus, raster in genus.items():
    print("Now coarsening", genus)
    print("Warping")
    gdal.Warp(
        "temp_coarse.tif",
        f'OpenFileGDB:{input_gdb}:{raster}',
        format="GTiff",
        outputBounds=(out_xmin, out_ymin, out_xmax, out_ymax),
        xRes=res,
        yRes=res,
        dstSRS=template_srs,
        creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
        outputType=gdal.GDT_Int16,
        resampleAlg=gdal.GRA_Average
    )
    print("Setting nodata")
    # Set nan values correctly
    gdal_calc.Calc(
        calc="np.where(a > 0, b, -32768)",
        user_namespace={"np": np},
        NoDataValue=-32768,
        a=forest_mask,
        b="temp_coarse.tif",
        outfile=os.path.join(output_dir, f"{genus}.tif"),
        type="Int16",
        overwrite=True
    )


os.remove("temp_coarse.tif")
        