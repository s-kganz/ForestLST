from osgeo import gdal, ogr, osr
from osgeo_utils import gdal_calc
import numpy as np
import os
import tempfile
ogr.UseExceptions()

# Get parameters
if 'snakemake' in globals():
    from snakemake.script import snakemake
    input_gdb = os.path.join(snakemake.config["data_in"], "nidrm", "L48_BA.gdb")
    input_totals_gdb = os.path.join(snakemake.config["data_in"], "nidrm", "L48_totals.gdb")
    output_dir = os.path.join(snakemake.config["data_working"], "genus_basal_area")
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
    "abies": ["b10", "b11", "b12", "b15", "b17", "b19", "b20"],
    #"picea": "b90",
    "pinus": ["b101", "b102", "b105", "b108", "b109", 
              "b113", "b114", "b116", "b117", "b119", 
              "b121", "b122", "b125", "b126", "b129"],
    "populus": ["b746", "b750"],
    "pseudotsuga": ["b202"],
    "tsuga": ["b263", "b264"]
}

for genus, spp in genus.items():
    # Coarsen and clip all the layers we care about,
    # then sum up everything in the genus
    print("Now coarsening", genus)
    print("Warping")
    with tempfile.TemporaryDirectory() as tmpdir:
        for sp in spp:
            gdal.Warp(
                os.path.join(tmpdir, f"{sp}.tif"),
                f'OpenFileGDB:{input_gdb}:{sp}',
                format="GTiff",
                outputBounds=(out_xmin, out_ymin, out_xmax, out_ymax),
                xRes=res,
                yRes=res,
                dstSRS=template_srs,
                creationOptions=["BIGTIFF=YES", "COMPRESS=DEFLATE"],
                outputType=gdal.GDT_Int16,
                resampleAlg=gdal.GRA_Average
            )
        print("Summing spp")
        var_to_sp = {f"a{i}": os.path.join(tmpdir, f"{sp}.tif") for i,sp in enumerate(spp)}
        print(var_to_sp)
        gdal_calc.Calc(
            calc=" + ".join(var_to_sp.keys()),
            NoDataValue=-32768,
            outfile=os.path.join(output_dir, f"{genus}.tif"),
            type="Int16",
            overwrite=True,
            **var_to_sp
        )
        