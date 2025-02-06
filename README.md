## Predicting tree mortality for California and beyond

As climate change continues, the evaporative demand on plant life also increases [(Anderegg et al., 2019)](https://www.nature.com/articles/nclimate1635). This predisposes trees to die through a combination of processes [(Choat et al., 2018)](https://www.nature.com/articles/s41586-018-0240-x):

 - Carbon starvation
 - Hydraulic failure
 - Attack by boring insects

Mortality events have been documented worldwide, but are particularly severe in the southwestern US. This project has two goals:
 1. Improve annual predictions of the distribution of drought-induced tree mortality.
 2. Identify drivers of mortality over time to determine which mechanisms are most important.

## Setup
First, clone the repository
```
git clone https://github.com/s-kganz/ForestLST
cd ForestLST
```
We use the pangeo pytorch-notebook docker image for development, plus a few other packages. Refer to [this repo](https://github.com/pangeo-data/pangeo-docker-images) for instructions on getting this set up.

Most of the data cleaning scripts use GDAL on the command line. GDAL is included in the above docker image, but if you find that you can't run GDAL commands check [this page](https://gdal.org/en/latest/api/python_bindings.html) for guidance on modifying your environment.

If you want to use any of the scripts that work with `earthaccess`, you will have to set up accounts with the respective providers. **This is not necessary unless you want to recreate the steps we took to build the mortality datasets.** Once you have done so, do the following:
 - Create a file named `.netrc` in your home directory. Add your `earthaccess` credentials to the file in the following format
```machine urs.earthdata.nasa.gov login <your_username> password <your_password>```
Now you should be able to authenticate with `earthacess.login(strategy="netrc")`.

For access to development data on GCS, you will have to also set up the Google Cloud SDK. You can have the SDK point to the conda environment we just created, or let it install the bundled Python. Follow the directions [here](https://cloud.google.com/sdk/docs/install) for your machine and then run the following two commands.
```
gcloud init
gcloud auth application-default login
```
Make sure you select the correct cloud project and Google account.

## Datasets

We have developed a benchmark forest mortality dataset we call `west_mort`. It is still under development, but is far enough along that we give some documentation details. **What follows is under active development and is not guaranteed to work!** `west_mort` covers all of the USFS regions 3, 4, 5, and 6 at 4 km resolution over a roughly 20-year period. We used an entirely open-source workflow to generate the dataset. Given the scale of data involved, the cleaning steps are spread across several scripts and notebooks. These are defined under `scripts/west_mort`. We use snakemake to define the data cleaning workflow, and the snakefile is at `scripts/west_mort/main.yml`. Since prior work on forest mortality has focused on California, we clip out a portion of `west_mort` for California, which we call `ca_mort`. This lets us directly compare model performance with others in the literature.

## License
See file `LICENSE`. This repo uses the MIT license to support collaboration, and I strongly encourage you to reach out to me if you want to work on this problem!

