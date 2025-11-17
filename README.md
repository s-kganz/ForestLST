## Predicting tree mortality in the western United States

As climate change continues, the evaporative demand on plant life also increases [(Anderegg et al., 2019)](https://www.nature.com/articles/nclimate1635). This predisposes trees to die through a combination of processes [(Choat et al., 2018)](https://www.nature.com/articles/s41586-018-0240-x):

 - Carbon starvation
 - Hydraulic failure
 - Attack by boring insects

Mortality events have been documented worldwide, but are particularly severe in the western US. This project has two goals:
 1. Improve annual predictions of the distribution of drought-induced tree mortality.
 2. Identify drivers of mortality over time to determine which mechanisms are most important.

## Setup
First, clone the repository
```
git clone https://github.com/s-kganz/ForestLST
cd ForestLST
```
Then, create the environment. `environment.yml` is a streamlined version of the [CryoCloud Python image](https://github.com/CryoInTheCloud/hub-image), plus a few libraries pip install'd on top.
```
conda env create -f environment.yml
```
If you want to use any of the scripts that work with `earthaccess`, you will have to set up a NASA Earthdata account. **This is not necessary unless you want to recreate the steps we took to build the mortality datasets.** Once you have done so, do the following:
 - Create a file named `.netrc` in your home directory. Add your `earthaccess` credentials to the file in the following format
```machine urs.earthdata.nasa.gov login <your_username> password <your_password>```
Now you should be able to authenticate with `earthacess.login(strategy="netrc")`.

## Datasets
If you don't care about recreating the model-ready datasets we used in this project, the netcdf files are available in the github release and on Zenodo under the `mort_datasets` directory.

## Recreating results
We provided a static version of the `data_out` directory at submission time on github/Zenodo. Placing this in the project directory will let you recreate all the paper figures with `notebooks/plots.ipynb`. Other notebooks do the following:

 - `ads_iou.ipynb`: calculate annual overlap between sequential ADS survey polygons.
 - `gbm_[westmort|soap_teak].ipynb`: fit gradient-boosted regression models to the continental and local mortality datasets.
 - `repeability_iou.ipynb`: calculate overlap among the survey polygons in [Coleman et al. (2018)](https://doi.org/10.1016/j.foreco.2018.08.020).
 - `survey_prop.ipynb`: calculate the ratio of survey probability given past mortality history.
 - `variograms.ipynb`: calculate Moran's $I$ and temporal autocorrelation at a variety of spatial and temporal lags.

## License
See file `LICENSE`. This repo uses the MIT license to support collaboration, and I strongly encourage you to reach out to me if you want to work on this problem!

