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
git clone https://github.com/UW-MLGEO/MLGEO2024_ForestMort
cd MLGEO2024_ForestMort
```
Next, install the environment
```
conda env create --file=environment.yml
```
If you want to use any of the scripts that work with Earth Engine or `earthaccess`, you will have to set up accounts with the respective providers.

## License
See file `LICENSE`. This repo uses the MIT license to support collaboration, and I strongly encourage you to reach out to me if you want to work on this problem!

