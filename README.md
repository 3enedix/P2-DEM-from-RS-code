# P2-DEM-from-RS-code
## Description
Code accompanying the paper "Towards a dynamic DTM for sandy beaches: integrating global DEMs and satellite time series with Kalman filtering".

Data will be published in 4TU data repository with DOI 10.4121/a1eec304-e5eb-4e3e-a014-473051af107b.

The goal of the study is to create a time-variable Digital Elevation Model (DTM) for sandy beaches relying only on satellite data. The elevations of the intertidal zone are derived with the "waterline method", combining satellite-derived shorelines with sea level from altimetry in combination with a tide model. These intertidal elevations are then used together with global DEMs in a Kalman filter and smoother framework in order to achieve a long time series of the largest possible beach extent. The approach was tested at three study sites, Terschelling (NL), Duck (US) and Narrabeen (AUS). The results were not accurate enough to reproduce time series of volume changes as compared to the validation data, nor to separate the influence of sea level rise and morphodynamics on shoreline migration.

## Structure of the code repository
Each notebook name starts with a number that corresponds with the number of the corresponding data folder. However, some notebooks make use of data in other folders, so it is advised to check the required folders specified at the beginning of a notebook. Notebooks starting with "1x_" contain files related to the study site at Terschelling, "2x_" are the notebooks for Duck, "3x_" are the notebooks for Narrabeen. Notebooks starting with "0x_" contain general code for preparatory analysis e.g. for tides and validation data.


