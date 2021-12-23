# ml-cloud-detection
Using Machine Learning for cloud detection

## Overview
This repository includes ML model development workflow for detection of clouds & cloud shadows
in Lansat images. The [SPARCS](https://www.usgs.gov/landsat-missions/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation-data) dataset is used for training a **LightGBM** supervised ML model for cloud detection.  


## Modeling Workflow

The entire modeling workflow is defined in the jupyter notebook "cloud_detection_notebook.ipynb". The workflow includes data splitting, data preparation, model training, validation & testing and model generalization for future use. 
## Environment Setup
A conda environment is used for this repository and may be reproduced from the `environment.yml` file. Please recreate and load the environment to successfully re-run the jupyter notebook and associated python files. 

## Running Cloud Detection in Prediction Mode
The trained model is stored as a pickle file. It is possible to re-run the cloud detection prediction on any Landsat image as long as the file formats are compatible with SPARCS. The function `detect_clouds` in `cloud_detection.py` provides a generalizable way of using the trained model for cloud detection. The prediction output is a binary mask of predicted cloud & shadow. This function was tested on an image from the test set and output is shared as `LC80150242014146LGN00_23_pred_cloud_mask.png`. This output can be compared with the true cloud & shadow mask (manually labelled in SPARCS) provided in `LC80150242014146LGN00_23_true_cloud_mask.png`.


