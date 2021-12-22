"""
    Module for preparation for features for Cloud Detection model training and validation 
"""

import numpy as np
from pandas.core.frame import DataFrame
import xarray
import rioxarray
from PIL import Image
from pathlib import Path
import pandas as pd

DATA_PATH = Path("C:/Users/TPPA/Documents/codes/data/l8cloudmasks/sending/") # change location as per use

def create_dataset_from_scene(scene_id: str) -> DataFrame:
    """Convert a single Landsat image/scene/tile into a dataset of 
    input features and output lables (works for the SPARCS dataset only). 

    Args:
        scene_id (str): Image / scene id

    Returns:
        DataFrame: Dataset with input features (10 bands) and output labels
    """

    # Step 1 --- Convert scene input features into DataFrame
    input_file_name = DATA_PATH / f"{scene_id}_data.tif"
    xds = rioxarray.open_rasterio(input_file_name)
    band_names = xds["band"].values
    df = pd.DataFrame(columns=list(band_names))
    for band_name in list(band_names):
        df[band_name] = xds.sel(band=band_name).values.flatten()

    # Step 2 --- Extract pixel labels from PNG file and append to DataFrame
    label_file_name = DATA_PATH / f"{scene_id}_mask.png"
    my_image = Image.open(label_file_name)
    my_labels = np.asarray(my_image)
    my_labels = my_labels.flatten()
    my_binary_labels = np.where((my_labels==0) | (my_labels==1) | (my_labels==5), 1, 0) # convert to binary
    df['label'] = my_labels
    df['bin_label'] = my_binary_labels

    return df
    

scene_id = "LC80020622013244LGN00_32"
create_dataset_from_scene(scene_id)