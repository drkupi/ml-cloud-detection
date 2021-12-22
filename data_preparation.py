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
import glob
from random import shuffle
from sklearn.model_selection import train_test_split


DATA_PATH = Path("C:/Users/TPPA/Documents/codes/data/l8cloudmasks/sending/") # change location as per use

def get_scene_ids() -> list:
    """Get ids of the unique images in the SPARCS dataset

    Returns:
        list: [description]
    """

    input_file_list = glob.glob(str(DATA_PATH) + "\*_data.tif")
    scene_ids = [Path(file_name).name.split("_data")[0] for file_name in input_file_list] 
    
    return scene_ids


def create_label_data(scene_ids: list) -> tuple:
    """Prepare data labels for training of cloud detection algorithm

    Args:
        scene_ids (list): List of names of Image ids of SPARCS dataset

    Returns:
        tuple: image labels, binary image labels and cloud proportions per image 
    """
    cloud_ratios = {}
    img_labels = {}
    img_binary_labels = {}
    for scene_id in scene_ids:
        label_file_name = DATA_PATH / f"{scene_id}_mask.png"
        my_image = Image.open(label_file_name)
        my_labels = np.asarray(my_image)
        my_labels = my_labels.flatten()
        my_binary_labels = np.where((my_labels==0) | (my_labels==1) | (my_labels==5), 1, 0) # convert to binary
        cloud_ratios[scene_id] = my_binary_labels.sum() / len(my_binary_labels)
        img_binary_labels[scene_id] = my_binary_labels
        img_labels[scene_id] = my_labels

    return img_labels, img_binary_labels, cloud_ratios 


def create_train_validation_test_sets(scene_ids: list, cloud_ratios: dict) -> tuple:
    """Generic function for ensuring that training, validation and test sets have 
    similar proportions of cloud pixels

    Args:
        scene_ids (list): List of image / scene ids
        cloud_ratios (dict): Dictionary with proportions of cloud pixels in each scene / image

    Returns:
        tuple: Tuple of lists with training, validation and test set image ids
    """

    avg_cloud_ratio = np.asarray(list(cloud_ratios.values())).mean()
    threshold = 0.3
    ratio_check = 0

    while ratio_check < 2:
        if ratio_check == 0:
            train_ids, temp_ids = train_test_split(scene_ids, test_size=30)
            train_ratios = {k:v for k, v in cloud_ratios.items() if k in train_ids}
            avg_train_ratio = np.asarray(list(train_ratios.values())).mean()
            if abs((avg_train_ratio - avg_cloud_ratio)/avg_cloud_ratio) <= threshold:
                ratio_check = 1
        if ratio_check == 1:
            val_ids, test_ids = train_test_split(temp_ids, test_size=15)
            val_ratios = {k:v for k, v in cloud_ratios.items() if k in val_ids}
            avg_val_ratio = np.asarray(list(val_ratios.values())).mean()
            if abs((avg_val_ratio - avg_cloud_ratio)/avg_cloud_ratio) <= threshold:
                ratio_check = 2
    
    return train_ids, val_ids, test_ids


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
    

scene_ids = get_scene_ids()
img_labels, img_binary_labels, cloud_ratios = create_label_data(scene_ids)
create_train_validation_test_sets(scene_ids, cloud_ratios)
