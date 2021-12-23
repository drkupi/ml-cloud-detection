import numpy as np
from data_preparation import create_dataset_from_scene
import pickle
import lightgbm as lgb
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def detect_clouds(scene_id: str) -> np.array:

    # Step 1 --- Load image input feature data and output
    df = create_dataset_from_scene(scene_id)

    # Step 2 --- Load model and make prediction
    model = pickle.load(open('cloud_detection_model.pkl', 'rb'))
    X = df.drop(['label', 'bin_label'], axis=1)
    y_true = df['bin_label'].to_numpy()
    y_pred = model.predict(X)

    # Step 3 --- Convert model output into image
    label_file = f"{scene_id}_photo.png"
    my_image = Image.open(label_file)
    my_labels = np.asarray(my_image)
    nrows = my_labels.shape[0]
    ncols = my_labels.shape[1]
    y_true_2d = np.reshape(y_true, (nrows, ncols))
    y_pred_2d = np.reshape(y_pred, (nrows, ncols))
    plt.imsave(f"{scene_id}_true_cloud_mask.png", y_true_2d, cmap=cm.gray)
    plt.imsave(f"{scene_id}_pred_cloud_mask.png", y_pred_2d, cmap=cm.gray)

    return y_pred_2d
