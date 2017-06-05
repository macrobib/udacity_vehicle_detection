import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import glob
import cv2


class dataset_handler:
    dataset_path_non_vehicle = '../dataset/non-vehicles/'
    dataset_path_vehicle = '../dataset/vehicles/'
    train_set = []
    validation_set = []
    # Dataset consisting of non-vehicle images.
    folders_non_vehicle = [dataset_path_non_vehicle + "Extras",
                           dataset_path_non_vehicle + "GTI"]

    # Dataset consisting of vehicle images.
    folders_vehicle = [dataset_path_vehicle + "GTI_Far",
                       dataset_path_vehicle + "GTI_Left",
                       dataset_path_vehicle + "GTI_MiddleClose",
                       dataset_path_vehicle + "GTI_Right",
                       dataset_path_vehicle + "KITTI_extracted"]

    def __init__(self, split_ratio=0.2):
        self.split_ratio = split_ratio

    def load_dataset(self, debug=False):
        """Load dataset."""
        self.create_balanced_train_test_split()
        if debug:
            self.data_look()

    def get_validation_set(self):
        """Get validation set."""
        return self.train_set

    def get_training_set(self):
        """Get training dataset."""
        return self.train_set

    # In the function, each folder of vehicle/non-vehicle dataset is read in,
    # Split is made to create training and validation set per folder and subsequently
    # each component of training/validation subsets are merged and returned.
    def create_balanced_train_test_split(self):
        """Create balanced training/validation dataset."""

    def data_look(self, car_list=None, notcar_list=None):
        """Return characteristics of the dataset."""
        data_dict = {}
        data_dict['n_cars'] = len(car_list)
        data_dict['n_notcars'] = len(notcar_list)
        test_image = mpimage.imread(car_list[0])
        data_dict['image_shape'] = test_image.shape
        data_dict['data_type'] = test_image.dtype
        return data_dict
