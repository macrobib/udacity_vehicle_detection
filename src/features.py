import cv2
import copy
import numpy as np
import matplotlib.image as img
from skimage.feature import hog

# Global hog feature extraction parameters.
hog_pr = {
    "cspace": "RGB",
    "orient": 9,
    "pix_per_cell": 8,
    "cell_per_block": 2,
    "hog_channel": 4
}

class features:
    """Common feature extraction/augmentation functions."""
    def __init__(self, cspace="RGB", bins=12):
        self.colorspace = cspace
        self.bins = 12

    def __convert_colorspace(self, image, cspace):
        """Aux function: do colorpsace change."""
        if cspace == "YUV":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == "HLS":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        else:
            pass
        return image

    def __color_hist(self, image, cspace=None, bins = None, bins_range=(0, 256)):
        if bins is None:
            bins = self.bins
        img = copy.copy(image)
        if cspace is not None:
            img = self.__convert_colorspace(image, cspace)
        bin_1 = np.histogram(img[:, :, 0], bins=bins, range=bins_range)
        bin_2 = np.histogram(img[:, :, 1], bins=bins, range=bins_range)
        bin_3 = np.histogram(img[:, :, 2], bins=bins, range=bins_range)
        feat = np.concatenate([bin_1, bin_2, bin_3])
        return feat

    def __bin_spatial(self, image, size=(32, 32), cspace=None):
        img = copy.copy(image)
        if cspace is not None:
            img = self.__convert_colorspace(image, cspace)
        features = cv2.resize(img, size).ravel()
        return features

    def __hog_features(self, image, orient, pix_p_cell, cell_p_blk, vis=False, feature_vec=True):
        """Get the hog features."""
        hog_image = None
        if vis:
            features, hog_image = hog(image, orientations=orient, pixels_per_cell=(pix_p_cell, pix_p_cell),
                                      cells_per_block=(cell_p_blk, cell_p_blk), transform_sqrt=True, visualise=vis,
                                      feature_vector=feature_vec)
        else:
            features, hog_image = hog(image, orientations=orient, pixels_per_cell=(pix_p_cell, pix_p_cell),
                                      cells_per_block=(cell_p_blk, cell_p_blk), transform_sqrt=True, visualise=vis,
                                      feature_vector=feature_vec)
        return features, hog_image

    def extract_features(self, image, cspace="RGB", size=(32, 32), bins=32, hist_range = (0, 256)):
        """Common function to abstract out all the feature capture."""
        feature_1 = self.__color_hist(image, cspace, bins, hist_range)
        feature_2 = self.__bin_spatial(image, size, cspace)
        feature_3 = []
        if hog_pr["hog_channel"] == 4:
            # Find and concatenate hog for all channels.
            for cl in range(3):
                feature_3.append(self.__hog_features(image[:, :, cl], hog_pr["orient"], hog_pr["pix_per_cell"],
                                                     hog_pr["cell_per_block"], True, True, ))
            feature_3 = np.ravel(feature_3)
        feature_3 = self.__hog_features(image[:, :, hog_pr["hog_channel"]], hog_pr["orient"], hog_pr["pix_per_cell"],
                                                     hog_pr["cell_per_block"], True, True, )
        feat = np.concatenate((feature_1, feature_2, feature_3))
        return feat

