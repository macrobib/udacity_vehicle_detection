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
    "hog_channel": 4,
    "bins": 12,
    "spatial_size": (64, 64)
}

class features:
    """Common feature extraction/augmentation functions."""
    def __init__(self):
        self.colorspace = hog_pr["cspace"]
        self.bins = hog_pr["bins"]
        self.hog_img = None
        self.hog_feat = True
        self.spatial_feat = True
        self.hist_feat = True
        self.feature_vec = True
        self.transform_sqrt = True
        self.hog_channel    = hog_pr["hog_channel"]
        self.orient         = hog_pr["orient"]
        self.pix_per_cell   = hog_pr["pix_per_cell"]
        self.cell_per_block = hog_pr["cell_per_block"]
        self.spatial_size   = hog_pr["spatial_size"]

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

    def __hog_features(self, image, vis=False):
        """Get the hog features."""
        if vis:
            feat, hog_image = hog(image, orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=self.transform_sqrt,
                                      visualise=True, feature_vector=self.feature_vec)
            return features, hog_image
        else:
            feat = hog(image, orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=self.transform_sqrt,
                           visualise=False, feature_vector=self.feature_vec)
            return feat

    def extract_features(self, image, cspace="RGB", size=(32, 32), bins=32, hist_range = (0, 256)):
        """Common function to abstract out all the feature capture."""
        feature_1 = self.__color_hist(image, cspace, bins, hist_range)
        feature_2 = self.__bin_spatial(image, size, cspace)
        feature_3 = []
        if hog_pr["hog_channel"] == 4:
            # Find and concatenate hog for all channels.
            for cl in range(3):
                feature_3.append(self.__hog_features(image[:, :, cl], vis=False))
            feature_3 = np.ravel(feature_3)
        feature_3 = self.__hog_features(image[:, :, hog_pr["hog_channel"]], vis=False)
        feat = np.concatenate((feature_1, feature_2, feature_3))
        return feat

    def single_image_features(self, img):
        """Extract feature set for given image."""
        """Gather feature set for given single image as a combination of spatial, histogram and hog features."""
        # Compute spatial feature if flag is set.
        hog_features = []
        spatial_features = []
        hist_features = []
        if self.hog_img is None and self.hog_feat == True:
            if self.hog_channel == 'ALL':
                for channel in range(img.shape[2]):
                    hog_features.extend(self.__hog_features(img[:, :, channel], vis=False))
            else:
                hog_features = self.__hog_features(img[:, :, self.hog_channel], vis=False)
        elif self.hog_feat == True:
            hog_features = self.hog_img

        if self.spatial_feat == True:
            spatial_features = self.__bin_spatial(img, size=self.spatial_size)
            # img_features.append(spatial_features)
        # Compute color histogram if flags are enabled.
        if self.hist_feat == True:
            hist_features = self.__color_hist(img, bins=self.bins)
            # img_features.append(hist_features)
            # Computer hog features if flag is set.
            # img_features.append(hog_features)
        img_features = np.hstack((spatial_features, hist_features, hog_features))
        return img_features


