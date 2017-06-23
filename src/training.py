import numpy as np
import cv2
import glob
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.features import features as ft

class training:
    """Main class with feature selection and training."""
    def __init__(self, path, kernel='linear', classifier="svm"):
        self.datasetpath = path
        self.kernel = kernel
        self.clf = None
        self.classifier = classifier
        self.ft = ft()

        dataset_path_non_vehicle = '../dataset/non-vehicles/'
        dataset_path_vehicle = '../dataset/vehicles/'
        self.folders_non_vehicle = [dataset_path_non_vehicle + "Extras",
                               dataset_path_non_vehicle + "GTI"]

        self.folders_vehicle = [dataset_path_vehicle + "GTI_Far",
                           dataset_path_vehicle + "GTI_Left",  # ]
                           dataset_path_vehicle + "GTI_MiddleClose",
                           dataset_path_vehicle + "GTI_Right",
                           dataset_path_vehicle + "KITTI_extracted"]

    def __grid_search(self, kernel, C=[1, 10]):
        """Find the best fit parameters"""
        params = {'kernel': kernel, 'C':C}
        svr = LinearSVC()
        self.clf = GridSearchCV(svr, params)

    def train(self, features, labels):
        """Main function to implement training.
        features: class object abstracting the feature selection and normalization.
        """
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(features,
                labels, test_size=0.2, random_state=rand_state)
        if self.classifier == "svm":
            if self.clf is None:
                self.__grid_search(('linear'), [1, 10])
                self.clf.fit(features, labels)

    def retrieve_file_names(self, folder):
        files = glob.glob(folder + '/' + '*.png')
        return files

    def extract_features(self, imgs):
        features = list()
        print("Extract features: Start")
        len_imgs = len(imgs)
        if imgs:
            for image in imgs:
                img = cv2.imread(image)
                img = cv2.resize(img, (64, 64), cv2.INTER_NEAREST)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                single_feature = ft.single_image_features(img)
                features.append(single_feature)
        print("Extract features: End,  {0} images processed.".format(len_imgs))
        return features


    def extract_combined_features(self, folders):
        images = []
        for folder in folders:
            val = self.retrieve_file_names(folder)
            images = images + val
        car_features = self.extract_features(images)
        return car_features, images

    def extract_dataset(self, debug=False):
        global car_features
        global noncar_features
        car_features, images_car = self.extract_combined_features(self.folders_vehicle)
        noncar_features, images_noncar =self.extract_combined_features(self.folders_non_vehicle)
        if debug:
            print("Car feature shape: ", np.array(car_features).shape)
            print("Non car feature shape: ", np.array(noncar_features).shape)

    def training(self):
        global car_features
        global noncar_features
        global svc
        global X_test
        global Y_test
        global scaling_param
        print("Starting training on the dataset..\n")
        self.extract_dataset(False)
        Y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
        X = np.vstack((car_features, noncar_features)).astype(np.float64)
        scaling_param = StandardScaler().fit(X)
        X_scaled = scaling_param.transform(X)
        print("Scaled training data shape..", X_scaled.shape)
        rand_state = np.random.randint(0, 1000)
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=rand_state)
        svc.fit(X_train, Y_train)
        ft.save_model()
