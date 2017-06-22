import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class training:
    """Main class with feature selection and training."""
    def __init__(self, path, kernel='linear', classifier="svm"):
        self.datasetpath = path
        self.kernel = kernel
        self.clf = None
        self.classifier = classifier

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


    def extract_features(self, imgs, cspace='RGB', spatial_size=(16, 16),
                         hist_bins=32, hist_range=(0, 256), debug=False):
        features = list()
        scale = 1.5
        color_space = cspace
        spatial_size = spatial_size
        hist_bins = hist_bins
        h_range = hist_range
        orient = 8
        pix_per_cell = 8
        cell_per_block = 2
        hog_channel = 'ALL'
        spatial_feat = True
        hist_feat = True
        hog_feat = True

        print("Extract features: Start")
        len_imgs = len(imgs)
        if imgs:
            for image in imgs:
                img = cv2.imread(image)
                img = cv2.resize(img, (64, 64), cv2.INTER_NEAREST)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                single_feature = single_image_features(img, spatial_size=spatial_size,
                                                       hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                                       cell_per_block=cell_per_block, hog_channel=hog_channel,
                                                       spatial_feat=spatial_feat,
                                                       hist_feat=hist_feat, hog_feat=hog_feat)
                features.append(single_feature)
        print("Extract features: End,  {0} images processed.".format(len_imgs))
        return features


    def extract_combined_features(self, folders, cspace='RGB', spatial_size=(16, 16),
                                  hist_bins=32, hist_range=(0, 256)):
        images = []
        for folder in folders:
            val = retrieve_file_names(folder)
            images = images + val
        car_features = extract_features(images, cspace, spatial_size,
                                        hist_bins, hist_range)
        return car_features, images

    def extract_dataset(self):
        global car_features
        global noncar_features

        car_features, images_car = extract_combined_features(folders_vehicle, cspace=default_cspace,
                                                             spatial_size=(16, 16),
                                                             hist_bins=32, hist_range=(0, 256))
        noncar_features, images_noncar = extract_combined_features(folders_non_vehicle, cspace=default_cspace,
                                                                   spatial_size=(16, 16), hist_bins=32,
                                                                   hist_range=(0, 256))
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
        extract_dataset(False)
        Y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
        X = np.vstack((car_features, noncar_features)).astype(np.float64)
        scaling_param = StandardScaler().fit(X)
        X_scaled = scaling_param.transform(X)
        print("Scaled training data shape..", X_scaled.shape)
        rand_state = np.random.randint(0, 1000)
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=rand_state)
        svc.fit(X_train, Y_train)
        save_model(svc, scaling_param, 8, 2, 8, default_cspace)
