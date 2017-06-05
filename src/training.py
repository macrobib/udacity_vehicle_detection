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
