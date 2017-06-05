from sklearn import svm

def simple_trial():
    x = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC()
    clf.fit(x, y)
    print(clf.predict([[2, 2]]))

def svm_multiclass():
    """Multi class example with SVM.
    SVC and NuSVC uses One against One classifier, for n set of classes, n(n-1)/2 classifiers are created.
    """
    x = [[0], [1], [2], [3]]
    y = [0, 1, 2, 3]
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(x, y)
    print(clf.predict([[5]]))


# simple_trial()
svm_multiclass()