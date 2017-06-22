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

# with Pool(processes=4) as pool:
#     res_bottom = pool.apply_async(search_windows, (frame, coord_windows_bottom, svc, scaling_param, spatial_size, hist_bins, hist_range,
#                                   orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat, ))
#
#     res_middle = pool.apply_async(search_windows, (frame, coord_windows_middle, svc, scaling_param, spatial_size, hist_bins, hist_range,
#                                   orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat, ))
#
#     res_top = pool.apply_async(search_windows, (frame, coord_windows_top, svc, scaling_param, spatial_size, hist_bins, hist_range,
#                                   orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat, ))
#     det_bottom = res_bottom.get()
#     det_middle = res_middle.get()
#     det_top = res_top.get()
#     detected_windows = det_bottom + det_middle + det_top
# Retrieve the hog features for three segments in one chunk.
# img_segment_top = cv2.resize(frame[:, :, -1], (0, 0),  fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST)
# img_segment_middle = cv2.resize(frame[:, :, -1], (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_NEAREST)
# img_segment_bottom = cv2.resize(frame[:, :, -1], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
channel = 1
hog_feature_top = None
hog_feature_middle = None
hog_feature_bottom = None
# if hog_channel == 'ALL':
#     channel = 3
# for i in range(channel):
#     hog_feature_top.append(get_hog_features(img_segment_top, orient,
#                                        pix_per_cell, cell_per_block, vis=False, feature_vec=False,
#                                        transform_sqrt=True))
#     hog_feature_middle.append(get_hog_features(img_segment_middle, orient,
#                                           pix_per_cell, cell_per_block, vis=False, feature_vec=False,
#                                           transform_sqrt=True))
#     hog_feature_bottom.append(get_hog_features(img_segment_bottom, orient,
#                                           pix_per_cell, cell_per_block, vis=False, feature_vec=False,
#                                           transform_sqrt=True))