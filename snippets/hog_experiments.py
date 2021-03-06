import random
import glob
import cv2
import os
import math
import pickle
from moviepy.editor import VideoFileClip 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from scipy.ndimage.measurements import label
from multiprocessing import Pool
from skimage import color, exposure
from collections import deque

dataset_path_non_vehicle = '../dataset/non-vehicles/'
dataset_path_vehicle = '../dataset/vehicles/'
car_features = None
noncar_features = None
X_test = None
Y_test = None
scaling_param = None
svc = LinearSVC(loss='hinge')
heatmap = None
# Frame coordinate windows:
coord_windows_bottom = None
coord_windows_middle = None
coord_windows_top = None
boxes_prev= None
heatmap_prev = np.zeros((720, 1280))
heatmap_deque = deque(maxlen=10)
centroid_deque = deque(maxlen=10)
default_cspace = 'YCrCb'
# keep track of centroids in terms of direction,  previous centroid, window size(w, h).
centroid_tracker = {}
prev_value = 0
cont_count = 0

frame_count = 0
model_stored = None

folders_non_vehicle = [dataset_path_non_vehicle + "Extras",
                       dataset_path_non_vehicle + "GTI"]

folders_vehicle = [dataset_path_vehicle + "GTI_Far",
                   dataset_path_vehicle + "GTI_Left",# ]
                   dataset_path_vehicle + "GTI_MiddleClose",
                   dataset_path_vehicle + "GTI_Right",
                   dataset_path_vehicle + "KITTI_extracted"]

search_ranges = {
    "bottom1": ([80, 400], [400, 670], [128, 128], [1280, 670-450]), # Tuple of search coordinates and window size.
    "bottom2": ([860, 1280], [400, 670], [128, 128], [1280, 670-450]),
    "middle1": ([40, 450], [400, 500], [96, 96], [1200, 550-420]),
    "middle2": ([880, 1280], [400, 500], [96, 96], [1080, 550 - 500]),
    "top":  ([400, 1280], [400,  540], [64, 64], [580, 600-400])
}

pickle_data = {
    "svc" : None,
    "orient": None,

}

# ******************************************Image handler utility functions ********************************************
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True, transform_sqrt=False):
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=transform_sqrt,
                                  visualise=True, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=transform_sqrt,
                       visualise=False, feature_vector=feature_vec)
        return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    feature_image = np.copy(img)
    channel1_hist = np.histogram(feature_image[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(feature_image[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(feature_image[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    feature_image = np.copy(img)
    feature_image = cv2.resize(feature_image, size, cv2.INTER_NEAREST)
    features = feature_image.ravel()
    return features

def extract_features(imgs, cspace='RGB', spatial_size=(16, 16),
                     hist_bins=32, hist_range=(0, 256), debug = False):
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


def extract_combined_features(folders, cspace='RGB', spatial_size=(16, 16),
                              hist_bins=32, hist_range=(0, 256)):
    images = []
    for folder in folders:
        val = retrieve_file_names(folder)
        images = images + val
    car_features = extract_features(images, cspace, spatial_size,
                                    hist_bins, hist_range)
    return car_features, images
# ***********************************************************************************************************************


# *************************************File handling APIs***************************************************************
def retrieve_file_names(folder):
    files = glob.glob(folder + '/' + '*.png')
    return files


def save_model(svc, scaling_param, pixel_per_cell, cell_per_block, orient, cspace, path=None):
    global model_stored
    if path is None:
        path = '../model/model.p'
    model = {
        "svc": svc,
        "scalar": scaling_param,
        "pixel_per_cell": pixel_per_cell,
        "cell_per_block": cell_per_block,
        "orient": orient,
        "cspace": cspace
    }
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    model_stored = model

def load_model(path = None):
    """Load a trained model."""
    model = None
    if path is None:
        path = '../model/model.p'
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
    return model


def data_look(car_list, notcar_list):
    """Return characteristics of the dataset."""
    data_dict = {}
    data_dict['n_cars'] = len(car_list)
    data_dict['n_notcars'] = len(notcar_list)
    test_image = mpimage.imread(car_list[0])
    data_dict['image_shape'] = test_image.shape
    data_dict['data_type'] = test_image.dtype
    return data_dict


def extract_dataset(visualize=False):
    global car_features
    global noncar_features

    car_features, images_car = extract_combined_features(folders_vehicle, cspace=default_cspace, spatial_size=(16, 16),
                                                         hist_bins=32, hist_range=(0, 256))
    noncar_features, images_noncar = extract_combined_features(folders_non_vehicle, cspace=default_cspace,
                                                               spatial_size=(16, 16), hist_bins=32, hist_range=(0, 256))
    print("Car feature shape: ", np.array(car_features).shape)
    print("Non car feature shape: ", np.array(noncar_features).shape)


def test_hog():
    folders = retrieve_file_names("../dataset/vehicles/GTI_Far")
    rand_index = random.randint(0, len(folders))

    rand_image = folders[rand_index]
    img = mpimage.imread(rand_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, hog_image = get_hog_features(gray, orient,
                                           pix_per_cell, cell_per_block,
                                           vis=True, feature_vec=False, transform_sqrt=True)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()
#***********************************************************************************************************************
# test_hog()

# extract_dataset(True)
# *****************************************************Training *******************************************************
def train_model():
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


def test_model():
    global X_test
    global Y_test
    global svc
    global scaling_param
    global model_stored
    svc = model_stored['svc']
    scaling_param = model_stored['scalar']
    pix_per_cell = model_stored['pixel_per_cell']
    cell_per_block = model_stored['cell_per_block']
    orient = model_stored['orient']
    cspace = model_stored['cspace']
    spatial_size = (16, 16)
    hist_bins = 32
    hist_range = (0, 256)
    # print("test accuracy: ", round(svc.score(X_test, Y_test), 4))
    img_path = "../test_images/non-car.png"
    feature = extract_features([img_path], cspace, spatial_size, hist_bins, hist_range, True)
    np_feature = np.array(feature)
    print("Test feature length and shape: ", len(feature),"--",  np_feature.shape)
    print("\n")
    prediction = svc.predict(feature)
    vis = mpimage.imread(img_path)
    plt.imshow(vis)
    title = None
    if prediction:
        print("prediction: ", prediction)
        title = "Car detected."
    else:
        title = "No car detected."
    plt.title(title)
    plt.show()
    print("SVC Prediction: ", svc.predict(feature))
    frame = cv2.imread("../test_images/test4.jpg")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = get_detected_windows(frame, True)
    frame = render_boxes(frame, boxes)
    plt.imshow(frame)
    plt.show()
# *********************************************************************************************************************


# ****************************************Drawing utilities ***********************************************************
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Draw a rectangle for given coordinates."""

    image_copy = np.copy(img)
    for box in bboxes:
        cv2.rectangle(image_copy, box[0], box[1], color, thick)
    return image_copy


def sliding_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64),
                   xy_overlap=(0.5, 0.5)):
    """Implementation of a sliding window search"""
    list_overall = []  # storage
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
        print("Got None..\n")
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
        print("Got None..\n")
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
        print("Got None..\n")
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
        print("Got None..\n")

    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Pixels per step in x and y direction.
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    nx_buffer = np.int(xy_window[0] * xy_overlap[0])
    ny_buffer = np.int(xy_window[1] * xy_overlap[1])
    nx_windows = np.int((xspan - nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer)/ny_pix_per_step)

    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate the window position.
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            list_overall.append(((startx, starty), (endx, endy)))
    return list_overall


def handle_colorspace(img, cspace):
    """Convert the image to given colorspace."""
    rgb_img = None
    if cspace == 'YUV':
        rgb_img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    elif cspace == 'HSV':
        rgb_img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    elif cspace == 'HLS':
        rgb_img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
    elif cspace == 'YCrCb':
        rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        rgb_img = img
    return rgb_img
# *********************************************************************************************************************


# ********************************Main pipeline functions.*************************************************************
def single_image_features(img, hog_img = None, spatial_size=(16, 16),
                          hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                          spatial_feat=True, hist_feat=True, hog_feat=True):
    """Gather feature set for given single image as a combination of spatial, histogram and hog features."""
    # Compute spatial feature if flag is set.
    hog_features = []
    if hog_img is None and hog_feat == True:
        if hog_channel == 'ALL':
            for channel in range(img.shape[2]):
                hog_features.extend(get_hog_features(img[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block, vis=False,
                                                     feature_vec=True, transform_sqrt=True))
        else:
            hog_features = get_hog_features(img[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True,
                                            transform_sqrt=True)

    elif hog_feat == True:
        hog_features = hog_img

    if spatial_feat == True:
        spatial_features = bin_spatial(img, size=spatial_size)
        # img_features.append(spatial_features)
    # Compute color histogram if flags are enabled.
    if hist_feat == True:
        hist_features = color_hist(img, nbins=hist_bins)
        # img_features.append(hist_features)
    # Computer hog features if flag is set.

        # img_features.append(hog_features)
    img_features = np.hstack((spatial_features, hist_features, hog_features))
    return img_features


def search_windows(img, windows, svc, scaling_param, hog_img = None, spatial_size=(16, 16),
                   hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                   spatial_feat=True, hist_feat=True, hog_feat=True):
    """Given a list of windows, search for possible detection using trained
    model and return a list of detected windows."""
    detected_windows = []
    hog_segment = []
    # hog_main_frame = []
    # if hog_channel == 'ALL':
    #     for i in range(3):
    #         hog_main_frame.append(get_hog_features(img[:, :, i], orient=orient, pix_per_cell=pix_per_cell,
    #                                            cell_per_block=cell_per_block, feature_vec=False, transform_sqrt=True))
    # else:
    #     hog_main_frame = get_hog_features(img[:, :, 0], orient=orient, pix_per_cell=pix_per_cell,
    #                                   cell_per_block=cell_per_block, feature_vec=False, transform_sqrt=True)
    # print('hog image shape: ', np.array(hog_img).shape)
    for window in windows:
        img_segment = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64), cv2.INTER_NEAREST)
        if hog_img is not None:
            if hog_channel == 'ALL':
                for i in range(3):
                    # All three channels are enabled, copy 3 segments in order.
                    hog_segment.append(hog_img[i][window[0][1]:window[1][1], window[0][0]:window[1][0]])
                    hog_segment = np.concatenate(hog_segment)
            else:
                hog_segment = hog_img[0][window[0][1]:window[1][1], window[0][0]:window[1][0]]
        # print("Image shape: ", img_segment.shape)
        features = single_image_features(img_segment,hog_img=None, spatial_size=spatial_size,
                                         hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, hog_channel=hog_channel,
                                         spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feat=hog_feat)
        np_features = (features.reshape(1, -1)).astype(np.float64)
        # print("Search window shape: ", np_features.shape)
        test_features = scaling_param.transform(np_features)
        # print("Search window shape: ", np_features.shape)
        prediction = svc.predict(test_features)
        if prediction == 1:
            detected_windows.append(window)
    return detected_windows


def visualize_prediction(img, prediction = None):
    """Plot and visualize prediction."""
    plt.imshow(img)
    if prediction:
        plt.title(prediction)
    plt.show()

def pipeline(image):
    output_image = None
    return output_image


def find_cars(img, ystart, ystop, scale,X_scaler, svc, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins):
    """Search for the cars."""
    draw_img = np.copy(img)
    print(draw_img.shape)
    img_to_search = img[ystart:ystop, :, :]
    trans_to_search = handle_colorspace(img_to_search, 'YCrCb')
    if scale != 1:
        imshape = img.shape
        trans_to_search = cv2.resize(trans_to_search, (int(imshape[1]//scale)
                , int(imshape[0]//scale)), cv2.INTER_NEAREST)
    ch1 = trans_to_search[:, :, 0]
    ch2 = trans_to_search[:, :, 1]
    ch3 = trans_to_search[:, :, 2]
    # Define the number of blocks and steps.
    window = 64 # Default window size.
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1 # No of x blocks.
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 # No of y blocks.
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2 # Similar to approach above but no of cells instead of % overlap.
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute the hog feature for the entire image.
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False, transform_sqrt=True)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False, transform_sqrt=True)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False, transform_sqrt=True)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            single_features = []
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.concatenate((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Image patch for spatial and hist transforms.
            subimg = cv2.resize(trans_to_search[ytop:ytop+window, xleft:xleft+window], (32, 32), cv2.INTER_NEAREST)
            # Get color features.
            spatial_features = bin_spatial(subimg, spatial_size)
            single_features.append(spatial_features)
            hist_features = color_hist(subimg, nbins=hist_bins)
            single_features.append(hist_features)
            single_features.append(hog_features)

            # Normalize, tranform and test.
            stacked_features = np.concatenate(single_features).astype(np.float64)
            # print("stacked shape: ", stacked_features.shape)
            # print("\n")
            test_features = X_scaler.transform(stacked_features)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img, (xbox_left, ystart + ytop_draw), 
                        (xbox_left + win_draw, ystart + ytop_draw + win_draw), (0, 0, 255), 6)
    plt.imshow(draw_img)
    plt.show()
    return draw_img

def get_detected_windows(frame, debug=False):
    """Get coordinates for detected windows."""
    global coord_windows_bottom
    global coord_windows_middle
    global coord_windows_top

    if coord_windows_bottom is None:
        coord_windows_bottom = sliding_window(frame, x_start_stop=search_ranges["bottom1"][0],
                                              y_start_stop=search_ranges["bottom1"][1],
                                              xy_window=search_ranges["bottom1"][2], xy_overlap=(0.6, 0.6))
        coord_windows_bottom += sliding_window(frame, x_start_stop=search_ranges["bottom2"][0],
                                              y_start_stop=search_ranges["bottom2"][1],
                                              xy_window=search_ranges["bottom2"][2], xy_overlap=(0.6, 0.6))

    # if coord_windows_middle is None:
    #     coord_windows_middle = sliding_window(frame, x_start_stop=search_ranges["middle1"][0],
    #                                           y_start_stop=search_ranges["middle1"][1],
    #                                           xy_window=search_ranges["middle1"][2], xy_overlap=(0.4, 0.4))
    #     coord_windows_middle += sliding_window(frame, x_start_stop=search_ranges["middle2"][0],
    #                                           y_start_stop=search_ranges["middle2"][1],
    #                                           xy_window=search_ranges["middle2"][2], xy_overlap=(0.6, 0.6))

    if coord_windows_top is None:
        coord_windows_top = sliding_window(frame, x_start_stop=search_ranges["top"][0],
                                           y_start_stop=search_ranges["top"][1],
                                           xy_window=search_ranges["top"][2], xy_overlap=(0.6, 0.6))
    # coord_windows = coord_windows_top + coord_windows_middle + coord_windows_bottom
    if debug:
        debug_cpy = np.copy(frame)
        debug_cpy = draw_boxes(debug_cpy, coord_windows_top, color=(255, 0, 0))
        debug_cpy = draw_boxes(debug_cpy, coord_windows_bottom, color=(0, 0, 255))
        # debug_cpy = draw_boxes(debug_cpy, coord_windows_middle, color=(255,255,0))

        plt.imshow(debug_cpy)
        plt.show()

    frame = handle_colorspace(frame, default_cspace)
    detected_windows = search_window_abstracted(frame, coord_windows_bottom)
    # detected_windows += search_window_abstracted(frame, coord_windows_middle)
    detected_windows += search_window_abstracted(frame, coord_windows_top)
    return detected_windows

def search_window_abstracted(frame, coord_windows):
    scale = 1.5
    global svc
    global scaling_param
    global model_stored
    svc = model_stored['svc']
    scaling_param = model_stored['scalar']
    pix_per_cell = model_stored['pixel_per_cell']
    cell_per_block = model_stored['cell_per_block']
    orient = model_stored['orient']
    color_space = model_stored['cspace']
    spatial_size = (16, 16)
    hist_bins = 32
    hist_range = (0, 256)
    hog_channel = 'ALL'
    spatial_feat = True
    hist_feat = True
    hog_feat = True
    detected_windows = search_windows(frame, coord_windows, svc, scaling_param, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range,
                                      orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                      spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    return detected_windows


def init_centroid_tracker(n):
    """Initialize centroid tracker for n instances"""
    global centroid_tracker
    for i in range(n):
        box_size = [0, 0] # Box size of the window.
        centroid_pos = [-1, -1] # Centroid coordinates.
        centroid_delta = [0, 0] # Movement delta value of centroid, movement since previous frame in x and y direction.
        centroid_tracker[i] = [centroid_pos, box_size, centroid_delta]

def centroid(coord):
    return [(coord[0][0] + coord[1][0])/2, (coord[0][1] + coord[1][1])/2]


def get_box_size(coord):
    window_size = 64
    if coord[1] > 550:
        window_size = 128
    elif 450 < coord[1] < 550:
        window_size = 96
    elif 400 < coord[1] < 450:
        window_size = 64
    return window_size


def get_coordinate_from_centroid(coord):
    """Retrieve bounding box from centroid."""
    window_size_half = 32
    window_size_half = get_box_size(coord)/2

    x1 = max(0, (coord[0] - window_size_half))
    y1 = max(0, (coord[1] - window_size_half))
    x2 = min(1280, (coord[0] + window_size_half))
    y2 = min(720, (coord[0] + window_size_half))
    return ((x1, y1), (x2, y2))


def get_square_root_distance(coords):
    a = coords[0]
    b = coords[1]
    distance = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    return distance


def add_heat_map(heatmap, bbox_list):
    """Heat map based suppression of overlapping bounding boxes."""
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    """Threshold away the weak pixels."""
    heatmap[heatmap <= threshold] = 0
    return heatmap


def get_thresholded_boxes(labels):
    """Return the thresholded bounding boxes got from heatmap."""
    global centroid_tracker
    global prev_value
    global cont_count
    bboxes = []
    print("Label count: ", labels[1])
    if labels[1]:
        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            # Determine X and Y values of the pixels.
            cont_count = 0
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            min_coord = (np.min(nonzerox), np.min(nonzeroy))
            max_coord = (np.max(nonzerox), np.max(nonzeroy))
            bboxes.append((min_coord, max_coord))
            cent = centroid((min_coord, max_coord))
            tracker_instance = centroid_tracker[car_number - 1]
            if tracker_instance[0][0] != -1:
                sqr_distance = get_square_root_distance((tracker_instance[0][0], cent))
                wnd_size = get_box_size(cent)
                if sqr_distance <= wnd_size:
                    x_delta = cent[0] - tracker_instance[2][0]
                    y_delta = cent[1] - tracker_instance[2][1]
                    centroid_tracker[car_number][2] = (x_delta, y_delta)
                    centroid_tracker[car_number][1] = wnd_size
                    centroid_tracker[car_number][0] = cent
    else:
        cont_count += 1
        if cont_count % 10 == 0:
            for i in centroid_tracker.keys():
                delta = centroid_tracker[i][2]
                centroid_tracker[i][0][0] += delta[0]
                centroid_tracker[i][0][1] += delta[1]




    return bboxes

def render_boxes(frame, boxlist, debug = False):
    """Draw the final bounding boxes on image."""
    for box in boxlist:
        # print("centroid: ", centroid(box))
        cv2.rectangle(frame, box[0], box[1], (0, 0, 255), 6)
    if debug:
        plt.imshow(frame)
        plt.show()
    return frame


def patch_search(frame, coordinates):
    """Do a fine grained search on previous and new heatmap coordinates."""
    new_windows = []
    for coordinate in coordinates:
        x1, x2, y1, y2 = 0, 0, 0, 0
        x1 = coordinate[0][0] - 10
        if x1 < 0:
            x1 = 0
        x2 = coordinate[1][0] + 10
        if x2 > 1280:
            x2 = 1280
        y1 = coordinate[0][1] - 10
        if y1 < 0:
            y1 = 0
        y2 = coordinate[1][1] + 10
        if y2 > 720:
            y2 = 720

        new_coords = sliding_window(frame[y1:y2, x1:x2], x_start_stop=[None, None],
                       y_start_stop=[None, None],
                       xy_window=[64, 64], xy_overlap=(0.4, 0.4))
        new_windows += search_window_abstracted(frame[y1:y2, x1:x2], new_coords)
    return new_windows

def process_frame(frame):
    """Sliding window detection on the video frame."""
    global heatmap_prev
    global boxes_prev
    global frame_count
    global heatmap_deque

    frame_count += 1
    draw_boxes_on_frame = False
    print("Image shape: ", frame.shape)
    heatmap_current = np.zeros_like(frame[:, :, 0]).astype(np.float)
    if frame_count:
        detected_windows = get_detected_windows(frame, False)
        heatmap_current = add_heat_map(heatmap_current, detected_windows)
        heatmap_deque.append(heatmap_current)
        boxes_prev = detected_windows
        # if len(heatmap_deque) == 5:
        heatmap_new = sum(heatmap_deque)
        heatmap_new = apply_threshold(heatmap_new, 4)
        clipped_heatmap = np.clip(heatmap_new, 0, 255)
        labels = label(clipped_heatmap)
        bbox_list = get_thresholded_boxes(labels)
        boxes_prev = bbox_list
        draw_boxes_on_frame = True
    else:
        # Do a trageted search every alternate frames.
        new_windows = patch_search(frame, boxes_prev)
        heatmap_current = add_heat_map(heatmap_current, new_windows)
        heatmap_current = sum(heatmap_deque) + heatmap_current
        heatmap_current = apply_threshold(heatmap_current, 5)
        clipped_heatmap = np.clip(heatmap_current, 0, 255)
        labels = label(clipped_heatmap)
        bbox_list = get_thresholded_boxes(labels)
        if len(bbox_list):
            boxes_prev = bbox_list
        else:
            bbox_list = boxes_prev
    if draw_boxes_on_frame:
        frame = render_boxes(frame, bbox_list, False)
        print(cont_count)
    return frame


def video_pipeline(inputpath, outputpath):
    """Main video pipeline."""
    print("Start detection pipeline..\n")
    clip = VideoFileClip(inputpath)
    output_clip = clip.fl_image(process_frame)
    output_clip.write_videofile(outputpath, audio=True)
# *********************************************************************************************************************

def main():
    global model_stored
    if load_model() is None:
        train_model()
    else:
        model_stored = load_model()
    # test_model()
    init_centroid_tracker(10)
    video_pipeline("../videos/project_video.mp4", "../videos/output.mp4")

if __name__ == '__main__':
    main()
