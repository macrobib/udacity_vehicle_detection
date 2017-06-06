import random
import glob
import cv2
from moviepy.editor import VideoFileClip 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from skimage import color, exposure

dataset_path_non_vehicle = '../dataset/non-vehicles/'
dataset_path_vehicle = '../dataset/vehicles/'
car_features = None
noncar_features = None
X_test = None
Y_test = None
svc = LinearSVC()

folders_non_vehicle = [dataset_path_non_vehicle + "Extras",
                       dataset_path_non_vehicle + "GTI"]

folders_vehicle = [dataset_path_vehicle + "GTI_Far",
                   dataset_path_vehicle + "GTI_Left"]


# dataset_path_vehicle + "GTI_MiddleClose",
# dataset_path_vehicle + "GTI_Right",
# dataset_path_vehicle + "KITTI_extracted"]

def data_look(car_list, notcar_list):
    """Return characteristics of the dataset."""
    data_dict = {}
    data_dict['n_cars'] = len(car_list)
    data_dict['n_notcars'] = len(notcar_list)
    test_image = mpimage.imread(car_list[0])
    data_dict['image_shape'] = test_image.shape
    data_dict['data_type'] = test_image.dtype
    return data_dict


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


def color_hist(img, nbins=32, bins_range=(0, 256), colorspace='RGB'):
    # Compute the histogram of the color channels separately
    feature_image = None
    if colorspace:
        if colorspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif colorspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif colorspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        else:
            # print("Skipping colorspace.")
            feature_image = np.copy(img)
    channel1_hist = np.histogram(feature_image[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(feature_image[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(feature_image[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32), color_space='RGB'):
    feature_image = None
    if color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    else:
        # print("Considering default of RGB.")
        feature_image = np.copy(img)

    feature_image = cv2.resize(feature_image, size, cv2.INTER_NEAREST)
    features = feature_image.ravel()
    return features


def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    features = list()
    print("Extract features: Start")
    len_imgs = len(imgs)
    if imgs:
        for image in imgs:
            input_image = mpimage.imread(image)
            hist_features = color_hist(input_image, hist_bins, hist_range, cspace)
            spatial_features = bin_spatial(input_image, spatial_size, cspace)
            norm_feature = np.concatenate((hist_features, spatial_features))
            # vert_stack = np.vstack([hist_features, spatial_features])
            # norm_feature = StandardScaler().fit(vert_stack).transform(vert_stack)
            features.append(norm_feature.ravel())
    print("Extract features: End,  {0} images processed.".format(len_imgs))
    return features


def retrieve_file_names(folder):
    files = glob.glob(folder + '/' + '*.png')
    return files


def extract_combined_features(folders, cspace='RGB', spatial_size=(32, 32),
                              hist_bins=32, hist_range=(0, 256)):
    images = list()
    for folder in folders:
        print(folder)
        val = retrieve_file_names(folder)
        print(val)
        images = images + val
    car_features = extract_features(images, cspace, spatial_size,
                                    hist_bins, hist_range)
    return car_features, images


def extract_dataset(visualize=False):
    global car_features
    global noncar_features

    car_features, images_car = extract_combined_features(folders_vehicle, cspace='RGB', spatial_size=(32, 32),
                                                         hist_bins=32, hist_range=(0, 256))
    noncar_features, images_noncar = extract_combined_features(folders_non_vehicle, cspace='RGB',
                                                               spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256))
    if len(car_features):
        stacked = np.vstack((car_features, noncar_features)).astype(np.float64)
        X_scaler = StandardScaler().fit(stacked)
        scaled_X = X_scaler.transform(stacked)
        if visualize:
            car_ind = np.random.randint(0, len(images_car))
            # Plot an example of raw and scaled features
            fig = plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.imshow(mpimage.imread(images_car[car_ind]))
            plt.title('Original Image')
            plt.subplot(132)
            plt.plot(stacked[car_ind])
            plt.title('Raw Features')
            plt.subplot(133)
            plt.plot(scaled_X[car_ind])
            plt.title('Normalized Features')
            fig.tight_layout()
            plt.show()
        else:
            print("Empty feature list.")


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
                                           vis=True, feature_vec=False)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()


test_hog()


# extract_dataset(True)

def train_model():
    global car_features
    global noncar_features
    global svc
    global X_test
    global Y_test

    extract_dataset(True)
    Y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    X_scaled = X_scaler.transform(X)

    rand_state = np.random.randint(0, 100)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=rand_state)
    svc.fit(X_train, Y_train)


def test_model():
    global X_test
    global Y_test
    print("test accuracy: ", round(svc.score(X_test, Y_test), 4))
    img_path = "D:\pycharmprojects\CarND-Vehicle-Detection\dataset\\vehicles\GTI_Far\image0000.png"
    feature = extract_features([img_path])
    vis = mpimage.imread(img_path)
    plt.imshow(vis)
    plt.show()
    print("SVC Prediction: ", svc.predict(feature))


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Draw a rectangle for given coordinates."""
    image_copy = np.copy(img)
    for box in bboxes:
        cv2.rectangle(image_copy, box[0], box[1], color, thick)
    return image_copy


def multi_scale_window():
    """Multi scale moving window implementation."""


def sliding_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64),
                   xy_overlap=(0.5, 0.5)):
    """Implementation of a sliding window search"""
    list_overall = []  # storage
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Pixels per step in x and y direction.
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    nx_buffer = np.int(xy_window[0] * xy_overlap[0])
    ny_buffer = np.int(xy_window[1] * xy_overlap[1])
    nx_windows = np.int((xspan - nx_buffer))
    ny_windows = np.int((yspan - ny_buffer))

    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate the window position.
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            list_overall.append(((startx, endx), (starty, endy)))
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


def single_image_features(img, cspace='RGB', spatial_size=(32, 32),
                          hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                          spatial_feat=True, hist_feat=True, hog_feat=True):
    """Gather feature set for given single image as a combination of spatial, histogram and hog features."""
    img_features = []
    if cspace == 'RGB':
        feature_img = np.copy(img)
    else:
        feature_img = handle_colorspace(img, cspace)
    # Compute spatial feature if flag is set.
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_img, size=spatial_size)
    # Compute color histogram if flags are enabled.
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
    # Computer hog features if flag is set.
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel]),
                                    orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        img_features.append(hog_features)
    return np.concatenate(img_features)


def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32, 32),
                   hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                   spatial_feat=True, hist_feat=True, hog_feat=True):
    """Given a list of windows, search for possible detection using trained
    model and return a list of detected windows."""
    detected_windows = []
    for window in windows:
        img_segment = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = single_image_features(img_segment, cspace=color_space, spatial_size=spatial_size,
                                         hist_bins=hist_bins, orient=orient, pix_per_cells=pix_per_cell,
                                         cell_per_block=cell_per_block, hog_channel=hog_channel,
                                         spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feat=hog_feat)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            detected_windows.append(window)
    return detected_windows


def heatmap_handler(windows):
    """Takes an input of windows are merges based on heatmap."""
    pass


def pipeline(image):
    output_image = None
    return output_image


def find_cars(img, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins):
    """Search for the cars."""
    draw_img = np.copy(img)
    img_to_search = img[ystart:ystop, :, :, :];
    trans_to_search = handle_colorspace(img_to_search, 'YCrCb')
    if scale != 1:
        imshape = img.shape
        trans_to_search = cv2.resize(trans_to_search, (np.int(imshape[1])/scale)
                , np.int(imshape[0])/scale)
    ch1 = trans_to_search[:, :, 0]
    ch2 = trans_to_search[:, :, 1]
    ch3 = trans_to_search[:, :, 2]
    # Define the number of blocks and steps.
    window = 64 # Default window size.
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1 # No of x blocks.
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 # No of y blocks.
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2 # Similar to approach above but no of cells instead of % overlap.
    nxsteps = (nxblocks - nblocks_per_window) // cell_per_step
    nysteps = (nyblocks - nblock_per_window) // cell_per_step

    # Compute the hog feature for the entire image.
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cell_per_step
            xpos = xb*cell_per_step
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.concatenate(hog_feat1, hog_feat2, hog_feat3)

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Image patch for spatial and hist transforms.
            sumimg = cv2.resize(trans_to_search[ytop:ytop+window, xleft:xleft+window], (64, 64))
            # Get color features.
            spatial_features = bin_spatial(subimg, spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Normalize, tranform and test.
            test_features = X_scaler.tranform(np.hstack(spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img, (xbox_left, ystart + ytop_draw), 
                        (xbox_left + win_draw, ystart + ytop_draw + win_draw), (0, 0, 255), 6)
    return draw_img

def minimum_suppression():
    """Reduce the impact of overlapping bounding boxes."""
    pass

def add_heat_map(heatmap, bbox_list):
    """Heat map based suppression of overlapping bounding boxes."""
    for box in bbox_list:
        heatmap[box[0][1]:[1][1], box[0][0]:[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    """Threshold away the weak pixels."""
    heatmap[heatmap <= threshold] = 0
    return heatmap

def render_boxes(frame, boxlist):
    """Draw the final bounding boxes on image."""
    for box in boxlist:
        cv2.rectangle(frame, box[0], box[1], 6)
    return frame

def process_frame(frame):
    """Sliding window detection on the video frame."""
    bbox_list = get_detected_windows(frame)
    heatmap = add_heat_map(heatmap, bbox_list)
    pruned_list = apply_threshold(heatmap, 3)
    frame = render_boxes(frame, pruned_list)
    return frame

def video_pipeline(inputpath, outputpath):
    """Main video pipeline."""
    print("Start detection pipeline..\n")
    clip = VideoFileClip(inpath)
    output_clip = clip.fl_image(process_frame)
    output_clip.write_videofile(outputpath, audio=False)

def main():
    train_model()
    test_model()
    find_cars()


if __name__ == '__main__':
    main()
