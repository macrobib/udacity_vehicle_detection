import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

def spatial_bin(img, color_space='RGB', size=(32, 32)):

    feature_image = None
    if color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    else:
        print("Considering default of RGB.")
        feature_image = np.copy(img)

    feature_image = cv2.resize(feature_image, size, cv2.INTER_NEAREST)
    features  = feature_image.ravel()
    return features

image = mpimage.imread('../test_images/cutout1.jpg')
feature_vec = spatial_bin(image, color_space='RGB', size=(32, 32))

# Plot features
plt.plot(feature_vec)
plt.title('Spatially Binned Features')
plt.show()
