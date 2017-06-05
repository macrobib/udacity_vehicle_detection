import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3D(pixels, colors_rgb,
           axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Assign axis limits.
    print(*axis_limits[0], axis_limits[0])
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')
    return ax

def experiment_colorspace(image):

    img = cv2.imread(image)
    scale = max(img.shape[0], img.shape[1], 64) / 64
    img_small = cv2.resize(img,
                           (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

    # Convert the scaled image to desired colorspaces.
    img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    img_small_LUV = cv2.cvtColor(img_small, cv2.COLOR_BGR2LUV)
    img_small_rgb_scaled = img_small_RGB/255.

    plot3D(img_small_RGB, img_small_rgb_scaled)
    plt.show()

    plot3D(img_small_HSV, img_small_rgb_scaled, axis_labels=list("HSV"))
    plt.show()

    plot3D(img_small_LUV, img_small_rgb_scaled, axis_labels=list("LUV"))
    plt.show()


kiti_1 = "../test_images/kiti_1.png"
kiti_2 = "../test_images/kiti_2.png"
kiti_3 = "../test_images/kiti_3.png"
experiment_colorspace(kiti_1)
experiment_colorspace(kiti_2)
experiment_colorspace(kiti_3)