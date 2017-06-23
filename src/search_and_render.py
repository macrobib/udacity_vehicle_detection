import numpy as np


class Search:
    def __init__(self, feat_map):
        self.feat_map = feat_map
        self.search_ranges = feat_map.search_ranges
        self.coord_windows_bottom = None
        self.coord_windows_middle = None
        self.coord_windows_top = None

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        """Draw a rectangle for given coordinates."""

        image_copy = np.copy(img)
        for box in bboxes:
            cv2.rectangle(image_copy, box[0], box[1], color, thick)
        return image_copy

    def sliding_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64),
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
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)

        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate the window position.
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                list_overall.append(((startx, starty), (endx, endy)))
        return list_overall

    def get_detected_windows(self, frame, debug=False):
        """Get coordinates for detected windows."""


        if self.coord_windows_bottom is None:
            coord_windows_bottom = self.sliding_window(frame, x_start_stop=self.search_ranges["bottom1"][0],
                                                  y_start_stop=self.search_ranges["bottom1"][1],
                                                  xy_window=self.search_ranges["bottom1"][2], xy_overlap=(0.6, 0.6))
            coord_windows_bottom += self.sliding_window(frame, x_start_stop=self.search_ranges["bottom2"][0],
                                                   y_start_stop=self.search_ranges["bottom2"][1],
                                                   xy_window=self.search_ranges["bottom2"][2], xy_overlap=(0.6, 0.6))

        # if coord_windows_middle is None:
        #     coord_windows_middle = sliding_window(frame, x_start_stop=search_ranges["middle1"][0],
        #                                           y_start_stop=search_ranges["middle1"][1],
        #                                           xy_window=search_ranges["middle1"][2], xy_overlap=(0.4, 0.4))
        #     coord_windows_middle += sliding_window(frame, x_start_stop=search_ranges["middle2"][0],
        #                                           y_start_stop=search_ranges["middle2"][1],
        #                                           xy_window=search_ranges["middle2"][2], xy_overlap=(0.6, 0.6))

        if self.coord_windows_top is None:
            coord_windows_top = self.sliding_window(frame, x_start_stop=search_ranges["top"][0],
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

        frame = self.feature_map.handle_colorspace(frame, default_cspace)
        detected_windows = self.detected_windows(frame, coord_windows_bottom)
        # detected_windows += self.detected_windows(frame, coord_windows_middle)
        detected_windows += self.detected_windows(frame, coord_windows_top)
        return detected_windows

    def detected_windows(self, frame, coord_windows):
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
        detected_windows = self.search_windows(self, frame, coord_windows, svc, scaling_param, spatial_size=spatial_size,
                                          hist_bins=hist_bins, hist_range=hist_range,
                                          orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                          hog_channel=hog_channel,
                                          spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        return detected_windows