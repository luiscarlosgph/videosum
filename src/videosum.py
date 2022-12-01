"""
@brief   Module to summarise a video into N frames. There are a lot of methods
         for video summarisation, and a lot of repositories in GitHub, but
         none of them seems to work out of the box. This module contains a 
         simple way of doing it.
@author  Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    18 May 2022.
"""

import argparse
import numpy as np
import cv2
import tqdm
import math
import skimage.measure
import scipy
import scipy.spatial.distance
import time
import faiss
import seaborn as sns

# My imports
import videosum


class VideoSummariser():
    def __init__(self, algo, number_of_frames: int = 100, 
            width: int = 1920, height: int = 1080, fps=None,
            time_segmentation=False, segbar_height=32):
        """
        @brief   Summarise a video into a collage.
        @details The frames in the collage are evenly taken from the video. 
                 No fancy stuff.

        @param[in]  algo               Algorithm to select key frames.
        @param[in]  number_of_frames   Number of frames of the video you want
                                       to see in the collage.
        @param[in]  width              Width of the summary collage.
        @param[in]  height             Height of the summary collage.
        @param[in]  time_segmentation  Set to True to show a time segmentation
                                       under the video collage.
        @param[in]  segbar_height      Height in pixels of the time
                                       segmentation bar.
        """
        assert(algo in VideoSummariser.ALGOS)
        assert(number_of_frames > 0)
        assert(width > 0)
        assert(height > 0)
        
        # Store attributes
        self.algo = algo
        self.number_of_frames = number_of_frames
        self.width = width
        self.height = height
        self.fps = fps
        self.form_factor = float(self.width) / self.height
        self.time_segmentation = time_segmentation
        self.segbar_height = segbar_height

        # Compute the width and height of each collage tile
        self.tile_height = self.height
        nframes = VideoSummariser._how_many_rectangles_fit(self.tile_height, 
                                                          self.width, 
                                                          self.height)
        while nframes < self.number_of_frames: 
            self.tile_height -= 1
            nframes = VideoSummariser._how_many_rectangles_fit(self.tile_height, 
                                                               self.width, 
                                                               self.height)
        self.tile_width = int(round(self.tile_height * self.form_factor))

        # Compute how many tiles per row and column
        self.tiles_per_row = self.width // self.tile_width
        self.tiles_per_col = self.height // self.tile_height 

        # Initialise the array that holds the label of each frame
        self.labels_ = None

        # Initialise the array that holds the indices of the key frames
        self.indices_ = None
 
    @staticmethod
    def _how_many_rectangles_fit(tile_height, width, height):
        """
        @brief Given a certain tile height, this method computes how many
               of them fit in a collage of a given size. We assume that the
               form factor of the tiles in the collage has to be the same of 
               the collage itself.
        @param[in]  tile_height  Tile height.
        @param[in]  width        Width of the collage.
        @param[in]  height       Height of the collage.
        """
        # Compute form factor
        ff = float(width) / height

        # Compute rectangle height 
        tile_width = int(round(tile_height * ff))

        # Compute how many rectangles fit inside the collage
        tiles_per_row = width // tile_width 
        tiles_per_col = height // tile_height

        return tiles_per_row * tiles_per_col 


    def _insert_frame(self, im, i, j):
        """@brief Insert image into the collage."""

        # Resize frame to the right tile size
        im_resized = cv2.resize(im, (self.tile_width, self.tile_height), 
                                interpolation = cv2.INTER_LANCZOS4)

        # Insert image within the collage
        y_start = i * self.tile_height
        y_end = y_start + im_resized.shape[0] 
        x_start = j * self.tile_width
        x_end = x_start + im_resized.shape[1] 
        self.collage[y_start:y_end, x_start:x_end] = im_resized
    
    # Import the different summarisation methods from their corresponding files
    from ._methods.time import get_key_frames_time
    from ._methods.inception import get_key_frames_inception
    from ._methods.fid import get_key_frames_fid 
    from ._methods.scda import get_key_frames_scda

    def get_key_frames(self, input_path):
        return VideoSummariser.ALGOS[self.algo](self, input_path)

    def check_collage_validity(self, input_path, key_frames):
        """
        @brief Asserts that if there are enough frames to make a collage,
               we should have a full collage.
        @param[in]  input_path  Path to the video.
        @param[in]  key_frames  List of key frames.
        @returns None
        """
        nframes_in_video = videosum.VideoReader.num_frames(input_path, self.fps)
        nframes_in_collage = len(key_frames)
        nframes_requested = self.number_of_frames
        if nframes_in_video > nframes_requested:
            assert(nframes_in_collage == nframes_requested)

    def generate_segbar(self):
        # Create an empty segmentation bar
        segbar_width = self.collage.shape[1]
        segbar = np.full((self.segbar_height, segbar_width, 3), 255, 
            dtype=np.uint8)

        # Create the colour palette, one colour per cluster
        palette = np.array(sns.color_palette("Set3", self.number_of_frames))
        colours = (palette * 255.).astype(np.uint8)

        # Loop over segmentation bar columns
        for c in range(segbar_width):
            # Find the frame corresponding to this vertical bar
            frame_idx = int(round(float(c) * len(self.labels_) / segbar_width))
            frame_idx = np.clip(frame_idx, 0, len(self.labels_) - 1)

            # Find the cluster index of the frame
            cluster_idx = self.labels_[frame_idx]
            
            # Make the line of the colour of the cluster the frame belongs to
            segbar[:, c] = colours[cluster_idx]
        
        # Add the key frame vertical lines to the segmentation bar
        key_frame_line_colour = [0, 0, 0]
        for i in self.indices_:
            # Find the vertical bar corresponding to this frame index
            idx = int(round(i * segbar_width / len(self.labels_)))
            
            # Colour the column corresponding to this key frame
            segbar[:, idx] = key_frame_line_colour

        # Colour the background of the segmentation bar
        """
        if self.algo == 'time':
            # The whole bar of the same background colour
            # FIXME: use the distance to the key frames to segment it with
            #        colours (i.e. make use of self.labels_)
            palette = np.array(sns.color_palette("Set3", 1))
            bg_colour = (palette * 255.).astype(np.uint8)[0]
            segbar *= bg_colour
        else:
            palette = np.array(sns.color_palette("Set3", self.number_of_frames))
            colours = (palette * 255.).astype(np.uint8)
            for c, l in enumerate(self.labels_):
                # Convert frame index 'c' to bar index (the video will 
                # typically have more frames than the number of pixels
                # corresponding to the width of the summary image
                bar_idx = round(segbar.shape[1] * c / len(self.labels_))

                # Set colour for the column equivalent to the current frame
                segbar[:, bar_idx] = colours[l]
        """

        return segbar

    def summarise(self, input_path):
        """
        @brief Create a collage of the video.  

        @param[in]  input_path   Path to an input video.

        @returns a BGR image (numpy.ndarray) with a collage of the video.
        """
        # Initialise list of frame cluster labels
        self.labels_ = None

        # Get list of key frames
        key_frames = VideoSummariser.ALGOS[self.algo](self, input_path)

        # Ensure that summariser actually filled the labels
        assert(self.labels_ is not None)
        assert(len(self.labels_) == videosum.VideoReader.num_frames(input_path, self.fps))

        # If the video has more frames than those requested, 
        # the collage should have all the frames filled
        self.check_collage_validity(input_path, key_frames)
        
        # Warn the user about the collage having less frames than requested,
        # this is expected if the number of requested frames is higher than
        # the number of frames of the video, but a problem otherwise
        if len(key_frames) < self.number_of_frames:
            print('[WARN] The key frame selection algorithm selected ' \
                + 'less frames than you wanted.')

        # Reset collage image
        self.collage = np.zeros((self.tiles_per_col * self.tile_height, 
                                 self.tiles_per_row * self.tile_width, 3),
                                dtype=np.uint8)
        
        # Initialise collage counters
        i = 0
        j = 0
        for im in key_frames:
            # Insert image in the collage
            self._insert_frame(im, i, j) 

            # Update collage iterators
            j += 1
            if j == self.tiles_per_row:
                j = 0
                i += 1

        # Put dividing lines on the collage
        #for x in range(1, self.tiles_per_row):
        #    self.collage[:, x * self.tile_width] = 255

        if self.time_segmentation:
            # Create the time segmentation bar
            segbar = self.generate_segbar()

            # Glue the time segmentation bar under the collage
            collage_with_seg = np.zeros((self.collage.shape[0] + segbar.shape[0],
                self.collage.shape[1], 3), dtype=self.collage.dtype) 
            collage_with_seg[:self.collage.shape[0], :] = self.collage
            collage_with_seg[self.collage.shape[0]:, :] = segbar
            self.collage = collage_with_seg

        return self.collage
    
    # Class attribute: supported key frame selection algorithms
    ALGOS = {
        'time':      get_key_frames_time,
        'inception': get_key_frames_inception,
        'fid' :      get_key_frames_fid,
        'scda':      get_key_frames_scda,
    }


if __name__ == '__main__':
    raise RuntimeError('[ERROR] This is not a Python script.')
