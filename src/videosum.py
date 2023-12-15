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


class VideoSummarizer():
    def __init__(self, algo, number_of_frames: int = 100, 
            width: int = 1920, height: int = 1080, fps=None,
            time_segmentation=False, segbar_height=32, time_smoothing=0.,
            compute_fid=False):
        """
        @brief   Summarise a video into a collage.
        @details The frames in the collage are evenly taken from the video. 
                 No fancy stuff.

        @param[in]  algo               Algorithm to select key frames.
        @param[in]  number_of_frames   Number of frames of the video you want
                                       to see in the collage.
        @param[in]  width              Width of the summary collage.
        @param[in]  height             Height of the summary collage.
        @param[in]  fps                TODO.
        @param[in]  time_segmentation  Set to True to show a time segmentation
                                       under the video collage.
        @param[in]  segbar_height      Height in pixels of the time
                                       segmentation bar.
        @param[in]  time_smoothing     TODO.
        @param[in]  compute_fid        Set it to True if you want a report on
                                       the FID of the summary to the whole
                                       video.
        """
        # Sanity checks
        assert(algo in VideoSummarizer.ALGOS)
        assert(number_of_frames > 0)
        assert(width > 0)
        assert(height > 0)
        if (algo == 'time' and time_smoothing > 1e-6):
            raise ValueError('[ERROR] You cannot use time smoothing with the time algorithm.')
        
        # Store attributes
        self.algo = algo
        self.number_of_frames = number_of_frames
        self.width = width
        self.height = height
        self.fps = fps
        self.form_factor = float(self.width) / self.height
        self.time_segmentation = time_segmentation
        self.segbar_height = segbar_height
        self.time_smoothing = time_smoothing
        self.compute_fid = compute_fid

        # Compute the width and height of each collage tile
        self.tile_height = self.height
        nframes = VideoSummarizer._how_many_rectangles_fit(self.tile_height, 
                                                          self.width, 
                                                          self.height)
        while nframes < self.number_of_frames: 
            self.tile_height -= 1
            nframes = VideoSummarizer._how_many_rectangles_fit(self.tile_height, 
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
    
    @staticmethod
    def transition_indices(labels):
        """
        @brief This method provides the indices of the boundary frames.

        @details  After the summarisation, the video frames are clustered into
                  classes. This means that there will be a video frame that
                  belongs to class X followed by another frame that belongs to 
                  class Y. This method detects this transitions and returns a
                  list of the Y frames, i.e. the first frames after a class
                  transition.
        @param[in]  labels  Pass the self.labels_ produced after calling
                            the summarise() method.
        @returns a list of indices.
        """
        transition_frames = []

        # Loop over the labels of all the video frames
        prev_class = None
        for idx, l in enumerate(labels):
            if l != prev_class:
                transition_frames.append(idx)
                prev_class = l

        return transition_frames
    
    @staticmethod
    def frame_distance_matrix(n: int):
        """
        @brief Compute the normalised distance matrix of rows and columns. 
        @details The distance matrix of rows and columns is (assuming only four
                 frames):

                    0 1 2 3
                    1 0 1 2
                    2 1 0 1
                    3 2 1 0
                  
                 The normalised version is simply a minmax normalisation of the
                 matrix above.
        """
        # Create lower triangular distance matrix
        dist = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(0, i):
                dist[i, j] = np.abs(i - j)

        # Fill the upper triangular part
        dist = dist + dist.T

        # Normalise matrix to [0, 1]
        return dist / np.max(dist)
    
    # Import the different summarisation methods from their corresponding files
    from ._methods.time import get_key_frames_time
    from ._methods.inception import get_key_frames_inception
    from ._methods.uid import get_key_frames_uid 
    from ._methods.scda import get_key_frames_scda

    def get_key_frames(self, input_path):
        return VideoSummarizer.ALGOS[self.algo](self, input_path,
            time_smoothing=self.time_smoothing)

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
            if nframes_in_collage != nframes_requested:
                print("[ERROR] There are {} frames ".format(nframes_in_video) \
                        + "in [{}], but the collage ".format(input_path) \
                        + "only has {} ".format(nframes_in_collage) \
                        + "out of the {} requested.".format(nframes_requested))

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
        key_frames = VideoSummarizer.ALGOS[self.algo](self, input_path,
            time_smoothing=self.time_smoothing)

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
        
        # Report the FID between the summary and the whole video if requested
        if self.compute_fid:
            print("[INFO] Summary FID: {}".format(self.fid_storyboard_vs_video(input_path, key_frames)))

        return self.collage

    def fid_storyboard_vs_video(self, input_path, key_frames):
        print("[INFO] Computing FID of the storyboard vs video ...")

        # Initialise InceptionV3 model
        model = videosum.InceptionFeatureExtractor('vector')

        # Compute the multivariate Gaussian of the summary
        story_fv = [model.get_latent_feature_vector(im) for im in key_frames]
        story_mu = np.mean(story_fv, axis=0)
        story_cov = np.cov(story_fv, rowvar=False)

        # Compute the multivariate Gaussian of the whole video 
        # (including the summary of course)
        video_fv = []
        reader = videosum.VideoReader(input_path, sampling_rate=self.fps, 
                                      pix_fmt='rgb24')
        w, h = reader.size
        for raw_frame in tqdm.tqdm(reader):
            im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()
            video_fv.append(model.get_latent_feature_vector(im))
        video_mu = np.mean(video_fv, axis=0)
        video_cov = np.cov(video_fv, rowvar=False)

        assert(story_mu.shape == video_mu.shape)
        assert(story_cov.shape == video_cov.shape)

        # Compute 2-Wasserstein distance
        emd = videosum.InceptionFeatureExtractor._calculate_frechet_distance(
            story_mu, story_cov, video_mu, video_cov)

        return emd
    
    # Class attribute: supported key frame selection algorithms
    ALGOS = {
        'time':      get_key_frames_time,
        'inception': get_key_frames_inception,
        'uid' :      get_key_frames_uid,
        'scda':      get_key_frames_scda,
    }


if __name__ == '__main__':
    raise RuntimeError('[ERROR] This is not a Python script.')
