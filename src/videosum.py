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
from .videoreader import VideoReader
from .summarizer import BaseSummarizer


class VideoSummarizer(BaseSummarizer):
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
 
    def check_collage_validity(self, input_path, key_frames):
        """
        @brief Asserts that if there are enough frames to make a collage,
               we should have a full collage.
        @param[in]  input_path  Path to the video.
        @param[in]  key_frames  List of key frames.
        @returns None
        """
        nframes_in_video = VideoReader.num_frames(input_path, self.fps)
        nframes_in_collage = len(key_frames)
        nframes_requested = self.number_of_frames
        if nframes_in_video > nframes_requested:
            if nframes_in_collage != nframes_requested:
                print("[ERROR] There are {} frames ".format(nframes_in_video) \
                        + "in [{}], but the collage ".format(input_path) \
                        + "only has {} ".format(nframes_in_collage) \
                        + "out of the {} requested.".format(nframes_requested))

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
        assert(len(self.labels_) == VideoReader.num_frames(input_path, self.fps))

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
        reader = VideoReader(input_path, sampling_rate=self.fps, 
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
    

if __name__ == '__main__':
    raise RuntimeError('[ERROR] This is not a Python script.')
