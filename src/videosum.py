"""
@brief   Module to summarise a video into N frames. There are a lot of methods
         for video summarisation, and a lot of repositories in GitHub, but
         none of them seems to work out of the box. This module contains a 
         simple way of doing it.
@author  Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    18 May 2022.
"""

import argparse
import imageio_ffmpeg 
import numpy as np
import cv2
import tqdm


class VideoSummariser():
    def __init__(self, number_of_frames: int = 100, 
            width: int = 1920, height: int = 1080):
        """
        @brief   Summarise a video into a collage.
        @details The frames in the collage are evenly taken from the video. 
                 No fancy stuff.

        @param[in]  input_path        Path to the input video file.
        @param[in]  number_of_frames  Number of frames of the video you want
                                      to see in the collage.
        """
        self.number_of_frames = number_of_frames
        self.width = width
        self.height = height
        self.form_factor = float(self.height) / self.width

        # Compute the width and height of each collage tile
        collage_area = self.width * self.height 
        denom = self.form_factor * number_of_frames
        self.tile_width = int(np.floor(np.sqrt((collage_area) / (denom))))
        self.tile_height = int(round(self.tile_width * self.form_factor))

        # Compute how many tiles per row and column
        self.tiles_per_row = int(np.floor(self.width / self.tile_width))
        self.tiles_per_col = int(np.floor(self.height / self.tile_height)) 

        print('self.tiles_per_col:', self.tiles_per_col)
        print('self.tiles_per_row:', self.tiles_per_row)
        print('self.tile_height:', self.tile_height)
        print('self.tile_width:', self.tile_width)

        # Create empty collage
        self.collage = np.zeros((self.tiles_per_col * self.tile_height, 
                                 self.tiles_per_row * self.tile_width, 3),
                                dtype=np.uint8)

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

    def summarise(self, input_path, output_path):
        """
        @brief Create a collage of the video.  

        @param[in]  input_path   Path to an input video.
        @param[in]  output_path  Path to an output image. 

        @returns a BGR image (numpy.ndarray) with a collage of the video.
        """
        
        # Initialise collage counters
        i = 0
        j = 0

        # Collect the collage frames from the video 
        reader = imageio_ffmpeg.read_frames(input_path, pix_fmt='rgb24')
        meta = reader.__next__()  # meta data, e.g. meta["size"] -> (width, height)
        w, h = meta['size']
        nframes = int(round(meta['duration'] * meta['fps']))
        interval = nframes // (self.tiles_per_row * self.tiles_per_col)
        counter = 0
        last_frame = None
        for raw_frame in tqdm.tqdm(reader):
            if counter == interval:
                counter = 0

                # Convert video frame into a BGR OpenCV/Numpy image
                im = np.frombuffer(raw_frame, 
                        dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()
            
                # Insert image in the collage
                self._insert_frame(im, i, j) 

                # Update collage iterators
                j += 1
                if j == self.tiles_per_row:
                    j = 0
                    i += 1
            else:
                last_frame = raw_frame
                counter += 1

        # Fill the bottom-right corner of the collage
        im = np.frombuffer(last_frame, 
                           dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()
        self._insert_frame(im, i, j) 

        # Put dividing lines on the collage
        #for x in range(1, self.tiles_per_row):
        #    self.collage[:, x * self.tile_width] = 255

        # Save collage to output file
        cv2.imwrite(output_path, self.collage)


if __name__ == '__main__':
    raise RuntimeError('[ERROR] This is not a Python script.')
