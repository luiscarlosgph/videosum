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
import math
import sklearn_extra.cluster
import skimage.measure
import scipy
import scipy.spatial.distance
import time

# My imports
import videosum


class VideoSummariser():
    def __init__(self, algo, number_of_frames: int = 100, 
            width: int = 1920, height: int = 1080):
        """
        @brief   Summarise a video into a collage.
        @details The frames in the collage are evenly taken from the video. 
                 No fancy stuff.

        @param[in]  algo              Algorithm to select key frames.
        @param[in]  number_of_frames  Number of frames of the video you want
                                      to see in the collage.
        @param[in]  width             Width of the summary collage.
        @param[in]  height            Height of the summary collage.
        """
        assert(algo in VideoSummariser.ALGOS)
        assert(number_of_frames > 0)
        assert(width > 0)
        assert(height > 0)

        self.algo = algo
        self.number_of_frames = number_of_frames
        self.width = width
        self.height = height
        self.form_factor = float(self.width) / self.height

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

    def get_key_frames_time(self, input_path):
        """
        @brief Get a list of key frames from the video. The key frames are
               simply evenly spaced along the video.
        
        @param[in]  input_path  Path to the video file.

        @returns a list of Numpy/OpenCV BGR images.
        """
        key_frames = []

        # Collect the collage frames from the video 
        reader = imageio_ffmpeg.read_frames(input_path, pix_fmt='rgb24')
        meta = reader.__next__()
        w, h = meta['size']
        nframes = int(math.floor(meta['duration'] * meta['fps']))
        interval = nframes // self.number_of_frames
        counter = interval
        for raw_frame in tqdm.tqdm(reader):
            # If we have collected all the frames we needed, mic out
            if len(key_frames) == self.number_of_frames:
                break
            
            # If this frame is a key frame...
            counter -= 1
            if counter == 0:
                counter = interval

                # Convert video frame into a BGR OpenCV/Numpy image
                im = np.frombuffer(raw_frame, 
                        dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()
            
                # Insert video frame in our list of key frames
                key_frames.append(im)

        return key_frames

    def get_key_frames_inception(self, input_path):
        """
        @brief Get a list of key video frames. 
        @details They key frames are selected by unsupervised clustering of 
                 latent feature vectors corresponding to the video frames.
                 The latent feature vector is obtained from Inception V3
                 trained on ImageNet. The clustering method used is kmedoids.  

        @param[in]  input_path  Path to the video file.
        
        @returns a list of Numpy/BGR images, range [0.0, 1.0], dtype = np.uint8. 
        """
        latent_vectors = []

        # Initialise Inception network model
        fid = videosum.FrechetInceptionDistance('vector')

        # Collect feature vectors for all the frames
        print('[INFO] Collecting feature vectors for all the frames ...')
        reader = imageio_ffmpeg.read_frames(input_path, pix_fmt='rgb24')
        meta = reader.__next__()
        w, h = meta['size']
        for raw_frame in tqdm.tqdm(reader):
            # Convert video frame into a BGR OpenCV/Numpy image
            im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

            # Compute latent feature vector for this video frame
            vec = fid.get_latent_feature_vector(im)

            # Add feature vector to our list
            latent_vectors.append(vec)
        print('[INFO] Done. Feature vectors computed.')

        # Cluster the feature vectors using the Frechet Inception Distance 
        print('[INFO] k-medoids clustering ...')
        X = np.array(latent_vectors)
        kmedoids = sklearn_extra.cluster.KMedoids(n_clusters=self.number_of_frames, 
            method='pam',
            init='k-medoids++',
            random_state=0).fit(X)
        indices = kmedoids.medoid_indices_.tolist()
        print('[INFO] k-medoids clustering finished.')

        # Retrieve the video frames corresponding to the cluster means
        print('[INFO] Retrieving key frames ...')
        key_frames = []
        counter = -1
        reader = imageio_ffmpeg.read_frames(input_path, pix_fmt='rgb24')
        for raw_frame in tqdm.tqdm(reader):
            counter += 1
            if counter in indices:
                # Convert video frame into a BGR OpenCV/Numpy image
                im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

                # Add key frame to list
                key_frames.append(im)
        print('[INFO] Key frames obtained.')

        return key_frames
        
    def get_key_frames_fid(self, input_path):
        """
        @brief Get a list of key video frames. 
        @details They key frames are selected by unsupervised clustering of 
                 latent feature vectors corresponding to the video frames.
                 The latent feature vector is obtained from Inception V3
                 trained on ImageNet. The clustering method used is kmedoids.  

        @param[in]  input_path  Path to the video file.
        
        @returns a list of Numpy/BGR images, range [0.0, 1.0], dtype = np.uint8. 
        """
        latent_vectors = []

        # Initialise Inception network model
        fid = videosum.FrechetInceptionDistance('vector')

        # Collect feature vectors for all the frames
        print('[INFO] Collecting feature vectors for all the frames ...')
        reader = imageio_ffmpeg.read_frames(input_path, pix_fmt='rgb24')
        meta = reader.__next__()
        w, h = meta['size']
        for raw_frame in tqdm.tqdm(reader):
            # Convert video frame into a BGR OpenCV/Numpy image
            im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

            # Compute latent feature vector for this video frame
            vec = fid.get_latent_feature_vector(im)

            # Add feature vector to our list
            latent_vectors.append(vec)
        print('[INFO] Done. Feature vectors computed.')

        # Compute distance matrix
        print('[INFO] Computing distance matrix ...')
        X = videosum.numba_fid(np.array(latent_vectors))
        print('[INFO] Done, distance matrix computed.')

        # Cluster the feature vectors using the Frechet Inception Distance 
        print('[INFO] k-medoids clustering ...')
        kmedoids = sklearn_extra.cluster.KMedoids(n_clusters=self.number_of_frames, 
            metric='precomputed',
            method='pam',
            init='k-medoids++',
            random_state=0).fit(X)
        indices = kmedoids.medoid_indices_.tolist()
        print('[INFO] k-medoids clustering finished.')

        # Retrieve the video frames corresponding to the cluster means
        print('[INFO] Retrieving key frames ...')
        key_frames = []
        counter = -1
        reader = imageio_ffmpeg.read_frames(input_path, pix_fmt='rgb24')
        for raw_frame in tqdm.tqdm(reader):
            counter += 1
            if counter in indices:
                # Convert video frame into a BGR OpenCV/Numpy image
                im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

                # Add key frame to list
                key_frames.append(im)
        print('[INFO] Key frames obtained.')

        return key_frames

    def get_key_frames_scda(self, input_path):
        """
        @brief Get a list of key video frames. 
        @details They key frames are selected by unsupervised clustering of 
                 latent feature vectors corresponding to the video frames.
                 The latent feature vector is obtained as explained in 
                 Wei et al. 2017:

                 "Selective Convolutional Descriptor Aggregation for
                  Fine-Grained Image Retrieval"
                 
                 The only difference with the paper is we use InceptionV3
                 to retrieve the latent tensor (as opposed to VGG-16).

        @param[in]  input_path  Path to the video file.
        
        @returns a list of Numpy/BGR images, range [0.0, 1.0], dtype = np.uint8. 
    
        """
        latent_vectors = []

        # Initialise Inception network model
        fid = videosum.FrechetInceptionDistance('tensor')

        # Collect feature vectors for all the frames
        reader = imageio_ffmpeg.read_frames(input_path, pix_fmt='rgb24')
        meta = reader.__next__()
        w, h = meta['size']
        for raw_frame in tqdm.tqdm(reader):
            # Convert video frame into a BGR OpenCV/Numpy image
            im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

            # Compute latent feature vector for this video frame
            tensor = fid.get_latent_feature_tensor(im)

            # Sum tensor over channels
            aggregation_map = tensor.sum(axis=0)

            # Compute mask map
            mean = aggregation_map.mean()
            mask_map = np.zeros_like(aggregation_map)
            mask_map[aggregation_map > mean] = 1

            # Connected component analysis
            cc = skimage.measure.label(mask_map) 

            # Largest connected component
            largest = np.argmax(np.bincount(cc.flat)[1:]) + 1
            largest_cc = scipy.ndimage.binary_fill_holes(cc == largest).astype(cc.dtype)

            # Selected descriptor set
            selected_descriptor_set = []
            for i in range(8):
                for j in range(8):
                    if largest_cc[i, j]:
                        selected_descriptor_set.append(tensor[:, i, j])
            selected_descriptor_set = np.array(selected_descriptor_set)

            # Compute avg&maxPool SCDA feature vector
            avgpool = np.mean(selected_descriptor_set, axis=0)
            maxpool = np.max(selected_descriptor_set, axis=0)
            scda = np.hstack([avgpool, maxpool])

            # Add feature vector to our list
            latent_vectors.append(scda)

        # Compute mean and variance of every vector
        X = np.array(latent_vectors)

        # Cluster the feature vectors using the Frechet Inception Distance 
        kmedoids = sklearn_extra.cluster.KMedoids(n_clusters=self.number_of_frames, 
            init='k-medoids++',
            random_state=0).fit(X)
        indices = kmedoids.medoid_indices_.tolist()

        # Retrieve the video frames corresponding to the cluster means
        key_frames = []
        counter = -1
        reader = imageio_ffmpeg.read_frames(input_path, pix_fmt='rgb24')
        for raw_frame in tqdm.tqdm(reader):
            counter += 1
            if counter in indices:
                # Convert video frame into a BGR OpenCV/Numpy image
                im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

                # Add key frame to list
                key_frames.append(im)

        return key_frames

    def get_key_frames(self, input_path):
        return VideoSummariser.ALGOS[self.algo](self, input_path)

    def summarise(self, input_path):
        """
        @brief Create a collage of the video.  

        @param[in]  input_path   Path to an input video.

        @returns a BGR image (numpy.ndarray) with a collage of the video.
        """
        # Get list of key frames
        key_frames = VideoSummariser.ALGOS[self.algo](self, input_path)

        #assert(len(key_frames) == self.number_of_frames)
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
