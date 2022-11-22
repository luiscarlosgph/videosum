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

        # Initialise the variable that holds the video frame count 
        self.frame_count_ = None

        # Initialise the array that holds the label of each frame
        self.labels_ = None
 
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
                 After this method is executed the list self.indices_ will 
                 hold the list of frame indices that represent the key frames.
        """
        key_frames = []

        # Initialise video reader
        reader = videosum.VideoReader(input_path, sampling_rate=self.fps)
        w, h = reader.size
        nframes = int(math.floor(reader.duration * reader.fps))

        # Collect the collage frames from the video 
        interval = nframes // self.number_of_frames
        counter = interval
        self.indices_ = []
        self.frame_count_ = 0
        for raw_frame in tqdm.tqdm(reader):
            # Increase the internal frame count of the video summariser
            self.frame_count_ += 1

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
                self.indices_.append(self.frame_count_ - 1)

        return key_frames

    def get_key_frames_inception(self, input_path, eps=1e-6):
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

        # Initialise video reader
        reader = videosum.VideoReader(input_path, sampling_rate=self.fps)
        w, h = reader.size

        # Initialise Inception network model
        model = videosum.FrechetInceptionDistance('vector')

        # Collect feature vectors for all the frames
        print('[INFO] Collecting feature vectors for all the frames ...')
        self.frame_count_ = 0
        for raw_frame in tqdm.tqdm(reader):
            self.frame_count_ += 1

            # Convert video frame into a BGR OpenCV/Numpy image
            im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

            # Compute latent feature vector for this video frame
            vec = model.get_latent_feature_vector(im)

            # Add feature vector to our list
            latent_vectors.append(vec)
        print('[INFO] Done. Feature vectors computed.')

        # Compute L2 distances
        X = np.array(latent_vectors, dtype=np.float32)
        mt = getattr(faiss, 'METRIC_L2')
        l2norm = np.clip(faiss.pairwise_distances(X, X, mt), 0, None)

        # Cluster the feature vectors using the Frechet Inception Distance 
        print('[INFO] k-medoids clustering ...')
        kmedoids = sklearn_extra.cluster.KMedoids(n_clusters=self.number_of_frames,
            metric='precomputed',
            method='pam',
            init='k-medoids++',
            random_state=0).fit(l2norm)
        indices = sorted(kmedoids.medoid_indices_.tolist(), reverse=True)
        self.indices_ = [x for x in indices]
        self.labels_ = kmedoids.labels_
        print('[INFO] k-medoids clustering finished.')

        # Retrieve the video frames corresponding to the cluster means
        print('[INFO] Retrieving key frames ...')
        key_frames = []
        counter = -1
        reader = imageio_ffmpeg.read_frames(input_path, pix_fmt='rgb24')
        for raw_frame in tqdm.tqdm(reader):
            counter += 1
            if counter == indices[-1]:
                # Convert video frame into a BGR OpenCV/Numpy image
                im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

                # Add key frame to list
                key_frames.append(im)
                
                # Remove the center we just found
                indices.pop()

            if not indices:
                break
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

        # Initialise video reader
        reader = videosum.VideoReader(input_path, sampling_rate=self.fps)
        w, h = reader.size

        # Initialise Inception network model
        model = videosum.FrechetInceptionDistance('vector')

        # Collect feature vectors for all the frames
        print('[INFO] Collecting feature vectors for all the frames ...')
        self.frame_count_ = 0
        for raw_frame in tqdm.tqdm(reader):
            self.frame_count_ += 1

            # Convert video frame into a BGR OpenCV/Numpy image
            im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

            # Compute latent feature vector for this video frame
            vec = model.get_latent_feature_vector(im)

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
        indices = sorted(kmedoids.medoid_indices_.tolist(), reverse=True)
        self.indices_ = [x for x in indices]
        self.labels_ = kmedoids.labels_
        print('[INFO] k-medoids clustering finished.')

        # Retrieve the video frames corresponding to the cluster means
        print('[INFO] Retrieving key frames ...')
        key_frames = []
        counter = -1
        reader = imageio_ffmpeg.read_frames(input_path, pix_fmt='rgb24')
        for raw_frame in tqdm.tqdm(reader):
            counter += 1
            if counter == indices[-1]:
                # Convert video frame into a BGR OpenCV/Numpy image
                im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

                # Add key frame to list
                key_frames.append(im)
                
                # Remove frame we just found
                indices.pop()
            if not indices:
                break
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

        # Initialise video reader
        reader = videosum.VideoReader(input_path, sampling_rate=self.fps)
        w, h = reader.size

        # Initialise Inception network model
        model = videosum.FrechetInceptionDistance('tensor')

        # Collect feature vectors for all the frames
        print('[INFO] Collecting feature vectors for all the frames ...')
        self.frame_count_ = 0
        for raw_frame in tqdm.tqdm(reader):
            self.frame_count_ += 1

            # Convert video frame into a BGR OpenCV/Numpy image
            im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

            # Compute latent feature vector for this video frame
            tensor = model.get_latent_feature_tensor(im)

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
        print('[INFO] Done. Feature vectors computed.')

        # Compute L2 distances
        print('[INFO] Computing distance matrix ...')
        X = np.array(latent_vectors, dtype=np.float32)
        mt = getattr(faiss, 'METRIC_L2')
        l2norm = np.clip(faiss.pairwise_distances(X, X, mt), 0, None)
        print('[INFO] Done, distance matrix computed.')

        # Cluster the feature vectors
        print('[INFO] k-medoids clustering ...')
        kmedoids = sklearn_extra.cluster.KMedoids(n_clusters=self.number_of_frames, 
            metric='precomputed',
            init='k-medoids++',
            random_state=0).fit(l2norm)
        print('[INFO] k-medoids clustering finished.')
        indices = sorted(kmedoids.medoid_indices_.tolist(), reverse=True)
        self.indices_ = [x for x in indices]
        self.labels_ = kmedoids.labels_

        # Retrieve the video frames corresponding to the cluster medoids
        print('[INFO] Retrieving key frames ...')
        key_frames = []
        counter = -1
        reader = imageio_ffmpeg.read_frames(input_path, pix_fmt='rgb24')
        for raw_frame in tqdm.tqdm(reader):
            counter += 1
            if counter == indices[-1]:
                # Convert video frame into a BGR OpenCV/Numpy image
                im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

                # Add key frame to list
                key_frames.append(im)
                
                # Remove frame we just found
                indices.pop()
            
            # Stop if we found all the frames
            if not indices:
                break
        print('[INFO] Key frames obtained.')

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

        # Create the time segmentation bar under the collage
        if self.time_segmentation:
            # Create an empty segmentation bar
            segbar = np.ones((self.segbar_height, self.collage.shape[1], 3), 
                dtype=np.uint8)
            
            # Colour the background of the segmentation bar
            if self.algo == 'time':
                # The whole bar of the same background colour
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
            
            # Add the key frame vertical lines to the segmentation bar
            for i in self.indices_:
                pos = round(self.collage.shape[1] * i / self.frame_count_)
                key_frame_line_colour = [0, 0, 0]
                segbar[:, pos] = np.array(key_frame_line_colour, dtype=np.uint8)

            # Stich the collage to the time segmentation bar
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
