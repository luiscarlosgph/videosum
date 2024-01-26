"""
@brief  This module contains the class UidSummarizer, whose main 
        purpose is to summarize a video as a storyboard where the 
        frame descriptor of each frame consists of the mean and the 
        variance of the InceptionV3 feature vector for the frame. 

@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   1 Dec 2022.
"""
import tqdm
import numpy as np
import sklearn_extra.cluster

# My imports
from .base_summarizer import BaseSummarizer 
from ..readers.base_reader import BaseReader


class UidSummarizer(BaseSummarizer):
    """
    @class UidSummarizer computes the distance between two images 
           their InceptionV3 latent space vectors are computed, a 
           univariate Gaussian is estimated for each of the two latent 
           vectors, and the 2-Wasserstein distance is computed between 
           the two Gaussians and used as clustering metric for 
           k-medoids. Uid stands for Univariate Inception Distance.
    """
    
    def __init__(self,
                 reader: BaseReader, 
                 number_of_frames: int = 100, 
                 width: int = 1920, 
                 height: int = 1080,
                 time_segmentation: bool = False, 
                 segbar_height: int = 32, 
                 time_smoothing: float = 0.,
                 compute_fid: bool = False):
        super().__init__(reader, number_of_frames, width, height, 
                         time_segmentation, segbar_height, 
                         time_smoothing, compute_fid)

    def get_key_frames_uid(self):
        """
        @brief Get a list of key video frames. 
        @details They key frames are selected by unsupervised 
                 clustering of latent feature vectors corresponding 
                 to the video frames.
                 The latent feature vector is obtained from 
                 Inception V3 trained on ImageNet. The clustering 
                 method used is kmedoids.  

        @param[in]  input_path      Path to the video file.
        @param[in]  time_smoothing  Weight of the time smoothing 
                                    factor. 
                                    Higher values make the clustering 
                                    closer to the 'time' method. 
                                    The maximum value is 1.
        
        @returns a list of Numpy/BGR images, range [0.0, 1.0], 
                 dtype = np.uint8. 
        """
        latent_vectors = []

        # Initialise video reader
        reader = videosum.VideoReader(input_path, 
                                      sampling_rate=self.fps, 
                                      pix_fmt='rgb24')
        w, h = reader.size

        # Initialise Inception network model
        model = videosum.InceptionFeatureExtractor('vector')

        # Collect feature vectors for all the frames
        print('[INFO] Collecting feature vectors for ' \
            + 'all the frames ...')
        self.frame_count_ = 0
        for raw_frame in tqdm.tqdm(reader):
            self.frame_count_ += 1

            # Convert video frame into a BGR OpenCV/Numpy image
            im = np.frombuffer(raw_frame, 
                dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

            # Compute latent feature vector for this video frame
            vec = model.get_latent_feature_vector(im)

            # Add feature vector to our list
            latent_vectors.append(vec)
        print('[INFO] Done. Feature vectors computed.')

        # Compute distance matrix
        print('[INFO] Computing distance matrix ...')
        X = videosum.numba_uid(np.array(latent_vectors))
        print('[INFO] Done, distance matrix computed.')

        # Minmax normalisation of distance matrix
        X /= np.max(X)

        # Time smoothing 
        fdm = videosum.VideoSummariser.frame_distance_matrix(
            X.shape[0])
        dist = (1. - time_smoothing) * X + time_smoothing * fdm

        # Cluster the feature vectors using the Frechet Inception 
        # Distance 
        print('[INFO] k-medoids clustering ...')
        kmedoids = sklearn_extra.cluster.KMedoids(
            n_clusters=self.number_of_frames, 
            metric='precomputed',
            method='pam',
            init='k-medoids++',
            random_state=0).fit(dist)
        self.indices_ = kmedoids.medoid_indices_
        self.labels_ = kmedoids.labels_
        print('[INFO] k-medoids clustering finished.')

        # Retrieve the video frames corresponding to the cluster 
        # means
        print('[INFO] Retrieving key frames ...')
        key_frames = []
        reader = videosum.VideoReader(input_path, 
            sampling_rate=self.fps, 
            pix_fmt='rgb24')
        counter = 0
        for raw_frame in tqdm.tqdm(reader):
            if counter in self.indices_:
                # Convert video frame into a BGR OpenCV/Numpy image
                im = np.frombuffer(raw_frame, 
                    dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

                # Add key frame to list
                key_frames.append(im)
                
            counter += 1
        print('[INFO] Key frames obtained.')

        return key_frames


if __name__ == '__main__':
    raise RuntimeError('[ERROR] uid.py cannot be executed as a script.')
