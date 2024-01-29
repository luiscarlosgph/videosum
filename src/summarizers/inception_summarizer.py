"""
@brief  This module contains the class InceptionSummarizer, whose
        main purpose is to convert a video into a storyboard 
        by means of clustering the feature vectors of the video
        frames obtained from the InceptionV3 model.

@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   29 Nov 2022.
"""
import numpy as np
import tqdm
import faiss
import sklearn_extra.cluster
import time

# My imports 
from .base_summarizer import BaseSummarizer 
from ..readers.base_reader import BaseReader
from ..inception_feature_extractor import InceptionFeatureExtractor


class InceptionSummarizer(BaseSummarizer):
    """
    @class InceptionSummarizer is a class that picks the key frames 
           of a video clustering the InceptionV3 feature vectors of 
           the frames of the video.
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

    def get_key_frames(self, eps=1e-6, batch_size=64):
        """
        @brief Get a list of key video frames. 
        @details They key frames are selected by unsupervised 
                 clustering of latent feature vectors corresponding 
                 to the video frames.
                 The latent feature vector is obtained from 
                 Inception V3 trained on ImageNet. 
                 The clustering method used is kmedoids.  

        @param[in]  input_path      Path to the video file.
        @param[in]  time_smoothing  Weight of the time smoothing 
                                    factor. 
                                    Higher values make the 
                                    clustering closer to the 'time' 
                                    method. The maximum value is 1.
        @param[in]  batch_size      Size of the batch of images that 
                                    will be passed to the Inception 
                                    Feature Extractor.
        
        @returns a list of Numpy/BGR images, range [0.0, 1.0], 
                 dtype = np.uint8. 
        """
        latent_vectors = []

        # Get video reader info
        w = self.reader.width
        h = self.reader.height

        # Initialise Inception network model
        model = InceptionFeatureExtractor('vector')

        # Collect feature vectors for all the frames
        print('[INFO] Collecting feature vectors for all ' \
            + 'the frames ...')
        tic = time.time() 
        finished = False
        while not finished:
            # Collect a batch of frames from the video
            frame_batch = []
            for raw_frame in reader:
                # Convert video frame into a BGR OpenCV/Numpy image
                im = np.frombuffer(raw_frame, 
                    dtype=np.uint8).reshape(
                        (h, w, 3))[...,::-1].copy()

                # Add frame to batch
                frame_batch.append(im)
                
                # Check if batch is full
                if len(frame_batch) == batch_size:
                    break
            
            # If we have less frames than expected, we are 
            # finished reading the video
            if len(frame_batch) < batch_size:
                finished = True
            
            # In case the video has a number of frames that is a 
            # multiple of the batch size
            if len(frame_batch) > 0:
                # Compute latent feature vector for this batch 
                # of video frames
                vec = model.get_latent_feature_vector(
                    np.array(frame_batch))

                # Add feature vectors to our list
                latent_vectors += [ vec[i] \
                    for i in range(vec.shape[0]) ]
        toc = time.time()
        print('[INFO] Feature vectors computed in {} seconds.'.format(
            toc - tic))

        # There is no point to cluster if we have less frames in 
        # the video than the number of frames that fit in the 
        # storyboard
        if len(latent_vectors) < self.number_of_frames:
            # All the frames go in the collage 
            self.indices_ = list(range(len(latent_vectors)))
            self.labels_ = list(range(len(latent_vectors)))
        else:
            # Compute L2 distances
            X = np.array(latent_vectors, dtype=np.float32)
            mt = getattr(faiss, 'METRIC_L2')
            l2norm = np.clip(faiss.pairwise_distances(X, X, mt), 0, 
                None)

            # Minmax normalisation of the distance matrix
            l2norm /= np.max(l2norm)

            # Compute the distance matrix with time smoothing 
            # if requested
            fdm = videosum.VideoSummariser.frame_distance_matrix(
                l2norm.shape[0])
            dist = (1. - self.time_smoothing) * l2norm \
                + self.time_smoothing * fdm

            # Cluster the feature vectors
            print('[INFO] k-medoids clustering ...')
            tic = time.time()
            kmedoids = sklearn_extra.cluster.KMedoids(
                n_clusters=self.number_of_frames,
                metric='precomputed',
                method='pam',
                init='k-medoids++',
                random_state=0).fit(dist)
            self.indices_ = kmedoids.medoid_indices_
            self.labels_ = kmedoids.labels_
            toc = time.time()
            msg_prefix = '[INFO] k-medoids clustering finished in'
            print('{} {} seconds.'.format(msg_prefix, toc - tic))

        # Retrieve the video frames corresponding to the cluster means
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
                    dtype=np.uint8).reshape(
                        (h, w, 3))[...,::-1].copy()

                # Add key frame to list
                key_frames.append(im)
                
            counter += 1
        print('[INFO] Key frames obtained.')

        return key_frames


if __name__ == '__main__':
    raise RuntimeError(
        '[ERROR] inception_summarizer.py cannot be run as a script.')
