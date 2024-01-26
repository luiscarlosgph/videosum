"""
@brief   Video frame clustering based on Selective Convolutional Descriptor 
         Aggregation.

@details They key frames are selected by unsupervised clustering of 
         latent feature vectors corresponding to the video frames.
         The latent feature vector is obtained as explained in 
         Wei et al. 2017:

         "Selective Convolutional Descriptor Aggregation for
         Fine-Grained Image Retrieval"
                 
         The only difference with the paper is we use InceptionV3
         to retrieve the latent tensor (as opposed to VGG-16).

@author  Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    1 Dec 2022.
"""

import numpy as np
import tqdm
import sklearn_extra.cluster
import skimage.measure
import scipy.ndimage
import faiss
import numba

# My imports
from .base_summarizer import BaseSummarizer 
from ..readers.base_reader import BaseReader


class ScdaSummarizer(BaseSummarizer):
    """
    @class ScdaSummarizer is a class that picks the key frames of a video
           based on the clustering of SCDA features.
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
                         time_segmentation, segbar_height, time_smoothing, 
                         compute_fid)

    def get_key_frames(self):
        """
        @brief Get a list of key video frames. 
        @returns a list of Numpy/BGR images, range [0.0, 1.0], 
                 dtype = np.uint8. 
        """

        @numba.jit(nopython=True)
        def numba_cosine_dist(X):
            dist = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
            for i in range(X.shape[0]):
                for j in range(i + 1, X.shape[0]):
                    # Compute cosine similarity
                    u = X[i, :]
                    v = X[j, :]
                    u_l2 = np.sqrt(np.sum(u ** 2))
                    v_l2 = np.sqrt(np.sum(v ** 2))
                    cos_sim = np.dot(u, v) / (u_l2 * v_l2)
                    

                    # Compute cosine distance
                    #cos_dist = 1 - cos_sim
                    #dist[i, j] = min(0., max(2., cos_dist))
                    dist[i, j] = 1 - cos_sim 
            dist += dist.T
            return dist

        # Initialise video reader
        reader = videosum.VideoReader(input_path, sampling_rate=self.fps,
                                      pix_fmt='rgb24')
        w, h = reader.size

        # Initialise Inception network model
        model = videosum.InceptionFeatureExtractor('tensor')

        # Collect feature vectors for all the frames
        print('[INFO] Collecting feature vectors for all the frames ...')
        latent_vectors = []
        for raw_frame in tqdm.tqdm(reader):
            # Convert video frame into a BGR OpenCV/Numpy image
            im = np.frombuffer(raw_frame, 
                dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

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
            largest_cc = scipy.ndimage.binary_fill_holes(
                cc == largest).astype(cc.dtype)

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

        # Compute cosine distances
        X = np.array(latent_vectors, dtype=np.float32)
        cos_dist = numba_cosine_dist(X)

        # Minmax normalisation of the l2norm distances
        cos_dist /= np.max(cos_dist)

        # Compute the distance matrix with time smoothing if requested
        fdm = videosum.VideoSummariser.frame_distance_matrix(
            cos_dist.shape[0])
        dist = (1. - self.time_smoothing) * cos_dist \
            + self.time_smoothing * fdm

        # Cluster the feature vectors
        print('[INFO] k-medoids clustering ...')
        kmedoids = sklearn_extra.cluster.KMedoids(
            n_clusters=self.number_of_frames, 
            metric='precomputed',
            init='k-medoids++',
            random_state=0).fit(dist)
        print('[INFO] k-medoids clustering finished.')
        indices = sorted(kmedoids.medoid_indices_.tolist(), reverse=True)
        self.indices_ = [x for x in indices]
        self.labels_ = kmedoids.labels_

        # Retrieve the video frames corresponding to the cluster means
        print('[INFO] Retrieving key frames ...')
        key_frames = []
        reader = videosum.VideoReader(input_path, sampling_rate=self.fps, 
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


