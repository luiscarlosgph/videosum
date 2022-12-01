"""
@brief  Summarise video as a storyboard where the frame descriptors are the
        mean and the variance of the InceptionV3 feature vector. The 
        distance metric used for clustering is Wasserstein-2. 

@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   1 Dec 2022.
"""
import tqdm
import numpy as np
import sklearn_extra.cluster
import imageio_ffmpeg

# My imports
import videosum

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
        reader = videosum.VideoReader(input_path, sampling_rate=self.fps, 
                                      pix_fmt='rgb24')
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
        self.indices_ = kmedoids.medoid_indices_
        self.labels_ = kmedoids.labels_
        print('[INFO] k-medoids clustering finished.')

        # Retrieve the video frames corresponding to the cluster means
        print('[INFO] Retrieving key frames ...')
        key_frames = []
        reader = videosum.VideoReader(input_path, sampling_rate=self.fps, 
                                      pix_fmt='rgb24')
        counter = 0
        for raw_frame in tqdm.tqdm(reader):
            if counter in self.indices_:
                # Convert video frame into a BGR OpenCV/Numpy image
                im = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()

                # Add key frame to list
                key_frames.append(im)
                
            counter += 1
        print('[INFO] Key frames obtained.')

        return key_frames


if __name__ == '__main__':
    raise RuntimeError('[ERROR] fid.py cannot be executed as a script.')
