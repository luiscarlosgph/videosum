"""
@brief  Method get_key_frames_inception() of the class VideoSummariser.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   29 Nov 2022
"""
import numpy as np

# My imports 
import videosum

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


if __name__ == '__main__':
    raise RuntimeError('[ERROR] inception.py cannot be run as a script.')
