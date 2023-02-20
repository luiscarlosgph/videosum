"""
@brief   This module provides a class to easily compute the Frechet Inception 
         Distance between to Numpy/OpenCV BGR images.

@author  Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    21 Jul 2022.
"""

import numpy as np
import torch
import scipy
import numba

# My imports
import videosum


@numba.jit(nopython=True)
def numba_uid(X, eps=1e-6):
    """
    @brief   The Frechet Inception Distance is computed between two 
             distributions. Here we just have a list of feature vectors. 
             Inspired by FID, in this function we define the 
             Univariate Inception Distance (UID), and compute a distance
             matrix from each vector in X to each other vector in X.

    @details In this function we assume that each feature vector contains
             many samples from the same univariate Gaussian distribution.
             That is, to compare two feature vectors we estimate the mean
             and std of each feature vector and compare the univariate 
             Gaussian distributions: 

             uid = sqrt((u_1 - u_2)**2 + (sigma_1 - sigma_2)**2)
             
    @details a distance matrix.
    """
    uid = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            # Compute means
            mu_1 = np.mean(X[i, :])
            mu_2 = np.mean(X[j, :])

            # Compute covariances (variances because it's univariate) 
            C_1 = np.std(X[i, :]) + eps
            C_2 = np.std(X[j, :]) + eps

            # Compute UID
            uid[i, j] = np.sqrt(((mu_1 - mu_2) ** 2) + ((C_1 - C_2) ** 2))
    uid += uid.T
    return uid


class InceptionFeatureExtractor():
    def __init__(self, mode: str, device: str = 'cuda'):
        self.mode = mode
        self.device = device
        
        # Initialise model
        if self.mode == 'vector':
            self.model = videosum.InceptionV3().to(device)
        elif self.mode == 'tensor':
            self.model = videosum.InceptionV3(output_vector=False).to(device)
        else:
            raise ValueError('[ERROR] Output mode not recognized.')
        self.model.eval()
    
    def _get_latent_features(self, im):
        """
        @brief  Given a BGR image, this method computes a 1D feature vector.

        @param[in]  im  Numpy/OpenCV BGR image, shape (H, W, 3),
                        dtype = np.uint8. 
                        Could also be a batch of images with 
                        shape (B, H, W, 3).
        
        @returns a Numpy ndarray of 2048 channels.
        """
        single_image = True if len(im.shape) == 3 else False

        # Convert the image to RGB in range [0, 1]
        im = im[...,::-1].copy().astype(np.float32) / 255.

        # Convert image from Numpy to Pytorch and upload it to GPU
        im = torch.from_numpy(im).to(self.device)

        # Convert image to shape (B, 3, H, W)
        if single_image:
            im = im.permute(2, 0, 1).unsqueeze(0)
        else:
            im = im.permute(0, 3, 1, 2)

        # Run inference on the image and get a vector of shape (2048,)
        with torch.no_grad():
            if single_image:
                output = self.model(im)[0].flatten().cpu().detach().numpy()
            else:
                bs, chan, _, _ = im.shape
                output = torch.squeeze(self.model(im)[0]).cpu().detach().numpy()

        return output

    def get_latent_feature_vector(self, im):
        assert(self.mode == 'vector')
        return self._get_latent_features(im)
    
    def get_latent_feature_tensor(self, im):
        assert(self.mode == 'tensor')
        #return self._get_latent_features(im).reshape((8, 8, 2048))
        return self._get_latent_features(im).reshape((2048, 8, 8))

    @staticmethod
    def frechet_inception_distance(vec1, vec2):
        return InceptionFeatureExtractor._calculate_frechet_distance(
            np.mean(vec1), np.cov(vec1), np.mean(vec2), np.cov(vec2))
    
    #@staticmethod
    #def frechet_inception_distance(mu_sigma_1, mu_sigma_2):
    #    """
    #    @brief Compute the Frechet inception distance between two 
    #           distributions.
    # 
    #    @param[in]  mu_sigma_1  Numpy array of arrays [[means], [covs]] for 
    #                            distribution 1.
    #    @param[in]  mu_sigma_2  Numpy array of arrays [[means], [covs]] for 
    #                            distribution 2.
    #    """
    #    return InceptionFeatureExtractor._calculate_frechet_distance(
    #        mu_sigma_1[0], mu_sigma_1[1], mu_sigma_2[0], mu_sigma_2[0])

    @staticmethod 
    def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        @brief Numpy implementation of the Frechet Distance.
               The code of this method was taken from the pytorch-fid package.
        
        @details The Frechet distance between two multivariate Gaussians 
                 X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is:

                 d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

        Stable version by Dougal J. Sutherland.

        @param[in]  mu1     Numpy array containing the activations of a layer 
                            of the inception net for generated samples.
        @param[in]  mu2     The sample mean over activations, precalculated 
                            on an representative data set.
        @param[in]  sigma1  The covariance matrix over activations for 
                            generated samples.
        @param[in]  sigma2  The covariance matrix over activations, precalculated 
                            on an representative data set.
    
        @returns the Frechet distance.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean) 


if __name__ == '__main__':
    raise RuntimeError('[ERROR] The FID module is not a script.')
