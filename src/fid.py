"""
@brief   This module provides a class to easily compute the Frechet Inception 
         Distance between to Numpy/OpenCV BGR images.

@author  Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    21 Jul 2022.
"""

import numpy as np
import torch
import scipy

# My imports
import videosum


class FrechetInceptionDistance():
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = videosum.InceptionV3().to(device)
        self.model.eval()
    
    def get_latent_feature_vector(self, im):
        """
        @brief  Given an image, this method computes a 1D feature vector.

        @param[in]  im  Numpy/OpenCV BGR image, shape (H, W, 3),
                        dtype = np.uint8.
        
        @returns a Numpy feature vector of shape (2048,).
        """
        # Convert the image to RGB in range [0, 1]
        im = im[...,::-1].copy().astype(np.float32) / 255.

        # Convert image from Numpy to Pytorch and upload it to GPU
        im = torch.from_numpy(im).to(self.device)

        # Convert image to shape (1, 3, H, W)
        im = im.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            # Run inference on the image and get a vector of shape (2048,)
            output = self.model(im)[0].flatten().cpu().detach().numpy()

        return output
    
    @staticmethod
    def frechet_inception_distance(mu_sigma_1, mu_sigma_2):
        """
        @brief Compute the Frechet inception distance between two 
               distributions.

        @param[in]  mu_sigma_1  Numpy array of arrays [[means], [covs]] for 
                                distribution 1.
        @param[in]  mu_sigma_2  Numpy array of arrays [[means], [covs]] for 
                                distribution 2.
        """
        return FrechetInceptionDistance._calculate_frechet_distance(
            mu_sigma_1[0], mu_sigma_1[1], mu_sigma_2[0], mu_sigma_2[0])

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
