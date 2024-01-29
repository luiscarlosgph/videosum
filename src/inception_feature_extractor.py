"""
@brief   This module provides a class to easily obtain the feature vector
         of an image in the latent space of Inception V3.

@author  Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    29 Jan 2023.
"""

import numpy as np
import torch
import scipy
import numba

# My imports
from .inception import InceptionV3


class InceptionFeatureExtractor():
    """
    @class InceptionFeatureExtractor is a class designed to generate the
           feature vector/tensor of a given input image in the latent space
           of the InceptionV3 model.
    """

    def __init__(self, mode: str, device: str = 'cuda'):
        self.mode = mode
        self.device = device

        print('DEBUG: inception feature extractor 1')
        
        # Initialise model
        if self.mode == 'vector':
            print('DEBUG: inception feature extractor 2')
            self.model = InceptionV3().to(device)
            print('DEBUG: inception feature extractor 3')
        elif self.mode == 'tensor':
            self.model = InceptionV3(output_vector=False).to(device)
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

    
if __name__ == '__main__':
    raise RuntimeError('[ERROR] The inception_feature_extractor.py module is not a script.')
