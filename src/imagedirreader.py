"""
@brief  Image reader class to read a folder of images as a video.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   6 Dec 2023.
"""
import os
import natsort
import PIL
import numpy as np

# My imports
from .reader import Reader


class ImageDirReader(Reader):
    """
    @class ImageDirReader is meant to make the process of reading a video that
           is stored as a folder of images easy.
    """

    def __init__(self, path: str):
        """
        @param[in]  path     Path to the video.
        @param[in]  pix_fmt  Pixel format string compatible with
                             imageio_ffmpeg.
        """
        # Sanity check: make sure that the path points to a folder 
        if not os.path.isdir(path):
            raise ValueError('[ERROR] The path should point to a folder.')

        # Store parameters
        self.path = path
        
        # Get the list of files in the folder
        self._files = natsort.natsorted(os.listdir(folder_path))
        self._index = 0
        
        # We don't know yet the size of the images of this "video"
        self._width = None
        self._height = None

    def __next__(self):
        """
        @returns the next image in the folder.
        """
        im = None
        if self._index < len(self._files):
            im_path = os.path.join(self.path, self._files[self._index])
            im = np.array(PIL.Image.open(im_path).convert('RGB'))
            self._height = im.shape[0]
            self._width = im.shape[1]
            self._index += 1
        return im

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


if __name__ == '__main__':
    raise RuntimeError('[ERROR] The module videosum.imagereader ' \
        + 'is not a script.')
