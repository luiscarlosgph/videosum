"""
@brief  Unit tests to check the videosum package.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   7 June 2022.
"""
import unittest
import cv2
import numpy as np

# My imports
import videosum

class TestVideosum(unittest.TestCase):

    def test_collage(self):
        """
        @brief Simple test to check that the collage still works with
               newer versions of the dependencies.
        """
        # Load video
        width = 640
        height = 480
        nframes = 9
        vs = videosum.VideoSummariser(nframes, width, height)

        # Make collage
        new_collage = vs.summarise('test/data/test.mp4')

        # Compare the collage with the one stored
        old_collage = cv2.imread('test/data/test.png')

        self.assertTrue(np.allclose(new_collage, old_collage))


if __name__ == '__main__':
    unittest.main()
