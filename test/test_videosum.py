"""
@brief  Unit tests to check the videosum package.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   7 June 2022.
"""
import unittest
import cv2
import numpy as np
import seaborn as sns
import tempfile
import uuid
import imageio_ffmpeg 
import os

# My imports
import videosum


def create_toy_video(num_colours: int = 16, width: int = 640, height: int = 480,
                     fps=30, pix_fmt_in='rgb24', pix_fmt_out='yuv420p'):
    """
    @returns the path to the toy video.
    """
    # Choose 16 random colours
    palette = np.round(np.array(sns.color_palette("Set3", num_colours)) * 255.).astype(np.uint8)

    # Produce all the frames of the video, one per colour
    frames = []
    for i in range(num_colours):
        for j in range(fps):
            im = np.ones((height, width, 3), dtype=np.uint8) * palette[i]
            frames.append(im)
    
    # Build path to the output video inside the temp folder
    filename = str(uuid.uuid4()) + '.mp4'
    path = os.path.join(tempfile.gettempdir(), filename)
    print('Video path:', path)

    # Save the video to the temporary folder
    writer = imageio_ffmpeg.write_frames(path, (width, height), 
                                         pix_fmt_in='rgb24', 
                                         pix_fmt_out='yuv420p',
                                         fps=fps)
    writer.send(None)  # seed the generator
    for frame in frames:
        writer.send(frame)
    writer.close()
     
    return path


class TestVideosum(unittest.TestCase):
    

    def test_time_summary(self, eps=1e-6):
        """
        @brief Simple test to check that the collage still works with
               newer versions of the dependencies.
        """
        # Create dummy video
        video_path = create_toy_video()

        # Load video
        width = 640
        height = 480
        nframes = 16
        vs = videosum.VideoSummariser('time', nframes, width, height, 
                                      time_segmentation=1, fps=1)

        # Make collage
        new_collage_path = 'test/data/time_dummy_new.png'
        new_collage = vs.summarise(video_path)
        cv2.imwrite(new_collage_path, new_collage)

        # Compare the collage with the one stored in the test folder
        old_collage_path = 'test/data/time_dummy.png'
        old_collage = cv2.imread(old_collage_path, cv2.IMREAD_UNCHANGED)
        diff = np.sum(np.abs(old_collage.astype(np.float32) - new_collage.astype(np.float32)))
        self.assertTrue(diff < eps)
        
        # Delete dummy video and new collage
        os.unlink(video_path)
        os.unlink(new_collage_path)


if __name__ == '__main__':
    unittest.main()
