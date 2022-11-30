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
import tempfile
import string
import random

# My imports
import videosum


def create_toy_video(path, num_colours: int = 16, width: int = 640, 
                     height: int = 480, fps=30, pix_fmt_in='rgb24', 
                     pix_fmt_out='yuv420p'):
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
    
    # Save the video to the temporary folder
    writer = imageio_ffmpeg.write_frames(path, (width, height), 
                                         pix_fmt_in='rgb24', 
                                         pix_fmt_out='yuv420p',
                                         fps=fps)

    # Seed the generator
    writer.send(None)  

    # Write video to file
    for frame in frames:
        writer.send(frame)
    writer.close()
     

def random_temp_file_path(length=10):
    """
    @param[in]  length  Length of the filename.
    @returns a path to a temporary file without extension located in /tmp/...
    """
    tmpdir = tempfile.gettempdir()
    letters = string.ascii_lowercase
    fname = ''.join(random.choice(letters) for i in range(length))
    return os.path.join(tmpdir, fname)


class TestVideosum(unittest.TestCase):

    def test_video_reader_1fps(self, duration=16, fps=1, eps=1e-6):
        # Create a dummy video of 16s at 1fps
        path = random_temp_file_path() + '.mp4'
        create_toy_video(path, fps=fps)

        # Check that the video is read as expected
        reader = videosum.VideoReader(path, sampling_rate=1)
        self.assertTrue(np.abs(reader.duration - duration) < eps)
        self.assertTrue(np.abs(reader.fps - fps) < eps)

        # Count the number of frames, should be 16 * 1
        count = 0
        for i in reader:
            count += 1
        self.assertTrue(np.abs(count - duration * fps) < eps)

        # Delete dummy video
        os.unlink(path)

    def test_video_reader_30fps(self, duration=16, fps=30, eps=1e-6):
        # Create a dummy video of 16s at 30fps
        path = random_temp_file_path() + '.mp4'
        create_toy_video(path, fps=fps)

        # Check that the video is read as expected
        reader = videosum.VideoReader(path, sampling_rate=fps)
        self.assertTrue(np.abs(reader.duration - duration) < eps)
        self.assertTrue(np.abs(reader.fps - fps) < eps)
        
        # Count the number of frames, should be 16 * 1
        count = 0
        for i in reader:
            count += 1
        self.assertTrue(np.abs(count - duration * fps) < eps)

        # Delete dummy video
        os.unlink(path)

    def test_reading_30fps_as_1fps(self, duration=16, src_fps=30, dst_fps=1, 
                              eps=1e-6):
        # Create a dummy video of 16s at 30fps
        path = random_temp_file_path() + '.mp4'
        create_toy_video(path, fps=src_fps)

        # Check that the video is read as expected
        reader = videosum.VideoReader(path, sampling_rate=dst_fps)
        self.assertTrue(np.abs(reader.duration - duration) < eps)
        self.assertTrue(np.abs(reader.fps - dst_fps) < eps)
        
        # Count the number of frames, should be 16 * 1
        count = 0
        for i in reader:
            count += 1
        self.assertTrue(np.abs(count - duration * dst_fps) < eps)

        # Remove temporary video file
        os.unlink(path)

    def test_reading_1fps_as_30fps(self, duration=16, src_fps=1, dst_fps=30,
                                   eps=1e-6):
        # Create a dummy video of 16s at 1fps
        path = random_temp_file_path() + '.mp4'
        create_toy_video(path, fps=src_fps)

        # Check that the video is read as expected
        reader = videosum.VideoReader(path, sampling_rate=dst_fps)
        self.assertTrue(np.abs(reader.duration - duration) < eps)
        
        # Count the number of frames, should be 16 * 30
        count = 0
        for i in reader:
            count += 1
        self.assertTrue(np.abs(count - duration * dst_fps) < eps)

        # Remove temporary video file
        os.unlink(path)

    def test_reading_30fps_as_1fps(self, duration=16, src_fps=30, dst_fps=1,
                                   eps=1e-6):
        # Create a dummy video of 16s at 30fps
        path = random_temp_file_path() + '.mp4'
        create_toy_video(path, fps=src_fps)

        # Check that the video is read as expected
        reader = videosum.VideoReader(path, sampling_rate=dst_fps)
        self.assertTrue(np.abs(reader.duration - duration) < eps)
        
        # Count the number of frames, should be 16 * 1
        count = 0
        for i in reader:
            count += 1
        self.assertTrue(np.abs(count - duration * dst_fps) < eps)

        # Remove temporary video file
        os.unlink(path)

    def test_same_frames_read_at_different_fps(self, eps=1e-6):
        # Create a dummy video of 16s at 30fps
        path = random_temp_file_path() + '.mp4'
        create_toy_video(path, fps=30)

        # Read frames at 30fps
        frames_30fps = []
        reader_30fps = videosum.VideoReader(path, sampling_rate=30)
        w, h = reader_30fps.size
        for raw_frame in reader_30fps:
            im = np.frombuffer(raw_frame, 
                               dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()
            frames_30fps.append(im)

        # Read frames at 1fps
        frames_1fps = []
        reader_1fps = videosum.VideoReader(path, sampling_rate=1)
        for raw_frame in reader_1fps:
            im = np.frombuffer(raw_frame, 
                               dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()
            frames_1fps.append(im)

        # Make sure that for each frame of the 1fps video there are 30 identical
        # frames in the 30fps video
        counter = 0
        for f1 in frames_1fps:
            f1 = f1.astype(np.float32)
            init_counter = counter 
            while counter < init_counter + 30:
                f30 = frames_30fps[counter].astype(np.float32)
                diff = np.abs(f30 - f1).sum()
                self.assertTrue(diff < eps)
                counter += 1

        # Remove temporary video file
        os.unlink(path)
         
    def test_time_summary(self, eps=1e-6):
        """
        @brief Simple test to check that the 'time' collage still works with
               newer versions of the dependencies.
        """
        # Create dummy video
        video_path = random_temp_file_path() + '.mp4'
        create_toy_video(video_path, fps=30)

        # Load video
        width = 640
        height = 480
        nframes = 16
        vs = videosum.VideoSummariser('time', nframes, width, height, 
                                      time_segmentation=1, fps=30)

        # Make collage
        new_collage_path = 'test/data/time_dummy.png'
        new_collage = vs.summarise(video_path)
        cv2.imwrite(new_collage_path, new_collage)

        # Compare the collage with the one stored in the test folder
        old_collage_path = 'test/data/collage.png'
        old_collage = cv2.imread(old_collage_path, cv2.IMREAD_UNCHANGED)
        diff = np.sum(np.abs(old_collage.astype(np.float32) - new_collage.astype(np.float32)))
        self.assertTrue(diff < eps)
        
        # Delete dummy video and new collage
        os.unlink(video_path)
        os.unlink(new_collage_path)

    def test_same_inception_collage_at_different_fps(self, eps=1e-6):
        """
        @brief TODO
        """
        # Create dummy video
        video_path = random_temp_file_path() + '.mp4'
        create_toy_video(video_path, fps=30)
        
        # Make collage at different fps
        width = 640
        height = 480
        nframes = 16
        vs_1fps = videosum.VideoSummariser('inception', nframes, width, height, 
                                           time_segmentation=1, fps=1)
        vs_30fps = videosum.VideoSummariser('inception', nframes, width, height, 
                                           time_segmentation=1, fps=30)
        collage_1fps = vs_1fps.summarise(video_path)
        collage_30fps = vs_30fps.summarise(video_path)

        # FIXME: debugging, remove this sentence to save the collage
        #collage_path_1fps = random_temp_file_path() + '.png'
        collage_path_30fps = random_temp_file_path() + '.png'
        #cv2.imwrite(collage_path_1fps, collage_1fps)
        cv2.imwrite(collage_path_30fps, collage_30fps)
        #print("1fps:", collage_path_1fps)
        print("30fps:", collage_path_30fps)

        #diff = np.sum(np.abs(collage_1fps.astype(np.float32) - collage_30fps.astype(np.float32)))

        # TODO

        # Make collage
        #new_collage_path = 'test/data/inception_dummy_new.png'

        collage_1fps = random_temp_file_path() + '.png'
        #new_collage = vs.summarise(video_path)
        #cv2.imwrite(new_collage_path, new_collage)

        # Compare the collage with the one stored in the test folder
        #old_collage_path = 'test/data/inception_dummy.png'
        #old_collage = cv2.imread(old_collage_path, cv2.IMREAD_UNCHANGED)
        #diff = np.sum(np.abs(old_collage.astype(np.float32) - new_collage.astype(np.float32)))
        #self.assertTrue(diff < eps)
        
        # Delete dummy video and new collage
        os.unlink(video_path)
        #os.unlink(new_collage_path)


if __name__ == '__main__':
    unittest.main()
