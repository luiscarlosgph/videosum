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
#import skimage.util

# My imports
import videosum


def create_toy_video(path, num_colours: int = 12, width: int = 640, 
                     height: int = 480, fps=30, pix_fmt_in='rgb24', 
                     pix_fmt_out='yuv420p'):
    """
    @brief Creates a video that displays 12 colours, one colour per second.
    @returns the path to the toy video.
    """
    # Choose 16 random colours
    palette = np.round(np.array(sns.color_palette("Set3", num_colours)) * 255.).astype(np.uint8)

    # Produce all the frames of the video, one per colour
    frames = []
    for i in range(num_colours):
        for j in range(fps):
            # Create video frame
            im = np.ones((height, width, 3), dtype=np.uint8) * palette[i]
            
            # Add frame to the video
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

    def test_readme_example(self):
        """
        @brief This is the example used in the README, so it must work.
        """
        # Create and save dummy video to disk
        video_path = random_temp_file_path() + '.mp4'
        create_toy_video(video_path, fps=30)

        # Create video reader
        vr = videosum.ReaderFactory(video_path)
        
        # Create video summarizer 
        nframes = 9
        width = 1920
        height = 1080
        vs = videosum.SummarizerFactory('time', vr, nframes, width, height)
        
        # Create storyboard
        im = vs.summarize()
        
        # Save storyboard to file
        cv2.imwrite('/tmp/storyabord.jpg', im)

        # Retrieve a list of Numpy/OpenCV BGR images corresponding to the 
        # key frames of the video
        key_frames = vs.get_key_frames()

        # Print the video frame indices of the key frames, available after 
        # calling summarize() or get_key_frames()
        print(vs.indices_)

        # Print the video frame cluster labels, available after calling 
        # summarize() or get_key_frames()
        print(vs.labels_)

    '''
    def test_video_reader_1fps(self, duration=12, fps=1, eps=1e-6):
        """
        @brief If the toy video displays 12 solid colour frames at 1fps, 
               we should read 12 frames.
        """
        # Create a dummy video of 12s at 1fps
        path = random_temp_file_path() + '.mp4'
        create_toy_video(path, fps=fps)

        # Check that the video is read as expected
        reader = videosum.VideoReader(path, sampling_rate=1)
        self.assertTrue(np.abs(reader.duration - duration) < eps)
        self.assertTrue(np.abs(reader.fps - fps) < eps)

        # Count the number of frames, should be 12 * 1
        count = 0
        for i in reader:
            count += 1
        self.assertTrue(np.abs(count - duration * fps) < eps)

        # Delete dummy video
        os.unlink(path)

    def test_video_reader_30fps(self, duration=12, fps=30, eps=1e-6):
        """
        @brief We should read 12 * 30 frames from the toy video.
        """
        # Create a dummy video of 12s at 30fps
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

    def test_reading_30fps_as_1fps(self, duration=12, src_fps=30, dst_fps=1, 
                              eps=1e-6):
        """
        @brief Even when the toy video is recorded at 30 fps, if we read it
               at 1fps we should read 12 frames, one per second.
        """
        # Create a dummy video of 12s at 30fps
        path = random_temp_file_path() + '.mp4'
        create_toy_video(path, fps=src_fps)

        # Check that the video is read as expected
        reader = videosum.VideoReader(path, sampling_rate=dst_fps)
        self.assertTrue(np.abs(reader.duration - duration) < eps)
        self.assertTrue(np.abs(reader.fps - dst_fps) < eps)
        
        # Count the number of frames, should be 12 * 1
        count = 0
        for i in reader:
            count += 1
        self.assertTrue(np.abs(count - duration * dst_fps) < eps)

        # Remove temporary video file
        os.unlink(path)

    def test_reading_1fps_as_30fps(self, duration=12, src_fps=1, dst_fps=30,
                                   eps=1e-6):
        """
        @brief Even when the toy video is saved at 1fps we should be able to
               read it at 30fps (with duplicated frames).
        """
        # Create a dummy video of 12s at 1fps
        path = random_temp_file_path() + '.mp4'
        create_toy_video(path, fps=src_fps)

        # Check that the video is read as expected
        reader = videosum.VideoReader(path, sampling_rate=dst_fps)
        self.assertTrue(np.abs(reader.duration - duration) < eps)
        
        # Count the number of frames, should be 12 * 30
        count = 0
        for i in reader:
            count += 1
        self.assertTrue(np.abs(count - duration * dst_fps) < eps)

        # Remove temporary video file
        os.unlink(path)

    def test_same_frames_read_at_different_fps(self, eps=1e-6):
        # Create a dummy video of 12s at 30fps
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
        @brief Check that the summary for the 'time' method is identical to
               the one created by previous versions of the package (which
               was visually inspected and appeared correct).
        """
        # Create dummy video
        video_path = random_temp_file_path() + '.mp4'
        create_toy_video(video_path, fps=30)

        # Load video
        width = 640
        height = 480
        nframes = 12
        vs = videosum.VideoSummariser('time', nframes, width, height, 
                                      time_segmentation=1, fps=30)

        # Make collage
        new_collage_path = 'test/data/time_dummy.png'
        new_collage = vs.summarise(video_path)
        cv2.imwrite(new_collage_path, new_collage)

        # Compare the collage with the one stored in the test folder
        old_collage_path = 'test/data/test_time.png'
        old_collage = cv2.imread(old_collage_path, cv2.IMREAD_UNCHANGED)
        diff = np.sum(np.abs(old_collage.astype(np.float32) - new_collage.astype(np.float32)))
        self.assertTrue(diff < eps)
        
        # Delete dummy video and new collage
        os.unlink(video_path)
        os.unlink(new_collage_path)

    def test_inception_summary(self, eps=1e-6):
        """
        @brief Check that the summary for the 'inception' method is identical to
               the one created by previous versions of the package (which 
               appeared to be correct on visual inspection).
        """
        # Create dummy video
        video_path = random_temp_file_path() + '.mp4'
        create_toy_video(video_path, fps=30)
        
        # Load video
        width = 640
        height = 480
        nframes = 12
        vs = videosum.VideoSummariser('inception', nframes, width, height, 
                                      time_segmentation=1, fps=30)
        
        # Make collage
        new_collage_path = 'test/data/inception_dummy.png'
        new_collage = vs.summarise(video_path)
        cv2.imwrite(new_collage_path, new_collage)
        
        # Compare the collage with the one stored in the test folder
        old_collage_path = 'test/data/test_inception.png'
        old_collage = cv2.imread(old_collage_path, cv2.IMREAD_UNCHANGED)
        diff = np.sum(np.abs(old_collage.astype(np.float32) - new_collage.astype(np.float32)))
        self.assertTrue(diff < eps)

        # Delete dummy video and new collage
        os.unlink(video_path)
        os.unlink(new_collage_path)

    
    def test_same_inception_collage_at_different_fps(self, eps=1e-6):
        """
        @brief No matter the FPS of the video there are 12 solid colors in the
               frames of the video and they should be present in the collage.
        """
        # Create dummy video
        video_path = random_temp_file_path() + '.mp4'
        create_toy_video(video_path, fps=30)
        
        # Make collage at different fps
        width = 640
        height = 480
        nframes = 12
        vs_30fps = videosum.VideoSummariser('inception', nframes, width, height, 
                                           time_segmentation=0, fps=30)
        vs_60fps = videosum.VideoSummariser('inception', nframes, width, height, 
                                           time_segmentation=0, fps=60)
        collage_30fps = vs_30fps.summarise(video_path)
        collage_60fps = vs_60fps.summarise(video_path)
            
        # Make sure that there is no difference, the key frames should be the same
        diff = np.sum(np.abs(collage_30fps.astype(np.float32) - collage_60fps.astype(np.float32)))
        self.assertTrue(diff < eps)

        # Delete dummy video and new collage
        os.unlink(video_path)

    def test_that_indices_and_labels_are_correctly_generated(self, eps=1e-6):
        """
        @brief self.labels_ and self.indices_ must be generate after the video
               summarisation.
        """
        # Create dummy video
        video_path = random_temp_file_path() + '.mp4'
        create_toy_video(video_path, fps=30)
        
        # Create video summarisers for all the methods
        width = 640
        height = 480
        nframes = 12
        fps = 30
        methods = {}
        for method in videosum.VideoSummariser.ALGOS:
            methods[method] = videosum.VideoSummariser(method, nframes, width, height, 
                                                       time_segmentation=1, fps=fps)

        # Check that self.indices_ and self.labels_ are None as we have not
        # run any summary
        for method in methods:
            self.assertTrue(methods[method].indices_ is None)
            self.assertTrue(methods[method].labels_ is None)

        # Run all the summary methods
        for method in methods:
            # Summarise the video into a storyboard
            methods[method].summarise(video_path)

            # Assert that the number of labels corresponds to the number of 
            # frames clustered
            self.assertTrue(len(methods[method].labels_) == nframes * fps)

            # Assert that we have the indices of each cluster center
            indices = methods[method].indices_
            self.assertTrue(len(indices) == nframes)

            # The toy video displays 12 different colours, therefore the 
            # indices of the cluster centers should be different
            self.assertTrue(np.unique(indices).shape[0] == nframes)

        # Delete dummy video and new collage
        os.unlink(video_path)

    def test_fid_method(self, eps=1e-6):
        """
        @brief The 'fid' method should not produce a result different to that
               of the inception method as the toy video contains 12 different
               colours that should be shown in the collage.
        """
        # Create dummy video
        video_path = random_temp_file_path() + '.mp4'
        create_toy_video(video_path, fps=30)

        # Define collage properties
        width = 640
        height = 480
        nframes = 12
        
        # Produce the collage with the inception method
        inception_vs = videosum.VideoSummariser('inception', nframes, width, height, 
                                                time_segmentation=0, fps=30)
        inception_collage = inception_vs.summarise(video_path)

        # Produce the collage with the the fid method
        fid_vs = videosum.VideoSummariser('fid', nframes, width, height, 
                                          time_segmentation=0, fps=30)
        fid_collage = fid_vs.summarise(video_path)

        # Check that the collages produced by both methods are identical
        diff = np.abs(inception_collage.astype(np.float32) - fid_collage.astype(np.float32))
        self.assertTrue(np.sum(diff) < eps)

        # Delete dummy video and new collage
        os.unlink(video_path)
    
    def test_scda_method(self, eps=1e-6):
        """
        @brief The 'scda' method should produce a collage with 12 colours as
               that produced by the inception method. 
        """
        # Create dummy video
        video_path = random_temp_file_path() + '.mp4'
        create_toy_video(video_path, fps=30)

        # Define collage properties
        width = 640
        height = 480
        nframes = 12
        
        # Produce the collage with the inception method
        inception_vs = videosum.VideoSummariser('inception', nframes, width, height, 
                                                time_segmentation=0, fps=30)
        inception_collage = inception_vs.summarise(video_path)

        # Produce the collage with the the scda method
        scda_vs = videosum.VideoSummariser('scda', nframes, width, height, 
                                          time_segmentation=0, fps=30)
        scda_collage = scda_vs.summarise(video_path)

        # Check that the collages produced by both methods are identical
        diff = np.abs(inception_collage.astype(np.float32) - scda_collage.astype(np.float32))
        self.assertTrue(np.sum(diff) < eps)

        # Delete dummy video
        os.unlink(video_path)

    def test_transition_frames(self):
        """
        @brief If we create a video of 12s at 1fps, displaying a different
               colour for every frame, the key frames are the same as the 
               transition frames.
        """
        # Create dummy video
        video_path = random_temp_file_path() + '.mp4'
        create_toy_video(video_path, fps=1)

        # Define collage properties
        width = 640
        height = 480
        number_of_images = 12
        
        # Summarise video
        vs = videosum.VideoSummariser('inception', number_of_images, width, 
                                      height, time_segmentation=0, fps=1)
        inception_collage = vs.summarise(video_path)

        # Assert that there are 12 key frames
        self.assertTrue(len(vs.labels_) == number_of_images)

        # Assert that there are 12 transition frames
        ti = vs.transition_indices(vs.indices_)
        self.assertEqual(len(ti), number_of_images)

        # The key frames and the transition frames should be the same
        self.assertEqual(set(vs.indices_), set(ti))

        # Delete dummy video
        os.unlink(video_path)
    '''

if __name__ == '__main__':
    unittest.main()
