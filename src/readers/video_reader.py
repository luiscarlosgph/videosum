"""
@brief  Video reader class to wrap whatever way we choose to read videos 
        inside the same API.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   22 Nov 2022.
"""
import imageio_ffmpeg

# My imports
from .base_reader import BaseReader


class VideoReader(BaseReader):
    """
    @class VideoReader is a class that helps to read the frames contained
           in a video file such as an mp4.
    """

    def __init__(self, path: str, output_fps: int = None, 
                 pix_fmt: str = 'rgb24'):
        """
        @param[in]  path        Path to the video.
        @param[in]  output_fps  How many frames per second to get from the
                                video, i.e. sampling fps.
        @param[in]  pix_fmt     Pixel format string compatible with
                                imageio_ffmpeg.
        """
        # Store parameters
        self._path = path
        self._output_fps = output_fps
        self._pix_fmt = pix_fmt
        
        # Open video reader
        if self.output_fps is None:
            self.reader_ = imageio_ffmpeg.read_frames(self.path,
                input_params=['-hide_banner'],
                pix_fmt=self.pix_fmt)
        else:
            self.reader_ = imageio_ffmpeg.read_frames(self.path, 
                pix_fmt=self.pix_fmt, 
                input_params=['-hide_banner'],
                output_params=[
                    '-filter:v', "fps={}".format(self.output_fps),
                ])

        # Get videeo info
        self.meta_ = self.reader_.__next__()

    def __next__(self):
        return self.reader_.__next__()
    
    def num_frames(self):
        """
        @brief Function to count the number of frames in a video.
        @returns the number of frames as an integer.
        """
        # Create video reader
        if self.output_fps is None:
            reader = imageio_ffmpeg.read_frames(self.path, pix_fmt='rgb24', 
                input_params=['-hide_banner'])
        else:
            reader = imageio_ffmpeg.read_frames(self.path, pix_fmt='rgb24', 
                input_params=['-hide_banner'],
                output_params=['-filter:v', "fps={}".format(self.output_fps)])

        # Read videeo info
        _ = reader.__next__()

        # Count the number of frames
        count = 0
        for _ in reader:
            count += 1

        return count

    @property
    def width(self):
        return self.meta_['size'][0]

    @property
    def height(self):
        return self.meta_['size'][1]

    #@property
    #def size(self):
    #    return self.meta_['size']

    @property
    def duration(self):
        return self.meta_['duration']

    @property
    def input_fps(self):
        """
        @returns the actual FPS of the original video.
        """
        return self.meta_['fps']

    @property
    def output_fps(self):
        """
        @returns the FPS at which the reader will actually return the frames.
        """
        return self._output_fps

    @property
    def path(self):
        """
        @returns the path of the original input video.
        """
        return self._path 
    
    @property
    def pix_fmt(self):
        return self._pix_fmt


if __name__ == '__main__':
    raise RuntimeError('[ERROR] The module videosum.videoreader is not a script.')

