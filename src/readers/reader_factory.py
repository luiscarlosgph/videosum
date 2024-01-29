"""
@brief  This module contains a factory of video readers.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   4 Jan 2023.
"""
import os

# My imports
from .base_reader import BaseReader
from .video_reader import VideoReader
from .imagedir_reader import ImageDirReader


class ReaderFactory():
    """
    @class ReaderFactory receives the path to the input and other command
           arguments that come from the command line and creates a video
           reader according to the input provided.
    """
    def __new__(cls, path: str, output_fps: int = None,
                 pix_fmt: str = 'rgb24') -> BaseReader:
        """
        @param[in]  path        Path to the input file or directory.
        @param[in]  output_fps  How many frames per second to get from the
                                video, i.e. sampling fps.
        @param[in]  pix_fmt     Pixel format string compatible with
                                imageio_ffmpeg.

        @returns a constructed video reader.
        """
        # If it is a file, it must be a video file
        if os.path.isfile(path):
            return VideoReader(path, output_fps, pix_fmt)
        else:
            return ImageDirReader(path)
