"""
@brief  This module contains a factory of video readers.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   4 Jan 2023.
"""
import os

# My imports
import .basereader import BaseReader
import .videoreader import VideoReader
import .imagedirreader import ImageDirReader


class ReaderFactory():
    """
    @class ReaderFactory receives the path to the input and other command
           arguments that come from the command line and creates a video
           reader according to the input provided.
    """
    def __new__(cls, path: str, sampling_rate: int = None,
                 pix_fmt: str = 'rgb24') -> BaseReader:
        """
        @param[in]  path           Path to the input file or directory.
        @param[in]  sampling_rate  How many frames per second to get from the
                                   video, i.e. sampling fps.
        @param[in]  pix_fmt        Pixel format string compatible with
                                   imageio_ffmpeg.

        @returns a constructed video reader.
        """
        # If it is a file, it must be a video file
        if os.path.isfile(path):
            return VideoReader(path, sampling_rate, pix_fmt)
        else:
            return ImageDirReader(path)
