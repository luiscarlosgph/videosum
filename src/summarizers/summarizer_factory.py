"""
@brief  This module contains a factory of video summarizers.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   4 Jan 2023.
"""
import os

# My imports
from ..readers.base_reader import BaseReader
from .base_summarizer import BaseSummarizer
from .time_summarizer import TimeSummarizer


class SummarizerFactory():
    """
    @class SummarizerFactory creates a video summarizer that selects the
           key frames of the video according to a requested method.
    """

    def __new__(cls, 
                algo: str, 
                reader: BaseReader,
                number_of_frames: int = 100, 
                width: int = 1920, 
                height: int = 1080,
                time_segmentation: bool = False, 
                segbar_height: int = 32, 
                time_smoothing: float = 0., 
                compute_fid: bool = False) -> BaseSummarizer:
        """
        @param[in]  algo               Name of the summarization algorithm 
                                       that you want to use.
        @param[in]  reader             Already constructed video reader that 
                                       will be used by the summarizer to 
                                       read the frames.
        @param[in]  number_of_frames   Number of frames of the video 
                                       you want to see in the 
                                       storyboard.
        @param[in]  width              Width of the summary 
                                       storyboard.
        @param[in]  height             Height of the summary 
                                       storybard.
        @param[in]  time_segmentation  Set to True to show a time 
                                       segmentation under the 
                                       storyboard.
        @param[in]  segbar_height      Height in pixels of the time
                                       segmentation bar.
        @param[in]  time_smoothing     Weight in the range [0., 1.] that 
                                       regulates the importance of time for 
                                       clustering frames. A higher weight 
                                       will result in a segmentation of 
                                       frames over time closer to that of 
                                       the time method.
        @param[in]  compute_fid        Set it to True if you want a 
                                       report on the FID of the 
                                       summary to the whole video.
        @returns a constructed video summarizer for the requested method.
        """
        if algo == 'time': 
            return TimeSummarizer(reader, number_of_frames, width, height,
                time_segmentation, segbar_height, compute_fid)
        else:
            error_msg = "[ERROR] The summarization method {} is not " \
                + "recognized."
            raise ValueError(error_msg)


if __name__ == '__main__':
    raise RuntimeError('[ERROR] summarizer_factory.py cannot be run as a script.')
