"""
@brief  Class to summarise a video as a storyboard of frames evenly 
        spaced.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   3 Jan 2023.
"""
import math
import tqdm
import numpy as np

# My imports
from .base_summarizer import BaseSummarizer 


class TimeSummarizer(BaseSummarizer):
    """
    @class TimeSummarizer is a class that picks the key frames of a video
           in an evenly spaced manner.
    """

    def __init__(self, 
                 reader: BaseReader, 
                 number_of_frames: int = 100, 
                 width: int = 1920, 
                 height: int = 1080,
                 time_segmentation: bool = False, 
                 segbar_height: int = 32, 
                 compute_fid: bool = False):
        super().__init__(reader, number_of_frames, width, height, 
            time_segmentation, segbar_height, 0., compute_fid)

    def get_key_frames(self):
        """
        @brief Get a list of key frames from the video. 
               The key frames are simply evenly spaced along the 
               video.
        
        @returns a list of Numpy/OpenCV BGR images. 
                 After this method is executed the list self.indices_ 
                 will hold the list of frame indices that represent 
                 the key frames.
        """
        key_frames = []

        # Initialise video reader
        reader = videosum.VideoReader(input_path, 
            sampling_rate=self.fps)

        # Get the properties of the video
        w, h = reader.size
        nframes = videosum.VideoReader.num_frames(input_path, self.fps)
        
        # We know the key frames straight away 
        self.indices_ = np.round(np.linspace(0, nframes - 1, 
            num=self.number_of_frames)).astype(int)

        frame_count = 0
        for raw_frame in tqdm.tqdm(reader):
            if frame_count in self.indices_: 
                # Convert video frame into a BGR OpenCV/Numpy image
                im = np.frombuffer(raw_frame, 
                        dtype=np.uint8).reshape((
                            h, w, 3))[...,::-1].copy()

                # Insert video frame in our list of key frames
                key_frames.append(im)
            frame_count += 1

        # Build self.labels_ based on self.indices_
        self.labels_ = []
        for i in range(nframes):
            min_dist = np.inf
            min_idx = -1
            for j in range(len(self.indices_)):
                if np.abs(i - self.indices_[j]) < min_dist:
                    min_dist = np.abs(i - self.indices_[j])
                    min_idx = j
            self.labels_.append(min_idx)

        return key_frames


if __name__ == '__main__':
    raise RuntimeError('[ERROR] time_summarizer.py cannot be run as a script.')
