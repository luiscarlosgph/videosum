"""
@brief Method of the class VideoSUmmariser to summarise a video as a storyboard
       of frames evenly spaced.
"""
import math
import tqdm
import numpy as np

# My imports
import videosum


def get_key_frames_time(self, input_path):
        """
        @brief Get a list of key frames from the video. The key frames are
               simply evenly spaced along the video.
        
        @param[in]  input_path  Path to the video file.

        @returns a list of Numpy/OpenCV BGR images. 
                 After this method is executed the list self.indices_ will 
                 hold the list of frame indices that represent the key frames.
        """
        key_frames = []

        # Initialise video reader
        reader = videosum.VideoReader(input_path, sampling_rate=self.fps)
        w, h = reader.size
        nframes = int(math.floor(reader.duration * reader.fps))

        # Collect the collage frames from the video 
        interval = nframes // self.number_of_frames
        counter = interval
        self.labels_ = []
        self.indices_ = []
        self.frame_count_ = 0
        for raw_frame in tqdm.tqdm(reader):
            # Increase the internal frame count of the video summariser
            self.frame_count_ += 1

            # If we have collected all the frames we needed, mic out
            if len(key_frames) == self.number_of_frames:
                break
            
            # If this frame is a key frame...
            counter -= 1
            if counter == 0:
                counter = interval

                # Convert video frame into a BGR OpenCV/Numpy image
                im = np.frombuffer(raw_frame, 
                        dtype=np.uint8).reshape((h, w, 3))[...,::-1].copy()
            
                # Insert video frame in our list of key frames
                key_frames.append(im)
                self.indices_.append(self.frame_count_ - 1)

        # Build self.labels_ based on self.indices_
        for i in range(self.frame_count_):
            min_dist = np.inf
            min_idx = -1
            for j in range(len(self.indices_)):
                if np.abs(i - self.indices_[j]) < min_dist:
                    min_dist = np.abs(i - self.indices_[j])
                    min_idx = j
            self.labels_.append(min_idx)

        return key_frames


if __name__ == '__main__':
    raise RuntimeError('[ERROR] time.py cannot be run as a script.')
