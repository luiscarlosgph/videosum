"""
@brief  Video reader class to wrap whatever way we choose to read videos 
        inside the same API.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   22 Nov 2022.
"""
import imageio_ffmpeg


class VideoReader:

    def __init__(self, path: str, sampling_rate: int = None, pix_fmt='rgb24'):
        """
        @param[in]  path           Path to the video.
        @param[in]  sampling_rate  How many frames per second to get from the
                                   video, i.e. sampling fps.
        @param[in]  pix_fmt        Pixel format string compatible with
                                   imageio_ffmpeg.
        """
        # Store parameters
        self.path = path
        self.sampling_rate = sampling_rate
        self.pix_fmt = pix_fmt
        
        # Open video reader
        if self.sampling_rate is None:
            self.reader_ = imageio_ffmpeg.read_frames(self.path,
                input_params=['-hide_banner'],
                pix_fmt=self.pix_fmt)
        else:
            self.reader_ = imageio_ffmpeg.read_frames(self.path, 
                pix_fmt=self.pix_fmt, 
                input_params=['-hide_banner'],
                output_params=[
                    '-filter:v', "fps={}".format(self.sampling_rate),
                ])

        # Get videeo info
        self.meta_ = self.reader_.__next__()

    def __iter__(self):
        return self

    def __next__(self):
        return self.reader_.__next__()

    @property
    def width(self):
        return self.meta_['size'][0]

    @property
    def height(self):
        return self.meta_['size'][1]

    @property
    def size(self):
        return self.meta_['size']

    @property
    def duration(self):
        return self.meta_['duration']

    @property
    def fps(self):
        return self.meta_['fps']

    @staticmethod
    def num_frames(path: str, sampling_rate: int = None):
        """
        @brief Function to count the number of frames in a video.
        @param[in]  path           Path to the video file.
        @param[in]  sampling_rate  At how many fps you want to read the video.
        @returns the number of frames as an integer.
        """
        # Create video reader
        if sampling_rate is None:
            reader = imageio_ffmpeg.read_frames(path, pix_fmt='rgb24', 
                input_params=['-hide_banner'])
        else:
            reader = imageio_ffmpeg.read_frames(path, pix_fmt='rgb24', 
                input_params=['-hide_banner'],
                output_params=['-filter:v', "fps={}".format(sampling_rate)])

        # Read videeo info
        _ = reader.__next__()

        # Count the number of frames
        count = 0
        for _ in reader:
            count += 1

        return count


if __name__ == '__main__':
    raise RuntimeError('[ERROR] The module videosum.videoreader is not a script.')

