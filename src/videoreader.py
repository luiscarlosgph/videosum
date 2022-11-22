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
        
        # Initialise video reader
        reader = imageio_ffmpeg.read_frames(self.path, pix_fmt=self.pix_fmt) 
        self.meta_ = reader.__next__()
        reader.close()
        
        # Compute sampling interval if requested
        self.sampling_interval = 0
        if self.sampling_rate is not None:
            self.sampling_interval = int(round(float(self.meta_['fps']) / self.sampling_rate))

    def __iter__(self):
        # Initialise video reader
        reader = imageio_ffmpeg.read_frames(self.path, pix_fmt=self.pix_fmt) 
        reader.__next__()
        
        #self.iterator_ = self.reader_.__iter__()
        #return self
        for frame in reader:
            yield frame
        reader.close()


    #def __next__(self):
    #    # Skip some frames if requested
    #    if self.sampling_interval > 0:
    #        for i in range(self.sampling_interval):
    #            self.iterator_.__next__()
    #
    #    return self.iterator_.__next__()

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


if __name__ == '__main__':
    raise RuntimeError('[ERROR] The module videosum.videoreader is not a script.')

