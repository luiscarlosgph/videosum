"""
@brief  Reader base class. This class provides an interface to be implemented
        by children able to read videos in different formats, e.g. as a folder
        of images or as a video file.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   31 Dec 2023.
"""
import abc

class BaseReader(abc.ABC):
    """
    @class BaseReader defines the skeleton that any reader used by the video
           summarizer should implement.
    """
    @abstractmethod
    def __init__(self, path: str, *args, **kwargs)) -> None:
        """
        @param[in]  path  Path to the input. Might be a file or a directory
                          path. This will depend on every particular reader.
        """
        raise NotImplemented()

    @abstractmethod
    def __next__(self):
        """
        @returns the next frame of the video.
        """
        raise NotImplemented()

    @abstractmethod
    def num_frames(self) -> int:
        """
        @returns This method should return the total number of frames of the
                 video.
        """
        raise NotImplemented()

    def __iter__(self):
        return self

    @abstractproperty
    def width(self):
        """
        @returns This method should return the width of all the frames of 
                 the video.
        """ 
        raise NotImplemented()

    @abstractproperty
    def height(self):
        """
        @returns This method should return the height of all the frames of
                 the video.
        """
        raise NotImplemented()

    
if __name__ == '__main__':
    raise RuntimeError('[ERROR] The module videosum.reader is not a script.')

