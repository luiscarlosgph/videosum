"""
@brief This is the base class of a summarizer, which is a class meant to 
       hold methods to produce the storyboard. This class is agnostic to the
       method used to choose the key frames.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   31 Dec 2023.
"""
import abc

class BaseSummarizer(abc.ABC):
    """
    @class BaseSummarizer contains all the methods related to the 
           summarization process, except the actual method to choose which
           frames of the video are the key frames. 
    @details You are supposed to inherit from this class to build classes
             that can summarize different types of input, for example a video
             that comes as a folder of images where each image is a frame, or
             a video that comes in an actual video file.
    """

    def __init__(self, reader, number_of_frames: int = 100, 
            width: int = 1920, height: int = 1080,
            time_segmentation=False, segbar_height=32, time_smoothing=0.,
            compute_fid=False) -> None:
        """
        @brief This method constructs the 'base' part of the summarizer, 
               leaving the rest of the construction to the child classes, 
               which should construct the objects according to each child's 
               specific requirement.
        
        @param[in]  reader             TODO
        @param[in]  number_of_frames   Number of frames of the video you want
                                       to see in the storyboard.
        @param[in]  width              Width of the summary storyboard.
        @param[in]  height             Height of the summary storybard.
        @param[in]  time_segmentation  Set to True to show a time 
                                       segmentation under the storyboard.
        @param[in]  segbar_height      Height in pixels of the time
                                       segmentation bar.
        @param[in]  time_smoothing     TODO.
        @param[in]  compute_fid        Set it to True if you want a report on
                                       the FID of the summary to the whole
                                       video.
        """
        # Sanity checks
        assert(algo in VideoSummarizer.ALGOS)
        assert(number_of_frames > 0)
        assert(width > 0)
        assert(height > 0)
        if (algo == 'time' and time_smoothing > 1e-6):
            raise ValueError('[ERROR] You cannot use time smoothing ' \
                + 'with the time algorithm.')
        
        # Store attributes
        self.reader = reader
        self.number_of_frames = number_of_frames
        self.width = width
        self.height = height
        self.form_factor = float(self.width) / self.height
        self.time_segmentation = time_segmentation
        self.segbar_height = segbar_height
        self.time_smoothing = time_smoothing
        self.compute_fid = compute_fid
        
        # Compute the width and height of each collage tile
        self.tile_height = self.height
        nframes = BaseSummarizer._how_many_rectangles_fit(
            self.tile_height, self.width, self.height)
        while nframes < self.number_of_frames: 
            self.tile_height -= 1
            nframes = BaseSummarizer._how_many_rectangles_fit(
                self.tile_height, self.width, self.height)
        self.tile_width = int(round(self.tile_height * self.form_factor))
        
        # Compute how many tiles per row and column
        self.tiles_per_row = self.width // self.tile_width
        self.tiles_per_col = self.height // self.tile_height
        
        # Initialise the array that holds the label of each frame
        self.labels_ = None
        
        # Initialise the array that holds the indices of the key frames
        self.indices_ = None

    # Import the different summarisation methods from their corresponding 
    # files
    from ._methods.time import get_key_frames_time
    from ._methods.inception import get_key_frames_inception
    from ._methods.uid import get_key_frames_uid 
    from ._methods.scda import get_key_frames_scda
    
    @staticmethod
    def transition_indices(labels):
        """
        @brief This method provides the indices of the boundary frames.

        @details  After the summarisation, the video frames are clustered into
                  classes. 
                  This means that there will be a video frame that belongs to 
                  class X followed by another frame that belongs to class Y. 
                  This method detects this transitions and returns a list of 
                  the Y frames, i.e. the first frames after a class
                  transition.
        @param[in]  labels  Pass the self.labels_ produced after calling
                            the summarise() method.
        @returns a list of indices.
        """
        transition_frames = []

        # Loop over the labels of all the video frames
        prev_class = None
        for idx, l in enumerate(labels):
            if l != prev_class:
                transition_frames.append(idx)
                prev_class = l

        return transition_frames
    
    @staticmethod
    def frame_distance_matrix(n: int):
        """
        @brief Compute the normalised distance matrix for a video of n frames.
        @details This static method is useful because we want to use the 
                 matrix produced by this method as an additional term for the
                 cost function provided to the method that selects the key
                 frames.

                 The idea is that frames that are further apart, have a 
                 higher distance between them because they are less likely
                 to belong to the same cluster.
                 
                 The distance matrix of a video of n frames would be:

                    0 1 2 3
                    1 0 1 2
                    2 1 0 1
                    3 2 1 0
                 
                 This method does not return exactly this matrix, but a 
                 normalized version of it. That is, the matrix above but where
                 all the elements are divided by four, so that the maximum
                 value is 1.

        @param[in]  n  Number of frames in the video.
        @returns a cost matrix of n frames where the cost between two frames
                 is equal to the number of frames between them divided by the
                 total number of frames.
        """
        # Create lower triangular distance matrix
        dist = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(0, i):
                dist[i, j] = np.abs(i - j)

        # Fill the upper triangular part
        dist = dist + dist.T

        # Normalise matrix to [0, 1]
        return dist / np.max(dist)
    
    @staticmethod
    def _how_many_rectangles_fit(tile_height, width, height):
        """
        @brief Given a certain tile height, this method computes how many
               of them fit in a collage of a given size. We assume that the
               form factor of the tiles in the collage has to be the same of 
               the collage itself.
        @param[in]  tile_height  Tile height.
        @param[in]  width        Width of the collage.
        @param[in]  height       Height of the collage.
        """
        # Compute form factor
        ff = float(width) / height

        # Compute rectangle height 
        tile_width = int(round(tile_height * ff))

        # Compute how many rectangles fit inside the collage
        tiles_per_row = width // tile_width 
        tiles_per_col = height // tile_height

        return tiles_per_row * tiles_per_col
    
    def get_key_frames(self, input_path: str):
        """
        @brief TODO
        @details This is the key method that the child classes should 
                 implement.
        @param[in]  input_path  TODO
        """
        #return Summarizer.ALGOS[self.algo](self, input_path,
        #    time_smoothing=self.time_smoothing)
        raise NotImplemented()

    def generate_segbar(self):
        """
        @brief TODO
        """
        # Create an empty segmentation bar
        segbar_width = self.collage.shape[1]
        segbar = np.full((self.segbar_height, segbar_width, 3), 255, 
            dtype=np.uint8)

        # Create the colour palette, one colour per cluster
        palette = np.array(sns.color_palette("Set3", self.number_of_frames))
        colours = (palette * 255.).astype(np.uint8)

        # Loop over segmentation bar columns
        for c in range(segbar_width):
            # Find the frame corresponding to this vertical bar
            frame_idx = int(round(
                float(c) * len(self.labels_) / segbar_width))
            frame_idx = np.clip(frame_idx, 0, len(self.labels_) - 1)

            # Find the cluster index of the frame
            cluster_idx = self.labels_[frame_idx]
            
            # Make the line of the colour of the cluster the frame belongs to
            segbar[:, c] = colours[cluster_idx]
        
        # Add the key frame vertical lines to the segmentation bar
        key_frame_line_colour = [0, 0, 0]
        for i in self.indices_:
            # Find the vertical bar corresponding to this frame index
            idx = int(round(i * segbar_width / len(self.labels_)))
            
            # Colour the column corresponding to this key frame
            segbar[:, idx] = key_frame_line_colour

        return segbar

    def summarize(self, input_path: str) -> np.ndarray:
        """
        @brief This method should return a BGR image containing a storyboard 
               of the input video.
        @param[in]  input_path  Input path to be passed to the summarize()
                                method of a child that implements this
                                interface class.
        """
        raise NotImplemented()
    
    def _insert_frame(self, im, i, j):
        """@brief Insert image into the collage."""

        # Resize frame to the right tile size
        im_resized = cv2.resize(im, (self.tile_width, self.tile_height), 
                                interpolation = cv2.INTER_LANCZOS4)

        # Insert image within the collage
        y_start = i * self.tile_height
        y_end = y_start + im_resized.shape[0] 
        x_start = j * self.tile_width
        x_end = x_start + im_resized.shape[1] 
        self.collage[y_start:y_end, x_start:x_end] = im_resized

    # Class attribute: supported key frame selection algorithms
    ALGOS = {
        'time':      get_key_frames_time,
        'inception': get_key_frames_inception,
        'uid' :      get_key_frames_uid,
        'scda':      get_key_frames_scda,
    }
