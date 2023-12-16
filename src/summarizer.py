"""
TODO
"""
import abc

class BaseSummarizer(abc.ABC):
    """
    @class TODO
    """

    # Import the different summarisation methods from their corresponding files
    from ._methods.time import get_key_frames_time
    from ._methods.inception import get_key_frames_inception
    from ._methods.uid import get_key_frames_uid 
    from ._methods.scda import get_key_frames_scda
    
    @staticmethod
    def transition_indices(labels):
        """
        @brief This method provides the indices of the boundary frames.

        @details  After the summarisation, the video frames are clustered into
                  classes. This means that there will be a video frame that
                  belongs to class X followed by another frame that belongs to 
                  class Y. This method detects this transitions and returns a
                  list of the Y frames, i.e. the first frames after a class
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
        @brief Compute the normalised distance matrix of rows and columns. 
        @details The distance matrix of rows and columns is (assuming only four
                 frames):

                    0 1 2 3
                    1 0 1 2
                    2 1 0 1
                    3 2 1 0
                  
                 The normalised version is simply a minmax normalisation of the
                 matrix above.
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
    
    def get_key_frames(self, input_path):
        return Summarizer.ALGOS[self.algo](self, input_path,
            time_smoothing=self.time_smoothing)

    def generate_segbar(self):
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
            frame_idx = int(round(float(c) * len(self.labels_) / segbar_width))
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
