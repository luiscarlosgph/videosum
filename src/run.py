"""
@brief   Package to summarize a video into N frames. There are a lot of methods
         for video summarisation, and a lot of repositories in GitHub, but
         none of them seems to work out of the box. This package contains a 
         simple way of doing it.

@author  Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    18 May 2022.
"""

import argparse
import imageio_ffmpeg
import numpy as np
import os
import cv2
import tqdm
import time
import multiprocessing
import logging

# My imports
import videosum


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-i': 'Path to the video (required: True)',
        '-o': 'Output directory path (required: True)',
        '-n': 'Number of frames in the collage (required: True)',
        '-x': 'Width of the collage (required: True)',
        '-y': 'Height of the collage (required: True)',
        '-a': 'Algorithm for frame selection, available options: time, frechet (required: True)',
        '-t': 'Add time segmentation based on the key frame selection (required: False)',
        '-f': 'Sampling frequency in fps (required: False)', 
        '-s': 'Time smoothing factor (required: False)',
        '-p': 'Number of processes (required: False)',
        '-l': 'Path to the log file (required: False)',
        '-m': 'Compute FID between storyboard and video (required: False)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    # Create command line parser
    parser = argparse.ArgumentParser(description='Easy-to-use video summarisation.')
    parser.add_argument('-i', '--input', required=True, type=str, 
                        help=help('-i'))
    parser.add_argument('-o', '--output', required=True, type=str, 
                        help=help('-o'))
    parser.add_argument('-n', '--nframes', required=True, type=int, 
                        help=help('-n'))
    parser.add_argument('-x', '--width', required=True, type=int, 
                        help=help('-x'))
    parser.add_argument('-y', '--height', required=True, type=int, 
                        help=help('-y'))
    parser.add_argument('-a', '--algo', required=True, type=str,
                        help=help('-a'))
    parser.add_argument('-t', '--time-segmentation', required=False, 
                        default=False, type=int, help=help('-t'))
    parser.add_argument('-f', '--fps', required=False, 
                        default=None, type=float, help=help('-f'))
    parser.add_argument('-s', '--time-smoothing', required=False, 
                        default=0., type=float, help=help('-s'))
    parser.add_argument('-p', '--processes', required=False,
                        default=multiprocessing.cpu_count(), type=int, 
                        help=help('-p'))
    parser.add_argument('-l', '--log', required=False, default='summary.log',
                        type=str, help=help('-l'))
    parser.add_argument('-m', '--metric', required=False, default=False,
                        type=bool, help=help('-m'))

    # Read parameters
    args = parser.parse_args()

    # Modify parameters according to needs
    args.time_segmentation = bool(args.time_segmentation)
    
    return args


def validate_cmdline_params(args):
    """
    @brief Input directory must exist and output must not.
    """
    if not os.path.isfile(args.input) and not os.path.isdir(args.input):
        raise RuntimeError('[ERROR] Input file or folder does not exist.')

    if os.path.isfile(args.output):
        raise RuntimeError('[ERROR] Output file already exists.')

    if args.algo not in videosum.VideoSummarizer.ALGOS: 
        raise ValueError("[ERROR] The method {} does not exist.".format(args.algo))

    return args


def process_video(input_path, output_path, args):
    """
    @brief Summarizes a video into a storyboard.
    @param[in]  input_path   Path to the input video.
    @param[in]  output_path  Path to the image file where you want to save
                             the image containing the storyboard.
    @returns nothing.
    """
    # Create video summarizer
    vidsum = videosum.VideoSummarizer(args.algo, args.nframes, 
                                      args.width, args.height, 
                                      time_segmentation=args.time_segmentation,
                                      fps=args.fps,
                                      time_smoothing=args.time_smoothing,
                                      compute_fid=args.metric)

    try:
        # Summarise video
        im = vidsum.summarize(input_path)

        # Save summary to the output folder
        cv2.imwrite(output_path, im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    except IOError as e:
        logging.info("The video {} is broken. Skipping.".format(os.path.basename(input_path)))


def process_image_dir(input_path, output_path, args):
    """
    @brief Summarizes a video into a storyboard.
    @param[in]  input_path   Path to the input video.
    @param[in]  output_path  Path to the image file where you want to save
                             the image containing the storyboard.
    @returns nothing.
    """
    # Create video summarizer
    vidsum = videosum.ImageDirSummarizer(args.algo, args.nframes, 
                                         args.width, args.height, 
                                         time_segmentation=args.time_segmentation,
                                         time_smoothing=args.time_smoothing,
                                         compute_fid=args.metric)

    try:
        # Summarise video
        im = vidsum.summarize(input_path)

        # Save summary to the output folder
        cv2.imwrite(output_path, im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    except IOError as e:
        logging.info("The video {} is broken. Skipping.".format(os.path.basename(input_path)))


def setup_logging(logfile_path):
    """
    @brief Sets up the logging to file.

    @param[in]  logfile_path  Path to the logfile, typically passed on in the
                              command line.
    """
    logging.basicConfig(filename=logfile_path, filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S', level=logging.DEBUG)


def detect_input_type(args, image_extensions=['.png', '.jpg', '.jpeg']):
    """
    @brief   This function determines tbe type of input provided by the user. 
    @details There are four types of input accepted by this script: 
            
            1. 'video': the input path provided points to a video file.
            2. 'many_videos': the input path provided points to a directory
                              containing many video files. All of them will
                              be summarized.
            3. 'frame': the input path provided points to a directory 
                        containing many images that represent the video
                        frames. Alphabetic order is expected for the frames.
            4. 'many_frames': the input path provided points to a directory
                              containing other subdirectories. 
                              Each subdirectory represents a video, and it is
                              expected to contain the frames as images. The
                              name of the images will be used to get the 
                              frame order.

    @param[in]  args              Object returned by argparse.parse_args() 
                                  containing the command line parameters 
                                  provided by the user.
    @param[in]  image_extensions  List of extensions used to detect if a
                                  directory contains videos or frames.

    @returns 'video', 'many_videos', 'frame', 'many_frames'. 
    """
    # Get the path to the input
    path = args.path
    
    # If it is a file, it must be a video file
    if os.path.isfile(args.input):
        return 'video'
    elif os.path.isdir(args.input):
        # We need to find out whether it is a list of videos or a list of dirs 
        listing = [x for x in os.path.listdir(args.input) if not x.startswith('.')] 
       
        # Sanity check: the folder cannot be empty, it has to contain videos
        #               or folders of frames
        if len(listing) < 1: 
            raise ValueError('[ERROR] The input directory does not contain anything.')              
        
        # Find out whether the first item inside the folder is a video file or
        # a directory
        path_to_first_item = os.path.join(args.input, listing[0])
        if os.path.isfile(path_to_first_item):
            # We need to find out whether this item is an image or a video
            for ext in image_extensions:
                if path_to_first_item.endswith(ext):
                    return 'frame'
            return 'many_videos'
        else:
            return 'many_frames'
    else:
        raise ValueError('[ERROR] Input type not recognized.')


def main():
    # Read command line parameters
    args = parse_cmdline_params()
    validate_cmdline_params(args)

    # Setup logging
    setup_logging(args.log)

    # Detect the type of input provided by the user 
    input_type = detect_input_type(args)  
    
    # Summarize whatever the user wants to summarize
    if input_type == 'video': 
        # The input is a single video

        # Create video summarizer
        vidsum = videosum.VideoSummarizer(args.algo, args.nframes, 
                                          args.width, args.height, 
                                          time_segmentation=args.time_segmentation,
                                          fps=args.fps, 
                                          time_smoothing=args.time_smoothing,
                                          compute_fid=args.metric)
        # The input is a file
        tic = time.time()
        im = vidsum.summarize(args.input)
        toc = time.time()
        print("[INFO] Video summarized in {} seconds.".format(toc - tic))
        cv2.imwrite(args.output, im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    elif input_type == 'frame':
        # The input is a folder of images corresponding to video frames
        input_dir = args.input
        output_path = args.output

        # Create video summarizer
        vidsum = videosum.ImageDirSummarizer(args.algo, args.nframes,
            args.width, args.height,
            time_segmentation=args.time_segmentation,
            time_smoothing=args.time_smoothing, args.metric)

        try:
            # Summarise video
            im = vidsum.summarize(input_dir)

            # Save summary to the output folder
            cv2.imwrite(output_path, im, 
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        except IOError as e:
            logging.info("The video {} is broken. Skipping.".format(os.path.basename(input_path)))

    elif input_type == 'many_videos':
        # The input is a folder of videos
        input_dir = args.input
        output_dir = args.output
        
        # Create output folder
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # Gather the list of video filenames inside the folder
        # TODO: we should accept videos in different containers (not just mp4)
        videos = [x for x in os.listdir(input_dir) if x.endswith('.mp4')]
        prev_len = len(videos)

        # Filter out the videos that have been already summarized
        already_summarized = [x.split('.jpg')[0] + '.mp4' \
            for x in os.listdir(output_dir) if x.endswith('.jpg')]
        videos = [x for x in videos if x not in already_summarized] 
        print("[INFO] {} videos have been already summarized.".format(prev_len - len(videos)))
        
        # Build data input ready for batch processing
        data_inputs = []
        for v in videos:
            input_path = os.path.join(input_dir, v)
            output_path = os.path.join(output_dir, os.path.splitext(v)[0] + '.jpg')
            data_inputs.append((input_path, output_path, args))
        
        # Run batch processing
        pool = multiprocessing.Pool(processes=args.processes)
        pool.starmap(process_video, data_inputs)

    elif input_type == 'many_frames':
        # The input path contains several subdirectories, each 
        # representing a video and containing images that represent 
        # the video frames
        
        # The input is a directory of directories
        input_dir = args.input
        output_dir = args.output
        
        # Create output folder
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # Gather the list of directory names inside the input folder 
        video_dirs = os.listdir(input_dir)

        # Filter out the videos that have been already summarized
        already_summarized = [x.split('.jpg')[0] \
            for x in os.listdir(output_dir) if x.endswith('.jpg')]
        videos_dirs = [x for x in video_dirs \
            if x not in already_summarized] 

        # Build data input ready for batch processing
        data_inputs = []
        for v in video_dirs:
            input_path = os.path.join(input_dir, v)
            output_path = os.path.join(output_dir, v + '.jpg')
            data_inputs.append((input_path, output_path, args))
        
        # Run batch processing
        pool = multiprocessing.Pool(processes=args.processes)
        pool.starmap(process_image_dir, data_inputs)

    else:
        raise ValueError('[ERROR] Input type not recognized.')

        
if __name__ == '__main__':
    main()
