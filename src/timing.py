"""
@brief   Script to record the execution time of each of the summarisation 
         methods and print it in markdown for the README.
@author  Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    5 Dec 2022.
"""

import argparse
import imageio_ffmpeg 
import numpy as np
import os
import cv2
import tqdm
import time
import multiprocessing
import datetime

# My imports
import videosum


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-i': 'Path to the video (required: True)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    # Create command line parser
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('-i', '--input', required=True, type=str, 
                        help=help('-i'))

    # Read parameters
    args = parser.parse_args()

    return args


def main():
    # Read command line parameters
    args = parse_cmdline_params()
    
    # Check that the input video exists
    if not os.path.isfile(args.input):
        raise ValueError('[ERROR] The input video file does not exist.')

    # Get the number of frames in the video
    fps = 30
    vr = videosum.VideoReader(args.input)
    num_frames = vr.num_frames(args.input, fps)

    # Run all the summarisation methods
    num_collage_images = 16
    width = 1920  
    height = 1080
    iterations = 10
    timings = {}
    for i in range(iterations):
        for m in videosum.VideoSummarizer.ALGOS:
            vs = videosum.VideoSummarizer(m, num_collage_images, width, height, 
                                          fps=fps)
            tic = time.time() 
            vs.summarise(args.input)
            toc = time.time()
            elapsed = toc - tic
            if m not in timings:
                timings[m] = [elapsed / num_frames]
            else:
                timings[m].append(elapsed / num_frames)

    # Compute the average over iterations
    for m in videosum.VideoSummarizer.ALGOS:
        timings[m] = sum(timings[m])/ len(timings[m])

    # Print the summarisation time per frame
    print("\n")
    print('| Method | Summarisation time per frame (s) |')
    print('| ------ | -------------------------------- |')
    for m in videosum.VideoSummarizer.ALGOS:
        print("| {:10s} | {:1.3f} |".format(m, timings[m]))

    # One hour video sampled at 1fps example
    print("\n")
    print('| Method | Time for a 1h video sampled at 1fps |')
    print('| ------ | ----------------------------------- |')
    for m in videosum.VideoSummarizer.ALGOS:
        td = round(timings[m] * 3600)
        print("| {:10s} | {}s |".format(m, td))


if __name__ == '__main__':
    main()
