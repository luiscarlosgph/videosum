"""
@brief   Module to summarise a video into N frames. There are a lot of methods
         for video summarisation, and a lot of repositories in GitHub, but
         none of them seems to work out of the box. This module contains a 
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
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    # Create command line parser
    parser = argparse.ArgumentParser(description='Upload LEMON dataset to Synapse.')
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

    # Read parameters
    args = parser.parse_args()
    
    return args


def validate_cmdline_params(args):
    """
    @brief Input directory must exist and output must not.
    """
    if not os.path.isfile(args.input):
        raise RuntimeError('[ERROR] Input file does not exist.')
    if os.path.isfile(args.output):
        raise RuntimeError('[ERROR] Output file already exists.')
    return args


def main():
    # Read command line parameters
    args = parse_cmdline_params()
    validate_cmdline_params(args)

    vidsum = videosum.VideoSummariser(args.nframes, args.width, args.height)
    im = vidsum.summarise(args.input, args.output)


if __name__ == '__main__':
    main()
