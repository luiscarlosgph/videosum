"""
@brief   Script that reads the CSV results for FID of storyboard vs video and
         plots the FID score averaged across videos (y-axis) for every 
         method (lines) and number of frames (x-axis).

@author  Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    13 Jan 2023.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
import os


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-i': 'Path to the CSV file (required: True)',
        '-o': 'Path to save the output plot (required: True)',
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

    # Read parameters
    args = parser.parse_args()

    return args


def validate_cmdline_params(args):
    """
    @brief Input directory must exist and output must not.
    """
    if not os.path.isfile(args.input) and not os.path.isdir(args.input):
        raise RuntimeError('[ERROR] Input file or folder does not exist.')

    if os.path.isfile(args.output):
        raise RuntimeError('[ERROR] Output file already exists.')

    return args


def main():
    # Read command line parameters
    args = parse_cmdline_params()
    validate_cmdline_params(args)
    
    # Raw data dictionary
    data = {
        #'video'  : [],
        'Method' : [],
        'Size'   : [],
        'FID'    : [],
    }
    
    # Read CSV file and build Pandas dataframe
    with open(args.input) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                #data['video'].append(row[0])
                data['Method'].append(row[1])
                data['Size'].append(row[2])
                data['FID'].append(row[3])
            line_count += 1
    df = pd.DataFrame.from_dict(data)
    
    # Average FID scores over videos
    # TODO

    # Generate FID vs no. images for all the methods
    lp = sns.lineplot(data=df, x='Size', y='FID', hue='Method')
    fig = lp.get_figure()
    plt.legend(loc='upper right')
    fig.savefig(args.output)

    
if __name__ == '__main__':
    main()
