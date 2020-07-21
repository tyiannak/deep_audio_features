#!/usr/bin/env python3

import argparse
import os

if __name__ == '__main__':

    # Read arguments -- a list of folders
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        type=str, nargs='+', help='Input folders')
    parser.add_argument('-c', '--configuration_file',
                        type=str, required=False, help='Configurations')
    args = parser.parse_args()

    # Get argument
    folders = args.input
    configuration_file = None
    # Fix any type errors
    folders = [f.replace(',', '').strip() for f in folders]

    # Check that every folder exists
    for f in folders:
        if os.path.exists(f) is False:
            raise FileNotFoundError()

    # Create classes
    classes = [os.path.basename(f) for f in folders]
