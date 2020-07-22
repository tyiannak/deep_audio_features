#!/usr/bin/env python3
from models.cnn import CNN1
import argparse
import os
import sys

import config

# sys.path.insert(0, '/'.join(os.path.abspath(__file__).split(' /')[:-2]))


def train(folders=None):
    """Train a given model on a given dataset"""
    # Check that folders exist
    if folders is None:
        raise FileNotFoundError()

    # Create classes
    classes = [os.path.basename(f) for f in folders]

    model = CNN1(output_dim=len(classes))


if __name__ == '__main__':

    # Read arguments -- a list of folders
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        type=str, nargs='+', help='Input folders')

    args = parser.parse_args()

    # Get argument
    folders = args.input

    # Fix any type errors
    folders = [f.replace(',', '').strip() for f in folders]

    # Check that every folder exists
    for f in folders:
        if os.path.exists(f) is False:
            raise FileNotFoundError()

    train(folders)
