from models.cnn import CNN1
import argparse
import os
import sys
import numpy as np

import config
from utils import load_dataset

from dataloading import FeatureExtractorDataset

# sys.path.insert(0, '/'.join(os.path.abspath(__file__).split(' /')[:-2]))


def train(folders=None):
    """Train a given model on a given dataset"""
    # Check that folders exist
    if folders is None:
        raise FileNotFoundError()

    # Create classes
    classes = [os.path.basename(f) for f in folders]

    # Use data only for training
    X_train, y_train = load_dataset.load(
        folders=folders, test=False, validation=False)

    # Compute max sequence length
    max_seq_length = load_dataset.max_sequence_length(
        reload=False, X=X_train)

    # Load sets
    train_set = FeatureExtractorDataset(
        X=X_train, y=y_train, feature_extraction_method="MEL_SPECTROGRAM", oversampling=True, max_sequence_length=max_seq_length)


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
