import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
import numpy as np

import config
from config import EPOCHS, CNN_BOOLEAN, VARIABLES_FOLDER
from models.cnn import CNN1
from lib.training import train_and_validate
from utils import load_dataset
from dataloading.dataloading import FeatureExtractorDataset
import argparse
import torch
from torch.utils.data import DataLoader
from dataloading.dataloading import FeatureExtractorDataset
from lib.training import test
from sklearn.metrics import classification_report
from utils.model_editing import drop_layers
from bin.basic_test import test_model


def test_report(modelpath, folders, layers_dropped,
             zero_pad=config.ZERO_PAD, size=config.SPECTOGRAM_SIZE):


    model = torch.load(modelpath)
    max_seq_length = model.max_sequence_length
    files_test, y_test = load_dataset.load(
        folders=folders, test=False, validation=False)


    # Load sets
    test_set = FeatureExtractorDataset(X=files_test, y=y_test,
                                        fe_method=
                                        config.FEATURE_EXTRACTION_METHOD,
                                        oversampling=config.OVERSAMPLING,
                                        max_sequence_length=max_seq_length,
                                        zero_pad=zero_pad,
                                        size=size)

    test_loader = DataLoader(test_set, batch_size=1,
                              num_workers=4, drop_last=True, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    _, y_pred, y_true = test(model=model, dataloader=test_loader,
                       cnn=True,
                       classifier=True if layers_dropped == 0 else False)

    report = classification_report(y_true, y_pred)
    print("Classification report: ")
    print(report)


if __name__ == '__main__':

    # Read arguments -- a list of folders
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True,
                        type=str, help='Model')

    parser.add_argument('-i', '--input', required=True,
                        type=str, nargs='+', help='Input folders')

    parser.add_argument('-L', '--layers', required=False, default=0,
                        help='Number of final layers to cut. Default is 0.')
    args = parser.parse_args()

    # Get arguments
    model = args.model
    folders = args.input

    layers_dropped = int(args.layers)

    # Fix any type errors
    folders = [f.replace(',', '').strip() for f in folders]

    # Check that every folder exists
    for f in folders:
        if os.path.exists(f) is False:
            raise FileNotFoundError()

    # Test the model
    test_report(model, folders, layers_dropped)
