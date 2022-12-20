import argparse
import torch
from torch.utils.data import DataLoader
import sys, os
import pickle
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
import deep_audio_features.bin.basic_test
import deep_audio_features.bin.config
import numpy

if __name__ == '__main__':

    # Read arguments -- model
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True,
                        type=str, help='Model')

    parser.add_argument('-i', '--input', required=True,
                        type=str, help='Input file for testing')

    parser.add_argument('-s', '--segmentation', required=False,
                        action='store_true',
                        help='Return segment predictions')

    parser.add_argument('-L', '--layers', required=False, default=0,
                        help='Number of final layers to cut. Default is 0.')
    args = parser.parse_args()

    # Get arguments
    model = args.model
    ifile = args.input
    layers_dropped = int(args.layers)
    segmentation = args.segmentation

    # Test the model
    d, p = deep_audio_features.bin.basic_test.test_model(modelpath=model, 
                      ifile=ifile,
                      layers_dropped=layers_dropped,
                      test_segmentation=segmentation)
