import pickle
import argparse
import os
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.models.convAE import load_convAE
from deep_audio_features.models.cnn import load_cnn



def print_model_architecture(modelpath):
    print('Loading model...')
    with open(modelpath, "rb") as input_file:
        model_params = pickle.load(input_file)
    if "classes_mapping" in model_params:
        model, hop_length, window_length = load_cnn(modelpath)
    else:
        model, hop_length, window_length = load_convAE(modelpath)
    print("Model architecture:\n", model)




if __name__ == '__main__':
    # Read arguments -- a list of folders
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str,
                        help='Model to apply tranfer learning.')

    args = parser.parse_args()
    modelpath = args.model

    # Check that model file exists
    if os.path.exists(modelpath) is False:
        raise FileNotFoundError
    print_model_architecture(modelpath)