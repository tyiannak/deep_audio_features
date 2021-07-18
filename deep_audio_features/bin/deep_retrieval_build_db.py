import argparse
import torch
from torch.utils.data import DataLoader
import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.dataloading.dataloading import FeatureExtractorDataset
from deep_audio_features.lib.training import test
from deep_audio_features.utils.model_editing import drop_layers
from deep_audio_features.bin.basic_test import test_model
import deep_audio_features.bin.config
import os
import glob
import numpy as np
import pickle


def load_models(models_path):
    models = []
    for file in os.listdir(models_path):
        if file.endswith(".pt"):
            models.append(os.path.join(models_path, file))
    return models


def get_meta_features(audio_file, list_of_models):
    # TODO add other layers
    layers_dropped = 0

    feature_names = []
    features_temporal = []
    features = np.array([])
    for m in list_of_models:
        r, soft = test_model(modelpath=m,
                             ifile=audio_file,
                             layers_dropped=layers_dropped,
                             test_segmentation=True,
                             verbose=True)
        # long-term average the posteriors
        # (along different CNN segment-decisions)
        soft_average = np.mean(soft, axis=0)

        features = np.concatenate([features, soft_average])
        feature_names += [f'{os.path.basename(m).replace(".pt", "")}_{i}'
                          for i in range(len(soft_average))]

        # keep whole temporal posterior sequences as well
        features_temporal.append(soft)

    return features, features_temporal, feature_names


def compile_deep_database(data_folder, models_folder, db_path):
    audio_files = glob.glob(os.path.join(data_folder, '*.wav'))

    models = load_models(models_folder)

    all_features = []
    all_features_temporal = []
    for a in audio_files:
        f, f_temporal, f_names = get_meta_features(a, models)
        all_features.append(f)
        all_features_temporal.append(np.concatenate(f_temporal, axis=1).transpose())
    all_features = np.array(all_features)

    with open(db_path, 'wb') as f:
        pickle.dump(all_features, f)
        pickle.dump(all_features_temporal, f)
        pickle.dump(f_names, f)
        pickle.dump(audio_files, f)
        pickle.dump(models_folder, f)
    return


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', required=True,
                        type=str, help='Dir of models')
    parser.add_argument('-i', '--input', required=True,
                        type=str, help='Input file for testing')
    args = parser.parse_args()
    model_dir = args.model_dir
    ifile = args.input

    compile_deep_database(ifile, model_dir, "db")
