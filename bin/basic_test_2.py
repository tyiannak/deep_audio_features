import argparse
import torch
from torch.utils.data import DataLoader
from dataloading.dataloading import FeatureExtractorDataset
from lib.training import test
from utils.model_editing import drop_layers
import config
import os
import glob
import numpy as np
import pickle


def test_model(modelpath, ifile, layers_dropped, ** kwargs):
    """Loads a model and predicts each classes probability

Arguments:

        modelpath {str} : A path where the model was stored.

        ifile {str} : A path of a given wav file,
                      which will be tested.

Returns:

        y_pred {np.array} : An array with the probability of each class
                            that the model predicts.

    """

    # Restore model
    model = torch.load(modelpath, map_location=torch.device('cpu') )
    max_seq_length = model.max_sequence_length

    # Apply layer drop
    model = drop_layers(model, layers_dropped)
    model.max_sequence_length = max_seq_length

    zero_pad = model.zero_pad
    spec_size = model.spec_size
    fuse = model.fuse

#    print('Model:\n{}'.format(model))

    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Create test set
    test_set = FeatureExtractorDataset(X=[ifile],
                                       # Random class -- does not matter at all
                                       y=[0],
                                       fe_method="MEL_SPECTROGRAM",
                                       oversampling=False,
                                       max_sequence_length=max_seq_length,
                                       zero_pad=zero_pad,
                                       forced_size=spec_size,
                                       fuse=fuse,
                                       show_hist=False)

    # Create test dataloader
    test_loader = DataLoader(dataset=test_set, batch_size=1,
                             num_workers=4, drop_last=False,
                             shuffle=False)

    # Forward a sample
    out, y_pred, _ = test(model=model, dataloader=test_loader,
                       cnn=True,
                       classifier=True if layers_dropped == 0 else False)

    return out[0], y_pred[0]


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
    features = np.array([])
    for m in list_of_models:
        soft, r = test_model(modelpath=m,
                             ifile=audio_file,
                             layers_dropped=layers_dropped)
        features = np.concatenate([features, soft])
        feature_names += [f'{os.path.basename(m).replace(".pt", "")}_{i}'
                          for i in range(len(soft))]

    return features, feature_names


def compile_deep_database(data_folder, models_folder, db_path):
    audio_files = glob.glob(os.path.join(data_folder, '*.wav'))

    models = load_models(models_folder)

    all_features = []
    for a in audio_files:
        f, f_names = get_meta_features(a, models)
        all_features.append(f)
    all_features = np.array(all_features)

    with open(db_path, 'wb') as f:
        pickle.dump(all_features, f)
        pickle.dump(f_names, f)
        pickle.dump(audio_files, f)

    return


def search_deep_database(database_path, models_folder, query_wav):
    with open(database_path, 'rb') as f:
        all_features = pickle.load(f)
        f_names = pickle.load(f)
        audio_files = pickle.load(f)

    models = load_models(models_folder)
    f, f_names = get_meta_features(query_wav, models)

    import scipy.spatial.distance
    print(f.reshape(-1,1).shape)
    print(all_features.shape)
    d = scipy.spatial.distance.cdist(f.reshape(-1,1).T, all_features)[0]
    print([x for _, x in sorted(zip(d, audio_files))])
    print(d)


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

    #compile_deep_database(ifile, model_dir, "db")
    search_deep_database("db", model_dir,
                         "/Users/tyiannak/Downloads/database/135- Dr. alban - Sing Hallelujah!.mp3.wav")