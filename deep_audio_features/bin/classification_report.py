import argparse
import os
import torch
import sys
import pickle
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from torch.utils.data import DataLoader
from deep_audio_features.dataloading.dataloading import FeatureExtractorDataset
from deep_audio_features.utils import load_dataset
from deep_audio_features.lib.training import test
from deep_audio_features.models.cnn import load_cnn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from deep_audio_features.bin import config


def map_labels(idx2folder, class_mapping):
    label_mapping = {}
    for label_new, class_name in idx2folder.items():
        for label_old, c in class_mapping.items():
            if c in class_name :
                label_mapping[label_old] = label_new

    return label_mapping


def user_defined_labels(y_true, y_pred, folders, class_mapping):
    _, idx2folder = load_dataset.folders_mapping(folders=folders)
    for name in idx2folder:
        directories = idx2folder[name].split("/")
        if directories[-1] == "":
            idx2folder[name] = directories[-2]
        else:
            idx2folder[name] = directories[-1]
    print("\nUser defined class mapping: {}\n".format(idx2folder))
    label_mapping = map_labels(idx2folder, class_mapping)
    new_class_mapping = idx2folder

    y_true = [label_mapping[label] for label in y_true]
    y_pred = [label_mapping[label] for label in y_pred]
    return y_true, y_pred, new_class_mapping


def test_report(model_path, folders):
    """Warning: This function is using the file_system as a shared memory
    in order to run on a big amount of data, since due to batch_size = 1,
    the share strategy used in torch.multiprocessing results in memory errors
    """

    with open(model_path, "rb") as input_file:
        model_params = pickle.load(input_file)
    class_mapping = model_params["classes_mapping"]

    model, hop_length, window_length = load_cnn(model_path)

    max_seq_length = model.max_sequence_length
    files_test, y_test, class_mapping = load_dataset.load(
        folders=folders, test=False,
        validation=False, class_mapping=class_mapping)

    spec_size = model.spec_size
    zero_pad = model.zero_pad

    # Load sets
    test_set = FeatureExtractorDataset(X=files_test, y=y_test,
                                        fe_method=
                                        config.FEATURE_EXTRACTION_METHOD,
                                        oversampling=config.OVERSAMPLING,
                                        max_sequence_length=max_seq_length,
                                        zero_pad=zero_pad,
                                        forced_size=spec_size,
                                        fuse=model.fuse,
                                        hop_length=hop_length, window_length=window_length)

    test_loader = DataLoader(test_set, batch_size=1,
                              num_workers=4, drop_last=True, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    _, y_pred, y_true = test(model=model, dataloader=test_loader,
                       cnn=True,
                       classifier=True)

    y_true, y_pred, new_class_mapping = user_defined_labels(y_true, y_pred, folders, class_mapping)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n {}".format(cm))
    sorted_labels = sorted(new_class_mapping.items())
    labels, target_names = zip(*sorted_labels)

    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names)

    print("Classification report: ")
    print(report)


if __name__ == '__main__':

    # Read arguments -- a list of folders
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True,
                        type=str, help='Model')

    parser.add_argument('-i', '--input', required=True,
                        type=str, nargs='+', help='Input folders')

    args = parser.parse_args()

    # Get arguments
    model_path = args.model
    folders = args.input

    # Fix any type errors
    folders = [f.replace(',', '').strip() for f in folders]

    # Check that every folder exists
    for f in folders:
        if os.path.exists(f) is False:
            raise FileNotFoundError()

    # Test the model
    test_report(model_path, folders)
