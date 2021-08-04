import os
import numpy as np
import glob
from sklearn.model_selection import StratifiedShuffleSplit
import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.bin import config
import wave
import contextlib


def load(folders=None, test_val=[0.2, 0.2], test=True, validation=True):
    """Loads a dataset from some folders.

    Arguments
    ----------
        folders {list} : A list of folders containing all samples.
        test_val {list} : A list containing the percentages for test and validation split.
        test {boolean} : If False only train samples and labels are returned.
        validation {boolean} : If False only train and test samples and
        labels are returned.

    Returns
    --------
        X_train {list} : All filenames for train.

        y_train {list} : Labels for train.

        X_test {list} : Filenames for test.

        y_test {list} : Labels for train.

        if `validation` is `True` also returns the following:

        X_valid {list} : Filenames for validation.

        y_valid {list} : Labels for validation.

    """
    if folders is None:
        raise AssertionError()
    filenames = []
    labels = []

    # Match filenames with labels
    for folder in folders:
        for f in glob.iglob(os.path.join(folder, '*.wav')):
            filenames.append(f)
            labels.append(folder)

    # Convert labels to int
    folder2idx, idx2folder = folders_mapping(folders=folders)
    labels = list(map(lambda x: folder2idx[x], labels))

    # Split
    if test is False and validation is False:
        # Use this data only to train
        return filenames, labels

    # Get percentages
    test_p, val_p = test_val

    X_train_, y_train_ = [], []
    X_test, y_test = [], []
    X_train, y_train = [], []
    X_val, y_val = [], []
    # First split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_p, random_state=0)
    train_idx, test_idx = next(
        sss.split(filenames, labels))
    # Train
    for idx in train_idx:
        X_train_.append(filenames[idx])
        y_train_.append(labels[idx])
    # Test
    for idx in test_idx:
        X_test.append(filenames[idx])
        y_test.append(labels[idx])

    # If validation split is not needed return
    if validation is False:
        return X_train_, y_train_, X_test, y_test

    # If valuation is True split again
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_p, random_state=0)
    train_idx, val_idx = next(sss.split(X_train_, y_train_))
    # Train after both splits
    for idx in train_idx:
        X_train.append(X_train_[idx])
        y_train.append(y_train_[idx])
    # validation
    for idx in val_idx:
        X_val.append(X_train_[idx])
        y_val.append(y_train_[idx])

    return X_train, y_train, X_test, y_test, X_val, y_val


def compute_max_seq_len(reload=False, X=None, folders=None):
    """Return max sequence length for all files."""
    # TAKE THE WINDOW STEPS
    if reload is True:
        if folders is None:
            raise AssertionError()
        # Get all sample labels
        X_train, _, X_test, _, X_val, _ = load(folders=folders)
        X = X_train+X_test+X_val
    # DEFAULT
    else:
        if X is None:
            raise AssertionError()

    # Calculate and print max sequence number
    print(config.HOP_LENGTH, config.WINDOW_LENGTH)
    lengths = []
    for f in X:
        with contextlib.closing(wave.open(f, 'r')) as fp:
            frames = fp.getnframes()
            fs = fp.getframerate()
            duration = frames / float(fs)
            length = int((duration -
                          (config.WINDOW_LENGTH - config.HOP_LENGTH)) / \
                         (config.HOP_LENGTH) + 1)
            lengths.append(length)
    max_seq = np.max(lengths)
    print(f"Max sequence length in dataset: {max_seq}")
    return max_seq


def folders_mapping(folders):
    """Return a mapping from folder to class and a mapping from class to folder."""
    folder2idx = {}
    idx2folder = {}
    for idx, folder in enumerate(folders):
        folder2idx[folder] = idx
        idx2folder[idx] = folder
    return folder2idx, idx2folder


def get_categories_population_dictionary(labels, n_classes=9):
    """Return a mapping (category) -> Population."""
    mapping = {i: 0 for i in range(0, n_classes)}
    # Iterate each file and map
    for l in labels:
        if l >= n_classes:
            continue
        mapping[l] += 1
    return mapping
