import numpy as np
import glob2 as glob
from sklearn.model_selection import StratifiedShuffleSplit

from utils.sound_processing import get_melspectrogram, load_wav


def load(folders=None, test_val=[0.2, 0.2], validation=True):
    """Loads a dataset from some folders.

    Arguments
    ----------
        folders {list} : A list of folders containing all samples.

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
        for f in glob.iglob(''.join([folder, '*.wav'])):
            filenames.append(f)
            labels.append(folder)

    # Convert labels to int
    folder2idx, idx2folder = folders_mapping(folders=folders)
    labels = list(map(lambda x: folder2idx[x], labels))

    # Get percentages
    test_p, val_p = test_val

    # Split
    X_train_, y_train_ = [], []
    X_test, y_test = [], []
    X_train, y_train = [], []
    X_val, y_val = [], []
    # First split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_p)
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
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_p)
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


def max_sequence_length(reload=False, X=None, folders=None):
    """Return max sequence length for all files."""
    if reload is True:
        if folders is None:
            raise AssertionError()
        # Get all sample labels
        X_train, _, X_test, _, X_val, _ = load(folders=folders)
    # DEFAULT
    else:
        if X is None:
            raise AssertionError()
        # Files should be given by previous load
        X_train, X_test, X_val = X

    # Calculate and print max sequence number
    l = [np.shape(get_melspectrogram(load_wav(f)))[0]
         for f in (X_train+X_test+X_val)]
    max_seq = np.max(l)
    # print(f"Max sequence length in dataset: {max_seq}")
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
