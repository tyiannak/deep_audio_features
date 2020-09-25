
import torch
import numpy as np
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler
from utils import sound_processing
from tqdm import tqdm


class FeatureExtractorDataset(Dataset):
    """Custom PyTorch Dataset for preparing features from wav inputs."""

    def __init__(self, X, y, fe_method="MFCC",
                 oversampling=False, max_sequence_length=281):
        """Create all important variables for dataset tokenization

        Arguments:
        ----------
            X {list} : List of training samples.
            y {list} : List of training labels.
            fe_method {string} : The method that extracts the features.
            oversampling {bool} : Resampling technique to be applied.
            max_sequence_length {int} : Max sequence length of the set.
        """
        self.fe_method = fe_method
        self.max_sequence_length = max_sequence_length

        if oversampling is True:
            ros = RandomOverSampler()
            # Expand last dimension
            X, y = ros.fit_resample(np.reshape(X, (len(X), -1)), y)
            # Reshape again for use
            X = np.squeeze(X)

        # Depending on the extraction method get X

        features = []
        fss = []
        print("Feature extraction")
        for audio_file in tqdm(X):  # for each audio file
            # load the signal
            signal, fs = sound_processing.load_wav(audio_file)
            # get the features:
            if fe_method == "MEL_SPECTROGRAM":
                feature = sound_processing.get_melspectrogram(signal, fs=fs)
            else:
                feature = sound_processing.get_mfcc_with_deltas(signal, fs=fs)
            # append to list of features
            features.append(feature)
            fss.append(fs)
        X = features.copy()

        # Create tensor for labels
        self.y = torch.tensor(y, dtype=int)
        # Get all lengths before zero padding
        lengths = np.array([len(x) for x in X])
        self.lengths = torch.tensor(lengths)

        # Zero pad all samples
        X = self.zero_pad_and_stack(X)
        # Create tensor for features
        self.X = torch.from_numpy(X).type('torch.FloatTensor')
        print(np.shape(self.X))

    def __len__(self):
        """Returns length of FeatureExtractor dataset."""
        return len(self.X)

    def __getitem__(self, index):
        """Returns a _transformed_ item from the dataset

        Arguments:
            index {int} -- [Index of an element to be returned]

        Returns:
            (tuple)
                * sample [ndarray] -- [Features of an sample]
                * label [int] -- [Label of an sample]
                * len [int] -- [Original length of sample]
        """
        return self.X[index], self.y[index], self.lengths[index]

    def zero_pad_and_stack(self, X,):
        """
        This function performs zero padding on a list of features and forms
        them into a numpy 3D array

        Returns:
            padded: a 3D numpy array of shape
            num_sequences x max_sequence_length x feature_dimension
        """

        if self.fe_method == "MEL_SPECTROGRAM":
            max_length = self.max_sequence_length  # self.lengths.max()
        else: # MFCCs
            max_length = self.lengths.max()

        feature_dim = X[0].shape[-1]
        padded = np.zeros((len(X), max_length, feature_dim))

        for i in range(len(X)):
            if X[i].shape[0] < max_length:
                # Needs padding
                diff = max_length - X[i].shape[0]
                # pad
                X[i] = np.vstack((X[i], np.zeros((diff, feature_dim))))
            else:
                # Instead of raising an error just truncate the file
                X[i] = np.take(X[i], list(range(0, max_length)), axis=0)
            # Add to padded
            padded[i, :, :] = X[i]
        return padded
