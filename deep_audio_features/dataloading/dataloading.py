import torch
import numpy as np
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.utils import sound_processing
from tqdm import tqdm
from PIL import Image
import datetime


class FeatureExtractorDataset(Dataset):
    """Custom PyTorch Dataset for preparing features from wav inputs."""

    def __init__(self, X, y, fe_method="MFCC",
                 oversampling=False, max_sequence_length=281,
                 zero_pad=False, forced_size=None, pure_features=False,
                 fuse=False, show_hist=True, test_segmentation=False):
        """Create all important variables for dataset tokenization

        Arguments:
        ----------
            X {list} : List of training samples.
            y {list} : List of training labels.
            fe_method {string} : The method that extracts the features.
            oversampling {bool} : Resampling technique to be applied.
            max_sequence_length {int} : Max sequence length of the set.
            zero_pad {bool}: Apply zero padding (True) or resizing (False)
            forced_size {bool}: Force specific size when resizing
            pure_features {bool}: Keep pure features (neither zero
            padding nor resizing)
            fuse {bool}: Fuse spectrogram with chromagram or not
            show_hist {bool}: Whether to store a histogram of the files'
                sequence length
            test_segmentation {bool}: Whether to extract segment predictions
                of a large sequence. Applied only for 1-file testing for the
                time being.
        """
        self.fe_method = fe_method
        self.max_sequence_length = max_sequence_length
        self.fuse = fuse
        if fuse:
            print('--> Fusing spectrogram and chromagram')

        if oversampling is True:
            ros = RandomOverSampler()
            # Expand last dimension
            X, y = ros.fit_resample(np.reshape(X, (len(X), -1)), y)
            # Reshape again for use
            X = np.squeeze(X)

        # Depending on the extraction method get X

        features = []
        fss = []
        print("--> Extracting spectrogram associated features. . .")
        spec_sizes = []
        for audio_file in tqdm(X):  # for each audio file
            # load the signal
            signal, fs = sound_processing.load_wav(audio_file)
            # get the features:
            if fe_method == "MEL_SPECTROGRAM":
                feature = sound_processing.get_melspectrogram(signal,
                                                              fs=fs, fuse=fuse)
            else:
                feature = sound_processing.get_mfcc_with_deltas(signal,
                                                                fs=fs,
                                                                fuse=fuse)

            spec_sizes.append(feature.shape[0])

            # append to list of features
            features.append(feature)
            fss.append(fs)
        self.features = features

        spec_sizes = np.asarray(spec_sizes)

        if test_segmentation:
            # NOTE: Applied only to 1 file input
            print("--> Applying segmentation to the input file. . .")
            sequence = features[0]
            sequence_length = spec_sizes[0]
            segment_length = forced_size[0]
            progress = 0
            segments = []
            while progress < sequence_length:
                if progress + segment_length > sequence_length:
                    segments.append(sequence[progress:-1])
                else:
                    segments.append(sequence[progress: progress + segment_length + 1])

                progress = progress + segment_length

            print(len(segments))
            self.features = segments
            self.spec_size = forced_size

            self.y = torch.tensor(np.zeros(len(segments)), dtype=int)
            lengths = np.array([len(x) for x in segments])
            self.lengths = torch.tensor(lengths)

        else:
            if show_hist:
                self.plot_hist(spec_sizes, y)

            if forced_size is None:
                size_0 = int(np.mean(spec_sizes))
                size_1 = 140 if fuse else 128
                self.spec_size = (size_0, size_1)
            else:
                self.spec_size = forced_size
            # Create tensor for labels
            self.y = torch.tensor(y, dtype=int)
            # Get all lengths before zero padding
            lengths = np.array([len(x) for x in X])
            self.lengths = torch.tensor(lengths)

        if not pure_features:
            self.handle_lengths(zero_pad)

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

    def handle_lengths(self, zero_pad=False, size=None):

        if size is not None:
            spec_size = size
        else:
            spec_size = self.spec_size

        if zero_pad:
            X = self.zero_pad_and_stack()
            print('--> Using zero padding with max_length = {}'.format(self.max_sequence_length))
        else:
            X = self.resize(spec_size)
            print('--> Using resizing with new_size = {}'.format(spec_size))

        X = np.asarray(X)
        self.X = torch.from_numpy(X).type('torch.FloatTensor')

    def zero_pad_and_stack(self):
        """
        This function performs zero padding on a list of features and forms
        them into a numpy 3D array

        Returns:
            padded: a 3D numpy array of shape
            num_sequences x max_sequence_length x feature_dimension
        """
        X = self.features.copy()

        if self.fe_method == "MEL_SPECTROGRAM":
            max_length = self.max_sequence_length  # self.lengths.max()
        else: # MFCCs
            max_length = self.lengths.max()

        feature_dim = X[0].shape[-1]
        padded = np.zeros((len(X), max_length, feature_dim))

        out_X = []
        for i in range(len(X)):
            if X[i].shape[0] < max_length:
                # Needs padding
                diff = max_length - X[i].shape[0]
                # pad
                tmp = np.concatenate((X[i], np.zeros((diff, feature_dim))), axis=0)
                out_X.append(tmp)
            else:
                # Instead of raising an error just truncate the file
                tmp = np.take(X[i], list(range(0, max_length)), axis=0)
                out_X.append(tmp)
            # Add to padded
            padded[i, :, :] = tmp
        return padded

    def resize(self, size=None):

        if size is not None:
            spec_size = size
        else:
            spec_size = self.spec_size

        X = self.features.copy()
        x_resized = []
        for x in X:
            if x.shape[0] > 0:
                spec = Image.fromarray(x)
                spec = spec.resize(spec_size)
                spec = np.array(spec)
                x_resized.append(spec)
        return x_resized

    @staticmethod
    def group_data_by_label(spec_sizes, labels):
        classes = np.unique(labels)
        grouped_data = []
        for c in classes:
            grouped_data.append([])

        for idx, spec_size in enumerate(spec_sizes):
            grouped_data[labels[idx]].append(spec_size)

        return grouped_data

    def plot_hist(self, spec_sizes, labels):

        print('--> Plotting histogram of spectrogram sizes. ')
        labels = np.asarray(labels)
        grouped_data = self.group_data_by_label(spec_sizes, labels)
        plt.style.use('ggplot')

        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Histogram of spectrogram sizes', fontsize=14)

        axs[0].title.set_text('Overall')
        axs[0].set_xlabel('Spectrogram time dimension')
        axs[0].set_ylabel('Frequency')
        axs[0].hist(spec_sizes, bins='auto')

        axs[1].title.set_text('For each class')
        axs[1].set_xlabel('Spectrogram time dimension')
        axs[1].set_ylabel('Frequency')
        for idx, group in enumerate(grouped_data):
            label = 'Class {}'.format(idx)
            axs[1].hist(group, alpha=0.5, bins='auto', label=label)
        axs[1].legend()
        ct = datetime.datetime.now()
        plt.savefig(ct.strftime("%m_%d_%Y, %H:%M:%S") + ".png")



