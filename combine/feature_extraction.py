import os
import glob2 as glob
import numpy as np
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO
from utils.load_dataset import folders_mapping


def read_files(filenames):
    #Consider same sampling frequencies
    sequences = []
    for file in filenames:
        fs, samples = aIO.read_audio_file(file)
        sequences.append(samples)

    sequences = np.asarray(sequences)

    return sequences, fs


def extract_segment_features(sequences, sampling_rate, mid_window, mid_step,
                             short_window, short_step):
    segment_features_stats_all = []
    segment_features_all = []

    for seq in sequences:
        (segment_features_stats, segment_features,
         feature_names) = aF.mid_feature_extraction(
            seq, sampling_rate, mid_window, mid_step,
            short_window, short_step)
        segment_features_stats_all.append(segment_features_stats)
        segment_features_all.append(segment_features)

    segment_features_stats_all = np.asarray(segment_features_stats_all)
    segment_features_all = np.asarray(segment_features_all)

    return segment_features_stats_all, segment_features_all, feature_names


def extraction(folders):
    filenames = []
    labels = []

    # Match filenames with labels
    for folder in folders:
        for f in glob.iglob(os.path.join(folder, '*.wav')):
            filenames.append(f)
            labels.append(folder)

    folder2idx, idx2folder = folders_mapping(folders=folders)
    labels = list(map(lambda x: folder2idx[x], labels))
    labels = np.asarray(labels)

    sequences, sampling_rate = read_files(filenames)
    mid_window = sampling_rate
    mid_step = sampling_rate
    short_window = 0.05 * mid_window
    short_step = 0.05 * mid_step
    _, sequences_short_features, feature_names =\
        extract_segment_features(sequences, sampling_rate, mid_window,
                                 mid_step, short_window, short_step)

    sequences_short_features_stats = []
    for sequence in sequences_short_features:
        mu = np.mean(sequence, axis=1)
        std = np.std(sequence, axis=1)
        sequence_stats = np.concatenate((mu, std))
        sequences_short_features_stats.append(sequence_stats)
    sequences_short_features_stats = np.asarray(sequences_short_features_stats)
    X = sequences_short_features_stats
    y = labels

    return X, y
