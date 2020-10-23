import os
import glob2 as glob
import numpy as np
import copy
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from dataloading.dataloading import FeatureExtractorDataset
from bin import config
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


def extract_segment_features(filenames, labels, basic_features_params):
    segment_features_all = []

    sequences, sampling_rate = read_files(filenames)

    mid_window = basic_features_params['mid_window']
    mid_step = basic_features_params['mid_step']
    short_window = basic_features_params['short_window']
    short_step = basic_features_params['short_step']

    for seq in sequences:
        (segment_features_stats, segment_features,
         feature_names) = aF.mid_feature_extraction(
            seq, sampling_rate, round(mid_window * sampling_rate),
            round(mid_step * sampling_rate),
            round(sampling_rate * short_window),
            round(sampling_rate * short_step))
        segment_features_stats = np.asarray(segment_features_stats)
        segment_features_all.append(segment_features_stats)

    return segment_features_all, feature_names


def extraction(folders, modification):

    n_components = modification['n_components']
    filenames = []
    labels = []

    for folder in folders:
        for f in glob.iglob(os.path.join(folder, '*.wav')):
            filenames.append(f)
            labels.append(folder)

    folder2idx, idx2folder = folders_mapping(folders=folders)
    labels = list(map(lambda x: folder2idx[x], labels))
    labels = np.asarray(labels)

    # Match filenames with labels
    if modification['extract_basic_features']:
        print('Basic features . . .')
        sequences_short_features, feature_names =\
            extract_segment_features(filenames, labels,
                                     modification['basic_features_params'])

        sequences_short_features_stats = []
        for sequence in sequences_short_features:
            mu = np.mean(sequence, axis=1)
            sequences_short_features_stats.append(mu)

        sequences_short_features_stats = np.asarray(sequences_short_features_stats)

    if modification['extract_nn_features']:

        model_paths = modification['model_paths']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEVICE: {device}")

        data = FeatureExtractorDataset(X=filenames, y=labels,
                                       fe_method=
                                       config.FEATURE_EXTRACTION_METHOD,
                                       oversampling=config.OVERSAMPLING)

        models = []
        nn_features = []
        if 'dim_reduction' in modification:
            pcas = modification['dim_reduction']
        else:
            pcas = []

        for j, model_path in enumerate(model_paths):
            print('Extracting features using model: {}'.format(model_path))
            if device == 'cpu':
                model = copy.deepcopy(torch.load(model_path, map_location='cpu'))
            else:
                model = copy.deepcopy(torch.load(model_path))
            model.type = 'feature_extractor'

            models.append(model)
            data.handle_lengths(model.zero_pad, model.spec_size)

            data_loader = DataLoader(data, batch_size=1,
                                     num_workers=4, drop_last=False, shuffle=False)
            features = []
            for index, batch in enumerate(data_loader, 1):
                # Split each batch[index]
                inputs, _, _ = batch

                # Transfer to device
                inputs = inputs.to(device)

                # Forward through the network
                # Add a new axis for CNN filter features, [z-axis]
                inputs = inputs[:, np.newaxis, :, :]
                out = model.forward(inputs)
                out = out.squeeze()
                out = out.detach().clone().to('cpu').numpy()
                out = out.flatten()
                features.append(out)

            if 'dim_reduction' in modification:
                pca = pcas[j]
            else:
                pca = PCA(n_components=n_components)
                pcas.append(pca.fit(features))
            print('    Applied dimensonality reduction to CNN features')
            print('        Expressed variance for the new '
                  'features: {}'.format(np.sum(pca.explained_variance_ratio_)))
            principal_components = pca.transform(features)
            print(principal_components.shape)
            nn_features.append(principal_components)

            #nn_features.append(features)

        nn_features = np.asarray(nn_features)
        print(nn_features.shape)
        if modification['extract_basic_features']:
            acc = sequences_short_features_stats
            acc = np.concatenate((acc, nn_features[0]), axis=1)
        else:
            acc = nn_features[0]
        for idx, f in enumerate(nn_features):
            if idx == 0:
                continue
            acc = np.concatenate((acc, f), axis=1)

        out_features = acc

    else:
        out_features = sequences_short_features_stats
    out_features = np.asarray(out_features)

    if 'dim_reduction' in modification:
        return out_features, labels
    else:
        return out_features, labels, pcas
