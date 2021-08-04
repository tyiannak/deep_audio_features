import os
import glob
import numpy as np
import copy
import torch
from sklearn.decomposition import PCA
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.utils.load_dataset import folders_mapping
from deep_audio_features.utils import get_models
from deep_audio_features.dataloading.dataloading import FeatureExtractorDataset
from deep_audio_features.bin import config
from PIL import Image


def read_files(filenames):
    """Read file using pyAudioAnalysis"""

    #Consider same sampling frequencies
    sequences = []
    for file in filenames:
        fs, samples = aIO.read_audio_file(file)
        sequences.append(samples)

    sequences = np.asarray(sequences)

    return sequences, fs


def extract_segment_features(filenames, basic_features_params):
    """
    Extract segment features using pyAudioAnalysis

    Parameters
    ----------

    filenames :
        List of input audio filenames

    basic_features_params:
        Dictionary of parameters to consider.
        It must contain:
            - mid_window: window size for framing
            - mid_step: window step for framing
            - short_window: segment window size
            - short_step: segment window step

    Returns
    -------

    segment_features_all:
        List of stats on segment features
    feature_names:
        List of feature names

    """
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


def resize_image(image, new_size, device):

    im = Image.fromarray(image)
    im_resized = im.resize(new_size)
    im_resized = np.array(im_resized)

    im_resized = im_resized[np.newaxis, :, :]
    im_resized = torch.from_numpy(im_resized).type('torch.FloatTensor')
    im_resized = im_resized.to(device)
    im_resized = im_resized[:, np.newaxis, :, :]

    return im_resized


def extract_segment_nn_features(data, model, device,
                                segment_step, enough_memory=False):
    """
     Extract features from audio using outputs of fourth convolutional
     layer of a pretrained deep neural network. If the size of an image is
     greater than the fixed size that a CNN demands, overlapping is
     used. Else we resize the image.

       Parameters
       ----------

       data:
            List of features (spectrograms of images in the original size)
            usually give from FeatureExtractorDataset.features

       model:
            Pretrained Pytorch's CNN model

       device:
            Device to run the model.

       segment_step:
            Step of the segment window.

        enough_memory:
            Boolean indicating whether system memory is enough.
            It enables the script to store features for evey segment.
            The amount of memory needed differs for different datasets.
            A good starting point is 8gb for medium datasets with small original
            image sizes.

       Returns
       -------
        nn_features_stats:
            Mean value of the segments' features

        nn_segment_features:
            Features for each segment. Returned only if enough memory is True
    """

    segment_size = model.spec_size
    print('--> Model\'s segment size: {}'.format(segment_size))
    nn_features_stats = []
    nn_segment_features = []

    for index, x in enumerate(data, 1):
        current_position = 0
        cnt = 0

        spec_size = x.shape
        seg_features = []
        if spec_size[0] < segment_size[0]:
            segment = resize_image(x, segment_size, device)

            out = model.forward(segment)
            out = out.view(out.size(0), -1)
            out = out.squeeze()
            out = out.type(torch.float32).detach().clone().to('cpu').numpy()
            out = out.flatten()
            seg_features.append(out)

        else:
            while current_position + segment_size[0] <= spec_size[0]:
                cnt += 1
                # get current window
                segment = x[current_position:current_position + segment_size[0]]
                segment = resize_image(segment, segment_size, device)
                # update window position
                current_position = current_position + segment_step

                out = model.forward(segment)
                out = out.squeeze()
                out = out.type(torch.float32).detach().clone().to('cpu').numpy()
                out = out.flatten()
                seg_features.append(out)

        seg_features = np.array(seg_features, copy=False)
        if enough_memory:
            nn_segment_features.append(seg_features)
        mu = np.mean(seg_features, axis=0, dtype=np.float32)
        nn_features_stats.append(mu)
    return nn_features_stats, nn_segment_features


def extraction(input, modification, folders=True, show_hist=True):
    """
    Extracts features from images. It can combine basic features
    calculated by pyAudioAnalysis + extracted features using
    pretrained CNN models.

    Parameters
    ----------

    folders:
        List of input folders. Each folder is a different class.

    modification:
        Dictionary that contains config parameters, such as:
            extract_basic_features:
                Boolean indicating whether extract pyAudioAnalysis features or not
            basic_features_params:
                Dictionary containing the following parameters for basic
                feature extraction:
                    - mid_window
                    - mid_step
                    - short_window
                    - short_step
            extract_nn_features:
                Boolean indicating whether extract CNN features or not
            model_paths:
                List of paths for pretained CNN models to use for feature extraction
            download_models (boolean):
                if true the missing models will be downloaded
            google_drive_ids (list of strings):
                list containing the ids of the google drive files
            n_components:
                Number of components to use for PCA on the CNN features, for each model
            segment_step:
                Step of the segment window, used fro overlapping (see
                extract_segment_nn_features function)

    Returns
    -------

        out_features:
            Array of the final features.

        labels:
            Array of the labels

        pcas:
            List of pca models used for each CNN.
    """

    print('-----------------------------------------------------------------')
    n_components = modification['n_components']
    filenames = []
    labels = []

    if folders:
        for folder in input:
            for f in glob.iglob(os.path.join(folder, '*.wav')):
                filenames.append(f)
                labels.append(folder)

        folder2idx, idx2folder = folders_mapping(folders=input)
        labels = list(map(lambda x: folder2idx[x], labels))
        labels = np.asarray(labels)

    else:
        filenames = [input]
    # Match filenames with labels

    if modification['extract_basic_features']:
        print('--> Basic features . . .')
        sequences_short_features, feature_names =\
            extract_segment_features(filenames,
                                     modification['basic_features_params'])

        sequences_short_features_stats = []
        for sequence in sequences_short_features:
            mu = np.mean(sequence, axis=1)
            sequences_short_features_stats.append(mu)

        sequences_short_features_stats = np.asarray(sequences_short_features_stats)

    if modification['extract_nn_features']:
        get_models.download_missing(modification)
        model_paths = modification['model_paths']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEVICE: {device}")

        data = FeatureExtractorDataset(X=filenames, y=labels,
                                       fe_method=
                                       config.FEATURE_EXTRACTION_METHOD,
                                       oversampling=config.OVERSAMPLING,
                                       pure_features=True, show_hist=show_hist)

        models = []
        nn_features = []
        if 'dim_reduction' in modification:
            pcas = modification['dim_reduction']
        else:
            pcas = []

        for j, model_path in enumerate(model_paths):
            print('--> Extracting features using model: {}'.format(model_path))
            if torch.cuda.is_available():
                model = copy.deepcopy(torch.load(model_path))
            else:
                model = copy.deepcopy(torch.load(
                    model_path, map_location=torch.device('cpu')))

            model.type = 'feature_extractor'

            models.append(model)

            nn_features_stats, _ = extract_segment_nn_features(
                        data.features.copy(), model,
                device, modification['segment_step'])
            if 'dim_reduction' in modification:
                pca = pcas[j]
            else:
                print('--> Finding {} principal components using'
                      ' PCA:'.format(n_components))

                pca = PCA(n_components=n_components)
                pca.fit(nn_features_stats)
                pcas.append(pca)
            print('    Applied dimensonality reduction to CNN features')
            print('        Expressed variance for the new '
                  'features: {}'.format(np.sum(pca.explained_variance_ratio_)))
            principal_components = pca.transform(nn_features_stats)
            nn_features.append(principal_components)

        nn_features = np.asarray(nn_features)
        print('--> CNN features shape: {}'.format(nn_features.shape))
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

    print('-----------------------------------------------------------------')
    if not folders:
        return out_features
    elif modification['extract_nn_features'] and \
            'dim_reduction' not in modification:
        return out_features, labels, pcas
    else:
        return out_features, labels
