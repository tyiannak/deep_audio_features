import librosa
import os
import numpy as np
from config import WINDOW_LENGTH, HOP_LENGTH, SAMPLING_RATE


def load_wav(filename):
    """Return read file using librosa library."""
    if not os.path.exists(filename):
        raise FileNotFoundError
    # Load file using librosa
    loaded_file, _ = librosa.load(filename, sr=SAMPLING_RATE)
    return loaded_file


def get_mfcc(wav, sr=SAMPLING_RATE):
    """Return MFCC of a given file, opened using librosa library."""
    return librosa.feature.mfcc(wav, sr, n_mfcc=13, win_length=WINDOW_LENGTH,
                                hop_length=HOP_LENGTH)


def get_mfcc_with_deltas(wav, sr=SAMPLING_RATE):
    """Return MFCC, delta and delta-delta of a given wav."""
    mfcc = get_mfcc(wav, sr)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
    return np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta))


def get_features_mean_var(loaded_wav=None):
    """Compute features from a loaded file. Computes MFCC
    coefficients and then the mean and variance along all frames
    for each coefficient.

    Keyword Arguments:
        loaded_wav {np.array} -- Wav read using librosa (default: {None})

    Returns:
        np.array -- An array containing mean and variance for each one of
                    the MFCC features as (m1,v1,m2,v2, ..) across all frames.
    """
    # Check inputs
    if loaded_wav is None:
        return None
    # Compute mfcc, delta and delta delta
    mfccs = get_mfcc_with_deltas(loaded_wav)  # (39,#frames)
    # Compute mean along 1-axis
    mean = np.mean(mfccs, axis=1)  # (39,1)
    # Compute variance along 1-axis
    variance = np.var(mfccs, axis=1)  # (39,1)
    # Export the features - order 'F' to preserve (m1,v1,m2,v2, ... )
    return np.ravel((mean, variance), order='F')  # (78,) -- 1darray


def get_melspectrogram(loaded_wav=None, n_fft=None, hop_length=None):
    """Returns a mel spectrogram."""

    if loaded_wav is None:
        return None

    # Set some values
    if n_fft is None:
        n_fft = WINDOW_LENGTH
    if hop_length is None:
        hop_length = HOP_LENGTH
    # Get spectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=loaded_wav,
        sr=SAMPLING_RATE,
        n_fft=n_fft,
        hop_length=hop_length)
    # Convert to MEL-Scale
    spectrogram_dB = librosa.power_to_db(
        spectrogram, ref=np.max)  # (n_mel,t)
    # Transpose to return (time,n_mel)
    return spectrogram_dB.T


def preview_melspectrogram(spectrogram=None, filename='spectrogram.png'):
    """Save a given spectrogram as an image."""

    if spectrogram is None:
        raise AssertionError

    import matplotlib.pyplot as plt
    import librosa.display

    plt.figure(figsize=(10, 4))
    # Spectrogram already in mel scale
    librosa.display.specshow(spectrogram)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(filename)
    return plt


def read_mel_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)
    print(spectrogram)
    spectrogram = spectrogram[:128]
    return spectrogram.T
