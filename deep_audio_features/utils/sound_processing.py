import librosa
import os
import numpy as np
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.bin.config import WINDOW_LENGTH, HOP_LENGTH


def load_wav(filename):
    """Rea audio file and return audio signal and sampling frequency"""
    if not os.path.exists(filename):
        raise FileNotFoundError
    # Load file using librosa
    x, fs = librosa.load(filename, sr=None)
    return x, fs


def get_mfcc(x, fs):
    """Return MFCC of a given file, opened using librosa library."""
    return librosa.feature.mfcc(x, fs, n_mfcc=13,
                                win_length=int(WINDOW_LENGTH * fs),
                                hop_length=int(HOP_LENGTH * fs))


def get_mfcc_with_deltas(wav, fs):
    """Return MFCC, delta and delta-delta of a given wav."""
    mfcc = get_mfcc(wav, fs)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
    return np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta))


def get_features_mean_var(x=None):
    """Compute features from a loaded file. Computes MFCC
    coefficients and then the mean and variance along all frames
    for each coefficient.

    Keyword Arguments:
        x {np.array} -- Wav read using librosa (default: {None})

    Returns:
        np.array -- An array containing mean and variance for each one of
                    the MFCC features as (m1,v1,m2,v2, ..) across all frames.
    """
    # Check inputs
    if x is None:
        return None
    # Compute mfcc, delta and delta delta
    mfccs = get_mfcc_with_deltas(x)  # (39,#frames)
    # Compute mean along 1-axis
    mean = np.mean(mfccs, axis=1)  # (39,1)
    # Compute variance along 1-axis
    variance = np.var(mfccs, axis=1)  # (39,1)
    # Export the features - order 'F' to preserve (m1,v1,m2,v2, ... )
    return np.ravel((mean, variance), order='F')  # (78,) -- 1darray


def get_melspectrogram(x=None, fs=None, n_fft=None, hop_length=None, fuse=False):
    """Returns a mel spectrogram."""

    if x is None:
        return None
    # Set some values
    if n_fft is None:
        n_fft = int(WINDOW_LENGTH * fs)
    if hop_length is None:
        hop_length = int(HOP_LENGTH * fs)
    # Get spectrogram
    spectrogram = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=n_fft,
                                                 hop_length=hop_length)
    # Convert to MEL-Scale
    spectrogram_dB = librosa.power_to_db(spectrogram, ref=np.max)  # (n_mel,t)

    if fuse:
        chroma = librosa.feature.chroma_stft(y=x, sr=fs, n_fft=n_fft,
                                             hop_length=hop_length)
        chroma_dB = librosa.power_to_db(chroma, ref=np.max)
        out = np.concatenate((spectrogram_dB.T, chroma_dB.T), axis=1)
    else:
        # Transpose to return (time,n_mel)
        out = spectrogram_dB.T
    return out


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
