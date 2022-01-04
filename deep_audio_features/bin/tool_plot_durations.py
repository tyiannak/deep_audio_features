import contextlib
import wave
from numpy.lib import percentile
import plotly.express as px
import os
import pandas as pd
import argparse
import numpy as np

def get_wav_duration(fname):
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = []
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        elif fullPath.endswith(('.wav', '.mp3')):
            allFiles.append(fullPath)

    return allFiles


def plot_hist(dir, low, high):

    file_list = getListOfFiles(dir)
    durations = []
    print('--> Calculating durations')
    for file in file_list:
        d = get_wav_duration(file)
        if (d >= low) and (d <= high):
            durations.append(d)
    df = pd.DataFrame({'duration': durations})
    p5 = np.percentile(durations, 5.0)
    p95 = np.percentile(durations, 95.0)
    m = np.mean(durations)
    percentage = 100 * len(durations) / float(len(file_list))
    print('Statistics:')
    print(f'mu = {m:.2f} [{p5:.2f}, {p95:.2f}]')
    print(f'Percentage of data used: {percentage:.1f}%')
    print('--> Plotting histogram of audio durations')
    fig = px.histogram(df, x="duration")
    fig.update_layout(title_text='Audio durations in s', title_x=0.5)
    fig.show()

    fig.write_html(dir + "_durations.html")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        required=True,
                        default=None,
                        help='Input directory')
    parser.add_argument('-l',
                        '--low',
                        type=float,
                        required=False,
                        default=0,
                        help='Lowest duration to be used')
    parser.add_argument('-u',
                        '--upper',
                        type=float,
                        required=False,
                        default=10000,
                        help='Highest duration to be used')
    FLAGS = parser.parse_args()
    plot_hist(FLAGS.input, FLAGS.low, FLAGS.upper)
