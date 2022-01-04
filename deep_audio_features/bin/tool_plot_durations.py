import contextlib
import wave
import plotly.express as px
import os
import pandas as pd
import argparse


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

def plot_hist(dir):

    file_list = getListOfFiles(dir)
    durations = []
    print('--> Calculating durations')
    for file in file_list:
        durations.append(get_wav_duration(file))
    df = pd.DataFrame({'duration': durations})
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
    FLAGS = parser.parse_args()
    plot_hist(FLAGS.input)