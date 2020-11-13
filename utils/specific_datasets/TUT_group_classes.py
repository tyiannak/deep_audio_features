"""Place inside TUT folder and run it.
It will produce a folder named TUT
containing the classes separated in different  folders.
"""

import os
from shutil import copy
import wave
import contextlib
import pyaudioconvert as pac
from scipy.io import wavfile
import numpy as np

classes = ['beach', 'bus', 'cafe', 'car',
           'city_center', 'forest_path', 'grocery_store',
           'home', 'library', 'metro_station', 'office',
           'park', 'residential_area', 'train', 'tram']

path = os.getcwd() + '/TUT_classes'
print ("The current working directory is %s" % path)

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)


for c in classes:

    try:
        os.mkdir(path + '/' + c)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)


with open("meta.txt", encoding="utf-8") as file:
    data = [(l.rstrip("\n")).split() for l in file]


for idx, d in enumerate(data):
    src = d[0]
    file_name = src.split('/')[1]
    if d[1] == 'cafe/restaurant':
        dst = 'TUT_classes/cafe'
    else:
        dst = 'TUT_classes/' + d[1]
    new_file = dst + '/' + file_name

    copy(src, new_file)

    # convert to PCM - 16bit mono
    pac.convert_wav_to_16bit_mono(new_file, new_file)

    fs_wav, data_wav = wavfile.read(new_file)
    wavfile.write(new_file, fs_wav, data_wav.astype(np.int16))

    print(src)

    with contextlib.closing(wave.open(new_file, 'r')) as fp:
        continue
