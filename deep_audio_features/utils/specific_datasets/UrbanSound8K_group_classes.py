"""Place inside UrbanSound8K folder and run it.
It will produce a folder named UrbanSound8K_classes
containing the classes separated in different  folders.
"""

import os
import pandas as pd
from shutil import copy
import wave
import contextlib
import pyaudioconvert as pac
from scipy.io import wavfile
import numpy as np

classes = ['air_conditioner', 'car_horn', 'children_playing',
           'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
           'jackhammer', 'siren', 'street_music']

path = os.getcwd() + '/UrbanSound8K_classes'
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


data = pd.read_csv("metadata/UrbanSound8K.csv")


for idx, d in enumerate(data['slice_file_name']):
    src = 'audio/fold{}/'.format(data['fold'][idx]) + d
    dst = path + '/' + data['class'][idx]
    new_file = dst + '/' + d
    copy(src, dst)

    # convert to PCM - 16bit mono
    pac.convert_wav_to_16bit_mono(new_file, new_file)

    fs_wav, data_wav = wavfile.read(new_file)
    wavfile.write(new_file, fs_wav, data_wav.astype(np.int16))

    print(src)

    with contextlib.closing(wave.open(new_file, 'r')) as fp:
        continue
