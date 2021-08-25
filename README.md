
# deep_audio_features: training an using CNNs on audio classification tasks 
## 1 About
`deep_audio_features` is a Python library for training Convolutional Neural Netowrks 
as audio classifiers. The library provides a wrappers to pytorch for training CNNs
on audio classification tasks, and using the CNNs as feature extractors. 

## 2 Installation
Εither use the source code

```bash
git clone https://github.com/tyiannak/deep_audio_features
```

Or install using pip
```bash
pip3 install deep-audio-features -U 
```


## 3 Functionality

### 3.1 Training a CNN

To train a CNN you can use the following command:
```python
python3 deep_audio_features/bin/basic_training.py -i /path/to/folder1 /path/to/folder2
```
`-i` : select the folders where the data will be loaded from.
`-o` : select the exported file name.

Or call the following function in Python:
```python
from deep_audio_features.bin import basic_training as bt
bt.train_model(["low","medium","high"], "energy")
```
The code above reads the WAV files in 3 folders, uses the folder names as classnames, extracts 
spectrogram representations from the respective sounds, trains and validates the CNN and saves the 
trained model in `pkl/energy.pt`

### 3.2 Testing a CNN

```
python3 deep_audio_features/bin/basic_test.py -m /path/to/model/ -i /path/to/file (-s)
```
`-i` : select the folders  where the testing data will be loaded from.

`-m` : select a model to apply testing.

`-s`  : if included extracts segment level predictions of a sequence

Or call the following function in Python:
```python
from deep_audio_features.bin import basic_test as btest
d, p = btest.test_model("pkl/energy.pt", 'some_file.wav', layers_dropped=0, test_segmentation=False)
```
The code above will use the CNN trained befre to classify an audio signal stored in `some_file.wav`.
`d` stores the decision (class indices) and `p` the soft outputs of the classes. 
If `layers_dropped` is positive, `d` is empty an `p` contains the outputs of the N-layers_dropped layer (N is the total number of layers in the CNN).
E.g. if `layers_dropped`, `p` will contain the outputs of the last fully connected layer, before softmax.

### 3.3 Transfer learning 

To transfer knowledge from a pre-trained model and fit it on a new target task you can use the following command:
```
python3 deep_audio_features/bin/transfer_learning.py -m /path/to/model -i /path/to/folder1 /path/to/folder2 -s
```
`-m` : select a model to apply fine-tuning.
`-i` : select the folders where the data will be loaded from.
`-s` : select which strategy to use. `0` applies fine-tuning to all layers 
while `1` freezes `Conv2d` layers and fine-tunes `Linear` only.

Similarly, you will need the same params to call the `deep_audio_features.bin.transfer_learning.transfer_learning()` 
function to transfer knowledge from a task to another:
```python
from deep_audio_features.bin import transfer_learning as tl
tl.transfer_learning('pkl/emotion_energy.pt', ['test/low/', 'test/high'] , strategy=0)
```
(The model will be saved in a local filename based on the timestamp)

### 3.4 Combine CNN features

In `deep_audio_features/combine/config.yaml` choose 
(i) which CNN models you want to combine by 
modifying either the model_paths or the 
`google_drive_ids fields` (in case the models are stored in google drive),
 (ii) whether you want to combine different CNN 
 models (`extract_nn_features` boolean variable), 
 use hand-crafted audio features using the [pyAudioAnalysis library](https://github.com/tyiannak/pyAudioAnalysis)
 (`extract_basic_features` boolean variable), 
 or combine the aforementioned choices 
 (both variables set to `True`).

#### 3.4.1 Train a combination of CNNs
```
python3 deep_audio_features/combine/trainer.py -i 4class_small/music_small 4class_small/speech_small -c deep_audio_features/combine/config.yaml
```
or in Python:
```python
from deep_audio_features.combine import trainer
trainer.train(["4class_small/music_small", "4class_small/speech_small"], None, "config.yaml")
```

#### 3.4.2 Evaluate the combiner
```
python3 deep_audio_features/combine/classification_report.py -m pkl/SVM_Thu_Jul_29_20:06:51_2021.pt -i 4class_small/music_small 4class_small/speech_small
```
or in Python:
```python
from deep_audio_features.combine import classification_report
import pickle
modification = pickle.load(open("pkl/SVM_Thu_Jul_29_20:31:35_2021.pt", 'rb'))
classification_report.combine_test_report(["4class_small/music_small", "4class_small/speech_small"], modification)
```
#### 3.4.3 Predict on an unknown sample using the combiner
```
python3 deep_audio_features/combine/predict.py -m pkl/SVM_Thu_Jul_29_20:06:51_2021.pt -i 4class_balanced/speech/s_BDYDHQBQMX_30.0_31.0.wav
```
or in Python:
```python
from deep_audio_features.combine import predict
predict.predict("4class_balanced/speech/s_BDYDHQBQMX_30.0_31.0.wav", modification)
```
(load model as above)
