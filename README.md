## Setup



#### Install and path setup

Εither use the source code

```bash
git clone https://github.com/tyiannak/deep_audio_features
```

Or install using pip
```bash
pip3 install deep-audio-features -U 
```


#### CORE UTILITIES

##### Training 

To train a CNN you can use the following command:
```python
python3 deep_audio_features/bin/basic_training.py -i /path/to/folder1 /path/to/folder2
```
`-i` : select the folders where the data will be loaded from.
`-o` : select the exported file name.

Or call the following function in Python:
```python
from deep_audio_features.bin import basic_training as bt
bt.train_model(["low","medium","high"], "enegy")
```
The code above reads the WAV files in 3 folders, uses the folder names as classnames, extracts 
spectrogram representations from the respective sounds, trains and validates the CNN and saves the 
trained model in `pkl/energy.pt`

##### Testing

```
python3 deep_audio_features/bin/basic_test.py -m /path/to/model/ -i /path/to/file (-s)
```
`-i` : select the folders  where the testing data will be loaded from.

`-m` : select a model to apply testing.

`-s`  : if included extracts segment level predictions of a sequence

Or call the following function in Python:
```python
from deep_audio_features.bin import basic_test as btest
d, p = btest.test_model("pkl/enegy.pt", 'some_file.wav', layers_dropped=0, test_segmentation=False)
```
The code above will use the CNN trained befre to classify an audio signal stored in `some_file.wav`.
`d` stores the decision (class indices) and `p` the soft outputs of the classes. 
If `layers_dropped` is positive, `d` is empty an `p` contains the outputs of the N-layers_dropped layer (N is the total number of layers in the CNN).
E.g. if `layers_dropped`, `p` will contain the outputs of the last fully connected layer, before softmax.

##### Transfer learning

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
