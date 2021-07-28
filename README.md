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

##### training script

```python
python3 deep_audio_features/bin/basic_training.py -i /path/to/folder1 /path/to/folder2
```
`-i` : select the folders where the data will be loaded from.

`-o` : select the exported file name.

```python
from deep_audio_features.bin import basic_training as bt
bt.train_model(["low","medium","high"], "enegy")
```

##### testing script

```
python3 deep_audio_features/bin/basic_test.py -m /path/to/model/ -i /path/to/file (-s)
```
`-i` : select the folders  where the testing data will be loaded from.

`-m` : select a model to apply testing.

`-s`  : if included extracts segment level predictions of a sequence

TODO PYTHON SCRIPT


##### transfer learning script

```
python3 deep_audio_features/bin/transfer_learning.py -m /path/to/model -i /path/to/folder1 /path/to/folder2 -s
```
`-m` : select a model to apply fine-tuning.

`-i` : select the folders where the data will be loaded from.

`-s` : select which strategy to use. `0` applies fine-tuning to all layers while `1` freezes `Conv2d` layers and fine-tunes `Linear` only.

TODO PYTHON SCRIPT