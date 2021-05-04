## Setup



#### Install and path setup

```bash
git clone https://github.com/tyiannak/deep_audio_features
cd deep_audio_features
source scripts/setup_paths.sh
```



#### CORE UTILITIES

##### training script

```python
python3 bin/basic_training.py -i /path/to/folder1 /path/to/folder2
```
`-i` : select the folders where the data will be loaded from.

`-o` : select the exported file name.

##### testing script

```
python3 bin/basic_test.py -m /path/to/model/ -i /path/to/file (-s)
```
`-i` : select the folders  where the testing data will be loaded from.

`-m` : select a model to apply testing.

`-s`  : if included extracts segment level predictions of a sequence



##### transfer learning script

```
python3 bin/transfer_learning.py -m /path/to/model -i /path/to/folder1 /path/to/folder2 -s
```
`-m` : select a model to apply fine-tuning.

`-i` : select the folders where the data will be loaded from.

`-s` : select which strategy to use. `0` applies fine-tuning to all layers while `1` freezes `Conv2d` layers and fine-tunes `Linear` only.

