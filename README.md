## Setup

#### PATHS
```python
cd deep_audio_features
source scripts/setup_paths.py
```
--------
#### MAIN

#### - training:

```python
python3 scripts/basic_training.py -i /path/to/folder1 /path/to/folder2
```

#### - testing:

```
python3 scripts/basic_test.py -m /path/to/model/ -i /path/to/file (-s)
```

​		if option `-s` is applied to testing, the function applies the argmax function to the softmax outputs.

