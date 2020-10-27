# Combine Features

Combine basic features calculated by pyAudioAnalysis + extracted features using pretrained CNN models.

## Preperation

Modify the combine/config.yaml file. It contains the following parameters:

- which_classifier:
    Dictionary indicating which classifier to use and its parameters
- extract_basic_features:
    Boolean indicating whether extract pyAudioAnalysis features or not

- basic_features_params:
    Dictionary containing the following parameters for basic
- feature extraction:
    1. mid_window
    2. mid_step
    3. short_window
    4. short_step

- extract_nn_features:
    Boolean indicating whether extract CNN features or not

- model_paths:
    List of paths for pretained CNN models to use for feature extraction

- n_components:
    Number of components to use for PCA on the CNN features, for each model

- segment_step:
    Step of the segment window, used fro overlapping (see extract_segment_nn_features function)
 
## Training 

Example command from the main directory:


```bash
python3 combine/trainer.py -i 4class_balanced/music 4class_balanced/other 4class_balanced/silence 4class_balanced/speech
```

## Classification Report


Example command from the main directory:


```bash
python3 combine/classification_report.py -m pkl/SVM_basic.pt -i 4class_balanced/music 4class_balanced/other 4class_balanced/silence 4class_balanced/speech
```
