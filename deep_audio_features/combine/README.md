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

- download_models (boolean):
    If true the missing models will be downloaded

- google_drive_ids (list of strings):
    List containing the ids of the google drive files
            
- n_components:
    Number of components to use for PCA on the CNN features, for each model

- segment_step:
    Step of the segment window, used fro overlapping (see extract_segment_nn_features function)
 
## Training 

Trains a classifier using combined features (pyAudioAnalysis & CNN models' fetures) and GridSearchCV to find best parameters. Reads config.yaml to set running parameters.

Example command from the main directory:


```bash
python3 combine/trainer.py -i 4class_balanced/music 4class_balanced/other 4class_balanced/silence 4class_balanced/speech
```

## Classification Report

Prints and stores classification report for the input data.

Example command from the main directory:


```bash
python3 combine/classification_report.py -m pkl/SVM_basic.pt -i 4class_balanced/music 4class_balanced/other 4class_balanced/silence 4class_balanced/speech
```

## Predict

Predicts and prints the predicted class for the input file.

Example command from the main directory:

```bash
python3 combine/predict.py -m pkl/SVM_basic_\&_UrbanSound8K.pt -i filename.wav
```


