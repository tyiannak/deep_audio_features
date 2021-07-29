import argparse
import os
import pickle
import time
from sklearn import svm
from imblearn.pipeline import Pipeline
from collections import Counter
import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.bin.config import VARIABLES_FOLDER
import deep_audio_features.combine.feature_extraction


def train(folders, ofile=None, config_file=r'combine/config.yaml'):
    """
    Trains a classifier using combined features (pyAudioAnalysis & CNN
    models' fetures) and GridSearchCV to find best parameters. Reads
    config.yaml to set running parameters.

    Parameters
    ----------

    folders:
        List of input folders. Each folder is a different class.

    ofile:
        Output model name


    Returns
    -------

    modification:
        Dictionary that contains all config parameters for reproducibility
        + a Classifier key that contains the final classifier.
    """

    with open(config_file) as file:
        modification = yaml.load(file, Loader=yaml.FullLoader)

    if modification['which_classifier']['type'] == 'svm':
        classifier_parameters = modification['which_classifier']['parameters']
        kernel = classifier_parameters['kernel']
        metric = classifier_parameters['metric']
    else:
        print('Supports only SVM classifier')
        return modification

    print('\nExtracting features...')
    if modification['extract_nn_features'] and 'dim_reduction' not in modification:
        X, y, pcas = deep_audio_features.combine.feature_extraction.\
            extraction(folders, modification)
        modification['dim_reduction'] = pcas
    else:
        X, y = deep_audio_features.combine.feature_extraction.\
            extraction(folders, modification)
    print('X (train data) shape: {}'.format(X.shape))
    print('y (train labels) shape: {}'.format(y.shape))

    clf = svm.SVC(kernel=kernel, class_weight='balanced')
    svm_parameters = {'gamma': ['auto', 1e-3, 1e-4, 1e-5, 1e-6],
                      'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}

    print('The classifier is an SVM with {} kernel '.format(kernel))
    pipe = Pipeline(steps=[('SVM', clf)],
                    memory='sklearn_tmp_memory')
    print('Running GridSearchCV to find best SVM parameters...')

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)

    grid_clf = GridSearchCV(
        pipe, dict(SVM__gamma=svm_parameters['gamma'],
                   SVM__C=svm_parameters['C']), cv=cv,
                   scoring=metric, n_jobs=-1)

    grid_clf.fit(X, y)

    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]

    print("Best parameters: {}".format(clf_params))
    print("Best validation score:      "
          "{:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))

    timestamp = time.ctime()
    if ofile is None:
        ofile = f"{'SVM'}_{timestamp}.pt"
        ofile = ofile.replace(' ', '_')
    else:
        ofile = str(ofile)
        if '.pt' not in ofile or '.pkl' not in ofile:
            ofile = ''.join([ofile, '.pt'])

    if not os.path.exists(VARIABLES_FOLDER):
        os.makedirs(VARIABLES_FOLDER)
    out_path = os.path.join(
        VARIABLES_FOLDER, ofile)
    print(f"\nSaving model to: {out_path}\n")

    modification['Classifier'] = clf
    with open(out_path, 'wb') as handle:
        pickle.dump(modification, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return modification


if __name__ == '__main__':

    # Read arguments -- a list of folders
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        type=str, nargs='+', help='Input folders')
    parser.add_argument('-o', '--ofile', required=False, default=None,
                        type=str, help='Model name.')
    parser.add_argument('-c', '--config', required=False, default=None,
                        type=str, help='Config file.')


    args = parser.parse_args()

    # Get argument
    folders = args.input
    ofile = args.ofile
    config_file = args.config

    # Fix any type errors
    folders = [f.replace(',', '').strip() for f in folders]

    # Check that every folder exists

    for f in folders:
        if os.path.exists(f) is False:
            raise FileNotFoundError()

    model = train(folders, ofile, config_file)
