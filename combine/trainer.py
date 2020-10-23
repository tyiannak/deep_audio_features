import argparse
import os
import pickle
import time
from sklearn import svm
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.decomposition import PCA
import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from bin.config import VARIABLES_FOLDER
import feature_extraction


def train(folders, ofile=None, kernel='rbf', metric='f1_macro'):

    with open(r'combine/config.yaml') as file:
        modification = yaml.load(file, Loader=yaml.FullLoader)

    print('Extracting features...')
    X, y, pcas = feature_extraction.extraction(folders, modification)
    modification['dim_reduction'] = pcas
    print('X: {}'.format(X.shape))
    print('y: {}'.format(y.shape))
    print(Counter(y))

    clf = svm.SVC(kernel=kernel, class_weight='balanced')
    svm_parameters = {'gamma': ['auto', 1e-3, 1e-4, 1e-5, 1e-6],
                      'C': [1, 1e1, 1e2, 1e3, 1e4, 1e5]}

    #pca = PCA(n_components=0.99999)
    #pca.fit(X)
    print('The classifier is an SVM with {} kernel '.format(kernel))
    pipe = Pipeline(steps=[('SVM', clf)],
                    memory='sklearn_tmp_memory')
    print('Running GridSearchCV to find best SVM parameters...')

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)

    clf = GridSearchCV(
        pipe, dict(SVM__gamma=svm_parameters['gamma'],
                   SVM__C=svm_parameters['C']), cv=cv,
                   scoring=metric, n_jobs=-1)

    clf.fit(X, y)
    print("Best parameters found:")
    print(clf.best_params_)

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

    args = parser.parse_args()

    # Get argument
    folders = args.input
    ofile = args.ofile

    # Fix any type errors
    folders = [f.replace(',', '').strip() for f in folders]

    # Check that every folder exists

    for f in folders:
        if os.path.exists(f) is False:
            raise FileNotFoundError()

    model = train(folders, ofile)
