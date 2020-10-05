import argparse
import os
import pickle
import time
from sklearn import svm
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from bin.config import VARIABLES_FOLDER
import feature_extraction


def train(folders, ofile=None):


    print('Extracting features...')
    X, y = feature_extraction.extraction(folders)

    scaler = StandardScaler()
    pca = PCA()
    clf = svm.SVC()
    svm_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6],
                      'C': [1, 1e1, 1e2, 1e3, 1e4, 1e5]}
    n_components = [0.94, 0.95, 0.96, 0.97, 0.98, X.shape[1]]

    print('The classifier is an SVM using StandardScaler and PCA '
          'for preprocessing')
    pipe = Pipeline(steps=[('PCA', pca), ('SCALER', scaler), ('SVM', clf)],
                    memory='sklearn_tmp_memory')
    print('Running GridSearchCV to find best SVM parameters...')
    clf = GridSearchCV(
        pipe, dict(PCA__n_components=n_components,
                   SVM__kernel=svm_parameters['kernel'],
                   SVM__gamma=svm_parameters['gamma'],
                   SVM__C=svm_parameters['C']), cv=5,
                   scoring='f1_macro', n_jobs=-1)

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

    pickle.dump(clf, open(out_path, 'wb'))

    return clf


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
