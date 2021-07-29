import argparse
import os
import pickle
import time
from sklearn.metrics import classification_report
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.bin.config import VARIABLES_FOLDER
import deep_audio_features.combine.feature_extraction


def combine_test_report(folders, modification, ofile=None):
    """
    Prints and stores classification report for the
    input data.

    Parameters
    ----------
    folders:
        List of input folders. Each folder is a different class.

    ofile:
        Name of the output file that contains the classification report

    modification:
        Dictionary that contains all config parameters for reproducibility
        + a Classifier key that contains the classifier to be used.

    Returns
    -------

    Classification report as produced from sklearn.metrics.classification_report
    """

    model = modification['Classifier']

    print('Extracting features...')
    X, y = deep_audio_features.combine.\
        feature_extraction.extraction(folders, modification)
    print("Detailed classification report:")
    y_true, y_pred = y, model.predict(X)
    print(classification_report(y_true, y_pred))
    timestamp = time.ctime()
    if ofile is None:
        ofile = f"SVM_classification_report_{timestamp}.pt"
        ofile = ofile.replace(' ', '_')
    else:
        ofile = str(ofile)
        if '.pt' not in ofile or '.pkl' not in ofile:
            ofile = ''.join([ofile, '.pt'])

    if not os.path.exists(VARIABLES_FOLDER):
        os.makedirs(VARIABLES_FOLDER)
    out_path = os.path.join(
        VARIABLES_FOLDER, ofile)
    print(f"\nSaving classification report to: {out_path}\n")

    pickle.dump(classification_report, open(out_path, 'wb'))

    return classification_report


if __name__ == '__main__':

    # Read arguments -- a list of folders
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_modification', required=True,
                        type=str, help='Model')

    parser.add_argument('-i', '--input', required=True,
                        type=str, nargs='+', help='Input folders')
    parser.add_argument('-o', '--ofile', required=False, default=None,
                        type=str, help='Model name.')

    args = parser.parse_args()

    # Get argument
    model_modification = args.model_modification
    folders = args.input
    ofile = args.ofile

    # Fix any type errors
    folders = [f.replace(',', '').strip() for f in folders]

    # Check that every folder exists

    for f in folders:
        if os.path.exists(f) is False:
            raise FileNotFoundError()

    modification = pickle.load(open(model_modification, 'rb'))

    classification_report = combine_test_report(folders, modification, ofile)
