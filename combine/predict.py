import argparse
import pickle
import feature_extraction


def predict(ifile, modification):
    """
    Prints the predicted class for the input file.

    Parameters
    ----------
    ifile:
        Filename of the input file.

    modification:
        Dictionary that contains all config parameters for reproducibility
        + a Classifier key that contains the classifier to be used.
    """

    model = modification['Classifier']

    print('Extracting features...')
    X = feature_extraction.extraction(ifile, modification,
                                      folders=False,
                                      show_hist=False)

    y_pred = model.predict(X)
    print("Predicted class: {}".format(y_pred[0]))


if __name__ == '__main__':

    # Read arguments -- a list of folders
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_modification', required=True,
                        type=str, help='Model')

    parser.add_argument('-i', '--input', required=True,
                        type=str, help='Input file for testing')

    args = parser.parse_args()

    # Get argument
    model_modification = args.model_modification
    ifile = args.input

    modification = pickle.load(open(model_modification, 'rb'))

    predict(ifile, modification)
