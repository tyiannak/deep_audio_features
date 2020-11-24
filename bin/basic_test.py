import argparse
import torch
from torch.utils.data import DataLoader
from dataloading.dataloading import FeatureExtractorDataset
from lib.training import test
from utils.model_editing import drop_layers
import config


def test_model(modelpath, ifile, layers_dropped, ** kwargs):
    """Loads a model and predicts each classes probability

Arguments:

        modelpath {str} : A path where the model was stored.

        ifile {str} : A path of a given wav file,
                      which will be tested.

Returns:

        y_pred {np.array} : An array with the probability of each class
                            that the model predicts.

    """

    # Restore model
    model = torch.load(modelpath)
    max_seq_length = model.max_sequence_length

    # Apply layer drop
    model = drop_layers(model, layers_dropped)
    model.max_sequence_length = max_seq_length

    zero_pad = model.zero_pad
    spec_size = model.spec_size
    fuse = model.fuse

    print('Model:\n{}'.format(model))

    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Create test set
    test_set = FeatureExtractorDataset(X=[ifile],
                                       # Random class -- does not matter at all
                                       y=[0],
                                       fe_method="MEL_SPECTROGRAM",
                                       oversampling=False,
                                       max_sequence_length=max_seq_length,
                                       zero_pad=zero_pad,
                                       spec_size=spec_size,
                                       fuse=fuse)

    # Create test dataloader
    test_loader = DataLoader(dataset=test_set, batch_size=1,
                             num_workers=4, drop_last=False,
                             shuffle=False)

    # Forward a sample
    out, y_pred, _ = test(model=model, dataloader=test_loader,
                       cnn=True,
                       classifier=True if layers_dropped == 0 else False)

    # [0] only for 1 sample to remove [[value]]
    print(out[0])
    #  If model has all layers can correctly predict a class
    print(y_pred)

    return y_pred[0]


if __name__ == '__main__':

    # Read arguments -- model
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True,
                        type=str, help='Model')

    parser.add_argument('-i', '--input', required=True,
                        type=str, help='Input file for testing')

    parser.add_argument('-L', '--layers', required=False, default=0,
                        help='Number of final layers to cut. Default is 0.')
    args = parser.parse_args()

    # Get arguments
    model = args.model
    ifile = args.input
    layers_dropped = int(args.layers)

    # Test the model
    test_model(modelpath=model, ifile=ifile, layers_dropped=layers_dropped)
