import argparse
import torch
from torch.utils.data import DataLoader
from dataloading.dataloading import FeatureExtractorDataset
from lib.training import test
from utils.model_editing import drop_layers
import config
import os
import glob


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
    model = torch.load(modelpath, map_location=torch.device('cpu') )
    max_seq_length = model.max_sequence_length

    # Apply layer drop
    model = drop_layers(model, layers_dropped)
    model.max_sequence_length = max_seq_length

    zero_pad = model.zero_pad
    spec_size = model.spec_size
    fuse = model.fuse

#    print('Model:\n{}'.format(model))

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
                                       forced_size=spec_size,
                                       fuse=fuse,
                                       show_hist=False)

    # Create test dataloader
    test_loader = DataLoader(dataset=test_set, batch_size=1,
                             num_workers=4, drop_last=False,
                             shuffle=False)

    # Forward a sample
    out, y_pred, _ = test(model=model, dataloader=test_loader,
                       cnn=True,
                       classifier=True if layers_dropped == 0 else False)

    return out[0], y_pred[0]


def compile_deep_database(data_folder, models_folder, layers_dropped):
    audio_files = glob.glob(os.path.join(data_folder, '*.wav'))

    models = []
    for file in os.listdir(models_folder):
        if file.endswith(".pt"):
            models.append(os.path.join(models_folder, file))

    for a in audio_files:
        for m in models:
            soft, r = test_model(modelpath=m, ifile=a,
                                 layers_dropped=layers_dropped)
            print(soft)
            feature_names = { f'{os.path.basename(m).replace(".pt", "")}_{i}':
                                 soft[i] for i in range(len(soft))}
            print(feature_names)


    return


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', required=True,
                        type=str, help='Dir of models')
    parser.add_argument('-i', '--input', required=True,
                        type=str, help='Input file for testing')
    parser.add_argument('-L', '--layers', required=False, default=0,
                        help='Number of final layers to cut. Default is 0.')
    args = parser.parse_args()
    model_dir = args.model_dir
    ifile = args.input
    layers_dropped = int(args.layers)

    compile_deep_database(ifile, model_dir, layers_dropped)