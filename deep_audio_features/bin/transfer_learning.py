"""
Transfer learning script for audio segment models
Example:

python3 bin/transfer_learning.py -m pkl/some_model.pt -i music speech -s 1

-m : is the initial model
-i : is the list of folders that contain audio segments (each folder --> class)
-s : 0 for retraining all network weights and 1 for training only the linear
layers
"""

import argparse
import os
import time
import torch
import sys
import pickle
from torch.utils.data import DataLoader


sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.models.convAE import load_convAE
from deep_audio_features.models.cnn import load_cnn
from deep_audio_features.bin.config import EPOCHS, CNN_BOOLEAN, VARIABLES_FOLDER, ZERO_PAD, \
    FORCE_SIZE, SPECTOGRAM_SIZE, FEATURE_EXTRACTION_METHOD, OVERSAMPLING, \
    FUSED_SPECT, BATCH_SIZE
from deep_audio_features.lib.training import train_and_validate
from deep_audio_features.utils import load_dataset
from deep_audio_features.utils.model_editing import fine_tune_model, \
    print_require_grad_parameter
from deep_audio_features.dataloading.dataloading import FeatureExtractorDataset

# sys.path.insert(0, '/'.join(os.path.abspath(__file__).split(' /')[:-2]))


def transfer_learning(modelpath=None, ofile=None, folders=None, strategy=False,
                      zero_pad=ZERO_PAD, forced_size=None, layers_freezed=0):
    """Transfer learning from all folders given to a model."""

    # Arguments check
    if folders is None:
        raise FileNotFoundError()

    if modelpath is None:
        raise FileNotFoundError()


    if not isinstance(layers_freezed, int):
        raise AttributeError("variable `layers_freezed` should be int !")

    # Create classes
    #classes = [os.path.basename(f) for f in folders]

    # Check if the model is already loaded
    # and load it to 'cpu' to get some free GPU for Dataloaders
    if isinstance(modelpath, str):
        print('Loading model...')
        with open(modelpath, "rb") as input_file:
            model_params = pickle.load(input_file)
        if "classes_mapping" in model_params:
            task = "classification"
            model, hop_length, window_length = load_cnn(modelpath)
        else:
            task = "representation"
            model, hop_length, window_length = load_convAE(modelpath)
    else:
        print('Model already loaded...\nMoving it to CPU...')

    model.to('cpu')

    # Get max_seq_length from the model
    max_seq_length = model.max_sequence_length
    print(f"Setting max sequence length: {max_seq_length}...")
    zero_pad = model.zero_pad
    fuse = model.fuse

    # Use data only for training and validation
    files_train, y_train, files_eval, y_eval, classes_mapping = load_dataset.load(
        folders=folders, test=True, validation=False)

    print("New model class mapping: {}".format(classes_mapping))
    # ====== DATASETS =================================
    # Load sets
    if forced_size is None:
        spec_size = model.spec_size
    else:
        spec_size = forced_size

    train_set = FeatureExtractorDataset(X=files_train, y=y_train,
                                        fe_method=FEATURE_EXTRACTION_METHOD,
                                        oversampling=OVERSAMPLING,
                                        max_sequence_length=max_seq_length,
                                        zero_pad=zero_pad,
                                        forced_size=spec_size,
                                        fuse=fuse,
                                        hop_length=hop_length, window_length=window_length)

    eval_set = FeatureExtractorDataset(X=files_eval, y=y_eval,
                                       fe_method=FEATURE_EXTRACTION_METHOD,
                                       oversampling=OVERSAMPLING,
                                       max_sequence_length=max_seq_length,
                                       zero_pad=zero_pad,
                                       forced_size=spec_size,
                                       fuse=fuse,
                                       hop_length=hop_length, window_length=window_length)

    # Add dataloader
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, num_workers=4, drop_last=True,
        shuffle=True)

    valid_loader = DataLoader(
        eval_set, batch_size=BATCH_SIZE, num_workers=4, drop_last=True,
        shuffle=True)

    # ======= MODEL =================================================

    # Finetune
    model = fine_tune_model(model=model, output_dim=len(classes_mapping),
                            strategy=strategy, deepcopy=True, layers_freezed=layers_freezed)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    #model.to('cpu')

    print_require_grad_parameter(model)

    # Send model to device
    model.to(device)
    print(model)
    # Number of parameters to be updated
    print('Model parameters:{}'.format(sum(p.numel()
                                           for p in model.parameters()
                                           if p.requires_grad)))

    ##################################
    # TRAINING PIPELINE
    ##################################
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=0.001,
                                  weight_decay=.02)

    if task == "classification":
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        loss_function = torch.nn.MSELoss()


    best_model, train_losses, valid_losses, \
    train_metric, val_metric, \
    val_comparison_metric, _epochs = train_and_validate(
        model=model, train_loader=train_loader,
        valid_loader=valid_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        epochs=EPOCHS,
        task=task,
        cnn=CNN_BOOLEAN,
        validation_epochs=5,
        early_stopping=True)

    timestamp = time.ctime()
    timestamp = timestamp.replace(" ", "_")

    if task == "classification":

        print('All validation accuracies: {} \n'.format(val_metric))
        best_index = val_comparison_metric.index(max(val_comparison_metric))
        best_model_acc = val_metric[best_index]
        print('Best model\'s validation accuracy: {}'.format(best_model_acc))
        best_model_f1 = val_comparison_metric[best_index]
        print('Best model\'s validation f1 score: {}'.format(best_model_f1))
        best_model_loss = valid_losses[best_index]
        print('Best model\'s validation loss: {}'.format(best_model_loss))

    else:

        print('All validation errors: {} \n'.format(val_comparison_metric))
        best_index = val_comparison_metric.index(min(val_comparison_metric))
        best_model_error = val_comparison_metric[best_index]
        print('Best model\'s validation error: {}'.format(best_model_error))
        best_model_loss = valid_losses[best_index]
        print('Best model\'s validation loss: {}'.format(best_model_loss))

    if ofile is None:
        ofile = f"{best_model.__class__.__name__}_{_epochs}_{timestamp}.pt"
    else:
        ofile = str(ofile)
        if '.pt' not in ofile or '.pkl' not in ofile:
            ofile = ''.join([ofile, '.pt'])

    if not os.path.exists(VARIABLES_FOLDER):
        os.makedirs(VARIABLES_FOLDER)
    modelname = os.path.join(
        VARIABLES_FOLDER, ofile)

    print(f"\nSaving model to: {modelname}\n")
    best_model = best_model.to("cpu")
    # Save model for later use
    model_params = {
        "height": best_model.height, "width": best_model.width,
        "zero_pad": zero_pad, "spec_size": spec_size, "fuse": fuse,
        "max_sequence_length": max_seq_length,
        "type": best_model.type, "state_dict": best_model.state_dict(), "window_length": window_length,
        "hop_length": hop_length
    }
    if task == "classification":
        model_params["classes_mapping"] = classes_mapping
        model_params["output_dim"] = len(classes_mapping)
        model_params["validation_f1"] = best_model_f1
    else:
        model_params["representation_channels"] = best_model.representation_channels
        model_params["validation_error"] = best_model_error
    with open(modelname, "wb") as output_file:
        pickle.dump(model_params, output_file)

if __name__ == '__main__':

    # Read arguments -- a list of folders
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        type=str, nargs='+', help='Input folders')

    parser.add_argument('-m', '--model', required=True, type=str,
                        help='Model to apply tranfer learning.')

    parser.add_argument('-s', '--strategy', required=False, action='store_true',
                        help='If added update only linear layers')

    parser.add_argument('-l', '--layers_freezed', required=False, default=0,
                        type=int, help='Number of final layers to freeze their weights')
    parser.add_argument('-o', '--ofile', required=False, default=None,
                        type=str, help='Model name.')
    args = parser.parse_args()

    # Get argument
    folders = args.input
    modelpath = args.model
    strategy = args.strategy
    layers_freezed = args.layers_freezed
    ofile = args.ofile
    # Fix any type errors
    folders = [f.replace(',', '').strip() for f in folders]

    # Check that every folder exists
    for f in folders:
        if os.path.exists(f) is False:
            raise FileNotFoundError()

    # Check that model file exists
    if os.path.exists(modelpath) is False:
        raise FileNotFoundError

    # If everything is ok, time to start
    if FORCE_SIZE:
        transfer_learning(modelpath=modelpath, ofile=ofile, folders=folders,
                          strategy=strategy, forced_size=SPECTOGRAM_SIZE, layers_freezed=layers_freezed)
    else:
        transfer_learning(modelpath=modelpath, ofile=ofile, folders=folders, strategy=strategy, layers_freezed=layers_freezed)
