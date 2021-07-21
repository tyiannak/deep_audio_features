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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))

from deep_audio_features.bin.config import EPOCHS, CNN_BOOLEAN, VARIABLES_FOLDER, ZERO_PAD, \
    FORCE_SIZE, SPECTOGRAM_SIZE, FEATURE_EXTRACTION_METHOD, OVERSAMPLING, \
    FUSED_SPECT, BATCH_SIZE
from deep_audio_features.lib.training import train_and_validate
from deep_audio_features.utils import load_dataset
from deep_audio_features.utils.model_editing import fine_tune_model, \
    print_require_grad_parameter
from deep_audio_features.dataloading.dataloading import FeatureExtractorDataset

# sys.path.insert(0, '/'.join(os.path.abspath(__file__).split(' /')[:-2]))


def transfer_learning(model=None, folders=None, strategy=0,
                      zero_pad=ZERO_PAD, forced_size=None):
    """Transfer learning from all folders given to a model."""

    # Arguments check
    if folders is None:
        raise FileNotFoundError()

    if model is None:
        raise FileNotFoundError()

    if not isinstance(strategy, int):
        raise AttributeError("variable `strategy` should be int !")

    # Create classes
    classes = [os.path.basename(f) for f in folders]

    # Check if the model is already loaded
    # and load it to 'cpu' to get some free GPU for Dataloaders
    if isinstance(model, str):
        print('Loading model...')
        model = torch.load(model, map_location='cpu')
    else:
        print('Model already loaded...\nMoving it to CPU...')
        model.to('cpu')

    # Get max_seq_length from the model
    max_seq_length = model.max_sequence_length
    print(f"Setting max sequence length: {max_seq_length}...")

    # Use data only for training and validation
    files_train, y_train, files_eval, y_eval = load_dataset.load(
        folders=folders, test=True, validation=False)

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
                                        fuse=FUSED_SPECT)

    eval_set = FeatureExtractorDataset(X=files_eval, y=y_eval,
                                       fe_method=FEATURE_EXTRACTION_METHOD,
                                       oversampling=OVERSAMPLING,
                                       max_sequence_length=max_seq_length,
                                       zero_pad=zero_pad,
                                       forced_size=spec_size,
                                       fuse=FUSED_SPECT)

    # Add dataloader
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, num_workers=4, drop_last=True,
        shuffle=True)

    valid_loader = DataLoader(
        eval_set, batch_size=BATCH_SIZE, num_workers=4, drop_last=True,
        shuffle=True)

    # ======= MODEL =================================================

    # Finetune
    model = fine_tune_model(model=model, output_dim=len(classes),
                            strategy=strategy, deepcopy=True)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")
    # Send model to device
    model.to('cpu')
    print(model)
    print_require_grad_parameter(model)

    # Add max_seq_length to model
    print('Model parameters:{}'.format(sum(p.numel()
                                           for p in model.parameters()
                                           if p.requires_grad)))

    ##################################
    # TRAINING PIPELINE
    ##################################
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=0.001,
                                  weight_decay=.02)

    best_model, train_losses, valid_losses, train_accuracy, \
    valid_accuracy, valid_f1, _epochs = train_and_validate(model=model,
                                                 train_loader=train_loader,
                                                 valid_loader=valid_loader,
                                                 loss_function=loss_function,
                                                 optimizer=optimizer,
                                                 epochs=EPOCHS,
                                                 cnn=CNN_BOOLEAN,
                                                 validation_epochs=5,
                                                 early_stopping=True)

    timestamp = time.ctime()
    model_id = f"{best_model.__class__.__name__}_{_epochs}_{timestamp}.pt"
    modelname = os.path.join(
        VARIABLES_FOLDER, model_id)
    print(f"\nSaving model to: {model_id}\n")
    # Save model for later use
    torch.save(best_model, modelname)


if __name__ == '__main__':

    # Read arguments -- a list of folders
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        type=str, nargs='+', help='Input folders')

    parser.add_argument('-m', '--model', required=True, type=str,
                        help='Model to apply tranfer learning.')

    parser.add_argument('-s', '--strategy', required=False, default=0, type=int,
                        help='Strategy to apply in transfer learning: 0 or 1.')

    args = parser.parse_args()

    # Get argument
    folders = args.input
    modelpath = args.model
    strategy = args.strategy

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
        transfer_learning(model=modelpath, folders=folders,
                          strategy=strategy, forced_size=SPECTOGRAM_SIZE)
    else:
        transfer_learning(model=modelpath, folders=folders, strategy=strategy)
