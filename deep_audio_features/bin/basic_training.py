"""
Script for audio segment classifier training using CNNs on melgrams or mfccs

Example:

python3 bin/basic_training.py -i music speech silence
-i : is the list of folders that contain audio segments (each folder --> class)

model is saved in pkl folder (exact filename is printed after training)

"""

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.bin.config import EPOCHS, CNN_BOOLEAN, VARIABLES_FOLDER, ZERO_PAD, \
    FORCE_SIZE, SPECTOGRAM_SIZE, FEATURE_EXTRACTION_METHOD, OVERSAMPLING, \
    FUSED_SPECT, BATCH_SIZE
from deep_audio_features.models.cnn import CNN1
from deep_audio_features.lib.training import train_and_validate
from deep_audio_features.utils import load_dataset
from deep_audio_features.dataloading.dataloading import FeatureExtractorDataset


def train_model(folders=None, ofile=None, zero_pad=ZERO_PAD,
                forced_size=None):
    """Train a given model on a given dataset"""
    # Check that folders exist
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    if folders is None:
        raise FileNotFoundError()

    # Create classes
    classes = [os.path.basename(f) for f in folders]

    # Use data only for training and validation. Instead of using validation,
    # we just use test data. There is no difference.
    files_train, y_train, files_eval, y_eval = load_dataset.load(
        folders=folders, test=True, validation=False)

    # Compute max sequence length
    max_seq_length = load_dataset.compute_max_seq_len(reload=False,
                                                      X=files_train+files_eval)

    # Load sets
    print('-------Creating train set-------')
    if forced_size is None:
        train_set = FeatureExtractorDataset(X=files_train, y=y_train,
                                            fe_method=
                                            FEATURE_EXTRACTION_METHOD,
                                            oversampling=OVERSAMPLING,
                                            max_sequence_length=max_seq_length,
                                            zero_pad=zero_pad,
                                            fuse=FUSED_SPECT)

    else:
        train_set = FeatureExtractorDataset(X=files_train, y=y_train,
                                            fe_method=
                                            FEATURE_EXTRACTION_METHOD,
                                            oversampling=OVERSAMPLING,
                                            max_sequence_length=max_seq_length,
                                            zero_pad=zero_pad,
                                            forced_size=forced_size,
                                            fuse=FUSED_SPECT)

    if forced_size is None:
        spec_size = train_set.spec_size
    else:
        spec_size = forced_size
    print('-------Creating validation set-------')
    eval_set = FeatureExtractorDataset(X=files_eval, y=y_eval,
                                       fe_method=
                                       FEATURE_EXTRACTION_METHOD,
                                       oversampling=OVERSAMPLING,
                                       max_sequence_length=max_seq_length,
                                       zero_pad=zero_pad,
                                       forced_size=spec_size,
                                       fuse=FUSED_SPECT)

    print('-----------------------------------')
    # Add dataloader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              num_workers=4, drop_last=True, shuffle=True)

    valid_loader = DataLoader(eval_set, batch_size=BATCH_SIZE,
                              num_workers=4, drop_last=True, shuffle=True)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")
    # Create model and send to device

    if zero_pad:
        height = max_seq_length
        width = train_set.X[0].shape[1]
    else:
        height = spec_size[0]
        width = spec_size[1]
    model = CNN1(height=height, width=width, output_dim=len(classes),
                 zero_pad=zero_pad, spec_size=spec_size, fuse=FUSED_SPECT)
    model.to(device)
    # Add max_seq_length to model
    model.max_sequence_length = max_seq_length

    print('Model parameters:{}'.format(sum(p.numel()
                                           for p in model.parameters()
                                           if p.requires_grad)))

    ##################################
    # TRAINING PIPELINE
    ##################################
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=0.002,
                                  weight_decay=.02)

    best_model, train_losses, valid_losses, train_accuracy,\
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
    print('All validation accuracies: {} \n'.format(valid_accuracy))
    best_index = valid_f1.index(max(valid_f1))
    best_model_acc = valid_accuracy[best_index]
    print('Best model\'s validation accuracy: {}'.format(best_model_acc))
    best_model_f1 = valid_f1[best_index]
    print('Best model\'s validation f1 score: {}'.format(best_model_f1))
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
    # Save model for later use
    torch.save(best_model, modelname)


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

    if FORCE_SIZE:
        train_model(folders, ofile, forced_size=SPECTOGRAM_SIZE)
    else:
        train_model(folders, ofile)
