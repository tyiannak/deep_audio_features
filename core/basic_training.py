import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

import config
from config import EPOCHS, CNN_BOOLEAN, VARIABLES_FOLDER
from models.cnn import CNN1
from lib.training import train_and_validate
from utils import load_dataset
from dataloading.dataloading import FeatureExtractorDataset

# sys.path.insert(0, '/'.join(os.path.abspath(__file__).split(' /')[:-2]))


def train_model(folders=None, ofile=None):
    """Train a given model on a given dataset"""
    # Check that folders exist
    if folders is None:
        raise FileNotFoundError()

    # Create classes
    classes = [os.path.basename(f) for f in folders]

    # Use data only for training and validation. Instead of using validation,
    # we just use test data. There is no difference.
    X_train, y_train, X_eval, y_eval = load_dataset.load(
        folders=folders, test=True, validation=False)

    # Compute max sequence length
    max_seq_length = load_dataset.max_sequence_length(
        reload=False, X=X_train+X_eval)

    # Load sets
    train_set = FeatureExtractorDataset(X=X_train, y=y_train,
                                        fe_method=
                                        config.FEATURE_EXTRACTION_METHOD,
                                        oversampling=config.OVERSAMPLING,
                                        max_sequence_length=max_seq_length)

    eval_set = FeatureExtractorDataset(X=X_eval, y=y_eval,
                                       fe_method=
                                       config.FEATURE_EXTRACTION_METHOD,
                                       oversampling=config.OVERSAMPLING,
                                       max_sequence_length=max_seq_length)

    # Add dataloader
    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, num_workers=4, drop_last=True,
        shuffle=True)

    valid_loader = DataLoader(
        eval_set, batch_size=config.BATCH_SIZE, num_workers=4, drop_last=True,
        shuffle=True)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")
    # Create model and send to device
    height = max_seq_length
    width = train_set.X[0].shape[1]
    model = CNN1(height=height, width=width, output_dim=len(classes))
    model.to(device)
    print(model)
    # Add max_seq_length to model
    model.max_sequence_length = max_seq_length

    print('Model parameters:{}'.format(sum(p.numel()
                                           for p in model.parameters()
                                           if p.requires_grad)))

    ##################################
    # TRAINING PIPELINE
    ##################################
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), weight_decay=.02)

    best_model, train_losses, valid_losses, train_accuracy, \
    valid_accuracy, _epochs = train_and_validate(model=model,
                                                 train_loader=train_loader,
                                                 valid_loader=valid_loader,
                                                 loss_function=loss_function,
                                                 optimizer=optimizer,
                                                 epochs=EPOCHS,
                                                 cnn=CNN_BOOLEAN,
                                                 validation_epochs=5,
                                                 early_stopping=True)
    timestamp = time.ctime()
    if ofile is None:
        ofile = f"{best_model.__class__.__name__}_{_epochs}_{timestamp}.pt"
    else:
        ofile = str(ofile)
        if '.pt' not in ofile or '.pkl' not in ofile:
            ofile = ''.join([ofile, '.pt'])
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

    train_model(folders, ofile)
