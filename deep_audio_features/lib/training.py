import torch
import sys
import math
from copy import deepcopy
import numpy as np
from sklearn.metrics import f1_score


def train_and_validate(model,
                       train_loader,
                       valid_loader,
                       loss_function,
                       optimizer,
                       epochs,
                       cnn=False,
                       validation_epochs=5,
                       early_stopping=False,
                       patience=10):
    """
    Trains the given <model>.
    Then validates every <validation_epochs>.
    Returns: <best_model> containing the model with best parameters.
    """

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min',
                                                           verbose=True)

    # obtain the model's device ID
    device = next(model.parameters()).device

    print(next(iter(train_loader)))

    EPOCHS = epochs
    VALIDATION_EPOCHS = validation_epochs

    # Store losses, models
    all_accuracy_training = []
    all_accuracy_validation = []
    all_train_loss = []
    all_valid_loss = []
    all_valid_f1 = []
    best_model = None
    best_model_epoch = 0
    min_loss = 1000
    f1_max = 0
    early_stop_counter = 0

    # Iterate for EPOCHS
    for epoch in range(1, EPOCHS + 1):

        scheduler.step(epoch)
        # ===== Training HERE =====
        train_loss, train_acc = train(epoch, train_loader, model,
                                      loss_function, optimizer, cnn=cnn)
        # Store statistics for later usage
        all_train_loss.append(train_loss)
        all_accuracy_training.append(train_acc)

        # ====== VALIDATION HERE ======

        valid_loss, valid_acc, valid_f1 = validate(epoch, valid_loader,
                                         model, loss_function,
                                         validation_epochs, cnn=cnn)

        # Find best model
        if best_model is None:
            # Initialize
            # Store model but on cpu
            best_model = deepcopy(model).to('cpu')
            best_model_epoch = epoch
            # Save new minimum
            f1_max = valid_f1
        # New model with lower loss
        elif valid_f1 > f1_max + 1e-5:
            # Update loss
            f1_max = valid_f1
            # Update best model, store on cpu
            best_model = deepcopy(model).to('cpu')
            best_model_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Store statistics for later usage
        all_valid_loss.append(valid_loss)
        all_accuracy_validation.append(valid_acc)
        all_valid_f1.append(valid_f1)
        # Make sure enough epochs have passed
        if epoch < 4 * VALIDATION_EPOCHS:
            continue

        # Early stopping enabled?
        if early_stopping is False:
            continue
        # If enabled do everything needed
        STOP = True

        # If validation loss is ascending two times in a row exit training
        if early_stop_counter > patience:
            print(f'\nIncreasing loss..')
            print(f'\nResetting model to epoch {best_model_epoch}.')
            # Remove unnessesary model
            model.to('cpu')
            best_model = best_model.to(device)
            # Exit 2 loops at the same time, go to testing
            return best_model, all_train_loss, all_valid_loss, \
                   all_accuracy_training, all_accuracy_validation,\
                   all_valid_f1, epoch

        # Small change in loss
        """
        if (abs(all_valid_loss[-1] - all_valid_loss[-2]) < 1e-3 and
                abs(all_valid_loss[-2] - all_valid_loss[-3]) < 1e-3):
            print(f'\nVery small change in loss..')
            print(f'\nResetting model to epoch {best_model_epoch}.')
            # Remove unnessesary model
            model.to('cpu')
            best_model = best_model.to(device)
            # Exit 2 loops at the same time, go to testing
            return best_model, all_train_loss, all_valid_loss, \
                   all_accuracy_training, all_accuracy_validation, epoch
        """
    print(f'\nTraining exited normally at epoch {epoch}.')
    # Remove unnessesary model
    model.to('cpu')
    best_model = best_model.to(device)
    return best_model, all_train_loss, all_valid_loss, \
           all_accuracy_training, all_accuracy_validation,\
           all_valid_f1, epoch


def train(_epoch, dataloader, model, loss_function, optimizer, cnn=False):
    # Set model to train mode
    model.train()
    training_loss = 0.0
    correct = 0

    # obtain the model's device ID
    device = next(model.parameters()).device

    # Iterate the batch
    for index, batch in enumerate(dataloader, 1):

        # Split the contents of each batch[i]
        inputs, labels, lengths = batch

        inputs = inputs.to(device)
        labels = labels.type('torch.LongTensor').to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass: y' = model(x)
        if cnn is False:
            y_pred = model.forward(inputs, lengths)
        else:
            # We got a CNN
            # Add a new axis for CNN filter features, [z-axis]
            inputs = inputs[:, np.newaxis, :, :]
            y_pred = model.forward(inputs)

        # print(f'\ny_preds={y_pred}')
        # print(f'\nlabels={labels}')
        # Compute loss: L = loss_function(y', y)
        loss = loss_function(y_pred, labels)

        labels_cpu = labels.detach().clone().to('cpu').numpy()
        # Get accuracy
        correct += sum([int(a == b)
                        for a, b in zip(labels_cpu,
                                        np.argmax(y_pred.detach().clone().to('cpu').numpy(),
                                                  axis=1))])
        # Backward pass: compute gradient wrt model parameters
        loss.backward()

        # Update weights
        optimizer.step()

        # Add loss to total epoch loss
        training_loss += loss.data.item()

        # print statistics
        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))

    # print statistics
    progress(loss=training_loss / len(dataloader.dataset),
             epoch=_epoch,
             batch=index,
             batch_size=dataloader.batch_size,
             dataset_size=len(dataloader.dataset))

    accuracy = correct/len(dataloader.dataset) * 100
    # Print some stats
    # print(
    #     f'\nTrain loss at epoch {_epoch} : {round(training_loss/len(dataloader), 4)}')
    # Return loss, accuracy
    return training_loss / len(dataloader.dataset), accuracy


def validate(_epoch, dataloader, model, loss_function,
             validation_epochs, cnn=False):
    """Validate the model."""

    # Put model to evalutation mode
    model.eval()

    correct = 0

    # obtain the model's device ID
    device = next(model.parameters()).device

    with torch.no_grad():
        valid_loss = 0

        pred_all = []
        actual_labels = []
        for index, batch in enumerate(dataloader, 1):

            # Get the sample
            inputs, labels, lengths = batch

            # Transfer to device
            inputs = inputs.to(device)
            labels = labels.type('torch.LongTensor').to(device)

            # Forward through the network
            if cnn is False:
                y_pred = model.forward(inputs, lengths)
            else:
                # We got CNN
                # Add a new axis for CNN filter features, [z-axis]
                inputs = inputs[:, np.newaxis, :, :]
                y_pred = model.forward(inputs)

            # Compute loss
            loss = loss_function(y_pred, labels)

            # Add validation loss
            valid_loss += loss.data.item()

            y_pred = np.argmax(y_pred.detach().clone().to('cpu').numpy(), axis=1)
            pred_all.append(y_pred)

            labels_cpu = labels.detach().clone().to('cpu').numpy()
            actual_labels.append(labels_cpu)
            # Get accuracy
            correct += sum([int(a == b)
                            for a, b in zip(labels_cpu, y_pred)])

        accuracy = correct / len(dataloader.dataset)

        labels = [item for sublist in actual_labels for item in sublist]
        preds = [item for sublist in pred_all for item in sublist]
        f1 = f1_score(labels, preds, average='macro')

        if _epoch % validation_epochs == 0:
            # Print some stats
            print('\nValidation results for epoch {}:'.format(_epoch))
            print('    --> loss: {}'.format(
                round(valid_loss/len(dataloader.dataset), 4)))
            print('    --> accuracy: {}'.format(round(accuracy, 4)))
            print('    --> f1 score: {}'.format(round(f1, 4)))

    return valid_loss / len(dataloader.dataset), accuracy, f1


def test(model, dataloader, cnn=False, classifier=True):
    """Tests a given model.

### Arguments:

        model {torch.nn.Module} : Model that will be tested.

        dataloader {torch.utils.data.DataLoader} : Test set dataloader.

        cnn {bool} : If True test function expects a cnn model.
                        This adds an extra axis for convolutional
                        layers features.

        classifier {bool} : If `True` the function returns the model's
                            class prediction. This makes the model work
                            as a classifier.

### Returns:

        posteriors {np.array} : An np.array containing output values
                            of the final layer of a model.

        y_pred {np.array} : If `classifier` is `True` returns the class
                             prediction else `False`.
        y_true {np.array} : Actual labels
    """
    # obtain the model's device ID
    device = next(model.parameters()).device

    correct = 0
    # Create empty array for storing predictions and labels
    posteriors = []
    y_pred = []
    y_true = []
    for index, batch in enumerate(dataloader, 1):
        # Split each batch[index]
        inputs, labels, lengths = batch

        # Transfer to device
        inputs = inputs.to(device)
        labels = labels.type('torch.LongTensor').to(device)
        # print(f'\n labels: {labels}')

        # Forward through the network
        if cnn is False:
            out = model.forward(inputs, lengths)
        else:
            # Add a new axis for CNN filter features, [z-axis]
            inputs = inputs[:, np.newaxis, :, :]
            out = model.forward(inputs)

        if classifier is False:
            posteriors.append(out.cpu().detach().numpy())
            y_pred.append(None)
            y_true.append(None)
        else:
            # Predict the one with the maximum probability
            predictions = torch.argmax(out, -1)
            # Save predictions
            y_pred.append(predictions.cpu().data.numpy())
            y_true.append(labels.cpu().data.numpy())
            posteriors.append(out[0].cpu().detach().numpy())
    if classifier is True:
        # Get metrics
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()

    return posteriors, y_pred, y_true


def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()
