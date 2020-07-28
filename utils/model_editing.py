import torch


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


def drop_layers(model, layers_dropped):
    """Removes the final `layers_dropped` from the `model`.

Arguments:

        model {torch.nn.Module} : The model to edit.

        layers_dropped {int} : The number of layers
                                to drop.

Returns:

        new_model {torch.nn.Module} : The new model.
    """
    if layers_dropped == 0:
        # Just do classification
        return model

    if layers_dropped not in list(range(1, len(list(model.children()))+1)):
        raise ArithmeticError(
            f'Please check out the number of layers to be removed ({layers_dropped}).')

    # Iterate to remove layers
    new_model_layers = []
    layers = list(model.children())[:-layers_dropped]
    flattened = False
    for layer in layers:
        # Add current layer
        new_model_layers.append(layer)
        for child in layer.children():
            # If layer contains a Linear layer flatten input before layer
            if isinstance(child, torch.nn.Linear) and flattened is False:
                # Add a flatten module. In order to do this remove last added layer
                new_model_layers = new_model_layers[:-1]
                new_model_layers.extend([Flatten(), layer])
                flattened = True
                break

    # If all linear layers where removed
    # make sure that the tensor is flattened
    if flattened is False:
        new_model_layers.append(Flatten())

    # Create model
    new_model = torch.nn.Sequential(*new_model_layers)
    return new_model
