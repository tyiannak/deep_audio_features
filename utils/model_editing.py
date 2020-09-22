import torch
import copy


def print_require_grad_parameter(model):
    """Iterate through `Conv2d` and `Linear` layers of a model.
Print whether a layer updates its weights or not.

# Arguments:

        model {torch.nn.Module} : The model to iterate.
"""
    print("\n--------------------------------")
    print("Layer -> Requires_grad ?")
    for ch1 in model.children():
        for layer in ch1.children():
            if isinstance(layer, torch.nn.Conv2d) or \
                    isinstance(layer, torch.nn.Linear):
                print(f"{layer} -> {layer.weight.requires_grad}")
    print("--------------------------------")
    print('\n\n')


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
            f'Please check out the number of layers to '
            f'be removed ({layers_dropped}).')

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
                # Add a flatten module.
                # In order to do this remove last added layer
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


def fine_tune_model(model=None, output_dim=None, strategy=0,
                    deepcopy=False, *args, **kwargs):
    """Fine tune a given model.

# Arguments:

        model {torch.nn.Module} : The model to edit.

        output_dim {int} : The number of new classes/dimensions.

        strategy {int} :

            * 0 : Return a model that updates all its
                    weights and has new output dimension.

            * 1 : Return a model that updates all its `Linear`
                    weights only and has new output dimension.

        deepcopy {bool} : If `True` a copy of `model` is returned.
                            Otherwise `model` is updated by reference
                            and returned.
    """
    # Attribute checks
    if model is None:
        raise AttributeError()
    if output_dim is None:
        raise AttributeError()
    if deepcopy is True:
        model = copy.deepcopy(model)

    if strategy == 0: # Train the entire model
        # Get all layers
        model_layers = [y for x in model.children() for y in x.children()]
        if model_layers == []:
            # Has not children level 2 => functional
            FUNCTIONAL = True
            raise NotImplementedError()
        named_children = list(model.named_children())

    elif strategy == 1: # Train only linear layers
        # Freeze all layers except for the linear
        model_layers = [y for x in model.children() for y in x.children()]
        if model_layers == []:
            raise NotImplementedError()
        named_children = list(model.named_children())
        for seq_layername, seq_layer in named_children:
            # Find all Conv2d layers and freeze weights
            if any([isinstance(c, torch.nn.Conv2d)
                    for c in seq_layer.children()]):
                for nested_layer in seq_layer.children():
                    # Skip all except Conv2d
                    if not isinstance(nested_layer, torch.nn.Conv2d):
                        continue
                    # Set grad off for bias as well as weights
                    try:
                        nested_layer.bias.requires_grad = False
                        nested_layer.weight.requires_grad = False
                    except Exception as e:
                        raise e("Error while trying to turn off gradients.")

    for seq_layername, seq_layer in named_children[::-1]:
        if any([isinstance(c, torch.nn.Linear)
                for c in seq_layer.children()]):
            newlayer = []
            for nested_layer in seq_layer.children():
                if not isinstance(nested_layer, torch.nn.Linear):
                    # Anything except linear is just added as it is
                    newlayer.append(nested_layer)
                else:
                    # If the layer is linear we need to parametrise it, so
                    # Get dimensions
                    input_dim = nested_layer.in_features
                    # Set new layer
                    newlayer.append(torch.nn.Linear(input_dim, output_dim))
            # Set the new Seq layer by replacing using attribute
            # **cant set manually due to generator is trying to access list
            setattr(model, seq_layername, torch.nn.Sequential(*newlayer))
            break

    return model
