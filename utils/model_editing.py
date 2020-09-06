import torch
import copy


def print_require_grad_parameter(model):
    """Iterate through `Conv2d` and `Linear` layers of a model.
Print whether a layer updates its weights or not.

### Arguments:

        model {torch.nn.Module} : The model to iterate.
"""
    for ch1 in model.children():
        for layer in ch1.children():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                print(f"{layer} -> {layer.weight.requires_grad}")
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


def fine_tune_model(model=None, output_dim=None, strategy=0, deepcopy=False, *args, **kwargs):
    """Fine tune a given model.

### Arguments:

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

    # Train the entire model
    if strategy == 0:
        # Just adjust the output dimentions

        # Get all layers
        model_layers = [y for x in model.children() for y in x.children()]

        # idx1 : children level 1 that contain linear
        # idx2 : children level 2 that contain -- are linear
        # linear : the class torch.nn.Linear()
        # Get a list of tuples. Last layer is [-1]
        idx1_idx2_linear = [(idx1, idx2, ch2) for idx1, ch1 in enumerate(model.children())
                            for idx2, ch2 in enumerate(ch1.children()) if isinstance(ch2, torch.nn.Linear)]

        if model_layers == []:
            # Has not children level 2 => functional
            FUNCTIONAL = True
            raise NotImplementedError()

        # Get indexes for linear layers
        linear_idx = [idx for idx, x in enumerate(
            model_layers) if isinstance(x, torch.nn.Linear)]

        # Get last linear layer variables
        last_linear = list(list(model.children())[
                           idx1_idx2_linear[-1][0]])[idx1_idx2_linear[-1][1]]

        # Get input dimension
        input_dim = last_linear.in_features

        # Set output dimension
        last_linear.out_features = output_dim

        # Get indexes for editing
        idx1_idx2_layer = [(idx1, idx2, ch2) for idx1, ch1 in enumerate(model.children())
                           for idx2, ch2 in enumerate(ch1.children())]
        # Set model's children prop: require_grad = True for all layers
        for idx1, idx2, layer in idx1_idx2_layer:
            try:
                model[idx1][idx2].weight.requires_grad = True
                model[idx1][idx2].bias.requires_grad = True
            except:
                # Dropout etc.
                pass

        return model

    # Train only linear layers
    if strategy == 1:
        # Freeze all layers except for the linear
        model_layers = [y for x in model.children() for y in x.children()]
        # indexes for editing
        idx1_idx2_layer = [(idx1, idx2, ch2) for idx1, ch1 in enumerate(model.children())
                           for idx2, ch2 in enumerate(ch1.children())]
        for idx1, idx2, layer in idx1_idx2_layer:
            if isinstance(layer, torch.nn.Conv2d):
                # Freeze bias
                try:
                    layer.bias.requires_grad = False
                except Exception as e:
                    print(
                        f"Failed to freeze all bias at model[{idx1}],[{idx2}]={layer}: {e}")
                # Freeze weight
                try:
                    layer.weight.requires_grad = False
                except Exception as e:
                    print(
                        f"Failed to freeze all weights at model[{idx1}],[{idx2}]={layer}: {e}")

        # idx1 : children level 1 that contain linear
        # idx2 : children level 2 that contain -- are linear
        # linear : the class torch.nn.Linear()
        # Get a list of tuples. Last layer is [-1]
        idx1_idx2_linear = [(idx1, idx2, ch2) for idx1, ch1 in enumerate(model.children())
                            for idx2, ch2 in enumerate(ch1.children()) if isinstance(ch2, torch.nn.Linear)]

        # Get indexes for linear layers
        linear_idx = [idx for idx, x in enumerate(
            model_layers) if isinstance(x, torch.nn.Linear)]

        # Get last linear layer variables
        last_linear = list(list(model.children())[
                           idx1_idx2_linear[-1][0]])[idx1_idx2_linear[-1][1]]

        # Get input dimension
        input_dim = last_linear.in_features

        # Set output dimension
        last_linear.out_features = output_dim

        return model
