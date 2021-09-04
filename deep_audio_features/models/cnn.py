import torch.nn as nn
import sys
import os
import pickle
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.bin.config import SPECTOGRAM_SIZE


class CNN1(nn.Module):
    def __init__(self, height, width, output_dim=7, first_channels=32,
                 kernel_size=5, stride=1, padding=2, zero_pad=False,
                 spec_size=SPECTOGRAM_SIZE,
                 fuse=False, type='classifier'):
        super(CNN1, self).__init__()
        self.zero_pad = zero_pad
        self.spec_size = spec_size
        self.fuse = fuse
        self.type = type
        self.num_cnn_layers = 4
        self.cnn_channels = 2
        self.output_dim = output_dim
        self.height = height
        self.width = width
        self.first_channels = first_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, first_channels, kernel_size=5,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(first_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2))
        if type == 'classifier':

            self.conv_layer2 = nn.Sequential(
                nn.Conv2d(first_channels, self.cnn_channels*first_channels,
                          kernel_size=5, stride=stride, padding=padding),
                nn.BatchNorm2d(self.cnn_channels*first_channels),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_layer3 = nn.Sequential(
                nn.Conv2d(self.cnn_channels * first_channels,
                          (self.cnn_channels ** 2) * first_channels,
                          kernel_size=5, stride=stride, padding=padding),
                nn.BatchNorm2d((self.cnn_channels ** 2) * first_channels),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

            self.conv_layer4 = nn.Sequential(
                nn.Conv2d((self.cnn_channels**2)*first_channels,
                          (self.cnn_channels**3) * first_channels,
                          kernel_size=5, stride=stride, padding=padding),
                nn.BatchNorm2d((self.cnn_channels ** 3) * first_channels),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.linear_layer1 = nn.Sequential(
                nn.Dropout(0.75),
                nn.Linear(self.calc_out_size(), 1024),
                nn.LeakyReLU()
            )

            self.linear_layer2 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(1024, 256),
                nn.LeakyReLU()
            )
            self.linear_layer3 = nn.Sequential(
                # nn.Dropout(0.2),
                nn.Linear(256, output_dim),
                nn.LeakyReLU()
            )

    def forward(self, x):
        # input: (batch_size,1,max_seq,features)
        # Each layer applies the following matrix tranformation
        # recursively: (batch_size,conv_output,max_seq/2 -1,features/2 -1)
        # CNN
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)

        if self.type == 'classifier':
            out = out.view(out.size(0), -1)
            # DNN -- pass through linear layers
            out = self.linear_layer1(out)
            out = self.linear_layer2(out)
            out = self.linear_layer3(out)

        return out

    def calc_out_size(self):
        height = int(self.height / 16)
        width = int(self.width / 16)
        kernels = (self.cnn_channels ** (self.num_cnn_layers - 1)) *\
            self.first_channels
        return kernels * height * width


def load_cnn(model_path):
    with open(model_path, "rb") as input_file:
        model_params = pickle.load(input_file)

    model = CNN1(height=model_params["height"], width=model_params["width"], output_dim=model_params["output_dim"],
                 zero_pad=model_params["zero_pad"], spec_size=model_params["spec_size"], fuse=model_params["fuse"],
                 type=model_params["type"])
    model.max_sequence_length = model_params["max_sequence_length"]
    model.load_state_dict(model_params["state_dict"])

    return model
