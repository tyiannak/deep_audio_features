import torch.nn as nn
import sys
import os
import pickle
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.bin.config import SPECTOGRAM_SIZE


class ConvEncoder(nn.Module):
    def __init__(self, height, width, representation_channels=100, first_channels=32,
                 kernel_size=3, stride=1, padding=2):
        super(ConvEncoder, self).__init__()

        self.num_cnn_layers = 4
        self.cnn_channels = 2
        self.height = height
        self.width = width
        self.first_channels = first_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.representation_channels = representation_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, first_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(first_channels),
            nn.LeakyReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(first_channels, self.cnn_channels * first_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(self.cnn_channels * first_channels),
            nn.LeakyReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.cnn_channels * first_channels, (self.cnn_channels ** 2) * first_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d((self.cnn_channels ** 2) * first_channels),
            nn.LeakyReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.conv4 = nn.Conv2d(
            (self.cnn_channels ** 2) * first_channels, representation_channels, kernel_size, stride=stride, padding=padding)
        self.pool4 = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        size_1 = x.size()
        x, idx_1 = self.pool1(x)
        x = self.conv2(x)
        size_2 = x.size()
        x, idx_2 = self.pool2(x)
        x = self.conv3(x)
        size_3 = x.size()
        x, idx_3 = self.pool3(x)
        x = self.conv4(x)
        size_4 = x.size()
        x, idx_4 = self.pool4(x)

        return x, [idx_4, idx_3, idx_2, idx_1], [size_4, size_3, size_2, size_1]

    def calc_out_size(self):
        height = int(self.height / 16)
        width = int(self.width / 16)
        kernels = (self.cnn_channels ** (self.num_cnn_layers - 1)) * \
                  self.first_channels
        return kernels * height * width


class ConvDecoder(nn.Module):
    def __init__(self, height, width, representation_channels=100, first_channels=32,
                 kernel_size=3, stride=1, padding=2):
        super(ConvDecoder, self).__init__()

        self.num_cnn_layers = 4
        self.cnn_channels = 2
        self.height = height
        self.width = width
        self.first_channels = first_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.representation_channels = representation_channels

        self.unpool0 = nn.MaxUnpool2d(kernel_size=2)

        self.unconv1 = nn.Sequential(
            nn.ConvTranspose2d(representation_channels, (self.cnn_channels ** 2) * first_channels, kernel_size,
                           stride=stride, padding=padding),
            nn.BatchNorm2d((self.cnn_channels ** 2) * first_channels),
            nn.LeakyReLU())
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2)

        self.unconv2 = nn.Sequential(
            nn.ConvTranspose2d((self.cnn_channels ** 2) * first_channels, self.cnn_channels * first_channels, kernel_size,
                           stride=stride, padding=padding),
            nn.BatchNorm2d(self.cnn_channels * first_channels),
            nn.LeakyReLU())
        self. unpool2 = nn.MaxUnpool2d(kernel_size=2)

        self.unconv3 = nn.Sequential(
            nn.ConvTranspose2d(self.cnn_channels * first_channels, first_channels, kernel_size, stride=stride,
                           padding=padding),
            nn.BatchNorm2d(first_channels),
            nn.LeakyReLU())
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2)

        self.unconv4 = nn.ConvTranspose2d(first_channels, 1, kernel_size, stride=stride, padding=padding)

    def forward(self, x, pool_indices, pool_sizes):
        x = self.unpool0(x, pool_indices[0], output_size=pool_sizes[0])
        x = self.unconv1(x)
        x = self.unpool1(x, pool_indices[1], output_size=pool_sizes[1])
        x = self.unconv2(x)
        x = self.unpool2(x, pool_indices[2], output_size=pool_sizes[2])
        x = self.unconv3(x)
        x = self.unpool3(x, pool_indices[3], output_size=pool_sizes[3])
        x = self.unconv4(x)

        return x

    def calc_out_size(self):
        height = int(self.height / 16)
        width = int(self.width / 16)
        kernels = (self.cnn_channels ** (self.num_cnn_layers - 1)) * \
                  self.first_channels
        return kernels * height * width


class ConvAutoEncoder(nn.Module):
    def __init__(self, height, width, representation_channels=10, first_channels=64,
                 kernel_size=5, stride=1, padding=2, zero_pad=False,
                 spec_size=SPECTOGRAM_SIZE, fuse=False):
        super(ConvAutoEncoder, self).__init__()

        self.num_cnn_layers = 4
        self.cnn_channels = 2
        self.height = height
        self.width = width
        self.first_channels = first_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.representation_channels = representation_channels
        self.zero_pad = zero_pad
        self.spec_size = spec_size
        self.fuse = fuse

        self.encoder = ConvEncoder(
            height, width, representation_channels=representation_channels,
            first_channels=first_channels, kernel_size=kernel_size,
            stride=stride, padding=padding)
        self.decoder = ConvDecoder(
            height, width, representation_channels=representation_channels,
            first_channels=first_channels, kernel_size=kernel_size,
            stride=stride, padding=padding)

    def forward(self, x):
        x, pool_indices, pool_sizes = self.encoder(x)
        #print(pool_sizes)
        representation = x.view(x.size(0), -1)
        x = self.decoder(x, pool_indices, pool_sizes)
        return x, representation

    def calc_out_size(self):
        height = int(self.height / 16)
        width = int(self.width / 16)
        kernels = (self.cnn_channels ** (self.num_cnn_layers - 1)) * \
                  self.first_channels
        return kernels * height * width


def load_convAE(model_path):
    with open(model_path, "rb") as input_file:
        model_params = pickle.load(input_file)
    print("Loaded model representation channels: {}".format(
        model_params["representation_channels"]))
    model = ConvAutoEncoder(
        height=model_params["height"], width=model_params["width"],
        representation_channels=model_params["representation_channels"],
        zero_pad=model_params["zero_pad"],
        spec_size=model_params["spec_size"], fuse=model_params["fuse"])
    model.max_sequence_length = model_params["max_sequence_length"]
    model.load_state_dict(model_params["state_dict"])

    return model, model_params["hop_length"], model_params["window_length"]
