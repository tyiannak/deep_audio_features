import torch.nn as nn
from bin.config import SPECTOGRAM_SIZE


class CNN1(nn.Module):
    def __init__(self, height, width, output_dim=7, first_channels=32,
                 kernel_size=5, stride=1, padding=2, zero_pad=False, spec_size=SPECTOGRAM_SIZE,
                 fuse=False, type='classifier'):
        super(CNN1, self).__init__()
        self.zero_pad = zero_pad
        self.spec_size = spec_size
        self.fuse = fuse
        self.type = type
        self.num_cnn_layers = 4
        self.cnn_channels = 2
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
                nn.Conv2d(self.cnn_channels * first_channels, (self.cnn_channels ** 2) * first_channels,
                          kernel_size=5, stride=stride, padding=padding),
                nn.BatchNorm2d((self.cnn_channels ** 2) * first_channels),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

            self.conv_layer4 = nn.Sequential(
                nn.Conv2d((self.cnn_channels**2)*first_channels, (self.cnn_channels**3) * first_channels,
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
        if self.type == 'classifier':
            out = self.conv_layer2(out)

            out = self.conv_layer3(out)
            out = self.conv_layer4(out)
            out = out.view(out.size(0), -1)

            # DNN -- pass through linear layers
            out = self.linear_layer1(out)
            out = self.linear_layer2(out)
            out = self.linear_layer3(out)
        else:
            out = out.view(out.size(0), -1)

        return out

    def calc_out_size(self):
        height = int(self.height / 16)
        width = int(self.width / 16)
        kernels = (self.cnn_channels ** (self.num_cnn_layers - 1)) *\
            self.first_channels
        return kernels * height * width
