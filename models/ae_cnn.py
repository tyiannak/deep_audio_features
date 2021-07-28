import torch.nn as nn

# def calc_out_size(self):
#         height = int(self.height / 16)
#         width = int(self.width / 16)
#         kernels = (self.cnn_channels ** (self.num_cnn_layers - 1)) *\
#             self.first_channels
#         return kernels * height * width


# # exact output size can be also specified as an argument
    
#     # >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
#     # >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
#     # >>> h = downsample(input)
#     # >>> h.size()
#     # torch.Size([1, 16, 6, 6])
#     # >>> output = upsample(h, output_size=input.size())
#     # >>> output.size()
#     # torch.Size([1, 16, 12, 12])

# Define the AE CNN architecture
class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        self.num_cnn_layers = 3
        self.cnn_channels = 2
        self.height = 201
        self.width = 128
        self.first_channels = 16

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        print()
        print("Start Encode: ", x.shape)
        x = self.encoder(x)
        print("Finished Encode: ", x.shape)
        x = self.decoder(x)
        print("Finished Decode: ", x.shape)
        return x

    def calc_out_size(self):
        height = int(self.height / 16)
        width = int(self.width / 16)
        kernels = (self.cnn_channels ** (self.num_cnn_layers - 1)) *\
            self.first_channels
        return kernels * height * width
