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
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # torch.Size([16, 32, 8, 12])


        )


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, 5, stride=3, padding=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, (5, 3), stride=3, padding=1, output_padding=(0, 1)),
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
