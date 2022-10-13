import torch

from .model import DCNN


class BinauralDCNN(DCNN):
    def forward(self, inputs):
        batch_size, binaural_channels, time_bins = inputs.shape

        inputs = inputs.flatten(end_dim=1)
        output = super().forward(inputs)
        output = output.unflatten(0, (batch_size, binaural_channels))

        return output
