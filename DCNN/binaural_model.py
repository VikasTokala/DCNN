import torch

from .model import DCNN


class BinauralDCNN(DCNN):
    def forward(self, inputs):
        output_left = super().forward(inputs[:, 0])
        output_right = super().forward(inputs[:, 1])

        output = torch.stack([output_left, output_right], dim=1)

        return output
