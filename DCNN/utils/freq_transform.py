import torch
import torch.nn as nn


class FTB(torch.nn.Module):
    """This is an attention layer based on frequency transformation"""

    def __init__(self,channels=24, f_length=257):
        super().__init__()
        self.channels = channels
        self.f_length = f_length