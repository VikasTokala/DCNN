import torch
import torch.nn as nn


class Stft(nn.Module):
    def __init__(self, n_dft=1024, hop_size=512, onesided=True,
                 is_complex=True):

        super().__init__()

        self.n_dft = n_dft
        self.hop_size = hop_size
        self.onesided = onesided
        self.is_complex = is_complex

    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels, time_steps)"

        y = torch.stft(x, self.n_dft, hop_length=self.hop_size, 
                       onesided=self.onesided, return_complex=True)
        
        y = y[:, 1:] # Remove DC component (f=0hz)

        # y.shape == (batch_size*channels, time, freqs)

        if not self.is_complex:
            y = torch.view_as_real(y)
            y = y.movedim(-1, 1) # move complex dim to front

        return y


class IStft(nn.Module):
    def __init__(self, n_dft=1024, hop_size=512, onesided=True):

        super().__init__()

        self.n_dft = n_dft
        self.hop_size = hop_size
        self.onesided = onesided

    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels, time_steps)"

        y = torch.istft(x, self.n_dft, hop_length=self.hop_size, 
                       onesided=self.onesided)

        return y
