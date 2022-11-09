import torch
import torch.nn as nn


class FAL(torch.nn.Module):
    """This is an attention layer based on frequency transformation"""

    def __init__(self, in_channels=24, f_length=256):
        super().__init__()
        self.in_channels = in_channels
        self.f_length = f_length
        self.c_fal_r = 5 # Channels to be used within the FTB
        self.out_channels = 4

        self.conv_1_multiply_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.c_fal_r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.c_fal_r),
            nn.ReLU()
        )
        self.conv_1D = nn.Sequential(
            nn.Conv1d(self.f_length * self.c_fal_r, self.in_channels, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU()
        )
        self.frec_fc = nn.Linear(self.f_length, self.f_length, bias=False)
        self.conv_1_multiply_1_2 = nn.Sequential(
            nn.Conv2d(2 * self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, inputs):
        seg_length = inputs.shape[-2]
        

        x = self.conv_1_multiply_1_1(inputs)  # [batch, c_ftb_r,time_bins,f]

        x = x.view(-1, self.f_length * self.c_fal_r, seg_length)  # [batch, c_ftb_r*f,time_bins]

        x = self.conv_1D(x)  # [batch, c_a, time_bins]

        x = x.view(-1, self.in_channels,seg_length,1)  # [batch, c_a, time_bins, 1]
        breakpoint()
        x = x * inputs  # [batch, c_a, time_bins, 1]*[batch, c_a, time_bins, f]

        x = self.frec_fc(x)  # [batch, c_a, time_bins, f]

        x = torch.cat((x, inputs), dim=1)  # [batch, 2*c_a, time_bins, f]

        outputs = self.conv_1_multiply_1_2(x)  # [batch, c_a, time_bins, f]

        return outputs
