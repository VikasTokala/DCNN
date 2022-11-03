import torch
import torch.nn as nn


class FAL(torch.nn.Module):
    """This is an attention layer based on frequency transformation"""

    def __init__(self, in_channels=24, f_length=256):
        super(FAL, self).__init__()
        self.in_channels = in_channels
        self.c_fal_r = 5 #Channels to be used within the FTB
        self.f_length = f_length

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
            nn.Conv2d(2 * self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

    def forward(self, inputs):
        _, _, _,seg_length = inputs.shape

        temp = self.conv_1_multiply_1_1(inputs)  # [B,c_ftb_r,segment_length,f]

        temp = temp.view(-1, self.f_length * self.c_fal_r, seg_length)  # [B,c_ftb_r*f,segment_length]

        temp = self.conv_1D(temp)  # [B,c_a,segment_length]

        temp = temp.view(-1, self.in_channels, seg_length, 1)  # [B,c_a,segment_length,1]

        temp = temp * inputs  # [B,c_a,segment_length,1]*[B,c_a,segment_length,f]

        temp = self.frec_fc(temp)  # [B,c_a,segment_length,f]

        temp = torch.cat((temp, inputs), dim=1)  # [B,2*c_a,segment_length,f]

        outputs = self.conv_1_multiply_1_2(temp)  # [B,c_a,segment_length,f]

        return outputs