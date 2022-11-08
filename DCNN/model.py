import torch
import torch.nn as nn
# import torch.nn.functional as F
from DCNN.feature_extractors import IStft, Stft

import DCNN.utils.complexPyTorch.complexLayers as torch_complex
from DCNN.utils.show import show_params, show_model
from DCNN.utils.complexnn import (
    ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm,
    NaiveComplexLSTM, complex_cat
)
from DCNN.utils.apply_mask import apply_mask



class DCNN(nn.Module):
    def __init__(
            self,
            rnn_layers=2, rnn_units=128,
            win_len=400, win_inc=100, fft_len=512, win_type='hann',
            masking_mode='E', use_clstm=False, use_cbn=False,
            kernel_size=5, kernel_num=[16, 32, 64, 128, 256, 256],
            **kwargs
    ):
        ''' 
            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag
        '''

        super().__init__()

        # for fft
        self.win_len = win_len # TODO: Remove
        self.win_inc = win_inc # TODO: Rename to hop_size
        self.fft_len = fft_len
        self.win_type = win_type # TODO: Remove

        self.rnn_units = rnn_units
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        # self.kernel_num = [2, 8, 16, 32, 128, 128, 128]
        # self.kernel_num = [2, 16, 32, 64, 128, 256, 256]
        self.kernel_num = [2] + kernel_num
        self.masking_mode = masking_mode
        self.use_clstm = use_clstm

        self.stft = Stft(self.fft_len, self.win_inc)
        self.istft = IStft(self.fft_len, self.win_inc)

        self._create_encoder(use_cbn)
        self._create_rnn(rnn_layers)
        self._create_decoder(use_cbn)
        
        show_model(self)
        show_params(self)
        self._flatten_parameters()

    def forward(self, inputs):

        # 0. Extract STFT
        x = cspecs = self.stft(inputs)
        x = x.unsqueeze(1) # Add a dummy channel
        '''
        means = torch.mean(cspecs, [1,2,3], keepdim=True)
        std = torch.std(cspecs, [1,2,3], keepdim=True )
        normed_cspecs = (cspecs-means)/(std+1e-8)
        out = normed_cspecs
        '''

        encoder_out = []
        # 1. Apply encoder
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            encoder_out.append(x)

        
        # 2. Apply RNN
        x = torch.view_as_real(x)
        x = x.movedim(-1, 1) # move complex dim to front
        x = x.flatten(start_dim=1, end_dim=2)
        x = self._forward_rnn(x)

        n_channels = x.shape[1]
        x = torch.complex(x[:, :n_channels//2], x[:, n_channels//2:])
      
        # 3. Apply decoder
        for idx in range(len(self.decoder)):
            x = complex_cat([x, encoder_out[-1 - idx]], 1)
            x = self.decoder[idx](x)
            #x = x[..., 1:]
        
        # 4. Apply mask
        out_spec = apply_mask(x, cspecs, self.masking_mode)

        # 5. Invert STFT
        out_spec = torch.complex(out_spec[:, 0], out_spec[:, 1])
        out_wav = self.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp_(out_wav, -1, 1)

        return out_wav  # out_spec, out_wav

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def _create_encoder(self, use_cbn):
        self.encoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    # nn.ConstantPad2d([0, 0, 0, 0], 0),
                    
                    torch_complex.ComplexConv2d(
                        self.kernel_num[idx]//2,
                        self.kernel_num[idx + 1]//2,
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    # torch_complex.NaiveComplexBatchNorm2d(self.kernel_num[idx + 1]), # if not use_cbn else ComplexBatchNorm(self.kernel_num[idx + 1]),
                    torch_complex.ComplexReLU()
                )
            )

    def _create_decoder(self, use_cbn):
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        torch_complex.ComplexConvTranspose2d(
                            self.kernel_num[idx], #* 2,
                            self.kernel_num[idx - 1]//2,
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                        # torch_complex.NaiveComplexBatchNorm2d(self.kernel_num[idx - 1]), #if not use_cbn else ComplexBatchNorm(self.kernel_num[idx - 1]),
                        torch_complex.ComplexReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        torch_complex.ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                    )
                )
    
    def _create_rnn(self, rnn_layers):
        # TODO: Make bidirectional a hyperparameter
        bidirectional = False
        fac = 2 if bidirectional else 1
        
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if self.use_clstm:
            rnns = []
            for idx in range(rnn_layers):
                rnns.append(
                    NaiveComplexLSTM(
                        input_size=hidden_dim *
                        self.kernel_num[-1] if idx == 0 else self.rnn_units,
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim=hidden_dim *
                        self.kernel_num[-1] if idx == rnn_layers - 1 else None,
                    )
                )
                # What is *rnns - Runs the layers sequentially - to be used when append is used
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_units,
                num_layers=2,
                dropout=0.0,
                bidirectional=bidirectional,
                batch_first=False
            )
            self.tranform = nn.Linear(
                self.rnn_units * fac, hidden_dim * self.kernel_num[-1])

    def _flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def _forward_rnn(self, x):
        batch_size, channels, dims, lengths = x.size()
        x = x.permute(3, 0, 1, 2)
        if self.use_clstm:
            r_rnn_in = x[:, :, :channels // 2]
            i_rnn_in = x[:, :, channels // 2:]
            r_rnn_in = torch.reshape(
                r_rnn_in, [lengths, batch_size, channels // 2 * dims])
            i_rnn_in = torch.reshape(
                i_rnn_in, [lengths, batch_size, channels // 2 * dims])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(
                r_rnn_in, [lengths, batch_size, channels // 2, dims])
            i_rnn_in = torch.reshape(
                i_rnn_in, [lengths, batch_size, channels // 2, dims])
            x = torch.cat([r_rnn_in, i_rnn_in], 2)

        else:
            # to [L, B, C, D]
            x = torch.reshape(x, [lengths, batch_size, channels * dims])
            x, _ = self.enhance(x)
            x = self.tranform(x)
            x = torch.reshape(x, [lengths, batch_size, channels, dims])

        x = x.permute(1, 2, 3, 0)

        return x