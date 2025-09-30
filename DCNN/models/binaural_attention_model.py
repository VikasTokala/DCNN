import torch
from .model import DCNN
from DCNN.utils.apply_mask import apply_mask
import matplotlib.pyplot as plt
import librosa.display
import matplotlib
import numpy as np
from DCNN.feature_extractors import Stft,IStft
from DCNN.feature_maps import plot_averaged_magnitude

def numfmt(x, pos): # your custom formatter function: divide by 100.0
    s = '{}'.format(x / 1000)
    return s

win_len = 400
win_inc = 100
fft_len = 512
fbins = int(fft_len/2 + 1)
fs = 16000

stft = Stft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
istft = IStft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)


class BinauralAttentionDCNN(DCNN):

    def forward(self, inputs):
        # batch_size, binaural_channels, time_bins = inputs.shape
        cspecs_l = self.stft(inputs[:, 0])
        cspecs_r = self.stft(inputs[:, 1])
        cspecs = torch.stack((cspecs_l, cspecs_r), dim=1)

        # breakpoint()

        # encoder_out_l = self.encoder(attention_enc[:, 0, :, :].unsqueeze(1))
        # encoder_out_r = self.encoder(attention_enc[:, 1, :, :].unsqueeze(1))

        encoder_out_l = self.encoder(cspecs_l.unsqueeze(1))
        encoder_out_r = self.encoder(cspecs_r.unsqueeze(1))
        

        
        
      
        
        
    
        

        # encoder_out = torch.cat((encoder_out_l[-1], encoder_out_r[-1]), dim=1)
        # attention_enc = self.attention_enc(encoder_out)
        # breakpoint()
        # attention_enc = encoder_out_l[-1]*encoder_out_r[-1].conj()
        # attention_enc = self.attention(cspecs)
        # _, attn_len, _, _ = attention_enc.shape
        # encoder_attn_l = attention_enc[:, :attn_len//2, :, :]
        # encoder_attn_r = attention_enc[:, attn_len//2:, :, :]
        # breakpoint()
        # 2. Apply RNN
        # x_l_rnn = self.rnn(encoder_out_l[-1])
        # x_r_rnn = self.rnn(encoder_out_r[-1])
        # breakpoint()
        attention_in = torch.cat((encoder_out_l[-1],encoder_out_r[-1]), dim=1)
        
        #
        # breakpoint()
        # x_l_mattn = self.mattn(encoder_out_l[-1])
        # x_r_mattn = self.mattn(encoder_out_r[-1])
        # breakpoint()

        x_attn = self.mattn(attention_in)
        # rnn_out = torch.cat((x_l_rnn, x_r_rnn), dim=1)
        # breakpoint()
        # attention_dec = self.attention_enc(rnn_out)

        # _, dec_attn_len, _, _ = attention_dec.shape
        # decoder_attn_l = attention_dec[:, :dec_attn_len//2, :, :]
        # decoder_attn_r = attention_dec[:, dec_attn_len//2:, :, :]
        x_l_mattn = x_attn[:,:128,:,:]
        x_r_mattn = x_attn[:,128:,:,:]
        # x_l_mattn = x_attn[:,:64,:,:]
        # x_r_mattn = x_attn[:,64:,:,:]
        # 3. Apply decoder
        # x_l = self.decoder(x_l_rnn, encoder_out_l)
        # x_r = self.decoder(x_r_rnn, encoder_out_r)
        x_l = self.decoder(x_l_mattn, encoder_out_l)
        x_r = self.decoder(x_r_mattn, encoder_out_r)
        # breakpoint()
        
        # plot_averaged_magnitude(x_l[0][0].abs().detach().numpy())
        # 4. Apply mask
        out_spec_l = apply_mask(x_l[:, 0], cspecs_l, self.masking_mode)
        out_spec_r = apply_mask(x_r[:, 0], cspecs_r, self.masking_mode)
        # breakpoint()
        
        # plot_averaged_magnitude((encoder_out_l[0][0][0].abs().detach().numpy()),title='Encoder Output - Layer 1',clabel='Magnitude',
        #                     fig_name='Encoder1.pdf',ylab='Frequency dimension', xlab='Time dimension')
        # plot_averaged_magnitude((encoder_out_l[1][0][0].abs().detach().numpy()),title='Encoder Output - Layer 2',clabel='Magnitude',
        #                     fig_name='Encoder2.pdf',ylab='Frequency dimension', xlab='Time dimension')
        # plot_averaged_magnitude((encoder_out_l[2][0][0].abs().detach().numpy()),title='Encoder Output - Layer 3',clabel='Magnitude',
        #                     fig_name='Encoder3.pdf',ylab='Frequency dimension', xlab='Time dimension')
        # plot_averaged_magnitude((encoder_out_l[3][0][0].abs().detach().numpy()),title='Encoder Output - Layer 4',clabel='Magnitude',
        #                     fig_name='Encoder4.pdf',ylab='Frequency dimension', xlab='Time dimension')
        # plot_averaged_magnitude((encoder_out_l[4][0][0].abs().detach().numpy()),title='Encoder Output - Layer 5',clabel='Magnitude',
        #                     fig_name='Encoder5.pdf',ylab='Frequency dimension', xlab='Time dimension')
        # plot_averaged_magnitude((encoder_out_l[5][0][0].abs().detach().numpy()),title='Encoder Output - Layer 6',clabel='Magnitude',
        #                     fig_name='Encoder6.pdf',ylab='Frequency dimension', xlab='Time dimension')
        
        
        # plot_averaged_magnitude(x_l[0][0].abs().detach().numpy(),title='Estimated CRM - Decoder output',clabel='Magnitude',
        #                     fig_name='Decoder.pdf',ylab='Frequency dimension', xlab='Time dimension')
        
        # plot_averaged_magnitude(librosa.amplitude_to_db(out_spec_l[0].abs().detach().numpy()),title='Enhanced Signal',clabel='Magnitude[dB]',
        #                     fig_name='EnhanSig.pdf',ylab='Frequency dimension', xlab='Time dimension')
        # plot_averaged_magnitude(librosa.amplitude_to_db(cspecs_l[0].abs().detach().numpy()),title='Noisy Signal',clabel='Magnitude[dB]',
        #                     fig_name='NoisySig.pdf',ylab='Frequency dimension', xlab='Time dimension')
        
        # plot_averaged_magnitude(librosa.amplitude_to_db(encoder_out_l[0][0][0].abs().detach().numpy()))
        # plot_averaged_magnitude(x_l[0][0].abs().detach().numpy())
        # plot_averaged_magnitude(librosa.amplitude_to_db(out_spec_l[0].abs().detach().numpy()))
        
        # breakpoint()
        # 5. Invert STFT
        out_wav_l = self.istft(out_spec_l)
        # breakpoint()
        # out_wav_l = torch.squeeze(out_wav_l, 1)
        # out_wav_l = torch.clamp_(out_wav_l, -1, 1)
        
        out_wav_r = self.istft(out_spec_r)
        # out_wav_r = torch.squeeze(out_wav_r, 1)
        # out_wav_r = torch.clamp_(out_wav_r, -1, 1)

        # breakpoint()
        
        out_wav = torch.stack([out_wav_l, out_wav_r], dim=1)
       
        return out_wav
