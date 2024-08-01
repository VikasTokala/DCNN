import torch

from DCNN.utils.apply_mask import apply_mask

from .model import DCNN
from DCNN.feature_extractors import Stft,IStft
from DCNN.feature_maps import plot_averaged_magnitude
import librosa


class BinauralDCRNN(DCNN):

    def forward(self, inputs):
        
        cspecs_l = self.stft(inputs[:, 0])
        cspecs_r = self.stft(inputs[:, 1])
        cspecs = torch.stack((cspecs_l, cspecs_r), dim=1)

    

        encoder_out_l = self.encoder(cspecs_l.unsqueeze(1))
        encoder_out_r = self.encoder(cspecs_r.unsqueeze(1))

        rnn_in = torch.cat((encoder_out_l[-1],encoder_out_r[-1]), dim=1)
        
        # breakpoint()
        # 2. Apply RNN
        # x_l_rnn = self.rnn(encoder_out_l[-1])
        # x_r_rnn = self.rnn(encoder_out_r[-1])
       
        x_rnn = self.rnn(rnn_in)
        
        x_l_rnn = x_rnn[:,:128,:,:]
        x_r_rnn = x_rnn[:,128:,:,:]

        # 3. Apply decoder
        x_l = self.decoder(x_l_rnn, encoder_out_l)
        # x_r = self.decoder(x_r_rnn, encoder_out_r)
    
    

        # 4. Apply mask
        out_spec_l = apply_mask(x_l[:, 0], cspecs_l, self.masking_mode)
        out_spec_r = apply_mask(x_l[:, 0], cspecs_r, self.masking_mode)

        # 5. Invert STFT
        out_wav_l = self.istft(out_spec_l)
       
        out_wav_r = self.istft(out_spec_r)

        plot_averaged_magnitude((encoder_out_l[0][0][0].abs().detach().numpy()),title='Encoder Output - Layer 1',clabel='Magnitude',
                            fig_name='Encoder1.pdf',ylab='Frequency dimension', xlab='Time dimension')
        plot_averaged_magnitude((encoder_out_l[1][0][0].abs().detach().numpy()),title='Encoder Output - Layer 2',clabel='Magnitude',
                            fig_name='Encoder2.pdf',ylab='Frequency dimension', xlab='Time dimension')
        plot_averaged_magnitude((encoder_out_l[2][0][0].abs().detach().numpy()),title='Encoder Output - Layer 3',clabel='Magnitude',
                            fig_name='Encoder3.pdf',ylab='Frequency dimension', xlab='Time dimension')
        plot_averaged_magnitude((encoder_out_l[3][0][0].abs().detach().numpy()),title='Encoder Output - Layer 4',clabel='Magnitude',
                            fig_name='Encoder4.pdf',ylab='Frequency dimension', xlab='Time dimension')
        plot_averaged_magnitude((encoder_out_l[4][0][0].abs().detach().numpy()),title='Encoder Output - Layer 5',clabel='Magnitude',
                            fig_name='Encoder5.pdf',ylab='Frequency dimension', xlab='Time dimension')
        plot_averaged_magnitude((encoder_out_l[5][0][0].abs().detach().numpy()),title='Encoder Output - Layer 6',clabel='Magnitude',
                            fig_name='Encoder6.pdf',ylab='Frequency dimension', xlab='Time dimension')
        
        
        plot_averaged_magnitude(x_l[0][0].abs().detach().numpy(),title='Estimated CRM - Decoder output',clabel='Magnitude',
                            fig_name='Decoder.pdf',ylab='Frequency dimension', xlab='Time dimension')
        
        plot_averaged_magnitude(librosa.amplitude_to_db(out_spec_l[0].abs().detach().numpy()),title='Enhanced Signal',clabel='Magnitude[dB]',
                            fig_name='EnhanSig.pdf',ylab='Frequency dimension', xlab='Time dimension')
        plot_averaged_magnitude(librosa.amplitude_to_db(cspecs_l[0].abs().detach().numpy()),title='Noisy Signal',clabel='Magnitude[dB]',
                            fig_name='NoisySig.pdf',ylab='Frequency dimension', xlab='Time dimension')
        
        # plot_averaged_magnitude(librosa.amplitude_to_db(encoder_out_l[0][0][0].abs().detach().numpy()))
        # plot_averaged_magnitude(x_l[0][0].abs().detach().numpy())
        # plot_averaged_magnitude(librosa.amplitude_to_db(out_spec_l[0].abs().detach().numpy()))
       
        breakpoint()

        
        out_wav = torch.stack([out_wav_l, out_wav_r], dim=1)
        
        return out_wav
