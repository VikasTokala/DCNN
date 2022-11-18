import torch

from DCNN.utils.apply_mask import apply_mask

from .model import DCNN
from DCNN.utils.freq_transform import FAL


class BinauralAttentionDCNN(DCNN):

    def forward(self, inputs):
        # batch_size, binaural_channels, time_bins = inputs.shape
        cspecs_l = self.stft(inputs[:, 0])
        cspecs_r = self.stft(inputs[:, 1])
        cspecs = torch.stack((cspecs_l, cspecs_r), dim=1)

        

        # breakpoint()

        # encoder_out_l = self.encoder(attention_out[:, 0, :, :].unsqueeze(1))
        # encoder_out_r = self.encoder(attention_out[:, 1, :, :].unsqueeze(1))

        encoder_out_l = self.encoder(cspecs_l.unsqueeze(1))
        encoder_out_r = self.encoder(cspecs_r.unsqueeze(1))
        # breakpoint()

        encoder_out = torch.cat((encoder_out_l[-1],encoder_out_r[-1]),dim=1)
        attention_out = self.attention(encoder_out)
        # breakpoint()
        # attention_out = encoder_out_l[-1]*encoder_out_r[-1].conj()
        # attention_out = self.attention(cspecs)
        _,attn_len,_,_ = attention_out.shape
        encoder_attn_l = attention_out[:,:attn_len//2,:,:]
        encoder_attn_r = attention_out[:,attn_len//2:,:,:]
        # breakpoint()
        # 2. Apply RNN
        x_l = self.rnn(encoder_attn_l)
        x_r = self.rnn(encoder_attn_r)

        # 3. Apply decoder
        x_l = self.decoder(x_l, encoder_out_l)
        x_r = self.decoder(x_r, encoder_out_r)

        # 4. Apply mask
        out_spec_l = apply_mask(x_l[:, 0], cspecs_l, self.masking_mode)
        out_spec_r = apply_mask(x_r[:, 0], cspecs_r, self.masking_mode)

        # 5. Invert STFT
        out_wav_l = self.istft(out_spec_l)
        out_wav_l = torch.squeeze(out_wav_l, 1)
        out_wav_l = torch.clamp_(out_wav_l, -1, 1)

        out_wav_r = self.istft(out_spec_r)
        out_wav_r = torch.squeeze(out_wav_r, 1)
        out_wav_r = torch.clamp_(out_wav_r, -1, 1)

        out_wav = torch.stack([out_wav_l, out_wav_r], dim=1)
        return out_wav
