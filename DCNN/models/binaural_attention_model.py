import torch

from DCNN.utils.apply_mask import apply_mask

from .model import DCNN
from DCNN.utils.freq_transform import FAL_enc, FAL_dec


class BinauralAttentionDCNN(DCNN):

    def forward(self, inputs):
     
        cspecs_l = self.stft(inputs[:, 0])
        cspecs_r = self.stft(inputs[:, 1])
        cspecs = torch.stack((cspecs_l, cspecs_r), dim=1)

      

        encoder_out_l = self.encoder(cspecs_l.unsqueeze(1))
        encoder_out_r = self.encoder(cspecs_r.unsqueeze(1))
       
        # 2. Apply Multihead attention
        
        x_l_mattn = self.mattn(encoder_out_l[-1])
        x_r_mattn = self.mattn(encoder_out_r[-1])
        
        # 3. Apply decoder
        
        x_l = self.decoder(x_l_mattn, encoder_out_l)
        x_r = self.decoder(x_r_mattn, encoder_out_r)

        # 4. Apply mask
        out_spec_l = apply_mask(x_l[:, 0], cspecs_l, self.masking_mode)
        out_spec_r = apply_mask(x_r[:, 0], cspecs_r, self.masking_mode)

        # 5. Invert STFT
        out_wav_l = self.istft(out_spec_l)
        
        out_wav_r = self.istft(out_spec_r)
      

        
        out_wav = torch.stack([out_wav_l, out_wav_r], dim=1)
        
        return out_wav
