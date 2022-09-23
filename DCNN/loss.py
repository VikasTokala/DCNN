import torch
import torch.functional as F

from torch.nn import Module
from DCNN.utils.complexnn import complex_div, complex_pow
from DCNN.utils.conv_stft import ConvSTFT
from torch_stoi import NegSTOILoss

EPS = 1e-7

class BinauralLoss(Module):
    def __init__(self, loss_mode="RTF", win_len=400,
                 win_inc=100, fft_len=512):

        super().__init__()
        self.loss_mode = loss_mode
        self.stft = STFT(win_len, win_inc, fft_len)

    def forward(self, model_output, targets):
        if self.loss_mode == "RTF":
            target_stft_l = self.stft(targets[:, 0])
            target_stft_r = self.stft(targets[:, 1])

            output_stft_l = self.stft(model_output[:, 0])
            output_stft_r = self.stft(model_output[:, 1])
            
            error = (output_stft_l/(output_stft_r + EPS) - target_stft_l/(target_stft_r + EPS))**2

            return error.abs().mean()

        else:
            raise NotImplementedError("Only loss available for binaural enhancement is 'RTF'")

class Loss(Module):
    def __init__(self, loss_mode="SI-SNR", win_len=400,
                 win_inc=100,
                 fft_len=512,
                 win_type="hann",
                 fix=True, sr=16000,
                 STOI_weight=1,
                 SNR_weight=0.1):
        super().__init__()
        self.loss_mode = loss_mode
        self.stft = ConvSTFT(win_len, win_inc, fft_len,
                             win_type, "complex", fix=fix)
        self.stoiLoss = NegSTOILoss(sample_rate=sr)
        self.STOI_weight = STOI_weight
        self.SNR_weight = SNR_weight

    def forward(self, model_output, targets):
        if self.loss_mode == "MSE":
            b, d, t = model_output.shape
            targets[:, 0, :] = 0
            targets[:, d // 2, :] = 0
            return F.mse_loss(model_output, targets, reduction="mean") * d

        elif self.loss_mode == "SI-SNR":
            # return -torch.mean(si_snr(model_output, targets))
            return -(si_snr(model_output, targets))

        elif self.loss_mode == "MAE":
            gth_spec, gth_phase = self.stft(targets)
            b, d, t = model_output.shape
            return torch.mean(torch.abs(model_output - gth_spec)) * d

        elif self.loss_mode == "STOI-SNR":
            loss_batch = self.stoiLoss(model_output, targets)
            return -(self.SNR_weight*si_snr(model_output, targets)) + self.STOI_weight*loss_batch.mean()


def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    # s1 = remove_dc(s1)
    # s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)


#stoi(x, y, fs_sig, extended=False)

class STFT(Module):
    def __init__(self, win_len=400, win_inc=100,
                 fft_len=512):
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len

        super().__init__()

    def forward(self, x):
        stft = torch.stft(x, self.fft_len, hop_length=self.win_inc,
                          win_length=self.win_len, return_complex=True)
        return stft
