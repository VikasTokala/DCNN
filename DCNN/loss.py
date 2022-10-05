import torch
import torch.functional as F

from torch.nn import Module
from DCNN.utils.conv_stft import ConvSTFT
from torch_stoi import NegSTOILoss
import matplotlib.pyplot as plt

EPS = 1e-6


class BinauralLoss(Module):
    def __init__(self, loss_mode="RTF", win_len=400,
                 win_inc=100, fft_len=512,sr=16000,rtf_weight=0.3,snr_weight=0.7,
                 ild_weight=0.1, ipd_weight=1, avg_mode="freq"):

        super().__init__()
        self.loss_mode = loss_mode
        self.stft = STFT(win_len, win_inc, fft_len)
        self.stoiLoss = NegSTOILoss(sample_rate=sr)
        self.istft = ISTFT(win_len, win_inc, fft_len)
        self.rtf_weight = rtf_weight
        self.snr_weight = snr_weight
        self.ild_weight = ild_weight
        self.ipd_weight = ipd_weight
        self.avg_mode = avg_mode


    def forward(self, model_output, targets):
        target_stft_l = self.stft(targets[:, 0])
        target_stft_r = self.stft(targets[:, 1])

        output_stft_l = self.stft(model_output[:, 0])
        output_stft_r = self.stft(model_output[:, 1])

        snr_l = si_snr(model_output[:, 0], targets[:, 0])
        snr_r = si_snr(model_output[:, 1], targets[:, 1])
        snr_loss = - (snr_l + snr_r)/2

        if self.loss_mode == "RTF":

            target_rtf_td_full = self.istft(target_stft_l/(target_stft_r + EPS))
            output_rtf_td_full = self.istft(output_stft_l/(output_stft_r + EPS))

            target_rtf_td = target_rtf_td_full[:,0:2047]
            output_rtf_td = output_rtf_td_full[:,0:2047]
            
            epsilon = target_rtf_td - ((target_rtf_td@(torch.transpose(output_rtf_td,0,1)))/(output_rtf_td@torch.transpose(output_rtf_td,0,1)))@output_rtf_td
            npm_error = torch.norm((epsilon/torch.max(epsilon)))/torch.norm((target_rtf_td)/torch.max(target_rtf_td))

            stoi_l = self.stoiLoss(model_output[:, 0], targets[:, 0])
            stoi_r = self.stoiLoss(model_output[:, 1], targets[:, 1])
            
            stoi_loss = (stoi_l+stoi_r)/2

            return self.rtf_weight *npm_error + self.snr_weight * snr_loss + 0*stoi_loss.mean()

        elif self.loss_mode == "ILD-SNR":
            ild_loss = ild_loss_db(target_stft_l, target_stft_r,
                                   output_stft_l, output_stft_r, avg_mode=self.avg_mode)
            return snr_loss + self.ild_weight*ild_loss

        elif self.loss_mode == "ILD-IPD-SNR":
            ild_loss = ild_loss_db(target_stft_l, target_stft_r,
                                   output_stft_l, output_stft_r, avg_mode=self.avg_mode)
            ipd_loss = ipd_loss_rads(target_stft_l, target_stft_r,
                                     output_stft_l, output_stft_r, avg_mode=self.avg_mode)
            
            return self.snr_weight*snr_loss + self.ild_weight*ild_loss + self.ipd_weight*ipd_loss
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
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=EPS, reduce_mean=True):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    snr_norm = snr #/max(snr)
    if reduce_mean:
        snr_norm = torch.mean(snr_norm) 
    
    return snr_norm


def ild_db(s1, s2, eps=EPS, avg_mode=None):
    s1 = _avg_signal(s1, avg_mode)
    s2 = _avg_signal(s1, avg_mode)

    l1 = 20*torch.log10(s1 + eps)
    l2 = 20*torch.log10(s2 + eps)

    ild_value = (l1 - l2).abs()


    return ild_value


def ild_loss_db(target_stft_l, target_stft_r,
                output_stft_l, output_stft_r, avg_mode=None):
    target_ild = ild_db(target_stft_l, target_stft_r, avg_mode=avg_mode)
    output_ild = ild_db(output_stft_l, output_stft_r, avg_mode=avg_mode)

    ild_loss = ((target_ild - output_ild).abs()).mean()
    return ild_loss


def ipd_rad(s1, s2, eps=EPS, avg_mode=None):
    s1 = _avg_signal(s1, avg_mode)
    s2 = _avg_signal(s1, avg_mode)

    ipd_value = ((s1 + eps)/(s2 + eps)).angle()

    return ipd_value


def ipd_loss_rads(target_stft_l, target_stft_r,
                  output_stft_l, output_stft_r, avg_mode=None):
    target_ild = ipd_rad(target_stft_l, target_stft_r, avg_mode=avg_mode)
    output_ild = ipd_rad(output_stft_l, output_stft_r, avg_mode=avg_mode)

    ild_loss = ((target_ild - output_ild).abs()).mean()
    return ild_loss


def _avg_signal(s, avg_mode):
    if avg_mode == "freq":
        return s.mean(dim=1)
    elif avg_mode == "time":
        return s.mean(dim=2)
    elif avg_mode == None:
        return s


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

class ISTFT(Module):
    def __init__(self, win_len=400, win_inc=100,
                 fft_len=512):
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len

        super().__init__()

    def forward(self, x):
        istft = torch.istft(x, self.fft_len, hop_length=self.win_inc,
                          win_length=self.win_len, return_complex=False)
        return istft