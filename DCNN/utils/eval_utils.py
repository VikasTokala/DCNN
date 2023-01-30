import torch
from DCNN.loss import BinaryMask
EPS= 1e-6






def ild_db(s1, s2, eps=EPS, avg_mode=None):
    # s1 = _avg_signal(s1, avg_mode)
    # s2 = _avg_signal(s2, avg_mode)

    l1 = 20*torch.log10(s1 + eps)
    l2 = 20*torch.log10(s2 + eps)
    ild_value = (l1 - l2)

    return ild_value



def ipd_rad(s1, s2, eps=EPS, avg_mode=None):
    # s1 = _avg_signal(s1, avg_mode)
    # s2 = _avg_signal(s2, avg_mode)

    ipd_value = ((s1 + eps)/(s2 + eps)).angle()
    # ipd_value = s1.angle() - s2.angle()
    return ipd_value

def _avg_signal(s, avg_mode):
    if avg_mode == "freq":
        return s.mean(dim=0)
    elif avg_mode == "time":
        return s.mean(dim=1)
    elif avg_mode == None:
        return s


def speechMask(stft_l,stft_r):
    # breakpoint()
    _,time_bins = stft_l.shape
    thresh_l,_ = (((stft_l.abs())**2)).max(dim=1) 
    thresh_l_db = 10*torch.log10(thresh_l) - 10
    thresh_l_db=thresh_l_db.unsqueeze(1).repeat(1,1,time_bins)
    
    thresh_r,_ = (((stft_r.abs())**2)).max(dim=1) 
    thresh_r_db = 10*torch.log10(thresh_r) - 10
    thresh_r_db=thresh_r_db.unsqueeze(1).repeat(1,1,time_bins)
    
    
    bin_mask_l = BinaryMask(threshold=thresh_l_db)
    bin_mask_r = BinaryMask(threshold=thresh_r_db)
    
    mask_l = bin_mask_l(20*torch.log10((stft_l.abs())))
    mask_r = bin_mask_r(20*torch.log10((stft_r.abs())))
    mask = torch.bitwise_and(mask_l.int(), mask_r.int())
    
    return mask