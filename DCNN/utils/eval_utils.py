import torch

EPS= 0






def ild_db(s1, s2, eps=EPS, avg_mode=None):
    # s1 = _avg_signal(s1, avg_mode)
    # s2 = _avg_signal(s2, avg_mode)

    l1 = 20*torch.log10(s1 + eps)
    l2 = 20*torch.log10(s2 + eps)
    ild_value = (l1 - l2).abs()

    return ild_value



def ipd_rad(s1, s2, eps=EPS, avg_mode=None):
    # s1 = _avg_signal(s1, avg_mode)
    # s2 = _avg_signal(s2, avg_mode)

    ipd_value = ((s1 + eps)/(s2 + eps)).angle()

    return ipd_value

def _avg_signal(s, avg_mode):
    if avg_mode == "freq":
        return s.mean(dim=0)
    elif avg_mode == "time":
        return s.mean(dim=1)
    elif avg_mode == None:
        return s