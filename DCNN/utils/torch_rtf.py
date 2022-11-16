import scipy.signal as sig 
import torch
import scipy.linalg as linalg



def rec_psd_est(x_stft, alpha=0.9): 

    if x_stft.dim() == 4:
        N_batch, N_freq, Nch, N_segs = x_stft.size()
    elif x_stft.dim() == 3: 
        N_freq, Nch, N_segs = x_stft.size()
        N_batch = 1
        x_stft = x_stft[None,:,:,:]
    else:
        raise ValueError('input stft has incorrect number of dimensions in tensor')

    psd = torch.full((N_batch, Nch, Nch, N_freq, N_segs), 0+0j)
    

    for b in range(N_batch):
        for f in range(N_freq):
            sig_stft_bin = x_stft[b,f,:,0]
            psd[b,:,:,f,0] = torch.outer(sig_stft_bin, torch.conj(sig_stft_bin))
            for n in range(1,N_segs):
                sig_stft_bin = x_stft[b,f,:,n]
                psd[b,:,:,f,n] = alpha*psd[b,:,:,f,n-1] + (1-alpha) * torch.outer(sig_stft_bin, torch.conj(sig_stft_bin))
    return psd



def gevd_rtf_est(ns_psd, n_psd, ref_ch):

    evals, evecs = linalg.eigh(ns_psd, n_psd)
    ref_ind = ref_ch - 1 
    rtf = (n_psd@evecs[:,-1]) / (n_psd@evecs[:,-1])[ref_ind]  
    return rtf




def test_gevd_rtf(y_stft, v_stft, ref_ch=1, alpha=0.9, info=False):

    if y_stft.dim() == 4:
        N_batch, N_freq, Nch, N_segs = y_stft.size()
    elif y_stft.dim() == 3: 
        N_freq, Nch, N_segs = y_stft.size()
        N_batch = 1
        y_stft = y_stft[None,:,:]
        x_stft = x_stft[None,:,:]
    else:
        raise ValueError('input stft has incorrect number of dimensions in tensor')

    rtf = torch.full((N_batch, N_freq, Nch, N_segs), 0+0j)

    ref_ind = ref_ch - 1
    encoded_channels = list(range(Nch))
    encoded_channels.remove(ref_ind)

    Pyy = rec_psd_est(y_stft)
    Pvv = rec_psd_est(v_stft)

    c=0
    d=0

    for b in range(N_batch):
        for f in range(1,N_freq):
            for n in range(N_segs):
                try:
                    rtf[b,f,:,n] = gevd_rtf_est(Pyy[b,:,:,f,n], Pvv[b,:,:,f,n], ref_ch)
                except:
                    try:
                        c+=1
                        Pv_rec_reg = Pvv[b,:,:,f,n] + 0.01*torch.min(Pvv[b,:,:,f,n])*torch.eye(Nch,Nch) 
                        rtf[b,f,:,n] = gevd_rtf_est(Pyy[b,:,:,f,n], Pv_rec_reg, ref_ch)
                    except:
                        d+=1 
                        rtf[b,f,:,n] = rtf[b,f,:,n-1]
        if c > 0:
            print(str(c) + ' rtf tf-bins required regularisation\n' + str(d) + ' were ignored after regularisation')

    return rtf


if __name__ == '__main__':

    y_stft = torch.rand((1, 4, 2, 6)) + 1j*torch.rand((1, 4, 2, 6))
    v_stft = torch.rand((1, 4, 2, 6)) + 1j*torch.rand((1, 4, 2, 6))

    rtf = test_gevd_rtf(y_stft, v_stft)