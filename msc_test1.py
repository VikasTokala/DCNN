import torch
import torch.fft as fft

import torch.nn as nn
import torch.nn.functional as F

def _periodogram(X:torch.Tensor, fs, detrend, scaling):
    """
    Compute the periodogram of a signal.
    """
    if X.dim() > 2:
        X = torch.squeeze(X)
    elif X.dim() ==1:
        X = X.unsqueeze(0)
        
    
    if detrend:
        X -= X.mean(-1, keepdim=True)
    
    N = X.size(-1)
    assert N % 2 == 0
    
    df = fs / N
    dt = df
    f = torch.arange(0, N / 2 +1) * df #[0:df:f/2]
    
    dual_side = fft.fft(X)
    half_idx = int(N/2 + 1 )
    single_side = dual_side[:,0:half_idx]
    win = torch.abs(single_side)
    
    ps = win ** 2
    
    if scaling == 'density':
        scale = N * fs
    elif scaling == 'spectrum':
        scale = N ** 2
    elif scaling is None:
        scale = 1
    else:
        raise ValueError('Unknown sacling: %r' % scaling)
    
    Pxx = ps / scale
    
    Pxx[:,1:-1]*= 2
    
    return  f, Pxx.squeeze()

def peridogram(X: torch.Tensor, fs=16e3, detrend=False, scaling='density', no_grad=True):
    
    if no_grad:
        with torch.no_grad():
            return _periodogram(X,fs,detrend,scaling)
    
    else:
        return _periodogram(X,fs,detrend,scaling)
    
def _get_window(window, nwlen):
    if window == 'hann':
        window = torch.hann_window(nwlen, dtype=torch.float32,  periodic=False)
    
    elif window == 'hamming':
        window = torch.hamming_window(nwlen, dtype=torch.float32, periodic=False)
    
    elif window == 'blackman':
        window = torch.blackman_window(nwlen, dtype=torch.float32,  periodic=False)
        
    elif window == 'boxcar':
        window =  torch.ones(nwlen, dtype=torch.float32)
    
    else:
        raise ValueError('Unknown window type: %r' % window)
    
    return window

def _pwelch(X:torch.Tensor, fs, detrend, scaling, window, nwlen, nhop):
    
    if scaling == 'density':
        scale = (fs * (window * window).sum().item())
    
    elif scaling == 'spectrum':
        scale = (window.sum().item() ** 2)
    
    elif scaling is None:
        scale = 1
    
    else:
        raise ValueError('Unknown scaling: %r' % scaling)
    
    N = 1
    # breakpoint()
    
    # print(X.size(-1))
    T = X.size(-1)
    
    X = X.view(N,1,1,T)
    unfold = nn.Unfold((1,nwlen),stride=nhop)
    X_fold = unfold(X)
    
    if detrend:
        X_fold -= X_fold.mean(1,keepdim= True)
    
    window = window.view(1,-1,1) 
    X_windowed = X_fold * window
    win_cnt = X_windowed.size(-1)
    
    X_windowed = X_windowed.transpose(1,2).contiguous()
    X_windowed = X_windowed.view(N*win_cnt,nwlen)
    
    f, pxx = _periodogram( X_windowed, fs=fs, detrend=False, scaling='density')
    
    # pxx /= scale
    
    pxx = pxx.view(N,win_cnt,-1)
    pxx = torch.mean(pxx, dim=1)
    
    return f, pxx

def pwelch(X:torch.Tensor, fs=16e3, detrend=False, scaling=None, window='hann', nwlen=400, nhop=100,no_grad = True):
    
    nhop = nwlen // 4 if nhop is None else nhop
    
    window =  _get_window(window=window,nwlen=nwlen)
    if no_grad:
        with torch.no_grad():
            return _pwelch(X,fs,detrend,scaling,window,nwlen,nhop)
    else:
        return _pwelch(X,fs,detrend,scaling,window,nwlen,nhop)
    

import torch
import scipy.signal as signal
import matplotlib.pyplot as plt
import torchaudio

# Define your data (e.g., two signals x and y)
bin_sig,sample_rate = torchaudio.load('/Users/vtokala/Documents/Research/di_nn/Dataset/clean_testset_1f/p278_051_0.wav')
x = bin_sig[0,:] # Replace with your left ear signal
y = bin_sig[1,:]  
# Define parameters for the Welch periodogram


# Compute the Welch periodograms for x and y using scipy.signal
frequencies_x, welch_x = pwelch(x, fs=16e3, detrend=False, scaling='density', window='hann', nwlen=400, nhop=100,no_grad = True)
frequencies_y, welch_y = pwelch(y, fs=16e3, detrend=False, scaling='density', window='hann', nwlen=400, nhop=100,no_grad = True)

# Convert the numpy arrays to PyTorch tensors
welch_x = torch.tensor(welch_x)
welch_y = torch.tensor(welch_y)

# Calculate the cross-power spectral density (CSD)
csd = torch.conj(welch_x) * welch_y

# Calculate the power spectral density (PSD)
psd_x = torch.abs(welch_x)
psd_y = torch.abs(welch_y)

# breakpoint()
# Compute the magnitude squared coherence (MSC)
msc = torch.abs(csd)**2 / (psd_x * psd_y)

# Plot the magnitude squared coherence
plt.figure(figsize=(8, 4))
plt.semilogy(frequencies_x, welch_x.squeeze())
plt.xlabel('Frequency (Hz)')
plt.ylabel('Welch X')
plt.title('Welch X')
plt.grid()
plt.show()
plt.figure(figsize=(8, 4))
plt.semilogy(frequencies_x, welch_y.squeeze())
plt.xlabel('Frequency (Hz)')
plt.ylabel('Welch Y')
plt.title('Welch Y')
plt.grid()
plt.show()
plt.figure(figsize=(8, 4))
plt.semilogy(frequencies_x, msc.squeeze())
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude Squared Coherence')
plt.title('Magnitude Squared Coherence')
plt.grid()
plt.show()