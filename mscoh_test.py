import torch
import torch.fft as fft
import torchaudio
import matplotlib.pyplot as plt

def mscohere(x, y, nperseg=None, noverlap=None, nfft=None):
    """
    Compute the magnitude squared coherence between two signals using PyTorch.
    
    Args:
        x (torch.Tensor): Input signal 1.
        y (torch.Tensor): Input signal 2.
        nperseg (int): Length of each segment for the FFT. Default is None.
        noverlap (int): Number of overlapping samples between segments. Default is None.
        nfft (int): Length of the FFT used. Default is None.
    
    Returns:
        torch.Tensor: Magnitude squared coherence between the two signals.
    """
    
    if nperseg is None:
        nperseg = min(len(x), 256)  # Default value for nperseg
        
    if noverlap is None:
        noverlap = nperseg // 2  # Default value for noverlap
    
    if nfft is None:
        nfft = 2 * nperseg  # Default value for nfft

    # Compute the spectrograms of x and y
    _, _, Sxx = spectrogram(x, y=None, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    _, _, Syy = spectrogram(y, y=None, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    
    # breakpoint()
    
    # Compute the cross-spectral density
    _, _, Sxy = spectrogram(x, y, nperseg, noverlap, nfft)
    
    # Compute the magnitude squared coherence
    coh = torch.abs(Sxy)**2 / (Sxx * Syy)
    
    return coh

def spectrogram(x, y=None, nperseg=None, noverlap=None, nfft=None):
    """
    Compute the spectrogram of a signal using PyTorch.
    
    Args:
        x (torch.Tensor): Input signal.
        y (torch.Tensor): If given, compute the cross-spectral density between x and y.
        nperseg (int): Length of each segment for the FFT. Default is None.
        noverlap (int): Number of overlapping samples between segments. Default is None.
        nfft (int): Length of the FFT used. Default is None.
    
    Returns:
        torch.Tensor: Frequencies, times, and spectrogram matrix.
    """
    
    if y is not None:
        # Compute cross-spectral density
        Pxy = cross_spectral_density(x, y, nperseg, noverlap, nfft)
        freqs, times = compute_freq_time(x, nperseg, noverlap)
        return freqs, times, Pxy
    
    # Compute power spectral density
    Pxx = power_spectral_density(x, nperseg, noverlap, nfft)
    freqs, times = compute_freq_time(x, nperseg, noverlap)
    return freqs, times, Pxx

def power_spectral_density(x, nperseg, noverlap, nfft):
    # Compute the power spectral density
    freqs, Pxx = compute_spectrum(x, y=None, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    return Pxx

def cross_spectral_density(x, y, nperseg, noverlap, nfft):
    # Compute the cross-spectral density between x and y
    freqs, Pxy = compute_spectrum(x, y, nperseg, noverlap, nfft)
    return Pxy

def compute_spectrum(x, y=None, nperseg=None, noverlap=None, nfft=None):
    # Compute the spectrum of x (or the cross-spectrum between x and y)
    if y is None:
        x_fft = fft.rfft(x, n=nfft)
        Pxx = torch.abs(x_fft)**2
        freqs = fft.rfftfreq(nfft)
    else:
        x_fft = fft.rfft(x, n=nfft)
        # breakpoint()
        y_fft = fft.rfft(y, n=nfft)
        Pxy = x_fft * torch.conj(y_fft)
        freqs = fft.rfftfreq(nfft)
    
    return freqs, Pxx if y is None else Pxy

def compute_freq_time(x, nperseg, noverlap):
    # Compute frequencies and times for the spectrogram
    n = len(x)
    sr=16e3
    nfft=512
    hop_size = nperseg - noverlap
    num_segments = (n - noverlap) // hop_size
    times = torch.arange(num_segments) * hop_size / sr
    freqs = torch.fft.rfftfreq(nfft, 1.0 / sr)
    return freqs, times



bin_sig,sample_rate = torchaudio.load('/Users/vtokala/Documents/Research/di_nn/Dataset/clean_trainset_1f/p226_002_0.wav')
# Example usage

print(bin_sig.shape)
x = bin_sig[0,:] # Replace with your left ear signal
y = bin_sig[1,:]
coh = mscohere(x, y, nperseg=400,noverlap=100,nfft=512)
coh_normalized = coh / coh.max()

# Frequency axis (assuming you have computed it in your mscohere function)
# You can replace this with the actual frequency values you used.
nfft=512
sr=16e3
print(coh.shape)
freqs = torch.fft.rfftfreq(nfft, 1.0 / sr)

# Plot the coherence vs. frequency
plt.figure(figsize=(8, 6))
plt.plot(freqs, coh_normalized)
plt.title('Coherence vs. Frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence')
plt.grid(True)
plt.show()