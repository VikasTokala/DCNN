import math
import torch
import torch.fft as fft
import matplotlib.pyplot as plt
import random
import torchaudio
import warnings
warnings.filterwarnings('error')

def welch_cross_spectral_density(signal1, signal2, segment_length, overlap, window_function=None, nfft=None, sampling_frequency=16000):
    assert len(signal1) == len(signal2), "Input signals must have the same length."

    signal_length = len(signal1)
    step_size = segment_length - overlap
    num_segments = (signal_length - overlap) // step_size

    # Calculate the sampling period
    sampling_period = 1 / sampling_frequency

    # Use segment_length as nfft if not provided
    if nfft is None:
        nfft = segment_length

    # Define a window function if not provided
    if window_function is None:
        window_function = torch.ones(segment_length)

    # Initialize the result CSD tensor
    csd = torch.zeros(nfft // 2 + 1)
    csd = torch.complex(csd,csd)

    # Iterate over segments and compute their cross-spectral densities
    for i in range(num_segments):
        start = i * step_size
        end = start + segment_length

        # Apply the window function to the segments
        windowed_segment1 = signal1[start:end] * window_function
        windowed_segment2 = signal2[start:end] * window_function

        # Compute the FFT of the windowed segments
        fft_result1 = fft.rfft(windowed_segment1, n=nfft)
        fft_result2 = fft.rfft(windowed_segment2, n=nfft)
        # breakpoint()
        # Compute the cross-spectral density for this segment
        csd_segment = (fft_result1 * fft_result2.conj()) / nfft

        # Add the segment's cross-spectral density to the overall result
        csd += csd_segment[:nfft // 2+1]

    # Average the cross-spectral densities over all segments
    csd /= num_segments

    return csd

def mean_squared_coherence(signal1, signal2, segment_length, overlap, window_function=None, nfft=None, sampling_frequency=16000):
    # Compute the cross-spectral density using Welch's method
    csd = welch_cross_spectral_density(signal1, signal2, segment_length, overlap, window_function, nfft, sampling_frequency)

    # Compute the power spectral densities of the individual signals
    psd1 = welch_cross_spectral_density(signal1, signal1, segment_length, overlap, window_function, nfft, sampling_frequency)
    psd2 = welch_cross_spectral_density(signal2, signal2, segment_length, overlap, window_function, nfft, sampling_frequency)

    # Compute the Mean Squared Coherence
    msc = (torch.abs(csd) ** 2) / ((psd1 * psd2)+1e-8)

    return msc

# Example usage
if __name__ == "__main__":
    # Generate two random signals for testing
    # signal_length = 1000
    # signal1 = torch.tensor([random.random() for _ in range(signal_length)])
    # signal2 = torch.tensor([random.random() for _ in range(signal_length)])
    
    # bin_sig,sample_rate = torchaudio.load('/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/iso_wgn.wav')
    bin_sig,sample_rate = torchaudio.load('/Users/vtokala/Documents/Research/di_nn/Dataset/clean_trainset_1f/p226_014_2.wav')
    signal1 = bin_sig[0,:]
    signal2 = bin_sig[1,:]
    signal1 = signal1.squeeze()
    signal2 = signal2.squeeze()
    # Parameters for Welch's method
    segment_length = 512
    overlap = 300
    # window_function = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(segment_length) / (segment_length - 1))
    window_function = torch.hann_window(segment_length)

    # Specify the desired nfft value (e.g., 256)
    nfft = 512

    # Specify the sampling frequency (16,000 Hz)
    sampling_frequency = 16000

    # Compute the Mean Squared Coherence with the specified nfft and sampling frequency
    psd1 = welch_cross_spectral_density(signal1, signal1, segment_length, overlap, window_function, nfft, sampling_frequency)
    psd2 = welch_cross_spectral_density(signal2, signal2, segment_length, overlap, window_function, nfft, sampling_frequency)
    cpsd = welch_cross_spectral_density(signal1, signal2, segment_length, overlap, window_function, nfft, sampling_frequency)
    msc = mean_squared_coherence(signal1, signal2, segment_length, overlap, window_function, nfft, sampling_frequency)
    frequency_axis = torch.linspace(0, sample_rate / 2, msc.size(0))
    # Plot the resulting Mean Squared Coherence
    
    # breakpoint()
    plt.figure(1)
    plt.plot(frequency_axis,(msc.abs()))
    plt.title("Mean Squared Coherence Estimate")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude")
    plt.show()
    print(cpsd.shape)
    plt.figure(3)
    plt.plot(frequency_axis,10*torch.log10(psd1.abs()))
    plt.title("PSD Channel 1")
    plt.xlabel("Frequency Hz")
    plt.ylabel("Magnitude")
    plt.show()
    
    plt.figure(4)
    plt.plot(frequency_axis,10*torch.log10(psd2.abs()))
    plt.title("PSD Channel 2")
    plt.xlabel("Frequency Hz")
    plt.ylabel("Magnitude")
    plt.show()
    
    plt.figure(5)
    plt.plot(frequency_axis,10*torch.log10(cpsd.abs()))
    plt.title("Cross PSD")
    plt.xlabel("Frequency Hz")
    plt.ylabel("Magnitude")
    plt.show()