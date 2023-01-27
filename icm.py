import torch
from DCNN.feature_extractors import Stft, IStft
from gammatone_fb import generate_mpgtf
from torch.autograd import Variable
import torch.nn.functional as F
import torchaudio
win_len = 400
win_inc = 100
fft_len = 512
fbins = int(fft_len/2 + 1)
import librosa.display as ld
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torchaudio.transforms as T
SR = 16000




stft = Stft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
istft = IStft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
psd = T.PSD()


def icm(stft_l, stft_r):
    
    crossPSD = stft_l * stft_r.conj()
    autoPSDL = stft_l * stft_l.conj()
    autoPSDR = stft_r * stft_r.conj()
    
    # breakpoint()
    
    icm_mag = torch.abs(crossPSD/(torch.sqrt(autoPSDL*autoPSDR)))
    
    return icm_mag


if __name__ == '__main__':
    
    input,_=torchaudio.load('/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Bin_ilpd_clean_test/p278_044_1.wav')
    
    stft_l = stft(input[0,:])
    stft_r = stft(input[1,:])
    
    # breakpoint()
    icms = icm(stft_l=stft_l, stft_r=stft_r)
    
    
    plt.figure()
    # D = librosa.amplitude_to_db(((icms).numpy()), ref=np.max)
    img=plt.imshow(icms.numpy())
    # img.set(title='Enhanced - Linear-frequency power spectrogram')
    # img.set.label_outer()
    
    plt.figure()
    D = librosa.amplitude_to_db((np.abs(stft_l).numpy()), ref=np.max)
    img = ld.specshow(D, y_axis='hz', x_axis='time',sr=16000)
    # img.set_t(title='Enhanced - Linear-frequency power spectrogram')
    # img.set.label_outer()
    
    plt.tight_layout()
    plt.show()
    