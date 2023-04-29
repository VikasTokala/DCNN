import matplotlib.pyplot as plt
from scipy.io import wavfile
import torchaudio

from DCNN.utils.eval_utils import speechMask
from DCNN.feature_extractors import Stft, IStft
import librosa.display
import torch
import numpy as np
win_len = 400
win_inc = 100
fft_len = 512
fbins = int(fft_len/2 + 1)
fs = 16000

stft = Stft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
istft = IStft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)

sig_noisy,_ = torchaudio.load('/Users/vtokala/Documents/Research/di_nn/DCNN/plot_sigs/p278_306_0_SNR_6_iwgn.wav')
sig_clean,_ = torchaudio.load('/Users/vtokala/Documents/Research/di_nn/DCNN/plot_sigs/p278_306_0.wav')
sig_enhanced,_ = torchaudio.load('/Users/vtokala/Documents/Research/di_nn/DCNN/plot_sigs/p278_306_0_DCCTN.wav')

dcctn_enhanced_stft_l = stft(sig_enhanced[0, :])
dcctn_enhanced_stft_r = stft(sig_enhanced[1, :])

target_stft_l = stft(sig_clean[0, :])
target_stft_r = stft(sig_clean[1, :])

noisy_stft_l = stft(sig_noisy[0, :])
noisy_stft_r = stft(sig_noisy[1, :])

mask = speechMask(target_stft_l,target_stft_r, threshold=20).squeeze(0)

fig, (ax1,ax2, ax3) = plt.subplots(1,3, sharey=True,figsize=(16, 5))
D = librosa.amplitude_to_db(np.abs(stft(sig_clean[0]).numpy()), ref=np.max)
img1 = librosa.display.specshow(D, y_axis='hz', x_axis='time',
                                sr=16000, ax=ax1,hop_length=win_inc, win_length=win_len)
ax1.set_title('Spectrogram of the clean speech \n (Left Ch.)',fontname="Times New Roman",fontweight='bold', fontsize='17')
ax1.label_outer()
ax1.set_xlabel('Time [s]', fontname="Times New Roman",fontweight='bold',fontsize='15')
ax1.set_ylabel('Frequency [Hz]', fontname="Times New Roman",fontweight='bold',fontsize='15')

D = librosa.amplitude_to_db(np.abs(stft(sig_clean[1]).numpy()), ref=np.max)
img = librosa.display.specshow(D, y_axis='hz', x_axis='time',
                                sr=16000, ax=ax2, hop_length=win_inc, win_length=win_len)
ax2.set_title('Spectrogram of the clean speech \n (Right Ch.)',fontname="Times New Roman",fontweight='bold',fontsize='17')
ax2.label_outer()
ax2.set_xlabel('Time [s]', fontname="Times New Roman",fontweight='bold',fontsize='15')
# ax2.set_ylabel('Frequency (Hz)', fontname="Times New Roman",fontweight='bold')
fig.colorbar(img1,ax=[ax1,ax2],format="%+2.0f dB")


D = mask.numpy()
img1 = librosa.display.specshow(D, y_axis='hz', x_axis='time',
                                sr=16000, ax=ax3,hop_length=win_inc, win_length=win_len)
ax3.set_title('IBM for the ILD and IPD Loss',fontname="Times New Roman",fontweight='bold', fontsize='17')
ax3.set_xlabel('Time [s]', fontname="Times New Roman",fontweight='bold', fontsize='15')
# ax3.set_ylabel('Frequency (Hz)', fontname="Times New Roman",fontweight='bold')
ax3.label_outer()
plt.savefig("specMask.pdf", format="pdf", bbox_inches="tight")
plt.show()



fig, (ax1,ax2) = plt.subplots(1,2, sharey=True,figsize=(13, 5))
D = librosa.amplitude_to_db(np.abs(stft(sig_clean[0]).numpy()), ref=np.max)
img1 = librosa.display.specshow(D, y_axis='hz', x_axis='time',
                                sr=16000, ax=ax1,hop_length=win_inc, win_length=win_len)
ax1.set(title='Spectrogram of the clean speech (Left Ch.)')

ax1.label_outer()

D = librosa.amplitude_to_db(np.abs(stft(sig_clean[1]).numpy()), ref=np.max)
img = librosa.display.specshow(D, y_axis='hz', x_axis='time',
                                sr=16000, ax=ax2, hop_length=win_inc, win_length=win_len)
ax2.set(title='Spectrogram of the clean speech (Right Ch.)')
ax2.label_outer()
fig.colorbar(img1,ax=[ax1,ax2],format="%+2.0f dB")

# plt.tight_layout()
plt.show()

# breakpoint()

fig, (ax1,ax2) = plt.subplots(1,2, sharey=True,figsize=(13, 5))
D = librosa.amplitude_to_db(np.abs(stft(sig_noisy[0]).numpy()), ref=np.max)
img1 = librosa.display.specshow(D, y_axis='hz', x_axis='time',
                                sr=16000, ax=ax1,hop_length=win_inc, win_length=win_len)
ax1.set(title='Spectrogram of the noisy speech (Left Ch.)')
ax1.label_outer()

D = librosa.amplitude_to_db(np.abs(stft(sig_noisy[1]).numpy()), ref=np.max)
img = librosa.display.specshow(D, y_axis='hz', x_axis='time',
                                sr=16000, ax=ax2, hop_length=win_inc, win_length=win_len)
ax2.set(title='Spectrogram of the noisy speech (Right Ch.)')
ax2.label_outer()
fig.colorbar(img1,ax=[ax1,ax2],format="%+2.0f dB")
plt.show()
fig, (ax1,ax2) = plt.subplots(1,2, sharey=True,figsize=(13, 5))
D = librosa.amplitude_to_db(np.abs(stft(sig_enhanced[0]).numpy()), ref=np.max)
img1 = librosa.display.specshow(D, y_axis='hz', x_axis='time',
                                sr=16000, ax=ax1,hop_length=win_inc, win_length=win_len)
ax1.set(title='Spectrogram of the enhanced speech (Left Ch.)')
ax1.label_outer()

D = librosa.amplitude_to_db(np.abs(stft(sig_enhanced[1]).numpy()), ref=np.max)
img = librosa.display.specshow(D, y_axis='hz', x_axis='time',
                                sr=16000, ax=ax2, hop_length=win_inc, win_length=win_len)
ax2.set(title='Spectrogram of the enhanced speech (Right Ch.)')
ax2.label_outer()
fig.colorbar(img1,ax=[ax1,ax2],format="%+2.0f dB")
plt.show()
# breakpoint()

fig, ax = plt.subplots()
D = mask.numpy()
img1 = librosa.display.specshow(D, y_axis='hz', x_axis='time',
                                sr=16000, ax=ax,hop_length=win_inc, win_length=win_len)
ax.set(title='IBM for the ILD and IPD Loss')
ax.label_outer()
# fig.colorbar(img1,ax=ax)
plt.show()