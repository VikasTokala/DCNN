from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from DCNN.loss import si_snr, ild_db, ipd_rad, _avg_signal
from DCNN.feature_extractors import Stft, IStft
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import yaml
import matplotlib.pyplot as plt
import numpy as np
from DCNN.datasets.base_dataset import BaseDataset
from DCNN.trainer import DCNNLightningModule
import torch.nn as nn
import torch
import librosa
import soundfile as sf
import warnings
warnings.simplefilter('ignore')


NOISY_DATASET_PATH = "/Users/vtokala/Documents/Research/di_nn/Dataset/noisy_testset_1f"
CLEAN_DATASET_PATH = '/Users/vtokala/Documents/Research/di_nn/Dataset/clean_testset_1f'
MODEL_CHECKPOINT_PATH = "/Users/vtokala/Documents/Research/di_nn/DCNN/checkpoints/monoloss_noattn.ckpt"

EPS = 1e-6

config = {
    "defaults": [
        "model",
        "training",
        {"dataset": "speech_dataset"}
    ]}

# Load the model from the checkpoint
with open('/Users/vtokala/Documents/Research/di_nn/config/config.yaml', 'w') as f_yaml:
    yaml.dump(config, f_yaml)
GlobalHydra.instance().clear()
initialize(config_path="config")
config = compose("config")


win_len = 400
win_inc = 100
fft_len = 512
avg_mode = 'freq'

stft = Stft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
istft = IStft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
stoi = ShortTimeObjectiveIntelligibility(fs=16000)


model = DCNNLightningModule(config)
model.eval()
torch.set_grad_enabled(False)
device = torch.device('cpu')
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
# checkpoint = torch.load(MODEL_CHECKPOINT_PATH)
model.load_state_dict(checkpoint["state_dict"], strict=False)


dataset = BaseDataset(NOISY_DATASET_PATH, CLEAN_DATASET_PATH, mono=False)

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True,
                                         drop_last=False)

dataloader = iter(dataloader)

noisy_snr_l = torch.zeros(len(dataloader))
noisy_snr_r = torch.zeros(len(dataloader))

enhanced_snr_l = torch.zeros(len(dataloader))
enhanced_snr_r = torch.zeros(len(dataloader))

noisy_stoi_l = torch.zeros(len(dataloader))
noisy_stoi_r = torch.zeros(len(dataloader))

enhanced_stoi_l = torch.zeros(len(dataloader))
enhanced_stoi_r = torch.zeros(len(dataloader))

masked_ild_error = torch.zeros(len(dataloader))
masked_ipd_error = torch.zeros(len(dataloader))

for i in range(len(dataloader)):  # Enhance 10 samples
    try:
        batch = next(dataloader)
    except StopIteration:
        break


    noisy_samples = (batch[0])
    clean_samples = (batch[1])[0]
    model_output = model(noisy_samples)[0]
    # mo = model_output.numpy()
    breakpoint()
    # sf.write('enhanced_'+ str(i+1) + '.wav', model_output.transpose().numpy(), samplerate=16000 )

    # breakpoint()
    
    
    noisy_snr_l[i] = si_snr(noisy_samples[0][0, :], clean_samples[0, :])
    noisy_snr_r[i] = si_snr(noisy_samples[0][1, :], clean_samples[1, :])

    enhanced_snr_l[i] = si_snr(model_output[0, :], clean_samples[0, :])
    enhanced_snr_r[i] = si_snr(model_output[1, :], clean_samples[1, :])

    noisy_stoi_l = stoi(model_output[0, :], clean_samples[0, :])
    noisy_stoi_r = stoi(model_output[1, :], clean_samples[1, :])

    noisy_stft_l = stft(noisy_samples[0][0, :])
    noisy_stft_r = stft(noisy_samples[0][1, :])

    enhanced_stft_l = stft(model_output[0, :])
    enhanced_stft_r = stft(model_output[1, :])

    target_stft_l = stft(clean_samples[0, :])
    target_stft_r = stft(clean_samples[1, :])

    target_ild = ild_db(target_stft_l, target_stft_r, avg_mode='freq')
    enhanced_ild = ild_db(enhanced_stft_l, enhanced_stft_r, avg_mode='freq')

    target_ipd = ipd_rad(target_stft_l, target_stft_r, avg_mode=avg_mode)
    enhanced_ipd = ipd_rad(enhanced_stft_l, enhanced_stft_r, avg_mode=avg_mode)

    mask = (target_stft_l.abs() + target_stft_r.abs())/2
    mask_avg = _avg_signal(mask, avg_mode)

    ipd_error = ((target_ipd - enhanced_ipd).abs())
    ild_error = ((target_ild - enhanced_ild).abs())

    masked_ild_error[i] = (mask_avg * ild_error).mean()
    masked_ipd_error[i] = (mask_avg * ipd_error).mean()

   

  
improved_snr_l = (enhanced_snr_l - noisy_snr_l)
improved_snr_r = (enhanced_snr_r - noisy_snr_r)

improved_stoi_l = (enhanced_stoi_l - noisy_stoi_l)
improved_snr_r = (enhanced_stoi_r - noisy_stoi_r)

# breakpoint()























# breakpoint()
# class evaluation(nn.Module):
#     def __init__(self, win_len=400,
#                  win_inc=100, fft_len=512, sr=16000, avg_mode="freq") -> None:
#         super().__init__()

#         self.stft = Stft(fft_len, win_inc, win_len)
#         self.istft = IStft(fft_len, win_inc, win_len)
#         self.model = DCNNLightningModule(config=config)
#         self.model.eval()
#         self.device = torch.device('cpu')
#         self.checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=self.device)
#         self.model.load_state_dict(self.checkpoint["state_dict"], strict=False)

#         self.dataset = BaseDataset(NOISY_DATASET_PATH, CLEAN_DATASET_PATH, mono=False)

#         self.dataloader = torch.utils.data.DataLoader(
#             self.dataset,
#             batch_size=1,
#             shuffle=False,
#             pin_memory=True,
#             drop_last=False,
#             num_workers=0)

#     def forward(self):

#         self.dataloader = iter(self.dataloader)

#         for i in range(len(self.dataloader)):  # Enhance 10 samples
#             try:
#                 batch = next(self.dataloader)
#             except StopIteration:
#                 break

#             print(batch[0][0].shape)
#             noisy_samples = (batch[0])
#             clean_samples = (batch[1])[0]
#             model_output = self.model(noisy_samples)[0]
#             print(noisy_samples.shape)
#             breakpoint()
#             target_stft_l = self.stft(clean_samples[0])
#             dsnr  = delsnr(noisy_samples,clean_samples)

#             return dsnr


# def delsnr(test_signal, ref_signal):

#     snr_l = si_snr(test_signal[:, 0], ref_signal[:, 0])
#     snr_r = si_snr(test_signal[:, 1], ref_signal[:, 1])

#     return snr_l, snr_r


# def shortfft(x):

#     stft = Stft(win_len=400,
#                 win_inc=100, fft_len=512)
#     X = stft(x)

#     return X


# def _istft(X):

#     istft = IStft(win_len=400,
#                   win_inc=100, fft_len=512)
#     x = istft(X)
#     return x
