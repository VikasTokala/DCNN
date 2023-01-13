from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchaudio import transforms as T
from DCNN.loss import si_snr
from DCNN.utils.eval_utils import ild_db, ipd_rad, _avg_signal
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
from mbstoi import mbstoi
import warnings
warnings.simplefilter('ignore')


NOISY_DATASET_PATH = "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Noisy_testset"
CLEAN_DATASET_PATH = '/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Clean_testset'
MODEL_CHECKPOINT_PATH = "/Users/vtokala/Documents/Research/di_nn/DCNN/checkpoints/monoloss_noattn.ckpt"
SR = 16000

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
amptodB = T.AmplitudeToDB(stype='amplitude')

win_len = 400
win_inc = 100
fft_len = 512
fbins = int(fft_len/2 + 1)
avg_mode = 'time'

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

# testset_len = len(dataloader)
testset_len = 10

noisy_snr_l = torch.zeros((testset_len,fbins))
noisy_snr_r = torch.zeros((testset_len,fbins))

enhanced_snr_l = torch.zeros((testset_len,fbins))
enhanced_snr_r = torch.zeros((testset_len,fbins))

noisy_stoi_l = torch.zeros((testset_len,fbins))
noisy_stoi_r = torch.zeros((testset_len,fbins))

enhanced_stoi_l = torch.zeros((testset_len,fbins))
enhanced_stoi_r = torch.zeros((testset_len,fbins))

masked_ild_error = torch.zeros((testset_len,fbins))
masked_ipd_error = torch.zeros((testset_len,fbins))

noisy_mbstoi = torch.zeros((testset_len,fbins))
enhanced_mbstoi = torch.zeros((testset_len,fbins))

avg_snr = torch.zeros(testset_len)

for i in range(testset_len):  # Enhance 10 samples
    try:
        batch = next(dataloader)
    except StopIteration:
        break


    noisy_samples = (batch[0])
    clean_samples = (batch[1])[0]
    model_output = model(noisy_samples)[0]
    # mo = model_output.numpy()
   

    # breakpoint()
    
    
    noisy_snr_l[i] = si_snr(noisy_samples[0][0, :], clean_samples[0, :])
    noisy_snr_r[i] = si_snr(noisy_samples[0][1, :], clean_samples[1, :])

    enhanced_snr_l[i] = si_snr(model_output[0, :], clean_samples[0, :])
    enhanced_snr_r[i] = si_snr(model_output[1, :], clean_samples[1, :])
    
    noisy_stoi_l[i] = stoi(noisy_samples[0][0, :], clean_samples[0, :])
    noisy_stoi_r[i] = stoi(noisy_samples[0][1, :], clean_samples[1, :])

    enhanced_stoi_l[i] = stoi(model_output[0, :], clean_samples[0, :])
    enhanced_stoi_r[i] = stoi(model_output[1, :], clean_samples[1, :])

    noisy_stft_l = stft(noisy_samples[0][0, :])
    noisy_stft_r = stft(noisy_samples[0][1, :])

    enhanced_stft_l = stft(model_output[0, :])
    enhanced_stft_r = stft(model_output[1, :])

    target_stft_l = stft(clean_samples[0, :])
    target_stft_r = stft(clean_samples[1, :])
    # breakpoint() 

    target_ild = ild_db(target_stft_l, target_stft_r, avg_mode=avg_mode)
    enhanced_ild = ild_db(enhanced_stft_l, enhanced_stft_r, avg_mode=avg_mode)

    target_ipd = ipd_rad(target_stft_l, target_stft_r, avg_mode=avg_mode)
    enhanced_ipd = ipd_rad(enhanced_stft_l, enhanced_stft_r, avg_mode=avg_mode)
    
    mask = (target_stft_l.abs() + target_stft_r.abs())/2
    psd_mag = (target_stft_l.abs() + target_stft_r.abs())/2
    psd_db = amptodB(psd_mag)
    # breakpoint()
    psd_db -= psd_db.min(1, keepdim=True)[0]
    psd_db /= psd_db.max(1, keepdim=True)[0] #Normalizing the dB values
    mask = psd_db
    mask_avg = _avg_signal(mask, avg_mode)

    ipd_error = ((target_ipd - enhanced_ipd).abs())
    ild_error = ((target_ild - enhanced_ild).abs())

    masked_ild_error[i,:] = (mask_avg * ild_error)
    masked_ipd_error[i,:] = (mask_avg * ipd_error)
    # breakpoint()
    avg_snr[i] = (noisy_snr_l[i]+noisy_snr_r[i])/2
   
    noisy_signals = noisy_samples[0].detach().cpu().numpy()
    clean_signals = clean_samples.detach().cpu().numpy()
    enhanced_signals = model_output.detach().cpu().numpy()
    noisy_mbstoi[i] = mbstoi(clean_signals[0,:], clean_signals[1,:],
               noisy_signals[0,:], noisy_signals[1,:], fsi=SR)
    enhanced_mbstoi[i] = mbstoi(clean_signals[0,:], clean_signals[1,:],
               enhanced_signals[0,:], enhanced_signals[1,:], fsi=SR)
    
    print('Processed Signal ', i+1 , ' of ', testset_len)

   

  
improved_snr_l = (enhanced_snr_l - noisy_snr_l)
improved_snr_r = (enhanced_snr_r - noisy_snr_r)

improved_stoi_l = (enhanced_stoi_l - noisy_stoi_l)
improved_stoi_r = (enhanced_stoi_r - noisy_stoi_r)

improved_mbstoi = enhanced_mbstoi - noisy_mbstoi


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

axs[0, 0].scatter(noisy_snr_l, improved_snr_l)
axs[0, 0].set_title("SI-SNR improvement - Left")
axs[0, 0].set_xlabel("Input Noisy SI-SNR")
axs[0, 0].set_ylabel("SI-SNR improvement")

axs[0, 1].scatter(noisy_snr_r, improved_snr_r)
axs[0, 1].set_title("SI-SNR improvement - Right")
axs[0, 1].set_xlabel("Input Noisy SI-SNR")
axs[0, 1].set_ylabel("SI-SNR improvement")

axs[1, 0].scatter(noisy_snr_l, improved_stoi_l)
axs[1, 0].set_title("STOI improvement - Left")
axs[1, 0].set_xlabel("Input Noisy SI-SNR")
axs[1, 0].set_ylabel("STOI improvement")

axs[1, 1].scatter(noisy_snr_r, improved_stoi_r)
axs[1, 1].set_title("STOI improvement - Right")
axs[1, 1].set_xlabel("Input Noisy SI-SNR")
axs[1, 1].set_ylabel("STOI improvement")
plt.savefig("snr_stoi_bin.png")

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

axs[0].scatter(avg_snr, masked_ipd_error)
axs[0].set_title("IPD error [rad]")
axs[0].set_xlabel("Input Noisy SI-SNR")
axs[0].set_ylabel("Frequency averaged IPD error in Radians")

axs[ 1].scatter(avg_snr, masked_ild_error)
axs[ 1].set_title("ILD error [dB]")
axs[ 1].set_xlabel("Input Noisy SI-SNR")
axs[ 1].set_ylabel("Frequency averaged ILD error in dB")
plt.savefig("ild_ipd_error_bin.png")

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))

axs.scatter(avg_snr, improved_mbstoi)
axs.set_title("MBSTOI improvement")
axs.set_xlabel("Input Noisy SI-SNR")
axs.set_ylabel("MBSTOI improvement")
plt.savefig("mbstoi_bin.png")





    