import torch
import torchaudio
import yaml
from DCNN.trainer import DCNNLightningModule
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
from train import train
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from DCNN.loss import si_snr
from DCNN.utils.eval_utils import ild_db, ipd_rad, speechMask
from DCNN.feature_extractors import Stft, IStft
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import yaml
import matplotlib.pyplot as plt
import numpy as np
from DCNN.datasets.base_dataset import BaseDataset
from DCNN.trainer import DCNNLightningModule
import torch.nn as nn
import librosa.display as ld
import librosa
from mbstoi import mbstoi
from torchaudio import transforms as T
import warnings
import soundfile as sf
from asteroid.dsp import vad
warnings.simplefilter('ignore')
config = {
    "defaults": [
        "model",
        "training",
        {"dataset": "speech_dataset"}
    ],
    "dataset": {
        "noisy_training_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Noisy_trainset",
        "noisy_validation_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Noisy_valset",
        "noisy_test_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Noisy_testset",
        "target_training_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Clean_trainset",
        "target_validation_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Clean_valset",
        "target_test_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Clean_testset"

    },
    "training": {
        "batch_size": 32,
        "n_epochs": 30,
        "n_workers": 4,
        "learning_rate": 0.0001,
        # "/kaggle/working/SE_DCNN/DCNN/checkpoints/weights-epoch=19-validation_loss=-17.90.ckpt",
        "train_checkpoint_path": None,
        "strategy": "ddp",
        "pin_memory": True,
        "accelerator": "null"
    },
    "model": {
        "rtf_weight": 0,
        "snr_weight": 1,
        "ild_weight": 1,  # 0.1,
        "ipd_weight": 10,  # 10,
        "stoi_weight": 10,  # 10
        "kurt_weight": 0,
        "avg_mode": "freq",
        "attention": True
    }
}

with open('config/config.yaml', 'w') as f_yaml:
    yaml.dump(config, f_yaml)
GlobalHydra.instance().clear()
initialize(config_path="./config")
config = compose("config")

MODEL_CHECKPOINT_PATH = "/Users/vtokala/Documents/Research/di_nn/DCNN/checkpoints/full_loss_Binaural_RNN.ckpt"
# MODEL_CHECKPOINT_PATH = "/kaggle/input/lss-resources/code/se/demo/last.ckpt"
model = DCNNLightningModule(config)
model.eval()
torch.set_grad_enabled(False)
device = torch.device('cpu')
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
# checkpoint = torch.load(MODEL_CHECKPOINT_PATH)
model.load_state_dict(checkpoint["state_dict"], strict=False)

NOISY_DATASET_PATH =  "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Bin_ilpd_noisy_test"
CLEAN_DATASET_PATH = '/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/bin_ilpd_clean_test'
SR = 16000

win_len = 500
win_inc = 100
fft_len = 512
fbins = int(fft_len/2 + 1)
fig={}
ax={}
stft = Stft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
istft = IStft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
stoi = ShortTimeObjectiveIntelligibility(fs=16000)

dataset = BaseDataset(NOISY_DATASET_PATH, CLEAN_DATASET_PATH, mono=False)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    drop_last=False
    
    
)

dataloader = iter(dataloader)

testset_len = len(dataloader)

noisy_snr_l = torch.zeros((testset_len))
noisy_snr_r = torch.zeros((testset_len))

enhanced_snr_l = torch.zeros((testset_len))
enhanced_snr_r = torch.zeros((testset_len))

noisy_stoi_l = torch.zeros((testset_len))
noisy_stoi_r = torch.zeros((testset_len))

enhanced_stoi_l = torch.zeros((testset_len))
enhanced_stoi_r = torch.zeros((testset_len))

masked_ild_error = torch.zeros((testset_len,fbins))
masked_ipd_error = torch.zeros((testset_len,fbins))

masked_target_ild = torch.zeros((testset_len,fbins))
masked_target_ipd = torch.zeros((testset_len,fbins))

rms_ild_error = torch.zeros((testset_len,fbins))
noisy_mbstoi = torch.zeros((testset_len))
enhanced_mbstoi = torch.zeros((testset_len))

avg_snr = torch.zeros(testset_len)

for i in range(testset_len):  # Enhance 10 samples
    try:
        batch = next(dataloader)
    except StopIteration:
        break


    noisy_samples = (batch[0])
    clean_samples = (batch[1])[0]
    model_output = model(noisy_samples)[0]
   
    
    clean_samples=(clean_samples)/(torch.max(clean_samples))
    model_output=(model_output)/(torch.max(model_output))
    sf.write('myfile.wav',clean_samples.transpose(0,1).numpy(),samplerate=16000)
    # mo = model_output.numpy()
    # breakpoint()
    noisy_snr_l[i] = si_snr(noisy_samples[0][0, :], clean_samples[0, :])
    noisy_snr_r[i] = si_snr(noisy_samples[0][1, :], clean_samples[1, :])

    enhanced_snr_l[i] = si_snr(model_output[0, :], clean_samples[0, :])
    enhanced_snr_r[i] = si_snr(model_output[1, :], clean_samples[1, :])
    # breakpoint()
    noisy_stft_l = stft(noisy_samples[0][0, :])
    noisy_stft_r = stft(noisy_samples[0][1, :])

    enhanced_stft_l = stft(model_output[0, :])
    enhanced_stft_r = stft(model_output[1, :])

    target_stft_l = stft(clean_samples[0, :])
    target_stft_r = stft(clean_samples[1, :])
    
    
    mask = speechMask(target_stft_l,target_stft_r).squeeze(0)
    target_ild = ild_db(target_stft_l.abs(), target_stft_r.abs())
    enhanced_ild = ild_db(enhanced_stft_l.abs(), enhanced_stft_r.abs())

    target_ipd = ipd_rad(target_stft_l, target_stft_r)
    enhanced_ipd = ipd_rad(enhanced_stft_l, enhanced_stft_r)

    ild_loss = (target_ild - enhanced_ild).abs()

    ipd_loss = (target_ipd - enhanced_ipd).abs()
   
    masked_ild_error[i,:] = (ild_loss*mask).sum(dim=1)/ mask.sum(dim=1)
    masked_ipd_error[i,:] = (ipd_loss*mask).sum(dim=1)/ mask.sum(dim=1)
    
    # target_ild = ild_db(target_stft_l.abs(), target_stft_r.abs())
    # enhanced_ild = ild_db(enhanced_stft_l.abs(), enhanced_stft_r.abs())

    # # breakpoint()
    # target_ipd = ipd_rad(target_stft_l, target_stft_r)
    # enhanced_ipd = ipd_rad(enhanced_stft_l, enhanced_stft_r)
    
    # masked_target_ild[i,:] = (target_ild * mask).sum(dim=1)/mask.sum(dim=1)
    # masked_target_ipd[i,:] = (target_ipd * mask).sum(dim=1)/mask.sum(dim=1)
    # # breakpoint()
    
    # masked_ild_error[i,:] = ((target_ild*mask).sum(dim=1)/mask.sum(dim=1) - ((enhanced_ild*mask).sum(dim=1)/mask.sum(dim=1)) ).abs() #/ mask.sum(dim=1)
    # masked_ipd_error[i,:] = ((target_ipd*mask).sum(dim=1)/mask.sum(dim=1) - ((enhanced_ipd*mask).sum(dim=1)/mask.sum(dim=1)) ).abs() #/ mask.sum(dim=1)
    
    # rms_ild_error[i,:] = torch.sqrt((torch.sum(target_ild*mask-enhanced_ild*mask,dim=1)/mask.sum(dim=1))**2)
    avg_snr[i] = (noisy_snr_l[i] + noisy_snr_r[i])/2
    # breakpoint()
    
    print('Processed Signal ', i+1 , ' of ', testset_len)
    
#     fig[i], ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
# #     D = librosa.amplitude_to_db(np.abs(stft(model_output[0]).numpy()), ref=np.max)
#     img = ld.specshow(mask_l.numpy(), y_axis='hz', x_axis='time',
#                                    sr=16000, ax=ax[0])
#     # ax.set(title='Mask - Linear-frequency power spectrogram')
#     # ax.label_outer()
    
# #     D = librosa.amplitude_to_db(np.abs(stft(model_output[1]).numpy()), ref=np.max)
#     img =ld.specshow(mask_r.numpy(), y_axis='hz', x_axis='time',sr=16000, ax=ax[1])
#     plt.show()
    
#     plt.figure()
#     plt.plot(clean_samples[0,:])

#  
improved_snr_l = (enhanced_snr_l - noisy_snr_l)
improved_snr_r = (enhanced_snr_r - noisy_snr_r)

f=np.fft.rfftfreq(n=fft_len, d=1./16000)
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
axs.errorbar( f,(masked_target_ipd.mean(dim=0)), ((masked_target_ipd.std(dim=0))))
axs.set_title("IPD error [rad]")
axs.set_xlabel("Frequency")
axs.set_ylabel("Frequency averaged IPD error in Radians")
plt.show()
# plt.savefig("ipd_error_bin.png")

fig1, axs1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
axs1.errorbar(f, masked_target_ild.mean(dim=0), masked_target_ild.std(dim=0))
axs1.set_title("ILD error [dB]")
axs1.set_xlabel("Frequency")
axs1.set_ylabel("Frequency averaged ILD error in dB")
plt.show()
# plt.savefig("ild_error_bin.png")

f=np.fft.rfftfreq(n=fft_len, d=1./16000)
fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
axs2.errorbar( f,(masked_ipd_error.mean(dim=0)), ((masked_ipd_error.std(dim=0))))
axs2.set_title("IPD error [rad]")
axs2.set_xlabel("Frequency")
axs2.set_ylabel("Frequency averaged IPD error in Radians")
plt.show()
# plt.savefig("ipd_error_bin.png")

fig3, axs3 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
axs3.errorbar(f, rms_ild_error.mean(dim=0), rms_ild_error.std(dim=0))
axs3.set_title("ILD error [dB]")
axs3.set_xlabel("Frequency")
axs3.set_ylabel("Frequency averaged ILD error in dB")
plt.show()



