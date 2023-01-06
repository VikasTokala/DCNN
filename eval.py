import torch
import torch.nn as nn
from DCNN.trainer import DCNNLightningModule
from DCNN.datasets.base_dataset import BaseDataset
import numpy as np
import matplotlib.pyplot as plt
import yaml
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
from DCNN.feature_extractors import Stft, IStft
from DCNN.loss import si_snr, ild_loss_db, ipd_loss_rads
# import librosa.display as ld

NOISY_DATASET_PATH = "/Users/vtokala/Documents/Research/di_nn/Dataset/noisy_testset_1f"
CLEAN_DATASET_PATH = '/Users/vtokala/Documents/Research/di_nn/Dataset/clean_testset_1f'
MODEL_CHECKPOINT_PATH = "/Users/vtokala/Documents/Research/di_nn/DCNN/checkpoints/full_loss_noattn.ckpt"

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


torch.set_grad_enabled(False)




# breakpoint()
class evaluation(nn.Module):
    def __init__(self, win_len=400,
                 win_inc=100, fft_len=512, sr=16000, avg_mode="freq") -> None:
        super().__init__()

        self.stft = Stft(fft_len, win_inc, win_len)
        self.istft = IStft(fft_len, win_inc, win_len)
        self.model = DCNNLightningModule(config=config)
        self.model.eval()
        self.device = torch.device('cpu')
        self.checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(self.checkpoint["state_dict"], strict=False)

        self.dataset = BaseDataset(NOISY_DATASET_PATH, CLEAN_DATASET_PATH, mono=False)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=0)

    def forward(self):          

        self.dataloader = iter(self.dataloader)

        for i in range(len(self.dataloader)):  # Enhance 10 samples
            try:
                batch = next(self.dataloader)
            except StopIteration:
                break

            print(batch[0][0].shape)
            noisy_samples = (batch[0])
            clean_samples = (batch[1])[0]
            model_output = self.model(noisy_samples)[0]
            print(noisy_samples.shape)
            breakpoint()
            target_stft_l = self.stft(clean_samples[0])
            dsnr  = delsnr(noisy_samples,clean_samples)

            return dsnr



def delsnr(test_signal, ref_signal):

    snr_l = si_snr(test_signal[:, 0], ref_signal[:, 0])
    snr_r = si_snr(test_signal[:, 1], ref_signal[:, 1])

    return snr_l, snr_r


def shortfft(x):

    stft = Stft(win_len=400,
                win_inc=100, fft_len=512)
    X = stft(x)

    return X


def _istft(X):

    istft = IStft(win_len=400,
                  win_inc=100, fft_len=512)
    x = istft(X)
    return x


if __name__ == "__main__":
    snr = evaluation ()
    print(snr)