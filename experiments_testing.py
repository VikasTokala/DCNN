import yaml
# !pip install tensorboardX
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
from train import train
import torch
from DCNN.trainer import DCNNLightningModule
from DCNN.utils import evalFunction
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from DCNN.matFileGen import writeMatFile
from DCNN.datasets.base_dataset import BaseDataset

config = {
    "defaults": [
        "model",
        "training",
        {"dataset": "speech_dataset"}
    ],
    "dataset":{
        "noisy_training_dataset_dir": "/kaggle/input/isotropicdataset/Noisy_trainset_iso",
        "noisy_validation_dataset_dir": "/kaggle/input/isotropicdataset/Noisy_valset_iso",
        "noisy_test_dataset_dir": "/kaggle/input/isotropic-testset/Isotropic_testset/SSN/Noisy_testset",
        "target_training_dataset_dir": "/kaggle/input/isotropicdataset/Clean_trainset_iso",
        "target_validation_dataset_dir": "/kaggle/input/isotropicdataset/Clean_valset_iso",
        "target_test_dataset_dir": "/kaggle/input/isotropic-testset/Isotropic_testset/SSN/Clean_testset"
#         "target_training_dataset_dir": "/kaggle/input/binauralspeech30db/Clean_trainset_30dB",
#         "target_validation_dataset_dir": "/kaggle/input/binauralspeech30db/Clean_valset_30dB",
#         "target_test_dataset_dir": "/kaggle/input/binauralspeech30db/Clean_testset_30dB"
    },
    "training":{
        "batch_size": 32,
        "n_epochs": 40,
        "n_workers": 4,
        "learning_rate":0.0001,
        "train_checkpoint_path": None, #"/kaggle/working/SE_DCNN/DCNN/checkpoints/weights-epoch=19-validation_loss=-17.90.ckpt",
        "strategy": "ddp",
        "pin_memory": True,
        "accelerator": "cuda"
    },
    "model":{
        "rtf_weight" : 0,
        "snr_weight" : 1,
        "ild_weight": 1, #0.1,
        "ipd_weight": 1, #10,
        "stoi_weight": 10, #10
        "kurt_weight": 0,
        "avg_mode": "time",
        "attention": True
    }
}

with open('config/config.yaml', 'w') as f_yaml:
    yaml.dump(config, f_yaml)
GlobalHydra.instance().clear()
initialize(config_path="./config")
config = compose("config")


evalMet = evalFunction.EvalMetrics()
MODEL_CHECKPOINT_PATH = "/Users/vtokala/Documents/Research/di_nn/DCNN/checkpoints/Complex-Attn-40E-iso-com.ckpt"
# MODEL_CHECKPOINT_PATH = "/kaggle/input/lss-resources/code/se/demo/last.ckpt"
model = DCNNLightningModule(config)
model.eval()
torch.set_grad_enabled(False)
device = torch.device('cpu')
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
# checkpoint = torch.load(MODEL_CHECKPOINT_PATH)
model.load_state_dict(checkpoint["state_dict"], strict=False)