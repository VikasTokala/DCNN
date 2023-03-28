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

TESTSET_LEN = 5

NOISY_DATASET_PATH =  "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Testset_Experiments/Isotropic_testset/SSN/Noisy_testset"
CLEAN_DATASET_PATH = '/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Testset_Experiments/Isotropic_testset/SSN/Clean_testset'


# noisy_snr_l, noisy_snr_r, enhanced_snr_r, enhanced_snr_l, masked_ild_error, masked_ipd_error, improved_mbstoi, improved_snr_l,improved_snr_r, improved_stoi_l,improved_stoi_r = evalMet(NOISY_DATASET_PATH,
                                                                                                                                                                                        # CLEAN_DATASET_PATH,
                                                                                                                                                                                        # model)


NOISY_DATASET_PATH =  "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Testset_Experiments/Isotropic_testset/SSN/Noisy_testset"
CLEAN_DATASET_PATH = '/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Testset_Experiments/Isotropic_testset/SSN/Clean_testset'

noisy_snr_l_ssn, noisy_snr_r_ssn, enhanced_snr_r_ssn, enhanced_snr_l_ssn, masked_ild_error_ssn, masked_ipd_error_ssn, improved_mbstoi_ssn, improved_snr_l_ssn,improved_snr_r_ssn, improved_stoi_l_ssn,improved_stoi_r_snn = evalMet(NOISY_DATASET_PATH,
                                                                                                                                                                                                                                    CLEAN_DATASET_PATH,model,TESTSET_LEN )
# breakpoint()
writeMatFile(noisy_snr_l_ssn, noisy_snr_r_ssn, enhanced_snr_r_ssn, enhanced_snr_l_ssn, masked_ild_error_ssn, masked_ipd_error_ssn, improved_mbstoi_ssn, improved_snr_l_ssn,improved_snr_r_ssn, improved_stoi_l_ssn,improved_stoi_r_snn,folPath='SSN')


NOISY_DATASET_PATH =  "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Testset_Experiments/Isotropic_testset/WGN/Noisy_testset"
CLEAN_DATASET_PATH = '/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Testset_Experiments/Isotropic_testset/WGN/Clean_testset'


noisy_snr_l_wgn, noisy_snr_r_wgn, enhanced_snr_r_wgn, enhanced_snr_l_wgn, masked_ild_error_wgn, masked_ipd_error_wgn, improved_mbstoi_wgn, improved_snr_l_wgn,improved_snr_r_wgn, improved_stoi_l_wgn,improved_stoi_r_wgn = evalMet(NOISY_DATASET_PATH,
                                                                                                                                                                                                                                     CLEAN_DATASET_PATH,
                                                                                                                                                                                                                                          model,TESTSET_LEN )
writeMatFile(noisy_snr_l_wgn, noisy_snr_r_wgn, enhanced_snr_r_wgn, enhanced_snr_l_wgn, masked_ild_error_wgn, masked_ipd_error_wgn, improved_mbstoi_wgn, improved_snr_l_wgn,improved_snr_r_wgn, improved_stoi_l_wgn,improved_stoi_r_wgn, folPath='WGN')


NOISY_DATASET_PATH =  "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Testset_Experiments/Isotropic_testset/Factory/Noisy_testset"
CLEAN_DATASET_PATH = '/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Testset_Experiments/Isotropic_testset/Factory/Clean_testset'



noisy_snr_l_fac, noisy_snr_r_fac, enhanced_snr_r_fac, enhanced_snr_l_fac, masked_ild_error_fac, masked_ipd_error_fac, improved_mbstoi_fac, improved_snr_l_fac,improved_snr_r_fac, improved_stoi_l_fac,improved_stoi_r_fac = evalMet(NOISY_DATASET_PATH,
                                                                                                                                                                                                                                     CLEAN_DATASET_PATH,
                                                                                                                                                                                                                                      model,TESTSET_LEN )

writeMatFile(noisy_snr_l_fac, noisy_snr_r_fac, enhanced_snr_r_fac, enhanced_snr_l_fac, masked_ild_error_fac, masked_ipd_error_fac, improved_mbstoi_fac, improved_snr_l_fac,improved_snr_r_fac, improved_stoi_l_fac,improved_stoi_r_fac,folPath='Factory')

NOISY_DATASET_PATH =  "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Testset_Experiments/Isotropic_testset/Office/Noisy_testset"
CLEAN_DATASET_PATH = '/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Testset_Experiments/Isotropic_testset/Office/Clean_testset'
noisy_snr_l_off, noisy_snr_r_off, enhanced_snr_r_off, enhanced_snr_l_off, masked_ild_error_off, masked_ipd_error_off, improved_mbstoi_off, improved_snr_l_off,improved_snr_r_off, improved_stoi_l_off,improved_stoi_r_off = evalMet(NOISY_DATASET_PATH,
                                                                                                                                                                                                                                    CLEAN_DATASET_PATH,
                                                                                                                                                                                                                                        model,TESTSET_LEN )

writeMatFile(noisy_snr_l_off, noisy_snr_r_off, enhanced_snr_r_off, enhanced_snr_l_off, masked_ild_error_off, masked_ipd_error_off, improved_mbstoi_off, improved_snr_l_off,improved_snr_r_off, improved_stoi_l_off,improved_stoi_r_off,folPath='Office') 
# x = ['WGN', 'SSN', 'Factory', 'Office']

fig = go.Figure()

fig.add_trace(go.Box(y=masked_ild_error_wgn.mean(dim=1),
    # x=x,
    name='wgn',
    # marker_color='#3D9970'
))
fig.add_trace(go.Box(
    y=masked_ild_error_ssn.mean(dim=1),
    # x=x,
    name='SSN',
    # marker_color='#3D9970'
))
fig.add_trace(go.Box(
    y=masked_ild_error_fac.mean(dim=1),
    # x=x,
    name='Factory',
    # marker_color='#3D9970'
))
fig.add_trace(go.Box(
    y=masked_ild_error_off.mean(dim=1),
    # x=x,
    name='Office',
    # marker_color='#3D9970'
))
fig.update_layout(yaxis=dict(tickmode='array',tick0=0, dtick=0.1))
# breakpoint()
fig.show()


# data = {
#         'ild_error':[masked_ild_error_wgn.mean(dim=0),masked_ild_error_ssn.mean(dim=0),masked_ild_error_fac.mean(dim=0),masked_ild_error_off.mean(dim=0)],
#         'noiseType':['WGN', 'SSN', 'Factory', 'Office']
# }

# df = pd.DataFrame(data=data)
# fig = px.box( y='ild_error', x='noiseType')
# fig.show()
