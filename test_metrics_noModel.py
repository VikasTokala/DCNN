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
from DCNN.datasets.test_dataset import BaseDataset
from pathlib import Path
from glob import glob
import os

parentdir = '/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/WASPAA_Testset/Enhanced_signals/WGN/VCTK/*/'

SNRfolders = glob(parentdir, recursive=True)

for i in range (len(SNRfolders)):
        
    NOISY_DATASET_PATH =  os.path.join(SNRfolders(i),"Noisy_testset")
    NOISY_DATASET_PATH_10k =  os.path.join(SNRfolders(i),"Noisy_testset_10kHz")
    CLEAN_DATASET_PATH =  os.path.join(SNRfolders(i),"Clean_testset")
    CLEAN_DATASET_PATH_10k =  os.path.join(SNRfolders(i),"Clean_testset_10kHz")
    CLEAN_DATASET_PATH_10k_eval =  os.path.join(SNRfolders(i),"Clean_testset_10k_eval")
    ENHANCED_DATASET_PATH_DCCTN = os.path.join(SNRfolders(i),"DCCTN")
    ENHANCED_DATASET_PATH_BLDC = os.path.join(SNRfolders(i),"BLDCCTN")
    ENHANCED_DATASET_PATH_BMWF = os.path.join(SNRfolders(i),"BMWF")
    ENHANCED_DATASET_PATH_BSOBM = os.path.join(SNRfolders(i),"BSOBM")
    
    
    dataset = BaseDataset(NOISY_DATASET_PATH, CLEAN_DATASET_PATH, mono=False)
    dataset_bsobm_noisy = BaseDataset(NOISY_DATASET_PATH_10k, CLEAN_DATASET_PATH_10k, mono=False,sr=10000)
    dataset_dcctn = BaseDataset(ENHANCED_DATASET_PATH_DCCTN, CLEAN_DATASET_PATH, mono=False)
    dataset_bldc = BaseDataset(ENHANCED_DATASET_PATH_BLDC, CLEAN_DATASET_PATH, mono=False)
    dataset_bmwf = BaseDataset(ENHANCED_DATASET_PATH_BMWF, CLEAN_DATASET_PATH, mono=False)
    dataset_bsobm = BaseDataset(ENHANCED_DATASET_PATH_BSOBM, CLEAN_DATASET_PATH_10k_eval, mono=False,sr=10000)
    

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    
    dataloader_bsobm_noisy = torch.utils.data.DataLoader(
    dataset_bsobm_noisy,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    drop_last=False)
    
    dataloader_dcctn = torch.utils.data.DataLoader(
        dataset_dcctn,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    
    dataloader_bldc = torch.utils.data.DataLoader(
        dataset_bldc,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    
    dataloader_bmwf = torch.utils.data.DataLoader(
        dataset_bmwf,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    
    dataloader_bsobm = torch.utils.data.DataLoader(
        dataset_bsobm,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    dataloader = iter(dataloader)
    dataloader_bsobm_noisy = iter(dataloader_bsobm_noisy)
    dataloader_bldc = iter(dataloader_bldc)
    dataloader_dcctn = iter(dataloader_dcctn)
    dataloader_bsobm = iter(dataloader_bsobm)
    dataloader_bmwf =  iter(dataloader_bmwf)
    
    

