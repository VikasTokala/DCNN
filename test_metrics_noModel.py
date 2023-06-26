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
from DCNN.utils.eval_utils import ild_db, ipd_rad, speechMask
from pathlib import Path
from glob import glob
import os
from DCNN.feature_extractors import Stft, IStft
import librosa
import librosa.display 

parentdir = '/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/WASPAA_Testset/Enhanced_signals/WGN/VCTK/*/'

SNRfolders = sorted(glob(parentdir, recursive=True))

win_len = 400
win_inc = 100
fft_len = 512
fbins = int(fft_len/2 + 1)

TESTSET_LEN =3
stft = Stft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
istft = IStft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
# breakpoint()
SNRlist = [-3,-6,0,12,15,3,6,9]

for i in range (len(SNRfolders)):
    print('Processed Signals of SNR ', SNRlist[i] )
    NOISY_DATASET_PATH =  os.path.join(SNRfolders[i],"Noisy_testset")
    NOISY_DATASET_PATH_10k =  os.path.join(SNRfolders[i],"Noisy_testset_10kHz")
    CLEAN_DATASET_PATH =  os.path.join(SNRfolders[i],"Clean_testset")
    CLEAN_DATASET_PATH_10k =  os.path.join(SNRfolders[i],"Clean_testset_10kHz")
    CLEAN_DATASET_PATH_10k_eval =  os.path.join(SNRfolders[i],"Clean_testset_10k_eval")
    ENHANCED_DATASET_PATH_DCCTN = os.path.join(SNRfolders[i],"DCCTN")
    ENHANCED_DATASET_PATH_DCCRN = os.path.join(SNRfolders[i],"DCCRN")
    ENHANCED_DATASET_PATH_BMWF = os.path.join(SNRfolders[i],"BMWF")
    ENHANCED_DATASET_PATH_BSOBM = os.path.join(SNRfolders[i],"BSOBM")
    
    
    dataset = BaseDataset(NOISY_DATASET_PATH, CLEAN_DATASET_PATH, mono=False)
    dataset_bsobm_noisy = BaseDataset(NOISY_DATASET_PATH_10k, CLEAN_DATASET_PATH_10k, mono=False,sr=10000)
    dataset_dcctn = BaseDataset(ENHANCED_DATASET_PATH_DCCTN, CLEAN_DATASET_PATH, mono=False)
    dataset_dccrn = BaseDataset(ENHANCED_DATASET_PATH_DCCRN, CLEAN_DATASET_PATH, mono=False)
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
    
    dataloader_dccrn = torch.utils.data.DataLoader(
        dataset_dccrn,
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
    dataloader_dccrn = iter(dataloader_dccrn)
    dataloader_dcctn = iter(dataloader_dcctn)
    dataloader_bsobm = iter(dataloader_bsobm)
    dataloader_bmwf =  iter(dataloader_bmwf)
    
    masked_ild_error_dcctn = torch.zeros((len(dataloader), fbins))
    masked_ipd_error_dcctn = torch.zeros((len(dataloader), fbins))
    
    masked_ild_error_dccrn = torch.zeros((len(dataloader), fbins))
    masked_ipd_error_dccrn = torch.zeros((len(dataloader), fbins))
    
    masked_ild_error_bsobm = torch.zeros((len(dataloader), fbins))
    masked_ipd_error_bsobm = torch.zeros((len(dataloader), fbins))
    
    masked_ild_error_bmwf = torch.zeros((len(dataloader), fbins))
    masked_ipd_error_bmwf = torch.zeros((len(dataloader), fbins))
    
    for j in range(len(dataloader)):#():
        try:
            batch = next(dataloader)
            batch_bsobm_noisy = next(dataloader_bsobm_noisy)
            batch_dccrn = next(dataloader_dccrn)
            batch_dcctn = next(dataloader_dcctn)
            batch_bsobm = next(dataloader_bsobm)
            batch_bmwf = next(dataloader_bmwf)
        except StopIteration:
            break
        
        noisy_samples = (batch[0])[0]
        noisy_sobm = (batch_bsobm_noisy[0])[0]
        clean_samples = (batch[1])[0]
        clean_bsobm = (batch_bsobm[1])[0]
        enhanced_dccrn = (batch_dccrn[0])[0]
        enhanced_dcctn = (batch_dcctn[0])[0]
        enhanced_bsobm = (batch_bsobm[0])[0]
        enhanced_bmwf = (batch_bmwf[0])[0]
        
        clean_samples=(clean_samples)/(torch.max(clean_samples))
        
        dcctn_enhanced_stft_l = stft(enhanced_dcctn[0, :])
        dcctn_enhanced_stft_r = stft(enhanced_dcctn[1, :])

        target_stft_l = stft(clean_samples[0, :])
        target_stft_r = stft(clean_samples[1, :])
        
        bsobm_target_stft_l = stft(clean_bsobm[0, :])
        bsobm_target_stft_r = stft(clean_bsobm[1, :])
        
        dccrn_enhanced_stft_l = stft(enhanced_dccrn[0, :])
        dccrn_enhanced_stft_r = stft(enhanced_dccrn[1, :])
        
        bsobm_enhanced_stft_l = stft(enhanced_bsobm[0, :])
        bsobm_enhanced_stft_r = stft(enhanced_bsobm[1, :])
        
        bmwf_enhanced_stft_l = stft(enhanced_bmwf[0, :])
        bmwf_enhanced_stft_r = stft(enhanced_bmwf[1, :])
        
        mask = speechMask(target_stft_l,target_stft_r, threshold=10).squeeze(0)
        mask_sobm = speechMask(bsobm_target_stft_l,bsobm_target_stft_r, threshold=10).squeeze(0)
        
        target_ild_16k = ild_db(target_stft_l.abs(), target_stft_r.abs())
        target_ild_10k = ild_db(bsobm_target_stft_l.abs(), bsobm_target_stft_r.abs())
        
        dcctn_enhanced_ild = ild_db(dcctn_enhanced_stft_l.abs(), dcctn_enhanced_stft_r.abs())
        dccrn_enhanced_ild = ild_db(dccrn_enhanced_stft_l.abs(), dccrn_enhanced_stft_r.abs())
        bsobm_enhanced_ild = ild_db(bsobm_enhanced_stft_l.abs(), bsobm_enhanced_stft_r.abs())
        bmwf_enhanced_ild = ild_db(bmwf_enhanced_stft_l.abs(), bmwf_enhanced_stft_r.abs())
        
        target_ipd_16k = ipd_rad(target_stft_l, target_stft_r)
        target_ipd_10k = ipd_rad(bsobm_target_stft_l, bsobm_target_stft_r)
        
        dcctn_enhanced_ipd = ipd_rad(dcctn_enhanced_stft_l, dcctn_enhanced_stft_r)
        dccrn_enhanced_ipd = ipd_rad(dccrn_enhanced_stft_l, dccrn_enhanced_stft_r)
        bsobm_enhanced_ipd = ipd_rad(bsobm_enhanced_stft_l, bsobm_enhanced_stft_r)
        bmwf_enhanced_ipd = ipd_rad(bmwf_enhanced_stft_l, bmwf_enhanced_stft_r)
        
        
        dcctn_ild_error = (target_ild_16k - dcctn_enhanced_ild).abs()
        dccrn_ild_error = (target_ild_16k - dccrn_enhanced_ild).abs()
        bmwf_ild_error = (target_ild_16k - bmwf_enhanced_ild).abs()
        bsobm_ild_error = (target_ild_10k - bsobm_enhanced_ild).abs()
        
        dcctn_ipd_error = (target_ipd_16k - dcctn_enhanced_ipd).abs()
        dccrn_ipd_error = (target_ipd_16k - dccrn_enhanced_ipd).abs()
        bmwf_ipd_error = (target_ipd_16k - bmwf_enhanced_ipd).abs()
        bsobm_ipd_error = (target_ipd_10k - bsobm_enhanced_ipd).abs()
        
        mask_sum=mask.sum(dim=1)
        mask_sum[mask_sum==0]=1
        
        mask_sum_sobm=mask_sobm.sum(dim=1)
        mask_sum_sobm[mask_sum_sobm==0]=1
    
        masked_ild_error_dcctn[j,:] = (dcctn_ild_error*mask).sum(dim=1)/ mask_sum
        masked_ipd_error_dcctn[j,:] = (dcctn_ipd_error*mask).sum(dim=1)/ mask_sum
        
        masked_ild_error_dccrn[j,:] = (dccrn_ild_error*mask).sum(dim=1)/ mask_sum
        masked_ipd_error_dccrn[j,:] = (dccrn_ipd_error*mask).sum(dim=1)/ mask_sum
        
        masked_ild_error_bmwf[j,:] = (bmwf_ild_error*mask).sum(dim=1)/ mask_sum
        masked_ipd_error_bmwf[j,:] = (bmwf_ipd_error*mask).sum(dim=1)/ mask_sum
        
        masked_ild_error_bsobm[j,:] = (bsobm_ild_error*mask_sobm).sum(dim=1)/ mask_sum_sobm
        masked_ipd_error_bsobm[j,:] = (bsobm_ipd_error*mask_sobm).sum(dim=1)/ mask_sum_sobm
        
    
        
        
        
        print('Processed Signal ', j+1 , ' of ', len(dataloader))
        
    folpath = os.path.join(SNRfolders[i],'ILD_IPD_Errors')
    writeMatFile(masked_ild_error_dcctn,masked_ipd_error_dcctn, folPath=folpath,method='DCCTN' )
    # breakpoint()
    writeMatFile(masked_ild_error_dccrn,masked_ipd_error_dccrn, folPath=folpath  , method='DCCRN')  
    writeMatFile(masked_ild_error_bsobm,masked_ipd_error_bsobm, folPath=folpath, method='BSOBM' )       
    writeMatFile(masked_ild_error_bmwf,masked_ipd_error_bmwf, folPath=folpath, method='BMWF'  )  
         
        
        
        
        
        
        
    
    

