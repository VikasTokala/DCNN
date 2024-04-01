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
from DCNN.writeMatFileIPD import writeMatFileIPD
from DCNN.datasets.test_dataset import BaseDataset
from DCNN.utils.eval_utils import ild_db, ipd_rad, speechMask
from pathlib import Path
from glob import glob
import os
from DCNN.feature_extractors import Stft, IStft
import librosa
import librosa.display 


parentdir = "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/ICASSP_Testset/audio_files_rb/*/"

SNRfolders = sorted(glob(parentdir, recursive=True))

win_len = 400
win_inc = 100
fft_len = 512
fbins = int(fft_len/2 + 1) - 50
fbins_ipd = 50

stft = Stft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
istft = IStft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
# breakpoint()
SNRlist = [-3,-6,0,12,15,3,6,9]
SNRnames =  ['m3', 'm6', '0','12','15','3','6','9']
for i in range (len(SNRfolders)):
    print('Processed Signals of SNR ', SNRlist[i] )
    # NOISY_DATASET_PATH =  os.path.join(SNRfolders[i],"Noisy_testset")
    # NOISY_DATASET_PATH_10k =  os.path.join(SNRfolders[i],"Noisy_testset_10k")
    CLEAN_DATASET_PATH =  os.path.join(SNRfolders[i],"Clean_testset")
    # CLEAN_DATASET_PATH_10k =  os.path.join(SNRfolders[i],"Clean_testset_10k")
    # CLEAN_DATASET_PATH_10k_eval =  os.path.join(SNRfolders[i],"Clean_testset_10k_eval")
    ENHANCED_DATASET_PATH_DCCTN_PL_SISNR = os.path.join(SNRfolders[i],"BCCRN_PL_ASLM_RVRB")
    # ENHANCED_DATASET_PATH_PL_SISDR = os.path.join(SNRfolders[i],"DCCTN_SDR_PL")
    # ENHANCED_DATASET_PATH_SISDR = os.path.join(SNRfolders[i],"DCCTN_SISDR")
    # ENHANCED_DATASET_PATH_BSOBM = os.path.join(SNRfolders[i],"BSOBM")
    ENHANCED_DATASET_PATH_SNR = os.path.join(SNRfolders[i],"BCCRN_SNR_ASLM_RVRB")
    ENHANCED_DATASET_PATH_BTN = os.path.join(SNRfolders[i],"BiTasNet_17M_ASLM_RVRB")
    # ENHANCED_DATASET_PATH_MMSE = os.path.join(SNRfolders[i],"DCCTN_MagMSE")
    
    # breakpoint()
    # dataset = BaseDataset(NOISY_DATASET_PATH, CLEAN_DATASET_PATH, mono=False)
    # dataset_bsobm_noisy = BaseDataset(NOISY_DATASET_PATH_10k, CLEAN_DATASET_PATH_10k, mono=False,sr=10000)
    dataset_dcctn_pl_sisnr = BaseDataset(ENHANCED_DATASET_PATH_DCCTN_PL_SISNR, CLEAN_DATASET_PATH, mono=False)
    # dataset_dcctn_pl_sisdr = BaseDataset(ENHANCED_DATASET_PATH_PL_SISDR, CLEAN_DATASET_PATH, mono=False)
    # dataset_dcctn_sisdr = BaseDataset(ENHANCED_DATASET_PATH_SISDR, CLEAN_DATASET_PATH, mono=False)
    # dataset_bsobm = BaseDataset(ENHANCED_DATASET_PATH_BSOBM, CLEAN_DATASET_PATH_10k_eval, mono=False,sr=10000)
    dataset_dcctn_snr = BaseDataset(ENHANCED_DATASET_PATH_SNR, CLEAN_DATASET_PATH, mono=False)
    dataset_bitas = BaseDataset(ENHANCED_DATASET_PATH_BTN, CLEAN_DATASET_PATH, mono=False)
    # dataset_dcctn_mmse = BaseDataset(ENHANCED_DATASET_PATH_MMSE, CLEAN_DATASET_PATH, mono=False)
    
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False)
    
    # dataloader_bsobm_noisy = torch.utils.data.DataLoader(
    # dataset_bsobm_noisy,
    # batch_size=1,
    # shuffle=False,
    # pin_memory=True,
    # drop_last=False)
    
    dataloader_dcctn_pl_sisnr = torch.utils.data.DataLoader(
        dataset_dcctn_pl_sisnr,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    
    # dataloader_dcctn_pl_sisdr = torch.utils.data.DataLoader(
    #     dataset_dcctn_pl_sisdr,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False)
    
    # dataloader_dcctn_sisdr = torch.utils.data.DataLoader(
    #     dataset_dcctn_sisdr,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False)
    
    dataloader_dcctn_snr = torch.utils.data.DataLoader(
        dataset_dcctn_snr,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    
    # dataloader_dcctn_pl_sisdr = torch.utils.data.DataLoader(
    #     dataset_dcctn_pl_sisdr,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False)
    
    dataloader_bitas = torch.utils.data.DataLoader(
        dataset_bitas,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    
    # dataloader_bsobm = torch.utils.data.DataLoader(
    #     dataset_bsobm,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False)
    
    # dataloader_dcctn_mmse = torch.utils.data.DataLoader(
    #     dataset_dcctn_mmse,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False)
    
 
    # dataloader = iter(dataloader)
    # dataloader_bsobm_noisy = iter(dataloader_bsobm_noisy)
    # dataloader_dcctn_pl_sisdr = iter(dataloader_dcctn_pl_sisdr)
    dataloader_dcctn_pl_sisnr = iter(dataloader_dcctn_pl_sisnr)
    # dataloader_bsobm = iter(dataloader_bsobm)
    # dataloader_dcctn_sisdr =  iter(dataloader_dcctn_sisdr)
    dataloader_dcctn_snr =  iter(dataloader_dcctn_snr)
    dataloader_bitas=  iter(dataloader_bitas)
    # dataloader_dcctn_mmse = iter(dataloader_dcctn_mmse)
    
    # mie_dcctn_pl_sisdr = torch.zeros((375,fbins))
    mie_dcctn_pl_sisnr = torch.zeros((375,fbins))
    mie_dcctn_sisdr = torch.zeros((375,fbins))
    mie_dcctn_snr = torch.zeros((375,fbins))
    mie_dcctn_mmse = torch.zeros((375,fbins))
    mie_bitas= torch.zeros((375,fbins))
    mie_bsobm = torch.zeros((375,fbins))
    
    mpe_dcctn_pl_sisnr = torch.zeros((375,fbins_ipd))
    mpe_dcctn_sisdr = torch.zeros((375,fbins_ipd))
    mpe_dcctn_snr = torch.zeros((375,fbins_ipd))
    mpe_dcctn_mmse = torch.zeros((375,fbins_ipd))
    mpe_bitas= torch.zeros((375,fbins_ipd))
    mpe_bsobm = torch.zeros((375,fbins_ipd))
    
    
    
    for j in range(len(dataloader_dcctn_pl_sisnr)):#():
        try:
            # batch = next(dataloader)
            # batch_bsobm_noisy = next(dataloader_bsobm_noisy)
            # batch_dcctn_pl_sisdr = next(dataloader_dcctn_pl_sisdr)
            batch_dcctn_pl_sisnr = next(dataloader_dcctn_pl_sisnr)
            # batch_dcctn_sisdr = next(dataloader_dcctn_sisdr)
            # batch_bsobm = next(dataloader_bsobm)
            batch_bitas = next(dataloader_bitas)
            batch_dcctn_snr = next(dataloader_dcctn_snr)
            # batch_dcctn_mmse = next(dataloader_dcctn_mmse)
            
        except StopIteration:
            break
        
    
        # noisy_samples = (batch[0])[0]
        # noisy_bsobm = (batch_bsobm_noisy[0])[0]
        clean_samples = (batch_dcctn_pl_sisnr[1])[0]
        # clean_bsobm = (batch_bsobm[1])[0]
        # dcctn_pl_sisdr = (batch_dcctn_pl_sisdr[0])[0]
        dcctn_pl_sisnr = (batch_dcctn_pl_sisnr[0])[0]
        # dcctn_sisdr = (batch_dcctn_sisdr[0][0])
        dcctn_snr = (batch_dcctn_snr[0])[0]
        # dcctn_mmse = (batch_dcctn_mmse[0])[0]
        bitas = (batch_bitas[0])[0]
        # bsobm = (batch_bsobm[0])[0]
        
        clean_samples=(clean_samples)/(torch.max(clean_samples))
        
        target_stft_l = stft(clean_samples[0, :])
        target_stft_r = stft(clean_samples[1, :])
        
        # bsobm_target_stft_l = stft(clean_bsobm[0, :])
        # bsobm_target_stft_r = stft(clean_bsobm[1, :])
        
        # dcctn_pl_sisdr_stft_l = stft(dcctn_pl_sisdr[0,:])
        # dcctn_pl_sisdr_stft_r = stft(dcctn_pl_sisdr[1,:])
        
        dcctn_pl_sisnr_stft_l = stft(dcctn_pl_sisnr[0,:])
        dcctn_pl_sisnr_stft_r = stft(dcctn_pl_sisnr[1,:])
        
        # dcctn_sisdr_stft_l = stft(dcctn_sisdr[0,:])
        # dcctn_sisdr_stft_r = stft(dcctn_sisdr[1,:])
        
        dcctn_snr_stft_l = stft(dcctn_snr[0,:])
        dcctn_snr_stft_r = stft(dcctn_snr[1,:])
        
        # dcctn_mmse_stft_l = stft(dcctn_mmse[0,:])
        # dcctn_mmse_stft_r = stft(dcctn_mmse[1,:])
        
        bitas_stft_l = stft(bitas[0,:])
        bitas_stft_r = stft(bitas[1,:])
        
        # bsobm_stft_l = stft(bsobm[0,:])
        # bsobm_stft_r = stft(bsobm[1,:])
        
        mask = speechMask(target_stft_l,target_stft_r, threshold=15).squeeze(0)
        # mask_sobm = speechMask(bsobm_target_stft_l,bsobm_target_stft_r, threshold=10).squeeze(0)
        # test=gcc_phat_itd(target_stft_l, target_stft_r, target_stft_l,target_stft_r)
        # breakpoint()n
        
        target_ild_16k = ild_db(target_stft_l.abs(), target_stft_r.abs())
        # target_ild_10k = ild_db(bsobm_target_stft_l.abs(), bsobm_target_stft_r.abs())
        
        # dcctn_pl_sisdr_ild = ild_db(dcctn_pl_sisdr_stft_l.abs(), dcctn_pl_sisdr_stft_r.abs())
        dcctn_pl_sisnr_ild = ild_db(dcctn_pl_sisnr_stft_l.abs(), dcctn_pl_sisnr_stft_r.abs())
        # dcctn_sisdr_ild = ild_db(dcctn_sisdr_stft_l.abs(), dcctn_sisdr_stft_r.abs())
        dcctn_snr_ild = ild_db(dcctn_snr_stft_l.abs(), dcctn_snr_stft_r.abs())
        bitas_ild = ild_db(bitas_stft_l.abs(), bitas_stft_r.abs())
        # bsobm_ild = ild_db(bsobm_stft_l.abs(), bsobm_stft_r.abs())
        # dcctn_mmse_ild = ild_db(dcctn_mmse_stft_l.abs(), dcctn_mmse_stft_r.abs())
        
        
        # dcctn_pl_sisdr_ild_error = (target_ild_16k - dcctn_pl_sisdr_ild).abs()
        dcctn_pl_sisnr_ild_error = (target_ild_16k - dcctn_pl_sisnr_ild).abs()
        # dcctn_sisdr_ild_error = (target_ild_16k - dcctn_sisdr_ild).abs()
        dcctn_snr_ild_error = (target_ild_16k - dcctn_snr_ild).abs()
        bitas_ild_error = (target_ild_16k - bitas_ild).abs()
        # bsobm_ild_error = (target_ild_10k - bsobm_ild).abs()
        # dcctn_mmse_ild_error = (target_ild_16k - dcctn_mmse_ild).abs()
        
        # breakpoint()
        mask = speechMask(target_stft_l,target_stft_r,threshold=15).squeeze()
        # breakpoint()
        mask = mask[50:,:]
        mask_sum=mask.sum(dim=1)
        mask_sum[mask_sum==0]=1
        
        dcctn_pl_sisnr_ild_error = dcctn_pl_sisnr_ild_error[50:,:]
        dcctn_snr_ild_error = dcctn_snr_ild_error[50:,:]
        bitas_ild_error = bitas_ild_error[50:,:]
        
        #  [:,50:,:]
        
        # mask_sum_sobm=mask_sobm.sum(dim=1)
        # mask_sum_sobm[mask_sum_sobm==0]=1
        
        
        # mie_dcctn_pl_sisdr[j,:] = (dcctn_pl_sisdr_ild_error*mask).sum(dim=1)/mask_sum
        mie_dcctn_pl_sisnr[j,:] = (dcctn_pl_sisnr_ild_error*mask).sum(dim=1)/mask_sum
        # mie_dcctn_sisdr[j,:] = (dcctn_sisdr_ild_error*mask).sum(dim=1)/mask_sum
        mie_dcctn_snr[j,:] = (dcctn_snr_ild_error*mask).sum(dim=1)/mask_sum
        mie_bitas[j,:] = (bitas_ild_error*mask).sum(dim=1)/mask_sum
        # mie_bsobm[j,:] = (bsobm_ild_error*mask_sobm).sum(dim=1)/mask_sum_sobm
        # mie_dcctn_mmse[j,:] = (dcctn_mmse_ild_error*mask).sum(dim=1)/mask_sum
        
        
        target_ipd_16k = ipd_rad(target_stft_l, target_stft_r)
        # target_ipd_10k = ipd_rad(bsobm_target_stft_l, bsobm_target_stft_r)
        
        # dcctn_pl_sisdr_ild = ipd_rad(dcctn_pl_sisdr_stft_l.abs(), dcctn_pl_sisdr_stft_r.abs())
        dcctn_pl_sisnr_ipd = ipd_rad(dcctn_pl_sisnr_stft_l, dcctn_pl_sisnr_stft_r)
        # dcctn_sisdr_ild = ipd_rad(dcctn_sisdr_stft_l.abs(), dcctn_sisdr_stft_r.abs())
        dcctn_snr_ipd = ipd_rad(dcctn_snr_stft_l, dcctn_snr_stft_r)
        bitas_ipd = ipd_rad(bitas_stft_l, bitas_stft_r)
        # bsobm_ipd = ipd_rad(bsobm_stft_l, bsobm_stft_r)
        # dcctn_mmse_ipd = ipd_rad(dcctn_mmse_stft_l, dcctn_mmse_stft_r)
        # breakpoint()
        
        # dcctn_pl_sisdr_ild_error = (target_ild_16k - dcctn_pl_sisdr_ild).abs()
        
        dcctn_pl_sisnr_ipd_error = (target_ipd_16k - dcctn_pl_sisnr_ipd)
        dcctn_pl_sisnr_ipd_error = (torch.remainder(dcctn_pl_sisnr_ipd_error + torch.pi, 2 * torch.pi) - torch.pi).abs()*180/torch.pi
        # dcctn_sisdr_ild_error = (target_ild_16k - dcctn_sisdr_ild).abs()
        dcctn_snr_ipd_error = (target_ipd_16k - dcctn_snr_ipd)
        dcctn_snr_ipd_error = (torch.remainder(dcctn_snr_ipd_error + torch.pi, 2 * torch.pi) - torch.pi).abs()*180/torch.pi
        bitas_ipd_error = (target_ipd_16k - bitas_ipd).abs()*180/np.pi
        # bsobm_ipd_error = (target_ipd_10k - bsobm_ipd).abs()*180/np.pi
        # dcctn_mmse_ipd_error = (target_ipd_16k - dcctn_mmse_ipd).abs()*180/np.pi
        
        # breakpoint()
        mask = speechMask(target_stft_l,target_stft_r,threshold=15).squeeze()
        # breakpoint()
        mask = mask[:fbins_ipd,:]
        mask_sum=mask.sum(dim=1)
        mask_sum[mask_sum==0]=1
        
        dcctn_pl_sisnr_ipd_error = dcctn_pl_sisnr_ipd_error[:fbins_ipd,:]
        dcctn_snr_ipd_error = dcctn_snr_ipd_error[:fbins_ipd,:]
        bitas_ipd_error = bitas_ipd_error[:fbins_ipd,:]
        mask_sum=mask.sum(dim=1)
        mask_sum[mask_sum==0]=1
        
        # mask_sum_sobm=mask_sobm.sum(dim=1)
        # mask_sum_sobm[mask_sum_sobm==0]=1
        
        # mie_dcctn_pl_sisdr[j,:] = (dcctn_pl_sisdr_ild_error*mask).sum(dim=1)/mask_sum
        mpe_dcctn_pl_sisnr[j,:] = (dcctn_pl_sisnr_ipd_error*mask).sum(dim=1)/mask_sum
        # mie_dcctn_sisdr[j,:] = (dcctn_sisdr_ild_error*mask).sum(dim=1)/mask_sum
        mpe_dcctn_snr[j,:] = (dcctn_snr_ipd_error*mask).sum(dim=1)/mask_sum
        mpe_bitas[j,:] = (bitas_ipd_error*mask).sum(dim=1)/mask_sum
        # mpe_bsobm[j,:] = (bsobm_ipd_error*mask_sobm).sum(dim=1)/mask_sum_sobm
        # mpe_dcctn_mmse[j,:] = (dcctn_mmse_ipd_error*mask).sum(dim=1)/mask_sum
        
        print('Processed Signal ', j+1 , ' of ', len(dataloader_dcctn_pl_sisnr))
        
    # writeMatFile(mie_dcctn_pl_sisdr, folPath=SNRfolders[i], method='pl_sisdr_'+ SNRnames[i])
    writeMatFile(mie_dcctn_pl_sisnr, folPath=SNRfolders[i], method='bccrn_pl_'+ SNRnames[i])  
    # writeMatFile(mie_dcctn_sisdr, folPath=SNRfolders[i], method='sisdr_'+ SNRnam8es[i])
    writeMatFile(mie_dcctn_snr, folPath=SNRfolders[i], method='bccrn_snr_'+ SNRnames[i])
    writeMatFile(mie_bitas, folPath=SNRfolders[i], method='bitas_'+ SNRnames[i] )
    # writeMatFile(mie_bsobm, folPath=SNRfolders[i], method='bsobm_'+ SNRnames[i])
    # writeMatFile(mie_dcctn_mmse, folPath=SNRfolders[i], method='mmse_'+ SNRnames[i])
    
        
    # writeMatFile(mie_dcctn_pl_sisdr, folPath=SNRfolders[i], method='pl_sisdr_'+ SNRnames[i])
    writeMatFileIPD(mpe_dcctn_pl_sisnr, folPath=SNRfolders[i], method='bccrn_pl_'+ SNRnames[i])  
    # writeMatFile(mie_dcctn_sisdr, folPath=SNRfolders[i], method='sisdr_'+ SNRnam8es[i])
    writeMatFileIPD(mpe_dcctn_snr, folPath=SNRfolders[i], method='bccrn_snr_'+ SNRnames[i])
    writeMatFileIPD(mpe_bitas, folPath=SNRfolders[i], method='bitas_'+ SNRnames[i])
    # writeMatFileIPD(mpe_bsobm, folPath=SNRfolders[i], method='bsobm_'+ SNRnames[i])
    # writeMatFileIPD(mpe_dcctn_mmse, folPath=SNRfolders[i], method='mmse_'+ SNRnames[i])
        
        
        
        