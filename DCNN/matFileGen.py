from scipy.io import savemat
import torch
import os










SAVE_PATH = '/kaggle/working/SE_DCNN/DCNN/MatFiles/'


def writeMatFile(noisy_snr_l, noisy_snr_r, enhanced_snr_r, enhanced_snr_l, masked_ild_error, masked_ipd_error, improved_mbstoi, improved_snr_l,improved_snr_r, improved_stoi_l,improved_stoi_r,folPath = 'General'):
    
   
    savemat(os.path.join(SAVE_PATH,folPath,'noisy_snr_l.mat'),{'noisy_snr_l':noisy_snr_l.numpy()})
    savemat(os.path.join(SAVE_PATH,folPath,'noisy_snr_r.mat'),{'noisy_snr_l':noisy_snr_r.numpy()})
    
    savemat(os.path.join(SAVE_PATH,folPath,'enhanced_snr_l.mat'),{'enhanced_snr_l':enhanced_snr_l.numpy()})
    savemat(os.path.join(SAVE_PATH,folPath,'enhanced_snr_r.mat'),{'enhanced_snr_r':enhanced_snr_r.numpy()})
    
    savemat(os.path.join(SAVE_PATH,folPath,'masked_ild_error.mat'),{'masked_ild_error':masked_ild_error.numpy()})
    savemat(os.path.join(SAVE_PATH,folPath,'masked_ipd_error.mat'),{'masked_ipd_error':masked_ipd_error.numpy()})
    
    savemat(os.path.join(SAVE_PATH,folPath,'improved_mbstoi.mat'),{'improved_mbstoi':improved_mbstoi.numpy()})
    
    savemat(os.path.join(SAVE_PATH,folPath,'improved_snr_l.mat'),{'improved_snr_l':improved_snr_l.numpy()})
    savemat(os.path.join(SAVE_PATH,folPath,'improved_snr_r.mat'),{'improved_snr_r':improved_snr_r.numpy()})
    
    savemat(os.path.join(SAVE_PATH,folPath,'improved_stoi_l.mat'),{'improved_stoi_l':improved_stoi_l.numpy()})
    savemat(os.path.join(SAVE_PATH,folPath,'improved_stoi_r.mat'),{'improved_stoi_r':improved_stoi_r.numpy()})
   
    print('MAT Files saved successfully!')
    
    