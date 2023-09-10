from scipy.io import savemat
import torch
import os










# SAVE_PATH = '/Users/vtokala/Documents/Research/di_nn/DCNN/MatFiles'


def writeMatFile( masked_ipd_error,folPath = 'General', method = 'DCCTN'):
    
   
    # savemat(os.path.join(SAVE_PATH,folPath,'noisy_snr_l.mat'),{'noisy_snr_l':noisy_snr_l.numpy()})
    # savemat(os.path.join(SAVE_PATH,folPath,'noisy_snr_r.mat'),{'noisy_snr_l':noisy_snr_r.numpy()})
    
    # savemat(os.path.join(SAVE_PATH,folPath,'enhanced_snr_l.mat'),{'enhanced_snr_l':enhanced_snr_l.numpy()})
    # savemat(os.path.join(SAVE_PATH,folPath,'enhanced_snr_r.mat'),{'enhanced_snr_r':enhanced_snr_r.numpy()})
    filename_ipd = 'masked_ipd_error_' + method + '.mat'
    # filename_ipd = 'masked_ipd_error_' + method + '.mat'
    savemat(os.path.join(folPath,filename_ipd),{'masked_ipd_error':masked_ipd_error.numpy()})
    # savemat(os.path.join(folPath,filename_ipd),{'masked_ipd_error':masked_ipd_error.numpy()})
    
    # savemat(os.path.join(SAVE_PATH,folPath,'improved_mbstoi.mat'),{'improved_mbstoi':improved_mbstoi.numpy()})
    
    # savemat(os.path.join(SAVE_PATH,folPath,'improved_snr_l.mat'),{'improved_snr_l':improved_snr_l.numpy()})
    # savemat(os.path.join(SAVE_PATH,folPath,'improved_snr_r.mat'),{'improved_snr_r':improved_snr_r.numpy()})
    
    # savemat(os.path.join(SAVE_PATH,folPath,'improved_stoi_l.mat'),{'improved_stoi_l':improved_stoi_l.numpy()})
    # savemat(os.path.join(SAVE_PATH,folPath,'improved_stoi_r.mat'),{'improved_stoi_r':improved_stoi_r.numpy()})
    
   
    print('MAT Files saved successfully!')
    
    