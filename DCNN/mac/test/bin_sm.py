import scipy.io as sio
import numpy as np
import h5py


class SignalModel():
    def __init__(self,Nch=2, N_rtf_t=1024):
        self.Nch = Nch
        self.N_rtf_t = N_rtf_t

        self.ytemp = sio.loadmat('/Users/vtokala/Documents/Research/di_nn/DCNN/mac/test/y.mat')
        self.gt_rtf_temp = sio.loadmat('/Users/vtokala/Documents/Research/di_nn/DCNN/mac/test/hrir.mat')

        
        self.y = self.ytemp['bcs']
        self.gt_rtf = self.gt_rtf_temp['inEar']

        print('\n Signal Model setup')