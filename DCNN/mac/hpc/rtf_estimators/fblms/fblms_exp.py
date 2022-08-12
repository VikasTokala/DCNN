import matplotlib.pyplot as plt 
import numpy as np
import os
import scipy.signal as sig 
home = os.path.expanduser('~')
import mac
import sys
from test import test_est

fs = 16000

if __name__ == '__main__':
    batch_size = int(sys.argv[1])
    sigs_batch = sys.argv[2:2+batch_size]
    SNR_db_list = [int(x) for x in sys.argv[2+batch_size :-1]]
    work_dir = sys.argv[-1]

    print(batch_size)
    print(work_dir)
    print(sigs_batch)
    print(SNR_db_list)

    Nch = 8
    ref_ch = 4
    noise_type = 'white'

    for SNR_db in SNR_db_list:

        mse_file_path = work_dir + 'results/' + noise_type + '_' + str(SNR_db) + 'db_SNR' + '_mse_yy' + '.txt'
        
        mse_file = open(mse_file_path, 'a')

        for ii, sig_path in enumerate(sigs_batch):

            sm = mac.sm.NuanceSignalModel(sig_path, noise_type=noise_type, Ts=0.5, len_w=1024, overlap=None, SNR_db=SNR_db)
            _, _, fblms_mse_yy = test_est.test_fblms_rtf(sm, ref_ch=ref_ch)

            mse_file.write(str(sig_path) + ', ' + ", ".join(map(str, fblms_mse_yy)) + '\n')
        
        mse_file.close()
