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

    print('batch size = ' + str(batch_size))
    print('sig batch = ' + str(sigs_batch))
    print('snr db list = ' + str(SNR_db_list))
    print('workdir = ' + str(work_dir))

    Nch = 8
    ref_ch = 4
    noise_type = 'ssn'
    len_w = 1024

    print('executing ipnlms_exp.py\n')

    for SNR_db in SNR_db_list:

        mse_file_path = work_dir + 'results/' + noise_type + '_' + str(SNR_db) + 'db_SNR' + '_mse_yy' + '.txt'
        
        mse_file = open(mse_file_path, 'a')

        for ii, sig_path in enumerate(sigs_batch):

            sm = mac.sm.NuanceSignalModel(sig_path, len_w, noise_type='ssn', Ts=0.5, overlap=None, SNR_db=SNR_db)
            _, _, ipnlms_mse_yy = test_est.test_ipnlms_rtf(sm, ref_ch=ref_ch)

            mse_file.write(str(sig_path) + ', ' + ", ".join(map(str, ipnlms_mse_yy)) + '\n')
        
        mse_file.close()
