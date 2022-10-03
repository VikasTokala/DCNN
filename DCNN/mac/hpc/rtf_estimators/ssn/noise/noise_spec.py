import mac
import os 
import matplotlib.pyplot as plt
import numpy as np
home = os.path.expanduser('~')

fs = 16000
len_w = 1024
Nch = 8
sig_dir = home + '/signals/NUANCE_SIGNALS/all/'
ave_spec_name = home + '/mac/hpc/rtf_estimators/ssn/noise_spec.pickle'
ave_ir_name = home + '/mac/hpc/rtf_estimators/ssn/noise_ir.pickle'

if __name__ == '__main__':

    ave_spec, ave_ir = mac.util.calc_ave_spec(sig_dir, Nch, len_w, ave_spec_name, ave_ir_name)
     
    v_ssn = mac.util.gen_ltas_noise(ave_ir, 1, (10000,8))

    plt.figure()
    plt.plot(v_ssn)
    plt.savefig(home + '/mac/hpc/rtf_estimators/ssn/v.png')

    noise_spec = np.abs(np.fft.rfft(v_ssn, axis=0, n=len_w))

    plt.figure()
    plt.plot(np.abs(ave_spec[:,0]), label='ave_spec')
    plt.legend()
    plt.savefig(home + '/mac/hpc/rtf_estimators/ssn/ave_specs.png')

    plt.figure()
    plt.plot(noise_spec, '--', label='noise_spec')
    plt.legend()
    plt.savefig(home + '/mac/hpc/rtf_estimators/ssn/noise_specs.png')
