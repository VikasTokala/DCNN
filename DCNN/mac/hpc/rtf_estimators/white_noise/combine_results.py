import numpy as np
import matplotlib.pyplot as plt
import os
home = os.path.expanduser('~')

def read_mse_files(folder, Nch, N_files):

    mse_neg6dB = np.zeros((Nch-1, N_files))
    mse_0dB = np.zeros((Nch-1, N_files))
    mse_6dB = np.zeros((Nch-1, N_files))
    mse_12dB = np.zeros((Nch-1, N_files))

    f = open(folder+'white_-6db_SNR_mse.txt', "r")
    for i,l in enumerate(f):
        l = l.split(', ')
        mse_neg6dB[:, i] = np.array(l[1:])
    f.close()

    f = open(folder+'white_0db_SNR_mse.txt', "r")
    for i,l in enumerate(f):
        l = l.split(', ')
        mse_0dB[:, i] = np.array(l[1:])
    f.close()

    f = open(folder+'white_6db_SNR_mse.txt', "r")
    for i,l in enumerate(f):
        l = l.split(', ')
        mse_6dB[:, i] = np.array(l[1:])
    f.close()

    f = open(folder+'white_12db_SNR_mse.txt', "r")
    for i,l in enumerate(f):
        l = l.split(', ')
        mse_12dB[:, i] = np.array(l[1:])
    f.close()

    mse_neg6dB_file_ave = np.mean(mse_neg6dB, axis=-1)
    mse_0dB_file_ave = np.mean(mse_0dB, axis=-1)
    mse_6dB_file_ave = np.mean(mse_6dB, axis=-1)
    mse_12dB_file_ave = np.mean(mse_12dB, axis=-1)

    mse_neg6dB_ch_ave = np.mean(mse_neg6dB_file_ave, axis=-1)
    mse_0dB_ch_ave = np.mean(mse_0dB_file_ave, axis=-1)
    mse_6dB_ch_ave = np.mean(mse_6dB_file_ave, axis=-1)
    mse_12dB_ch_ave = np.mean(mse_12dB_file_ave, axis=-1)

    return mse_neg6dB, mse_0dB, mse_6dB, mse_12dB, mse_neg6dB_file_ave, mse_0dB_file_ave, mse_6dB_file_ave, mse_12dB_file_ave, mse_neg6dB_ch_ave, mse_0dB_ch_ave, mse_6dB_ch_ave, mse_12dB_ch_ave 

if __name__ == '__main__':

    rtf_est_folder = home + '/mac/hpc/rtf_estimators/'

    fblms_folder = rtf_est_folder + 'fblms/results/'
    ipnlms_folder = rtf_est_folder + 'ipnlms/results/'
    gevd_folder = rtf_est_folder + 'gevd/results/'
    pm_gevd_folder = rtf_est_folder + 'pm_gevd/results/'
    pm_pevd_folder = rtf_est_folder + 'pm_pevd/results/'

    snr_values = [-6, 0, 6, 12]

    N_files = 200
    Nch = 8

    enc_ch = [1, 2, 3, 5, 6, 7, 8] # no ref channel :)

    fblms_neg6dB, fblms_0dB, fblms_6dB, fblms_12dB, fblms_neg6dB_file_ave, fblms_0dB_file_ave, fblms_6dB_file_ave, fblms_12dB_file_ave, fblms_neg6dB_ch_ave, fblms_0dB_ch_ave, fblms_6dB_ch_ave, fblms_12dB_ch_ave  = read_mse_files(fblms_folder, Nch, N_files)
    ipnlms_neg6dB, ipnlms_0dB, ipnlms_6dB, ipnlms_12dB, ipnlms_neg6dB_file_ave, ipnlms_0dB_file_ave, ipnlms_6dB_file_ave, ipnlms_12dB_file_ave, ipnlms_neg6dB_ch_ave, ipnlms_0dB_ch_ave, ipnlms_6dB_ch_ave, ipnlms_12dB_ch_ave  = read_mse_files(ipnlms_folder, Nch, N_files)
    gevd_neg6dB, gevd_0dB, gevd_6dB, gevd_12dB, gevd_neg6dB_file_ave, gevd_0dB_file_ave, gevd_6dB_file_ave, gevd_12dB_file_ave, gevd_neg6dB_ch_ave, gevd_0dB_ch_ave, gevd_6dB_ch_ave, gevd_12dB_ch_ave  = read_mse_files(gevd_folder, Nch, N_files)
    pm_gevd_neg6dB, pm_gevd_0dB, pm_gevd_6dB, pm_gevd_12dB, pm_gevd_neg6dB_file_ave, pm_gevd_0dB_file_ave, pm_gevd_6dB_file_ave, pm_gevd_12dB_file_ave, pm_gevd_neg6dB_ch_ave, pm_gevd_0dB_ch_ave, pm_gevd_6dB_ch_ave, pm_gevd_12dB_ch_ave  = read_mse_files(pm_gevd_folder, Nch, N_files)
    pm_pevd_neg6dB, pm_pevd_0dB, pm_pevd_6dB, pm_pevd_12dB, pm_pevd_neg6dB_file_ave, pm_pevd_0dB_file_ave, pm_pevd_6dB_file_ave, pm_pevd_12dB_file_ave, pm_pevd_neg6dB_ch_ave, pm_pevd_0dB_ch_ave, pm_pevd_6dB_ch_ave, pm_pevd_12dB_ch_ave  = read_mse_files(pm_pevd_folder, Nch, N_files)

    plt.figure()
    plt.plot(enc_ch, fblms_0dB_file_ave, label='fblms')
    plt.plot(enc_ch, ipnlms_0dB_file_ave, label='ipnlms')
    plt.plot(enc_ch, gevd_0dB_file_ave, label='gevd')
    plt.plot(enc_ch, pm_gevd_0dB_file_ave, label='pm_gevd')
    plt.plot(enc_ch, pm_pevd_0dB_file_ave, label='pm_pevd')
    plt.title('0dB SNR rtf estimation reconstruction error')
    plt.xlabel('channel')
    plt.ylabel('mse [dB]')
    plt.legend()
    plt.savefig(rtf_est_folder + 'results_combined_0dB.png')

    plt.figure()
    plt.plot(enc_ch, fblms_6dB_file_ave, label='fblms')
    plt.plot(enc_ch, ipnlms_6dB_file_ave, label='ipnlms')
    plt.plot(enc_ch, gevd_6dB_file_ave, label='gevd')
    plt.plot(enc_ch, pm_gevd_6dB_file_ave, label='pm_gevd')
    plt.plot(enc_ch, pm_pevd_6dB_file_ave, label='pm_pevd')
    plt.title('6dB SNR rtf estimation reconstruction error')
    plt.xlabel('channel')
    plt.ylabel('mse [dB]')
    plt.legend()
    plt.savefig(rtf_est_folder + 'results_combined_6dB.png')

    plt.figure()
    plt.plot(enc_ch, fblms_12dB_file_ave, label='fblms')
    plt.plot(enc_ch, ipnlms_12dB_file_ave, label='ipnlms')
    plt.plot(enc_ch, gevd_12dB_file_ave, label='gevd')
    plt.plot(enc_ch, pm_gevd_12dB_file_ave, label='pm_gevd')
    plt.plot(enc_ch, pm_pevd_12dB_file_ave, label='pm_pevd')
    plt.title('12dB SNR rtf estimation reconstruction error')
    plt.xlabel('channel')
    plt.ylabel('mse [dB]')
    plt.legend()
    plt.savefig(rtf_est_folder + 'results_combined_12dB.png')

    plt.figure()
    plt.plot(enc_ch, fblms_neg6dB_file_ave, label='fblms')
    plt.plot(enc_ch, ipnlms_neg6dB_file_ave, label='ipnlms')
    plt.plot(enc_ch, gevd_neg6dB_file_ave, label='gevd')
    plt.plot(enc_ch, pm_gevd_neg6dB_file_ave, label='pm_gevd')
    plt.plot(enc_ch, pm_pevd_neg6dB_file_ave, label='pm_pevd')
    plt.title('-6dB SNR rtf estimation reconstruction error')
    plt.xlabel('channel')
    plt.ylabel('mse [dB]')
    plt.legend()
    plt.savefig(rtf_est_folder + 'results_combined_-6dB.png')

    plt.show()

    pass
