import matplotlib.pyplot as plt 
import numpy as np
import os
import scipy.signal as sig 
home = os.path.expanduser('~')
import sys
sys.path.append("/Users/vtokala/Documents/Research")
import mac

fs = 16000

def test_ipnlms_rtf(sm, ref_ch=1):
    
    print('\ntesting ipnlms based rtf estimation')

    L_rtf = sm.N_rtf_t
    alpha = 0
    mu = 1

    L, Nch = np.shape(sm.y)
    ref_ind = ref_ch - 1
    encoded_channels = list(range(Nch))
    encoded_channels.remove(ref_ind)

    rtf, e, d_hat = mac.est.ipnlms_est(sm.y, ref_ch, alpha, mu, L_rtf=L_rtf, L_frame=L_rtf, caus_delay=0)

    mse = mac.eval.mse_db(sm.y[:,encoded_channels], d_hat[:,encoded_channels])
    print('mse between y and y_hat = ' + str(mse) + ' dB')
    print('Signal shape - ', sm.y.shape)
    print('RTF shape - ',rtf.shape)

    return rtf, d_hat, mse

def test_fblms_rtf(sm, ref_ch=1):

    print('\ntesting fblms rtf estimation')

    ref_ind = ref_ch - 1
    encoded_channels = list(range(sm.Nch))
    encoded_channels.remove(ref_ind)

    M = sm.N_rtf_t #length of the rtf

    rtf_f, e, y_hat = mac.est.fblms_est(sm.y, ref_ch, M)
    _,_,N_segs = np.shape(rtf_f)

    mse = mac.eval.mse_db(sm.y[:,encoded_channels], y_hat[:,encoded_channels])
    print('mse between y and y_hat = ' + str(mse) + ' dB')
    print(sm.y.shape)
    print(encoded_channels)

    rtf_td = np.zeros((M, sm.Nch, N_segs))
    for n in range(N_segs):
       rtf_td[:,:,n] = np.real(np.fft.ifft(rtf_f[:,:,n], axis=0))[:M,:]

    print(rtf_td.shape)
    return rtf_td, y_hat, mse

def test_gevd_rtf(sm, ref_ch=1):

    print('\ntesting subspace rtf estimation (Marcovich 2009)')
   
    ref_ind = ref_ch - 1
    encoded_channels = list(range(sm.Nch))
    encoded_channels.remove(ref_ind)

    # Signal model setup and inits 
    N_rtf = sm.len_w
    
    _, Nch, N_freq, N_segs = np.shape(sm.Py_rec)
    c=0
    d=0
    rtf = np.zeros((N_freq, Nch, N_segs), 'complex')

    # RTF estimation 
    for f in range(1,N_freq):
        for n in range(N_segs):
            try:
                rtf[f, :, n] = mac.est.gevd_rtf(sm.Py_rec[:,:,f,n], sm.Pv_rec[:,:,f,n], ref_ch)
            except:
                try:
                    c+=1
                    Pv_rec_reg = sm.Pv_rec[:,:,f,n] + 0.01*np.min(sm.Pv_rec[:,:,f,n])*np.eye(Nch,Nch) 
                    rtf[f,:,n] = mac.est.gevd_rtf(sm.Py_rec[:,:,f,n], Pv_rec_reg, ref_ch)
                except:
                    d+=1 
                    rtf[f,:,n] = rtf[f,:,n-1]
    if c > 0:
        print(str(c) + ' rtf tf-bins required regularisation\n' + str(d) + ' were ignored after regularisation')
    
    f = np.linspace(0,fs/2,N_freq+1)
    t = np.arange(0,N_segs+1) 

    Y_ref = np.zeros_like(sm.Y)
    Y_oracle = np.zeros_like(sm.Y)
    
    for ch in range(Nch):
        Y_ref[:,ch,:] = sm.Y[:,ref_ind,:] 

    Y_filt = Y_ref*rtf
    y_filt = (sig.istft(Y_filt, fs=fs, time_axis=-1, freq_axis=0, noverlap=sm.overlap)[1]).T

    mse = mac.eval.mse_db(sm.y[:,encoded_channels], y_filt[:,encoded_channels])
    print('mse y vs y_filt = ' + str(mse))


    rtf_td = np.zeros((N_rtf, Nch, N_segs))
    for n in range(N_segs):
        rtf_td[:,:,n] = np.real(np.fft.irfft(rtf[:,:,n], axis=0))

    return rtf_td, y_filt, mse

def test_pm_gevd_rtf(sm, ref_ch=1):

    print('\ntesting pm gevd rtf estimation (varzandeh 2017)')
   
    # Signal model setup and inits 
    ref_ind = ref_ch - 1
    encoded_channels = list(range(sm.Nch))
    encoded_channels.remove(ref_ind)
    N_rtf = sm.len_w

    _, Nch, N_freq, N_segs = np.shape(sm.Py_rec)
    
    rtf = np.zeros((N_freq, Nch, N_segs), 'complex') 

    # RTF estimation 
    for f in range(1,N_freq):
        u_k = np.ones((Nch, 1))
        for n in range(N_segs):
            rtf[f, :, n], u_k = mac.est.pm_gevd_rtf(sm.Py_rec[:,:,f,n], sm.Pv_rec[:,:,f,n], u_k, ref_ch)
    f = np.linspace(0,fs/2,N_freq+1)

    Y_ref = np.zeros_like(sm.Y)
    Y_oracle = np.zeros_like(sm.Y)
    
    for ch in range(Nch):
        Y_ref[:,ch,:] = sm.Y[:,ref_ind,:] 

    Y_filt = Y_ref*rtf
    y_filt = (sig.istft(Y_filt, fs=fs, time_axis=-1, freq_axis=0, noverlap=sm.overlap)[1]).T


    mse = mac.eval.mse_db(sm.y[:,encoded_channels], y_filt[:,encoded_channels])
    print('mse y vs y_filt = ' + str(mse))

    rtf_td = np.zeros((N_rtf, Nch, N_segs))
    for n in range(N_segs):
        rtf_td[:,:,n] = np.real(np.fft.irfft(rtf[:,:,n], axis=0))

    return rtf_td, y_filt, mse

def test_pm_pevd_rtf(sm, ref_ch=1):

    print('\ntesting pevd rtf estimation')

    # Signal model setup and inits 
     
    ref_ind = ref_ch - 1
    encoded_channels = list(range(sm.Nch))
    encoded_channels.remove(ref_ind)

    _, Nch, L_seg, N_segs = np.shape(sm.RYY_rec)

    rtf_td = np.zeros((L_seg, Nch, N_segs))
    rtf = np.zeros((L_seg//2 + 1, Nch, N_segs), 'complex')

    for s in range(N_segs):
        rtf[:,:,s], rtf_td[:,:,s] = mac.est.pm_pevd_rtf(sm.RYY_rec[:,:,:,s], ref_ch=ref_ch, N_iter=1)

    Y_ref = np.zeros_like(sm.Y)
    for ch in range(Nch):
        Y_ref[:,ch,:] = sm.Y[:,ref_ind,:]

    Y_filt = Y_ref*rtf[0::2, :, :]
    y_filt = (sig.istft(Y_filt, fs=fs, time_axis=-1, freq_axis=0, noverlap=sm.overlap)[1]).T

    mse = mac.eval.mse_db(sm.y[:,encoded_channels], y_filt[:,encoded_channels])
    print('mse y vs y_filt = ' + str(mse))

    return rtf_td, y_filt, mse

def test_pm_pgevd_rtf(sm, ref_ch=1):

    print('\ntesting pgevd rtf estimation')

    # Signal model setup and inits  
    ref_ind = ref_ch - 1
    encoded_channels = list(range(sm.Nch))
    encoded_channels.remove(ref_ind)

    _, Nch, L_seg, N_segs = np.shape(sm.RYY_rec)

    rtf_td = np.zeros((L_seg, Nch, N_segs))
    rtf = np.zeros((L_seg//2 + 1, Nch, N_segs), 'complex')

    A = sm.RYY_rec - sm.RVV_rec

    for s in range(N_segs):
        rtf[:,:,s], rtf_td[:,:,s] = mac.est.pm_pevd_rtf(A[:,:,:,s], ref_ch=ref_ch, N_iter=1)

    Y_ref = np.zeros_like(sm.Y)
    for ch in range(Nch):
        Y_ref[:,ch,:] = sm.Y[:,ref_ind,:]

    Y_filt = Y_ref*rtf[0::2, :, :]
    y_filt = (sig.istft(Y_filt, fs=fs, time_axis=-1, freq_axis=0, noverlap=sm.overlap)[1]).T

    mse = mac.eval.mse_db(sm.y[:,encoded_channels], y_filt[:,encoded_channels])
    print('mse y vs y_filt = ' + str(mse))

    return rtf_td, y_filt, mse

def sim_test_main():
    
    N_rtf_t = 1024
    
    ir_type = ['imp_imp', 
               'imp_half_sin',
               'half_sin_half_sin']

    # ir_type = ['imp_imp']

    for i, ir in enumerate(ir_type):

        sm = mac.sm.TestSignalModel(ir_type=ir, Ts=0.5, len_w=1024, overlap=None, SNR_db=6)

        rtf_fblms, y_hat_fblms, mse_fblms = test_ipnlms_rtf(sm)
        # rtf_gevd , y_hat_gevd, mse_gevd = test_gevd_rtf(sm)
        # rtf_pm_gevd , y_hat_pm_gevd, mse_pm_gevd = test_pm_gevd_rtf(sm)
        # rtf_pevd, y_hat_pevd, mse_pevd = test_pm_pevd_rtf(sm)    
        # rtf_ipnlms , y_hat_ipnlms, mse_ipnlms = test_ipnlms_rtf(sm)
        
        
        # mse_list = np.squeeze(np.array([mse_gevd, mse_pm_gevd, mse_pevd, mse_ipnlms, mse_fblms]))
        # label_list = np.array(['gevd','pm-gevd','pevd','ipnlms','fblms'])
        
        # plt.figure()
        # plt.bar(label_list, mse_list)
        # plt.title(ir)
        # plt.legend()

        plt.figure()
        plt.plot(rtf_fblms[:,1,-1])
        # plt.plot(rtf_ipnlms[:,1,-1])
        # plt.plot(rtf_pevd[:,1,-1])
        # plt.plot(rtf_pm_gevd[:,1,-1])
        # plt.plot(rtf_gevd[:,1,-1])
        #export PYTHONPATH="${PYTHONPATH}:/Users/vtokala/Documents/Research"

    plt.show()
    pass

def nuance_test_main():
    
    N_rtf_t = 1024
    Nch = 8
    ref_ch = 4
    SNR_db = 0
    sig_path = "/Users/dj/signals/LSN/RealDictation/ULA8Thin_Playback_D1/sig/121-121726-0000.wav"
    noise_type = 'white'

    ref_ind = ref_ch - 1
    encoded_channels = list(range(Nch))
    encoded_channels.remove(ref_ind)

    sm = mac.sm.NuanceSignalModel(sig_path, noise_type, len_w=1024, Ts=0.5, overlap=None, SNR_db=SNR_db)
    # _, _, pm_pevd_mse = test_pm_pevd_rtf(sm, ref_ch=ref_ch)
    # # _, _, pm_pgevd_mse = test_pm_pgevd_rtf(sm, ref_ch=ref_ch)
    # _, _, ipnlms_mse = test_ipnlms_rtf(sm, ref_ch=ref_ch)
    # _, _, pm_gevd_mse = test_pm_gevd_rtf(sm, ref_ch=ref_ch)
    # _, _, gevd_mse = test_gevd_rtf(sm, ref_ch=ref_ch)
    _, _, fblms_mse = test_fblms_rtf(sm, ref_ch=ref_ch)

    plt.figure()
    # plt.plot(encoded_channels, pm_pevd_mse, '-x', label='pm_pevd')
    # # plt.plot(encoded_channels, pm_pgevd_mse, '-x', label='pm_pgevd_mse')
    # plt.plot(encoded_channels, pm_gevd_mse, '-x', label='pm_gevd_mse')
    # plt.plot(encoded_channels, gevd_mse, '-x', label='gevd_mse')
    # plt.plot(encoded_channels, ipnlms_mse, '-x', label='ipnlms_mse')
    plt.plot(encoded_channels, fblms_mse, '-x', label='fblms_mse')

    plt.legend()   
    plt.xlabel('channels')
    plt.ylabel('mse [dB]')
    plt.title('Reconstruction error at ' + str(SNR_db) + 'dB SNR')
    # plt.savefig(home + '/mac/exp/recon/' + str(SNR_db) + 'dB_SNR.png')
    plt.show()

if __name__ == '__main__':

    sim_test_main()
    # nuance_test_main()