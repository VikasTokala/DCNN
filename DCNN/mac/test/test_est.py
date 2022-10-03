import matplotlib.pyplot as plt 
import numpy as np
import os
import scipy.signal as sig 
home = os.path.expanduser('~')
import mac

fs = 16000

def test_ipnlms_rtf(sm, ref_ch=1, info=False):
    
    print('\ntesting ipnlms based rtf estimation')

    L_rtf = sm.N_rtf_t
    alpha = 0
    mu = 1

    L, Nch = np.shape(sm.y)
    ref_ind = ref_ch - 1
    encoded_channels = list(range(Nch))
    encoded_channels.remove(ref_ind)

    rtf, e, y_hat, x_hat = mac.est.ipnlms_est(sm.y, sm.x, ref_ch, alpha, mu, L_rtf=L_rtf, L_frame=L_rtf, caus_delay=0)

    nmse_y = mac.eval.nmse_db(sm.y[:,encoded_channels], y_hat[:,encoded_channels])
    nmse_x = mac.eval.nmse_db(sm.x[:,encoded_channels], x_hat[:,encoded_channels])

    # print('mse between y and y_hat = ' + str(nmse_y) + ' dB')

    if info == True:
        print('iplms rtf time domain shape:' + str(np.shape(rtf)))

    return rtf, y_hat, nmse_y, nmse_x

def test_fblms_rtf(sm, ref_ch=1, info=False):

    print('\ntesting fblms rtf estimation')

    ref_ind = ref_ch - 1
    encoded_channels = list(range(sm.Nch))
    encoded_channels.remove(ref_ind)

    M = sm.N_rtf_t

    rtf_f, e, y_hat = mac.est.fblms_est(sm.y, ref_ch, M)
    _,_,N_segs = np.shape(rtf_f)

    nmse_y = mac.eval.nmse_db(sm.y[:,encoded_channels], y_hat[:,encoded_channels])
    nmse_x = mac.eval.nmse_db(sm.x[:,encoded_channels], y_hat[:,encoded_channels])

    # print('mse between y and y_hat = ' + str(nmse) + ' dB')

    rtf_td = np.zeros((M, sm.Nch, N_segs))
    for n in range(N_segs):
       rtf_td[:,:,n] = np.real(np.fft.ifft(rtf_f[:,:,n], axis=0))[:M,:]

    if info == True:
        print('fblms rtf time domain shape:' + str(np.shape(rtf_td)))
        print('fblms rtf frequency domain shape:' + str(np.shape(rtf_f)))

    return rtf_td, y_hat, nmse_y, nmse_x

def test_gevd_rtf(sm, ref_ch=1, info=False):

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
    X_ref = np.zeros_like(sm.Y)
    Y_oracle = np.zeros_like(sm.Y)
    
    for ch in range(Nch):
        Y_ref[:,ch,:] = sm.Y[:,ref_ind,:] 
        X_ref[:,ch,:] = sm.X[:,ref_ind,:] 

    Y_filt = Y_ref*rtf
    X_filt = X_ref*rtf
    y_filt = (sig.istft(Y_filt, fs=fs, time_axis=-1, freq_axis=0, noverlap=sm.overlap)[1]).T
    x_filt = (sig.istft(X_filt, fs=fs, time_axis=-1, freq_axis=0, noverlap=sm.overlap)[1]).T

    nmse_y = mac.eval.nmse_db(sm.y[:,encoded_channels], y_filt[:,encoded_channels])
    nmse_x = mac.eval.nmse_db(sm.x[:,encoded_channels], x_filt[:,encoded_channels])

    # print('mse y vs y_filt = ' + str(nmse_y))


    rtf_td = np.zeros((N_rtf, Nch, N_segs))
    for n in range(N_segs):
        rtf_td[:,:,n] = np.real(np.fft.irfft(rtf[:,:,n], axis=0))

    if info == True:
        print('gevd rtf time domain shape:' + str(np.shape(rtf_td)))
        print('gevd rtf frequency domain shape:' + str(np.shape(rtf)))

    return rtf_td, y_filt, nmse_y, nmse_x

def test_pm_gevd_rtf(sm, ref_ch=1, info=False):

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
    X_ref = np.zeros_like(sm.X)
    Y_oracle = np.zeros_like(sm.Y)
    
    for ch in range(Nch):
        Y_ref[:,ch,:] = sm.Y[:,ref_ind,:]
        X_ref[:,ch,:] = sm.X[:,ref_ind,:]

    Y_filt = Y_ref*rtf
    X_filt = X_ref*rtf
    y_filt = (sig.istft(Y_filt, fs=fs, time_axis=-1, freq_axis=0, noverlap=sm.overlap)[1]).T
    x_filt = (sig.istft(X_filt, fs=fs, time_axis=-1, freq_axis=0, noverlap=sm.overlap)[1]).T


    nmse_y = mac.eval.nmse_db(sm.y[:,encoded_channels], y_filt[:,encoded_channels])
    nmse_x = mac.eval.nmse_db(sm.x[:,encoded_channels], x_filt[:,encoded_channels])

    # print('mse y vs y_filt = ' + str(nmse_y))

    rtf_td = np.zeros((N_rtf, Nch, N_segs))
    for n in range(N_segs):
        rtf_td[:,:,n] = np.real(np.fft.irfft(rtf[:,:,n], axis=0))

    if info == True:
        print('pm-gevd rtf time domain shape:' + str(np.shape(rtf_td)))
        print('pm-gevd rtf frequency domain shape:' + str(np.shape(rtf)))

    return rtf_td, y_filt, nmse_y, nmse_x

def test_pm_pevd_rtf(sm, ref_ch=1, info=False):

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
    X_ref = np.zeros_like(sm.X)
    for ch in range(Nch):
        Y_ref[:,ch,:] = sm.Y[:,ref_ind,:]
        X_ref[:,ch,:] = sm.X[:,ref_ind,:]

    Y_filt = Y_ref*rtf[0::2, :, :]
    X_filt = X_ref*rtf[0::2, :, :]
    y_filt = (sig.istft(Y_filt, fs=fs, time_axis=-1, freq_axis=0, noverlap=sm.overlap)[1]).T
    x_filt = (sig.istft(X_filt, fs=fs, time_axis=-1, freq_axis=0, noverlap=sm.overlap)[1]).T

    nmse_y = mac.eval.nmse_db(sm.y[:,encoded_channels], y_filt[:,encoded_channels])
    nmse_x = mac.eval.nmse_db(sm.x[:,encoded_channels], x_filt[:,encoded_channels])
    
    # print('mse y vs y_filt = ' + str(nmse_y))

    if info == True:
        print('pm-pevd rtf time domain shape:' + str(np.shape(rtf_td)))
        print('pm-pevd rtf frequency domain shape:' + str(np.shape(rtf)))

    return rtf_td, y_filt, nmse_y, nmse_x

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
    X_ref = np.zeros_like(sm.X)
    for ch in range(Nch):
        X_ref[:,ch,:] = sm.X[:,ref_ind,:]

    Y_filt = Y_ref*rtf[0::2, :, :]
    X_filt = X_ref*rtf[0::2, :, :]
    y_filt = (sig.istft(Y_filt, fs=fs, time_axis=-1, freq_axis=0, noverlap=sm.overlap)[1]).T
    x_filt = (sig.istft(X_filt, fs=fs, time_axis=-1, freq_axis=0, noverlap=sm.overlap)[1]).T

    nmse_y = mac.eval.nmse_db(sm.y[:,encoded_channels], y_filt[:,encoded_channels])
    nmse_x = mac.eval.nmse_db(sm.x[:,encoded_channels], x_filt[:,encoded_channels])

    # print('mse y vs y_filt = ' + str(nmse_y))

    return rtf_td, y_filt, nmse_y, nmse_x

def sim_test_main():
    
    N_rtf_t = 1024
    
    ir_type = ['imp_imp', 
               'imp_half_sin',
               'half_sin_half_sin']

    # ir_type = ['imp_imp']

    for i, ir in enumerate(ir_type):

        sm = mac.sm.TestSignalModel(ir_type=ir, Ts=0.5, len_w=1024, overlap=None, SNR_db=6)

        rtf_fblms, y_hat_fblms, mse_fblms, _ = test_fblms_rtf(sm, info=True)
        rtf_gevd , y_hat_gevd, mse_gevd, _ = test_gevd_rtf(sm, info=True)
        rtf_pm_gevd , y_hat_pm_gevd, mse_pm_gevd, _= test_pm_gevd_rtf(sm, info=True)
        rtf_pevd, y_hat_pevd, mse_pevd, _ = test_pm_pevd_rtf(sm, info=True)    
        rtf_ipnlms , y_hat_ipnlms, mse_ipnlms, _ = test_ipnlms_rtf(sm, info=True)
        
        
        mse_list = np.squeeze(np.array([mse_gevd, mse_pm_gevd, mse_pevd, mse_ipnlms, mse_fblms]))
        label_list = np.array(['gevd','pm-gevd','pevd','ipnlms','fblms'])
        
        plt.figure()
        plt.bar(label_list, mse_list)
        plt.title(ir)
        plt.legend()

        # plt.figure()
        # plt.plot(rtf_fblms[:,1,-1])
        # plt.plot(rtf_ipnlms[:,1,-1])
        # plt.plot(rtf_pevd[:,1,-1])
        # plt.plot(rtf_pm_gevd[:,1,-1])
        # plt.plot(rtf_gevd[:,1,-1])

    plt.show()
    pass

def nuance_test_main():
    
    N_rtf_t = 1024
    Nch = 8
    ref_ch = 4
    SNR_db = 0
    sig_path = "/Users/dj/signals/LSN/RealDictation/ULA8Thin_Playback_D1/sig/121-121726-0000.wav"
    noise_type = 'ssn'

    ref_ind = ref_ch - 1
    encoded_channels = list(range(Nch))
    encoded_channels.remove(ref_ind)

    sm = mac.sm.NuanceSignalModel(sig_path, len_w=1024, noise_type=noise_type, Ts=0.5, overlap=None, SNR_db=SNR_db)

    # _, _, ipnlms_nmse_y, ipnlms_nmse_x = test_ipnlms_rtf(sm, ref_ch=ref_ch)
    _, _, pm_pevd_nmse_y, pm_pevd_nmse_x = test_pm_pevd_rtf(sm, ref_ch=ref_ch)
    _, _, pm_gevd_nmse_y, pm_gevd_nmse_x = test_pm_gevd_rtf(sm, ref_ch=ref_ch)
    _, _, gevd_nmse_y, gevd_nmse_x = test_gevd_rtf(sm, ref_ch=ref_ch)
    # _, _, fblms_mse = test_fblms_rtf(sm, ref_ch=ref_ch)
    # _, _, pm_pgevd_mse = test_pm_pgevd_rtf(sm, ref_ch=ref_ch)

    plt.figure()
    # plt.plot(encoded_channels, ipnlms_nmse_x, 'r-x', label='ipnlms_mse_x')
    # plt.plot(encoded_channels, ipnlms_nmse_y, 'r--x', label='ipnlms_mse_y')
    plt.plot(encoded_channels, pm_pevd_nmse_x, 'b-x', label='pm_pevd_x')
    plt.plot(encoded_channels, pm_pevd_nmse_y, 'b--x', label='pm_pevd_y')
    plt.plot(encoded_channels, pm_gevd_nmse_x, 'k-x', label='pm_gevd_mse_x')
    plt.plot(encoded_channels, pm_gevd_nmse_y, 'k--x', label='pm_gevd_mse_y')
    plt.plot(encoded_channels, gevd_nmse_x, 'g-x', label='gevd_mse_x')
    plt.plot(encoded_channels, gevd_nmse_y, 'g--x', label='gevd_mse_y')
    # plt.plot(encoded_channels, fblms_mse, '-x', label='fblms_mse')

    # # plt.plot(encoded_channels, pm_pgevd_mse, '-x', label='pm_pgevd_mse')

    plt.legend()   
    plt.xlabel('channels')
    plt.ylabel('mse [dB]')
    plt.title('Reconstruction error at ' + str(SNR_db) + 'dB SNR')
    plt.savefig(home + '/mac/exp/recon/' + str(SNR_db) + 'dB_SNR.png')

if __name__ == '__main__':

    # sim_test_main()
    nuance_test_main()