import numpy as np 
import scipy.signal as sig
import scipy.linalg as linalg
import sys
import os
home = os.path.expanduser('~')
import mac
import math
import warnings
import matplotlib.pyplot as plt

def ipnlms_est(sig_in, ref_ch, alpha, mu, L_rtf, L_frame, caus_delay, vad=[], eps=2**-50):

    if vad == []:
        vad = np.ones_like(sig_in)

    L, Nch = np.shape(sig_in)
    L = L + caus_delay 
    N_frames = int(np.ceil((L)/L_frame))
    L_pad = int(N_frames*L_rtf - (L))
    L = L + L_pad

    ref_ind = ref_ch - 1
    encoded_channels = list(range(Nch))
    encoded_channels.remove(ref_ind)

    des = np.zeros((L + caus_delay, Nch))
    des = np.concatenate(( np.zeros((caus_delay, Nch)), sig_in, np.zeros((L_pad, Nch)) ))    
    
    vad = np.concatenate(( np.zeros((caus_delay, Nch)), vad, np.zeros((L_pad, Nch)) ))  

    ref = np.zeros((L, 1))
    ref = np.concatenate(( sig_in[:, ref_ind], np.zeros((caus_delay + L_pad,)) ))

    des_hat = np.zeros_like(des)
    des_hat_ave = np.zeros_like(des)

    h = np.zeros((L_rtf, Nch))
    h_ave = np.zeros((L_rtf, Nch, N_frames))

    e = np.zeros_like(des) 
    e_ave = np.zeros((L, Nch))

    ref_toeplitz = np.zeros((L_rtf, L_frame))

    start_stop_bank = np.zeros((2*N_frames,1))

    for k in range(N_frames):

        start = k*L_frame # if in doubt review python indexing
        stop = start + L_frame # if in doubt review python indexing

        epsilon = np.sum(ref[start:stop]**2)/len(ref[start:stop])
        epsilon = epsilon*(1-alpha)/(2*L_rtf) + eps

        # Generate toeplitz matrix for the convolution operations
        col0 = np.concatenate(( np.expand_dims(np.array(ref[start]), 0), ref_toeplitz[:-1,-1] )) # ref_toeplitz will be 0 for first loop and consistent aftwerwards
        row0 = ref[start:stop]
        ref_toeplitz[:,:] = linalg.toeplitz(col0, row0)

        start_stop_bank[2*k] = start
        start_stop_bank[2*k+1] = stop

        for l in range(0, L_frame): 

            n = start + l

            x_frame = np.expand_dims(ref_toeplitz[:,l], 1)
            des_hat[n, :] = (x_frame.T)@h # estimate based on current weights
            e[n, :] = des[n,:] - des_hat[n,:] # error based on current weights 
            # Sparsity coefficients 
            K = ( 1 / (2*L_rtf)) * (1-alpha) * np.ones((L_rtf, Nch)) + (1+alpha)*np.abs(h) / (2*np.sum(np.abs(h)) + eps)
            # Calculation of new weights 
            h = h + (vad[n,:]*mu*e[n,:]*K*x_frame) / (x_frame.T@(K*x_frame) + epsilon)
            # recursive avereaging
            h_ave[:,:,k] += h/L_frame

        # calculation of frame based estimated signal and error 
        des_hat_ave[start:stop,:] = (ref_toeplitz).T@h_ave[:,:,k] # estimate based on averaged weights
        e_ave[start:stop,:] = des[start:stop,:] - des_hat_ave[start:stop,:]

    if L_pad != 0:
        des_hat_ave = des_hat_ave[:-L_pad,:] 
        e_ave = e_ave[:-L_pad,:]

    return h_ave, e_ave, des_hat_ave

def gevd_rtf(ns_psd, n_psd, ref_ch):
    """
    A function that implements the GEVD based RTF estimator from Marcovich 2009 paper for an individual STFT bin.
    It's an STFT domain estimator that requires estimates of the noisy signal PSD, noise PSD and the source activity.
    
    Input: 

    noisy_psd      -> MxM array containing the noisy signal PSD matrix
    noise_psd      -> MxM array containing the noise signal PSD matrix
    ref            -> channel to be used as reference channel for the RTF
    """

    warnings.filterwarnings("error")
    evals, evecs = linalg.eigh(ns_psd, n_psd)
    ref_ind = ref_ch - 1 
    rtf = (n_psd@evecs[:,-1]) / (n_psd@evecs[:,-1])[ref_ind]  
    return rtf

def pm_gevd_rtf(ns_psd, n_psd, u0, ref_ch):
    """
    A function that implements the GEVD based RTF estimator from Marcovich 2009 paper for an individual STFT bin.
    It's an STFT domain estimator that requires estimates of the noisy signal PSD, noise PSD and the source activity.
    
    Input: 

    noisy_psd      -> MxM array containing the noisy signal PSD matrix
    noise_psd      -> MxM array containing the noise signal PSD matrix
    ref            -> channel to be used as reference channel for the RTF
    """

    ref_ind = ref_ch - 1 
    Acs = ns_psd - n_psd
    u = np.squeeze((Acs@u0)/np.linalg.norm(Acs@u0))
    rtf = u/u[ref_ind]
    return rtf, u


def pm_pevd_rtf(RXX, ref_ch=1, N_iter=10):
    """
    A function that implements the GEVD based RTF estimator from Marcovich 2009 paper for an individual STFT bin.
    It's an STFT domain estimator that requires estimates of the noisy signal PSD, noise PSD and the source activity.
    
    Input: 

    noisy_psd      -> MxM array containing the noisy signal PSD matrix
    noise_psd      -> MxM array containing the noise signal PSD matrix
    ref            -> channel to be used as reference channel for the RTF
    """

    ref_ind = ref_ch - 1 

    M, _, L = np.shape(RXX)

    A = np.fft.rfft(RXX, axis=-1)
    # A = np.fft.rfft(RXX, axis=-1, n=N_rtf)
    A_shift = np.zeros_like(A)
    _,_,N_freqs = np.shape(A)

    time_shift = int((N_freqs-1))

    U = np.ones((M,N_freqs), 'complex')
    U_reverse_shift = np.ones_like(U)
    rtf = np.zeros((M,N_freqs), 'complex')

    for k in range(N_freqs): 
        A_shift[:,:,k] = A[:,:,k]*np.exp(-1j*2*np.pi*k*(-time_shift)/N_freqs)

    for k in range(N_freqs):
        for n in range(N_iter):
            U[:,k] = A_shift[:,:,k]@U[:,k]/np.linalg.norm(A_shift[:,:,k]@U[:,k])

    for k in range(N_freqs): 
        U_reverse_shift[:,k] = U[:,k]*np.exp(-1j*2*np.pi*k*(time_shift)/N_freqs)
        rtf[:,k] = U_reverse_shift[:,k] / U_reverse_shift[ref_ind,k]

    rtf = rtf.T
    rtf_td = np.fft.irfft(rtf, axis=0)

    return rtf, rtf_td

def rec_psd_est(sig_stft, alpha):
    '''
    Calculates an estimate of a signal PSD matrix by recursively averaging instantaneous estimates
    Input:
    sig_stft -> vector of the signal that is of interest  
    alpha    -> forgetting factor, higher updates slower, lower value updates the average faster
    '''
    N_freq, Nch, N_segs = np.shape(sig_stft)
    psd = np.zeros((Nch, Nch, N_freq, N_segs), 'complex')
    psd[:,:,:,-1] = np.ones((Nch, Nch, N_freq), 'complex')
    for n in range(N_segs):
        for f in range(N_freq):
            sig_stft_bin = np.expand_dims(np.squeeze(sig_stft[f,:,n]), 1)
            psd[:,:,f,n] = alpha*psd[:,:,f,n-1] + (1-alpha)*sig_stft_bin@(np.conj(sig_stft_bin)).T
    return psd


def static_stcov(X, w=None):
    # computes the space-time covariance matrix given the signal X
    p, T = X.shape  # p is the number of channels, T is the signal length
    if p > T:
        X = X.T
        p, T = X.shape
    zRxx = np.zeros([2 * T - 1, p, p], dtype=complex)

    for ii in range(p):
        for jj in range(p):
            if jj >= ii:
                zRxx[:, ii, jj] = sig.correlate(X[ii, :], X[jj, :]) / T
            else:
                zRxx[:, ii, jj] = np.flip(zRxx[:, jj, ii].conjugate())
    if w == None:
        w = T - 1
    zRxx = zRxx[T -1 - w: T + w, :, :]
    if np.max(np.imag(zRxx)) == 0:
        zRxx = np.real(zRxx)
    else:
        raise ValueError('imag stcov matrix, BONKERS!')
    return zRxx


def rec_stcov_est(x, len_w, alpha, overlap=None, w=None):
    # computes the space-time covariance matrix given the signal X
    L, Nch = np.shape(x)  # p is the number of channels, T is the signal length
    if Nch > L:
        x = x.T
        Nch, L = x.shape

    x_framed, N_frames = mac.util.td_framing(x, len_w, overlap=overlap)
    L_frames, _, _ = np.shape(x_framed)
    
    zRxx = np.zeros([Nch, Nch, 2 * L_frames, N_frames], dtype=complex)

    for f in range(N_frames):
        for ii in range(Nch):
            for jj in range(Nch):
                if jj >= ii:
                    zRxx[ii, jj, 0:-1, f] = alpha*zRxx[ii, jj, 0:-1, f-1] + (1-alpha)*sig.correlate(x_framed[:, ii, f], x_framed[:, jj, f])/L
                else:
                    zRxx[ii, jj, 0:-1, f] = np.flip(zRxx[jj, ii, 0:-1, f].conjugate())

    if w == None:
        w = L - 1
    zRxx = zRxx[:, :, L -1 - w: L + w, :]
    if np.max(np.imag(zRxx)) == 0:
        zRxx = np.real(zRxx)
    else:
        raise ValueError('imag stcov matrix, BONKERS!')
    return zRxx


def fblms_est(sig_in, ref_ch, M, alpha=0.7, gamma=0.1, vad=[]):
 
    L, Nch = np.shape(sig_in)

    N_frames = int(np.ceil(L/M))
    N_end_pad = int(N_frames*M - L)
    sig_pad = np.concatenate((np.zeros((M, Nch)), sig_in, np.zeros((N_end_pad, Nch))), axis=0)

    ref_ind = ref_ch - 1
    encoded_channels = list(range(Nch))
    encoded_channels.remove(ref_ind)
    
    if vad == []:
        vad = np.ones_like(sig_in)
    else: 
        raise ValueError('VAD controlled learning rate not implemented')

    y_hat_adapt = np.zeros_like(sig_pad)
    y_hat_pad = np.zeros_like(sig_pad)
    W = np.zeros((2*M, Nch, N_frames + 1), dtype='complex')
    P = 0.00001*np.ones((2*M, Nch, N_frames))

    ref_sig_pad = sig_pad[:, ref_ind]
    y_hat_adapt[:, ref_ind] = ref_sig_pad
    y_hat_pad[:, ref_ind] = ref_sig_pad
    e_pad = np.zeros_like(y_hat_adapt)

    for ch in range(Nch):
        for k in range(1,N_frames):

            start = k*M - M
            mid = k*M
            end = k*M + M #-1 but not needed due to python indexing

            u = sig_pad[start:end, ref_ind]
            U = np.diag(np.fft.fft(u, axis=0))

            y_hat_adapt[mid:end,ch] = np.real((np.fft.ifft(U@W[:,ch,k], axis=0)[-M:]).T)

            e_pad[mid:end,ch] = sig_pad[mid:end,ch] - y_hat_adapt[mid:end,ch]
            e_conc = np.concatenate((np.zeros((M,)), e_pad[mid:end,ch]))
            E = np.fft.fft(e_conc, axis=0)

            P[:,ch,k] = gamma*P[:,ch,k-1] + (1-gamma)*np.abs(U.diagonal())**2

            D = np.diag(1/P[:,ch,k])

            phi = np.fft.ifft(D@np.conj(U.T)@E, axis=0)[:M]
            phi_conc = np.concatenate((phi, np.zeros((M,))))

            W[:,ch,k+1] = W[:,ch,k] + alpha*np.fft.fft(phi_conc, axis=0)

            y_hat_pad[mid:end,ch] = np.real((np.fft.ifft(U@W[:,ch,k+1], axis=0)[-M:]).T)

        k+=1 
        start = k*M - M
        mid = k*M
        end = k*M + M #-1 but not needed due to python indexing
        u = sig_pad[start:end, ref_ind]
        U = np.diag(np.fft.fft(u, axis=0))
        y_hat_pad[-M:,ch] = np.real((np.fft.ifft(U@W[:,ch,k], axis=0)[-M:]).T)

    if N_end_pad == 0:
        y_hat = y_hat_pad[M:,:]
        e = e_pad[M:,:]
    else:
        y_hat = y_hat_pad[M:-N_end_pad,:]
        e = e_pad[M:-N_end_pad,:]
        
    return W[:,:,1:], e, y_hat



def joint_mle_psd_rtf():
    """
    A function to implement the Maximum Likelihood Estimation based joint estimation of RTFs and the covariance matrix.
    
    By coincidence it has been shown that the MLE of the RTF is given by the GEVD subspace method in mark09

    """

    return 0 