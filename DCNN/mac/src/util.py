import numpy as np 
import string 
import random 
import os
import soundfile as sf
import scipy.signal as sig
import pickle

fs = 16000

def td_framing(signal, L_frame, overlap=None):
    """
    splits signals into time domain frames
    """

    if overlap == None: 
        overlap = L_frame//2
    else:
        overlap = overlap
    overlap_factor = overlap/L_frame

    N_samples, Nch = np.shape(signal)
    if N_samples < L_frame:
        raise ValueError('signal shorter than frame length')
    
    if overlap != 0:
        # Split signals into frames and pad if not multiple of frame length     
        N_frames = int(np.ceil(N_samples/(L_frame*(1 - overlap_factor))) + 1)
        N_end_pad = int(N_frames*L_frame*(1 - overlap_factor) - N_samples)
        sig_pad = np.concatenate((signal, np.zeros((N_end_pad, Nch))), axis=0)
        N_samples = len(sig_pad)
        sig_framed = np.zeros((L_frame, Nch, N_frames))
        for f in range(0, N_frames-1):
            start_window = int(f*L_frame*(1 - overlap_factor))
            stop_window = start_window + L_frame
            sig_framed[:,:,f] = sig_pad[start_window:stop_window,:]
    else:
        # Split signals into frames and pad if not multiple of frame length     
        N_frames = int(np.ceil(N_samples/(L_frame)))
        N_end_pad = int(N_frames*L_frame*(1 - overlap_factor) - N_samples)
        sig_pad = np.concatenate((signal, np.zeros((N_end_pad, Nch))), axis=0)
        N_samples = len(sig_pad)
        sig_framed = np.zeros((L_frame, Nch, N_frames))
        for f in range(0, N_frames-1):
            start_window = f*L_frame + f
            stop_window = start_window + L_frame
            sig_framed[:,:,f] = sig_pad[start_window:stop_window,:]
    
    return sig_framed, N_frames

def undo_tf_framing(sig_framed):
    """
    undoes framing
    """
    L_frame, Nch, N_frames = np.shape(sig_framed)

    sig_pad = np.zeros((L_frame*N_frames, Nch))

    for f in range(0, N_frames-1):
        start_window = f*L_frame + f
        stop_window = start_window + L_frame
        sig_pad[start_window:stop_window,:] = sig_framed[:,:,f]

    return sig_pad

def randStr(chars = string.ascii_uppercase + string.digits, N=4):
    return ''.join(random.choice(chars) for _ in range(N))

def get_im_rirs(room):

    Nch = np.shape(room.rir)[0]
    h_len = 0 
    for ch in range(Nch):
        if np.shape(room.rir[ch])[1] > h_len:
            h_len = np.shape(room.rir[ch])[1] 

    h = np.zeros((h_len, Nch))
    for ch in range(Nch):
        pad = np.zeros((h_len - np.shape(room.rir[ch])[1],))
        h[:, ch] = np.concatenate((np.squeeze(np.transpose(room.rir[ch])), pad))

    return h 

def causality_pad(y, N_rtf, ref_ch):
    ref_ind = ref_ch - 1 
    L, Nch = np.shape(y)
    y_pad = np.concatenate((y, np.zeros(N_rtf//2, Nch)))
    enc_ch = [range(Nch)].remove(ref_ind)
    for ch in enc_ch:
        y_pad[:, ch] = np.roll(y_pad[:,ch], N_rtf//2)
    return y_pad

def causality_unpad(y, N_rtf, ref_ch):
    return 0

def rotate_list(l, n):
    return l[n:] + l[:n]

def calc_ave_spec(sig_dir, Nch, len_w, ave_spec_name=[], ave_ir_name=[]):

    sigs_list = []

    for file in os.listdir(sig_dir):
        if file.endswith('.wav'):
            sigs_list.append(sig_dir + '/' + file)

    N_sigs = len(sigs_list)

    ave_spec = np.zeros((len_w, Nch), 'complex')

    for s in sigs_list:
        x, _ = sf.read(s)
        _, _, X = sig.stft(x, axis=0, nperseg=len_w)
        ave_spec = np.mean(X, axis=-1)

        ave_spec = ave_spec/np.sum(ave_spec**2, axis=0)

        ave_ir = np.fft.irfft(ave_spec, axis=0)

    if ave_spec_name != []:
        pickle.dump(ave_spec, open(ave_spec_name, 'wb'))

    if ave_ir_name != []:
        pickle.dump(ave_ir, open(ave_ir_name, 'wb'))

    return ave_spec, ave_ir

def gen_ltas_noise(ave_ir, noise_power, noise_shape):
    if type(noise_shape) != tuple or len(noise_shape) != 2:
        raise ValueError('shape should be a two dimensional tuple length, channels')
    
    v_white = np.random.normal(0, noise_power, noise_shape)
    v_ltas = np.zeros_like(v_white)

    if type(ave_ir) == str:
        ave_ir_path = ave_ir
        pickle_file = open(ave_ir, 'rb')
        ave_ir = pickle.load(pickle_file)
    if type(ave_ir) == np.ndarray:
        pass
    else:
        raise ValueError('ave_ir is wrong not path to file or array')

    L, Nch = noise_shape

    for ch in range(Nch):
        v_ltas[:, ch] = np.convolve(v_white[:,ch], ave_ir[:,ch])[:L]

    ltas_power = np.mean(v_ltas**2, axis=0)
    norm_gain = noise_power/ltas_power

    v_ltas = v_ltas*norm_gain

    pickle_file.close()

    return v_ltas