import numpy as np 
import string 
import random 
import math

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

def casuality_unpad(y, N_rtf, ref_ch):
    return 0

def rotate_list(l, n):
    return l[n:] + l[:n]