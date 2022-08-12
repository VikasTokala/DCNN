def ipnlms_est(ref, sig_in, vad, alpha, mu, L_RTF, L_frame, eps=2**-50):
    
    ref, _, fs_ref = mac.util.clever_load(ref)
    x, _, fs_x = mac.util.clever_load(sig_in)

    if fs_ref != fs_x:
        sys.exit('sampling frequencies of the ref and input signal are not equal')

    N_samples, Nch = np.shape(x)

    if N_samples < L_frame:
        sys.exit('signal shorter than frame length')
    if vad == []:
        vad = np.ones_like(x)
    
    # Split signals into frames and pad if not multiple of frame length     
    N_frames = math.ceil(N_samples/L_frame)
    N_end_pad = N_frames*L_frame - N_samples
    x = np.concatenate((x, np.zeros((N_end_pad, Nch))), axis=0)
    ref = np.concatenate((ref, np.zeros((N_end_pad, 1))), axis=0)
    vad = np.concatenate((vad, np.zeros((N_end_pad, Nch))), axis=0)
    N_samples, Nch = np.shape(x)

    # Initialise vectors & matrices
    h = np.zeros((L_RTF, Nch))
    h_mp = np.zeros((L_RTF, Nch, N_frames))
    d_hat = np.zeros((N_samples, Nch)) 
    d_hat_frames = np.zeros((N_samples, Nch)) 
    e = np.zeros((N_samples, Nch))
    e_frame = np.zeros((N_samples, Nch))
    ref_toeplitz = np.zeros((L_RTF, L_frame))

    for f in range(0, N_frames):  

        start_window = f*L_frame
        stop_window = f*L_frame + L_frame - 1 

        epsilon = np.sum(ref[start_window:stop_window+1]**2)/len(ref[start_window:stop_window + 1])
        epsilon = epsilon*(1-alpha)/(2*L_RTF) + eps

        # Generate toeplitz matrix for the convolution operations
        col0 = np.concatenate((np.expand_dims(ref[start_window], 1), np.zeros((L_RTF -1, 1))))
        row0 = ref[start_window:stop_window+1]
        ref_toeplitz[:,:] = linalg.toeplitz(col0, row0)

        for l in range(0, L_frame -1):
            n = f*L_frame + 1 # absolute index based on frame f and distance into frame l
            x_frame = np.expand_dims(ref_toeplitz[:,l], 1)
            d_hat[n, :] = (x_frame.T)@h # estimate based on current weights
            e[n, :] = x[n,:] - d_hat[n,:] # error based on current weights 

            # Sparsity coefficients 
            K = ( 1 / (2*L_RTF)) * (1-alpha) * np.ones((L_RTF, Nch)) + (1+alpha)*np.abs(h) / (2*np.sum(np.abs(h)) + eps)

            # Calculation of new weights 
            h = h + (vad[n,:]*mu*e[n,:]*K*x_frame) / (x_frame.T@(K*x_frame) + epsilon)

            # recursive avereaging
            h_mp[:,:,f] += h/L_frame

        # calculation of frame based estimated signal and error 
        d_hat_frames[start_window:stop_window+1,:] = (x_frame.T)@h_mp # estimate based on averaged weights
        e_frame[start_window:stop_window+1,:] = x[start_window:stop_window+1,:] - d_hat_frames[start_window:stop_window+1,:]
    
    d_hat_frames = d_hat_frames[0:-N_end_pad,:]
    e_frame = e_frame[0:-N_end_pad,:]

    return h_mp, e_frame

def ipnlms_frame_est(ref_frame, sig_frame, h0, vad, alpha, mu, L_RTF, eps=2**-50):

    ref_frame, _, fs_ref = mac.util.clever_load(ref_frame)
    x_frame, _, fs_x = mac.util.clever_load(sig_frame)

    if fs_ref != fs_x:
        sys.exit('sampling frequencies of the ref and input signal are not equal')

    L_frame, Nch = np.shape(x_frame)

    if vad == []:
        vad = np.ones_like(x_frame)
    else:
        vad = mac.util.clever_load(vad)
        if len(vad) != len(x_frame):
            raise ValueError('VAD length not equal to frame length')

    # Initialise vectors & matrices
    h1 = np.zeros((L_RTF, Nch))
    h_rec_ave = np.zeros((L_RTF, Nch))
    d_hat0 = np.zeros((L_frame, Nch)) 
    d_hat1 = np.zeros((L_frame, Nch)) 
    d_hat_ave_frame = np.zeros((L_frame, Nch)) 
    e0 = np.zeros((L_frame, Nch))
    e1 = np.zeros((L_frame, Nch))
    e_frame = np.zeros((L_frame, Nch))
    ref_toeplitz = np.zeros((L_RTF, L_frame))

    # Generate toeplitz matrix for the convolution operations
    col0 = np.concatenate((np.expand_dims(ref_frame, 1), np.zeros((L_RTF -1, 1))))
    row0 = ref_frame
    ref_toeplitz = linalg.toeplitz(col0, row0)
        
    epsilon = np.sum(ref_frame**2)/len(ref_frame)
    epsilon = epsilon*(1-alpha)/(2*L_RTF) + eps

    for l in range(0, L_frame -1):
 
        d_hat0[:, :] = (ref_toeplitz[:, l].T)@h0 # estimate based on current weights
        e0[l, :] = x_frame[l, :] - d_hat0[l, :] # error based on current weights 

        # Sparsity coefficients 
        K = ( 1 / (2*L_RTF)) * (1-alpha) * np.ones((L_RTF, Nch)) + (1+alpha)*np.abs(h0) / (2*np.sum(np.abs(h0)) + eps)

        # Calculation of new weights 
        h1 = h0 + (vad[l,:]*mu*e0[l,:]*K*x_frame) / (x_frame.T@(K*x_frame) + epsilon)

        # recursive avereaging
        h_rec_ave += h1/L_frame

        # preparing for next loop
        h0 = h1

        # could include calculation of instantaneous rather than just frame based errors here in future

        d_hat1[:, :] = (ref_toeplitz[:, l].T)@h0 # estimate based on current weights
        e1[l, :] = x_frame[l, :] - d_hat0[l, :] # error based on current weights 

    d_hat_ave_frame = (ref_toeplitz[:, l].T)@h_rec_ave
    e_ave = x_frame - d_hat_ave_frame

    return (h_rec_ave, e_ave, e0)