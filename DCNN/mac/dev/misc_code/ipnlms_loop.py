def ipnlms_loop(des_sig, h0, ref_ch, alpha, mu, L_RTF, vad=[], eps=2**-50):

    L, Nch = np.shape(des_sig)
    ref_ind = ref_ch - 1
    ref_sig = np.expand_dims(des_sig[:, ref_ind],1)

    if vad == []:
        vad = np.ones_like(des_sig)

    if np.shape(h0) != (L_RTF, Nch):
        raise ValueError('inital rtf size inconsistent')

    # Initialise vectors & matrices
    # h = np.zeros((L_RTF, Nch))
    h = h0
    h_ave = np.zeros((L_RTF, Nch))
    d_hat = np.zeros((L, Nch)) 
    d_hat_ave = np.zeros((L, Nch)) 
    e = np.zeros((L, Nch))
    e_ave = np.zeros((L, Nch))
    ref_toeplitz = np.zeros((L_RTF, L))
    epsilon = np.sum(ref_sig**2)/len(ref_sig)
    epsilon = epsilon*(1-alpha)/(2*L_RTF) + eps

    # Generate toeplitz matrix for the convolution operations
    col0 = np.concatenate((np.expand_dims(ref_sig[0,:], 1), np.zeros((L_RTF -1, 1))))
    row0 = ref_sig
    ref_toeplitz[:,:] = linalg.toeplitz(col0, row0)

    for l in range(0, L-1):

        x_frame = np.expand_dims(ref_toeplitz[:,l], 1)
        d_hat[l, :] = (x_frame.T)@h # estimate based on current weights
        e[l, :] = des_sig[l,:] - d_hat[l,:] # error based on current weights 
        # Sparsity coefficients 
        K = ( 1 / (2*L_RTF)) * (1-alpha) * np.ones((L_RTF, Nch)) + (1+alpha)*np.abs(h) / (2*np.sum(np.abs(h)) + eps)
        # Calculation of new weights 
        h = h + (vad[l,:]*mu*e[l,:]*K*x_frame) / (x_frame.T@(K*x_frame) + epsilon)
        # recursive avereaging
        h_ave += h/L

    # calculation of frame based estimated signal and error 
    d_hat_ave = (ref_toeplitz).T@h_ave # estimate based on averaged weights
    e_ave = des_sig - d_hat_ave

    return h_ave, e_ave, d_hat_ave