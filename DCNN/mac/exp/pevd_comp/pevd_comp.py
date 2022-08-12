import mac 
import numpy as np
import matplotlib.pyplot as plt
import pypevd 

if __name__ == '__main__':


    alpha = 0.9
    Nch = 2 
    len_w = 2
    ref_ch = 1
    ref_ind = ref_ch - 1
    x = np.zeros((len_w, Nch))
    x[0,0] = 1   
    x[-1,1] = 1

    RXX = np.squeeze(mac.est.rec_stcov_est(x, len_w, alpha, overlap=0))
    pypevd.putility.vispmd(np.moveaxis(RXX,-1,0), ph=True, title='st-cov-eg-eigenvector', save=False)

    H, D = pypevd.pevd.smd(np.moveaxis(RXX,-1,0), maxIterations=10, trim=0, delta=1e-5)
    pypevd.putility.vispmd(H, ph=False, title='st-cov-eg-eigenvector', save=False)
    pypevd.putility.vispmd(D, ph=True, title='st-cov-eg-eigenvector', save=False)

    pass