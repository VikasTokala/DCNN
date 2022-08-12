import mac 
import numpy as np
import matplotlib.pyplot as plt
import pypevd 

if __name__ == '__main__':

    ir_type = 'imp_rtf'
    
    sm = mac.util.TestSignalModel(ir_type=ir_type, len_w=1024, SNR_db=20)

    # U = mac.util.iterative_pevd(sm.RYY, 5, 256)

    U = mac.util.iterative_pevd(sm.RXX, 50, 32)

    # pypevd.putility.vispmd(np.moveaxis(sm.RXX,-1,0), ph=False, title='st-cov-eg-eigenvector', save=False)

    H, D = pypevd.pevd.smd(np.moveaxis(sm.RXX,-1,0), maxIterations=10, trim=0, delta=1e-5)

    pypevd.putility.vispmd(H, ph=False, title='st-cov-eg-eigenvector', save=False)
    
    plt.figure()
    plt.plot(np.squeeze(np.real(U).T))
    plt.show()

    pass