import numpy as np
import matplotlib.pyplot as plt

def nmse_db(ref_sig, test_sig, plot=False):
    '''
    Calculates the channel averaged nmse of a signal relative to a reference
    '''

    if np.shape(ref_sig) != np.shape(test_sig):
        raise ValueError('shapes not compatible - check inputs')

    L, Nch = np.shape(ref_sig)
    if Nch > L:
        raise ValueError('Nch > L')   

    if np.max(np.abs(ref_sig)) == 0:
        return 0

    nmse_db = np.zeros((Nch,))
    error = np.zeros_like(ref_sig)
    for ch in range(Nch):
        error[:,ch] = ref_sig[:,ch] - test_sig[:,ch]
        error_db = 10*np.log10(np.mean((error[:, ch]**2), axis=0))
        ref_db = 10*np.log10(np.mean(((ref_sig[:,ch])**2), axis=0))
        nmse_db[ch] = error_db - ref_db

    if plot != False:
        for ch in range(Nch):
            plt.figure()
            plt.plot(ref_sig[:,ch], label='ref')
            plt.plot(test_sig[:,ch], label='error')
            plt.legend()
        plt.show()
    return nmse_db

def mse_db(ref_sig, test_sig):
    '''
    Calculates the channel averaged mse of a signal relative to a reference
    '''

    if np.shape(ref_sig) != np.shape(test_sig):
        raise ValueError('shapes not compatible - check inputs')

    L, Nch = np.shape(ref_sig)
    if Nch > L:
        raise ValueError('Nch > L')   

    if np.max(np.abs(ref_sig)) == 0:
        return 0

    mse_db = np.zeros((Nch,))
    for ch in range(Nch):
        mse_db[ch] = 10*np.log10( np.mean( ((ref_sig[:,ch] - test_sig[:,ch])**2), axis=0) )
    return mse_db