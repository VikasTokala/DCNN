import numpy as np

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
        mse_db[ch] = 10*np.log10(np.mean((ref_sig[:,ch] - test_sig[:,ch])**2, axis=0))
    return mse_db