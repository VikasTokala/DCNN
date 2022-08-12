import mac 
import numpy as np
import matplotlib.pyplot as plt
import pypevd 

if __name__ == '__main__':

    M = 2 
    N_iter = 10

    time_shift = 512
    L = 2*time_shift + 1 

    RXX = np.zeros((M, M, L))

    RXX[:,:,0] = np.array(([0, 1],[0, 0]))
    RXX[:,:,time_shift] = np.array(([1, 0],[0, 1]))
    RXX[:,:,-1] = np.array(([0, 0],[1, 0]))

    A = np.fft.rfft(RXX, axis=-1)
    A_shift = np.zeros_like(A)
    _,_,N_freqs = np.shape(A)

    U = np.ones((M,N_freqs), 'complex')
    U_reverse_shift = np.ones_like(U)
    RTF = np.zeros((M,N_freqs), 'complex')

    for k in range(N_freqs): 
        A_shift[:,:,k] = A[:,:,k]*np.exp(-1j*2*np.pi*k*(-time_shift)/N_freqs)

    for k in range(N_freqs):
        for n in range(N_iter):
            U[:,k] = A_shift[:,:,k]@U[:,k]/np.linalg.norm(A_shift[:,:,k]@U[:,k])

    for k in range(N_freqs): 
        U_reverse_shift[:,k] = U[:,k]*np.exp(-1j*2*np.pi*k*(time_shift)/N_freqs)
        RTF[:,k] = U_reverse_shift[:,k] / U_reverse_shift[0,k]

    u = np.fft.irfft(U_reverse_shift, axis=-1)
    rtf = np.fft.irfft(RTF, axis=-1)

    rtf_check, rtf_td_check = mac.est.pm_pevd_rtf(RXX, N_rtf=1024)

    plt.figure()
    plt.plot(u[0,:])
    plt.figure()
    plt.plot(u[1,:])

    # H, D = pypevd.pevd.smd(np.moveaxis(RXX,-1,0), maxIterations=10, trim=0, delta=1e-5)
    # plt.figure()
    # plt.plot(H[:,0,0])
    # plt.figure()
    # plt.plot(H[:,0,1])

    plt.figure()
    plt.plot(rtf[0,:])
    plt.plot(rtf_td_check[:,0], '--')
    plt.plot(rtf[1,:])
    plt.plot(rtf_td_check[:,1], '--')


    plt.show()