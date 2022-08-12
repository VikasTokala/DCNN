import numpy as np 
import scipy.signal as sig 
import matplotlib.pyplot as plt 
import pyroomacoustics as pra
from . import est
import soundfile as sf

class TestSignalModel():
    def __init__(self, ir_type, len_w, fs=16000, Ts=4, overlap=None, SNR_db=20, Nch=2, alpha=0.9, plot=False):

        self.ir_type = ir_type
        self.len_w = len_w
        self.fs = fs
        self.Ts = Ts
        self.Ls = int(Ts*fs)
        self.alpha = alpha
        self.Nch = Nch
        
        if overlap == None: 
            self.overlap = len_w//2
        else:
            self.overlap = overlap
        self.overlap_factor = self.overlap/self.len_w

        self.N_rtf_t = len_w 
        self.N_rtf_f = (self.N_rtf_t//2) + 1

        # Generate room response
        if self.ir_type=='sim':
            rt60 = 0.5  # seconds
            room_dim = [10, 6, 5]  # meters
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
            room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
            room.add_source([2.5, 3, 1.5])
            
            mic_locs = np.c_[
                [6.3, 4.5, 1.2],  # mic 1
                [6.3, 4.55, 1.2],  # mic 2
                ]
            room.add_microphone_array(mic_locs)
            room.compute_rir()
            self.h = np.squeeze(np.array(room.rir)).T
                        
            self.h = self.h/np.max(abs(self.h))
            self.h = self.h[0:self.N_rtf_t,:]

        elif self.ir_type == 'imp_imp':
            
            self.h = np.zeros((self.N_rtf_t,Nch))
            self.h[0,0] = 1
            self.h[50,1] = 0.5

        elif self.ir_type == 'imp_quarter_cos':
            
            self.h = np.zeros((self.N_rtf_t,Nch))

            w_proto = 0.5*np.cos(np.linspace(0, np.pi/2, 40))
            w_proto = np.concatenate((np.zeros((20,)), w_proto, np.zeros((int(self.N_rtf_t - 60),))))

            self.h[0,0] = 1
            self.h[:,1] = w_proto

        elif self.ir_type == 'imp_half_sin':
            
            self.h = np.zeros((self.N_rtf_t,Nch))

            w_proto = 0.5*np.sin(np.linspace(0, np.pi, 40))
            w_proto = np.concatenate((np.zeros((20,)), w_proto, np.zeros((int(self.N_rtf_t - 60),))))

            self.h[0,0] = 1
            self.h[:,1] = w_proto

        elif self.ir_type == 'half_sin_half_sin':
            
            self.h = np.zeros((self.N_rtf_t,Nch))
            w_proto = 0.5*np.sin(np.linspace(0, np.pi, 40))
            self.h[:,0] = np.concatenate((w_proto, np.zeros((int(self.N_rtf_t - 40),))))
            self.h[:,1] = np.concatenate((np.zeros((20,)), w_proto, np.zeros((int(self.N_rtf_t - 60),))))

        elif self.ir_type == 'nuance':
            self.h = np.zeros((self.N_rtf_t, Nch))

        else:
            raise ValueError('ir_type provided does not match possible options')

        if plot: 
            plt.figure()
            plt.plot(self.h)
            plt.title("Signal model impulses - " + str(ir_type))

        # Generate noise signals 
        mu, sigma = 0, 1.2
        s = np.random.normal(mu, sigma, self.Ls)
        self.Lx = self.Ls + len(self.h) -  1
        self.N_windows = int(np.ceil(self.Lx/(self.len_w*(1 - self.overlap_factor))) + 1)
        self.Lx = int((self.N_windows - 1)*self.len_w*(1 - self.overlap_factor))

        self.x = np.zeros((self.Lx, Nch)) # desired signal
        for ch in range(Nch):
            xch = np.convolve(self.h[:,ch], s)
            self.x[0:len(xch),ch] = xch

        SNR_factor = 10**(SNR_db/10)

        self.v = np.random.normal(mu, sigma/SNR_factor, (self.Lx, Nch)) # noise signal
        self.y = self.x + self.v # mixture signal

        f, t, self.X = sig.stft(self.x, fs=fs, nperseg=len_w, axis=0, padded=False, noverlap=self.overlap)
        assert (self.N_windows == np.shape(self.X)[-1])
        assert (self.N_rtf_f == np.shape(self.X)[0])
        _, _, self.V = sig.stft(self.v, fs=fs, nperseg=len_w, axis=0, padded=False, noverlap=self.overlap)
        self.Y = self.X + self.V # mixture signal

        # PSD matrix inits 
        self.Px_rec = est.rec_psd_est(self.X, alpha)
        self.Pv_rec = est.rec_psd_est(self.V, alpha)
        self.Py_rec = est.rec_psd_est(self.Y, alpha)

        self.RXX_ave = np.moveaxis(est.static_stcov(self.x),0,-1)
        self.RVV_ave = np.moveaxis(est.static_stcov(self.v),0,-1)
        self.RYY_ave = np.moveaxis(est.static_stcov(self.y),0,-1)

        self.RXX_rec = est.rec_stcov_est(self.x, self.len_w, alpha)
        self.RVV_rec = est.rec_stcov_est(self.v, self.len_w, alpha)
        self.RYY_rec = est.rec_stcov_est(self.y, self.len_w, alpha)

        self.H = np.fft.rfft(self.h, axis=0)

        print('\nSignal Model set up\n')

class NuanceSignalModel():
    
    def __init__(self, sig_path, noise_type, len_w, fs=16000, Ts=4, overlap=None, SNR_db=20, alpha=0.9, plot=False):

            self.len_w = len_w
            self.fs = fs
            self.Ts = Ts
            self.Ls = int(Ts*fs)
            self.alpha = alpha
            self.Nch = 8

            if overlap == None: 
                self.overlap = len_w//2
            else:
                self.overlap = overlap
            self.overlap_factor = self.overlap/self.len_w

            self.N_rtf_t = len_w 
            self.N_rtf_f = (self.N_rtf_t//2) + 1

            self.x, fs = sf.read(sig_path)

            self.Lx, self.Nch = np.shape(self.x)
            self.N_windows = int(np.ceil(self.Lx/(self.len_w*(1 - self.overlap_factor))) + 1)
            self.Lx_pad = int((self.N_windows - 1)*self.len_w*(1 - self.overlap_factor))

            L_pad = self.Lx_pad - self.Lx

            self.x = np.concatenate((self.x, np.zeros((L_pad, self.Nch))))

            pow_x = np.mean(self.x**2)
            SNR_factor = 10**(SNR_db/10)
            pow_v = pow_x/SNR_factor

            if noise_type == 'white':
                self.v = np.random.normal(0, pow_v, (self.Lx_pad, self.Nch)) # noise signal
            
            self.y = self.x + self.v
        
            f, t, self.Y = sig.stft(self.y, fs=fs, nperseg=len_w, axis=0, padded=False, noverlap=self.overlap)
            self.Py_rec = est.rec_psd_est(self.Y, alpha)
            self.RYY_ave = np.moveaxis(est.static_stcov(self.y),0,-1)
            self.RYY_rec = est.rec_stcov_est(self.y, self.len_w, alpha)

            f, t, self.X = sig.stft(self.x, fs=fs, nperseg=len_w, axis=0, padded=False, noverlap=self.overlap)
            self.Px_rec = est.rec_psd_est(self.X, alpha)
            self.RXX_ave = np.moveaxis(est.static_stcov(self.x),0,-1)
            self.RXX_rec = est.rec_stcov_est(self.x, self.len_w, alpha)

            f, t, self.V = sig.stft(self.v, fs=fs, nperseg=len_w, axis=0, padded=False, noverlap=self.overlap)
            self.Pv_rec = est.rec_psd_est(self.V, alpha)
            self.RVV_ave = np.moveaxis(est.static_stcov(self.v),0,-1)
            self.RVV_rec = est.rec_stcov_est(self.v, self.len_w, alpha)

            print('\nSignal Model set up\n')


class PingPongSignalModel():
    def __init__(self, ir_type, len_w, ping_pong_T=2, fs=16000, Ts=4, overlap=None, SNR_db=20, Nch=2, alpha=0.9, plot=False):
        return 0 