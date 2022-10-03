import soundfile as sf
import scipy.signal as sig
import numpy as np
import os
import sys

def convolution_ace_librispeech(signal_path, fs, output_folder, Nch=8):

    ace_folder = '/rds/general/user/dtj20/home/signals/ACE/Lin8Ch/'

    # Choose RIRs
    rir_path_1 = ace_folder + 'Meeting_Room_1/1/Lin8Ch_503_1_RIR.wav' 
    rir_path_2 = ace_folder + 'Meeting_Room_2/1/Lin8Ch_611_1_RIR.wav'
    rir_path_3 = ace_folder + 'Lecture_Room_1/1/Lin8Ch_508_1_RIR.wav'
    rir_path_4 = ace_folder + 'Lecture_Room_2/1/Lin8Ch_403a_1_RIR.wav'

    rir_paths = [rir_path_1, rir_path_2, rir_path_3, rir_path_4]

    # Load RIRs
    rir1, fs1 = sf.read(rir_paths[0])
    rir2, fs2 = sf.read(rir_paths[1])
    rir3, fs3 = sf.read(rir_paths[2])
    rir4, fs4 = sf.read(rir_paths[3])
    
    # Resample RIRs
    L1 = round(len(rir1) * float(fs) / fs1)
    L2 = round(len(rir2) * float(fs) / fs2)
    L3 = round(len(rir3) * float(fs) / fs3)
    L4 = round(len(rir4) * float(fs) / fs4)

    rir1 = sig.resample(rir1, L1)
    rir2 = sig.resample(rir1, L2)
    rir3 = sig.resample(rir1, L3)
    rir4 = sig.resample(rir1, L4)

    x, fs_sig = sf.read(signal_path)

    y1 = np.zeros((L1+len(x)-1, Nch))
    y2 = np.zeros((L2+len(x)-1, Nch))
    y3 = np.zeros((L3+len(x)-1, Nch))
    y4 = np.zeros((L4+len(x)-1, Nch))
    
    L = len(x)
    for ch in range(Nch):
        y1[:, ch] = np.convolve(x, rir1[:,ch])
        y2[:, ch] = np.convolve(x, rir2[:,ch])
        y3[:, ch] = np.convolve(x, rir3[:,ch])
        y4[:, ch] = np.convolve(x, rir4[:,ch])

    y_file_1 = output_folder + 'rir1/' + os.path.basename(signal_path)
    y_file_2 = output_folder + 'rir2/' + os.path.basename(signal_path)
    y_file_3 = output_folder + 'rir3/' + os.path.basename(signal_path)
    y_file_4 = output_folder + 'rir4/' + os.path.basename(signal_path)

    sf.write(y_file_1, y1, samplerate=fs)
    sf.write(y_file_2, y2, samplerate=fs)
    sf.write(y_file_3, y3, samplerate=fs)
    sf.write(y_file_4, y4, samplerate=fs)

if __name__ == '__main__':

    signal_path = sys.argv[1]
    fs = int(sys.argv[2])
    output_folder = sys.argv[3]

    convolution_ace_librispeech(signal_path, fs, output_folder)