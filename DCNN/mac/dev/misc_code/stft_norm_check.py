import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as sig

fs = 16000
T = 5
t = np.linspace(0, T, T*fs)
y = 90*np.sin(2*np.pi*1000*t)

f,s,Y = sig.stft(y,fs=fs,nperseg=4096)

plt.figure()
plt.plot(y)

plt.figure()
plt.pcolormesh(s, f, np.abs(Y), shading='auto')
plt.colorbar()
plt.show()

print(np.max(np.abs(Y)))