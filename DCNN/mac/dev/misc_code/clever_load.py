# def clever_load(sig_in):
#     if type(sig_in) == str:
#         path = sig_in
#         if sig_in.endswith('.wav'):
#             signal, fs = sf.read(sig_in)
#     if type(sig_in) == np.ndarray:
#         signal = sig_in 
#         path = None 
#         fs = None 

#     if signal.ndim == 1:
#         signal = np.expand_dims(signal,1)
    
#     return signal, path, fs 