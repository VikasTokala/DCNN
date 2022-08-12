import numpy as np 
import scipy.signal as sig
import scipy.linalg as linalg
import sys
import os
home = os.path.expanduser('~')
sys.path.append(home)
import mac
import math
import soundfile as sf 
import tempfile as tmp


def ipnlms_mac(input_signal, output_signal, ref_ch, vad, L_RTF, alpha, mu, mode = 'e_diff_ave_opus'):
    
    # Create an empty directory
    tmp_suffix = mac.util.randStr()
    tmpdir = tmp.mkdtemp(dir=output_signal, suffix=tmp_suffix)    
    os.mkdir(tmpdir)

    # Read signals 
    ref_ind = ref_ch - 1 
    input_signal, _, fs = mac.util.clever_load(input_signal)
    ref_signal = np.copy(input_signal[:, ref_ind])

    # Split signals into frames and pad if not multiple of frame length     
    N_samples, Nch = np.shape(input_signal)
    L_frame = L_RTF
    if N_samples < L_frame:
        sys.exit('signal shorter than frame length')
    if vad == []:
        vad = np.ones_like(input_signal)
    N_frames = math.ceil(N_samples/L_frame)
    N_end_pad = N_frames*L_frame - N_samples
    input_signal = np.concatenate((input_signal, np.zeros((N_end_pad, Nch))), axis=0)
    ref_signal = np.concatenate((ref_signal, np.zeros((N_end_pad, 1))), axis=0)
    vad = np.concatenate((vad, np.zeros((N_end_pad, Nch))), axis=0)
    N_samples, Nch = np.shape(input_signal)

    # Init filter weights & start loop
    h0 = np.zeros((L_RTF, Nch))   
    last_rtf_transmit = N_frames # init as this so re-transmission logic knows that on firts transmit it's okay to transmit
    for f in range(N_frames):
        start_window = f*L_frame
        stop_window = f*L_frame + L_frame - 1 
        ref_frame = ref_signal[start_window:stop_window+1]
        input_frame = input_signal[start_window:stop_window+1, :]
    
        # Estimate RTFs, returns RTFs and errors 
        h_frame, e_frame, e0 = mac.est.ipnlms_frame_est(ref_frame, input_frame, h0, [], alpha, mu, L_RTF)

        # Check if re-transmission is needed.        
        if mode == 'e_diff_ave_opus':
            e0_db = 10*np.log10((e0)**2)
            e_frame_db = 10*np.log10((e_frame)**2)
            error_reduction = e0_db - e_frame_db
            h_diff = np.mean((h_frame - h0)**2)/(32768**2)             
            # w_transmit = (e_norm + h_diff) 
            
            # NEED A VALUE SO WEIGHTS ARE RESET WHEN OPUS IS USED BECAUSE E0 IS NOT USEFUL

            if e0_db <= -25: # If error with old RTF is low --> no change
                os.system('touch ' + tmpdir + str(f) + '.none')
            
            else: # need some kind of updating --> choose between fallback and new RTF 
            
                if e_frame_db <= -25: # RTF is meeting performance curve --> Transmit
                    if abs(f-last_rtf_transmit) < None:  
                        packet_name = str(f) + '.wav' 
                        sf.write(packet_name, h_frame, fs)
                        last_rtf_transmit = f
                
                elif e_frame_db > -25: # RTF not meeting performance curve --> Check fallback
                    e_fallback_db = 10*np.log10(None)  # Generate fallback packet and error
                    if e_fallback_db < e_rtf_transmit: # Use fallback for certain ammount of time
                         
                        pass
    return 0

    # zip into .rtf file

