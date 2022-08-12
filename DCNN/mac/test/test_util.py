import mac
import numpy as np 
import matplotlib.pyplot as plt

def test_framing():

    L = 113
    Nch = 2 
    L_frame = 14
    overlap=L_frame//2

    x = np.zeros((L,Nch))
    x[20,0] = 1
    x[90,1] = 1

    x_framed, N_frames = mac.util.td_framing(x, L_frame, overlap=0)
    x_unframed = mac.util.undo_tf_framing(x_framed)
    assert np.max(np.abs((x - x_unframed[:L,:]))) == 0
    print('PASS')

    assert np.max(np.abs(x_unframed[L:,:])) == 0
    print('PASS')

    x_framed, N_frames = mac.util.td_framing(x, L_frame, overlap)

    jump = L_frame - overlap

    # for f in range(N_frames-2):
    #     assert np.max(np.abs(x[f*jump:f*jump + L_frame,:] - x_framed[:, :, f])) == 0
    # print('PASS')

if __name__ == '__main__':
    test_framing()
