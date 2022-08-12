import mac
import os

if __name__ == '__main__':
    home = os.path.expanduser('~')
    dataset_dir = home + '/mac/datasets/libriconv_8ch/'
    signals_dir = home + '/signals/LSP/'
    N_sigs = 10
    N_impulses = 50
    array_spacing = 0.05
    Nch = 8
    T_chunk = 4
    
    mac.nrtf.gen_libriconv_dataset(signals_dir, dataset_dir, N_sigs, T_chunk, N_impulses, array_spacing, Nch)    