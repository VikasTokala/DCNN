import ctypes
from termios import NL0, NL1
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import tracemalloc_domain
from numpy.lib.arraysetops import ediff1d
from torch._C import Value 
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.utils import data
from torch.utils.data import DataLoader, random_split, Dataset
import torch
from torch import nn 
import torch.nn.functional as F
import torch.optim as optim
import pyroomacoustics as pra
import os
import mac
import soundfile as sf
import scipy.signal as sig
import time
import networkx as nx
import pickle

home = os.path.expanduser('~')

def RTF_graph_cycles(Nch, path_order):

    graph_path = home+'/mac/datasets/graphs/'+str(Nch)+'ch_'+str(path_order)+'e.pickle'

    if os.path.isfile(graph_path):
        cycles_sorted = pickle.load(open(graph_path, 'rb'))
        print('loading graph from file')

    else:

        print('generating graph and saving to file')

        if path_order > Nch:
            raise ValueError('path order can not be greater than the number of microphones')
        G = nx.complete_graph(Nch, create_using=nx.DiGraph())
        cycles = sorted(nx.simple_cycles(G))
        for ii, c in enumerate(cycles):

            if len(c) > path_order:
                del cycles[ii]

        cycles_sorted = []

        for ch in range(Nch):
            cycles_sorted.append([])

        for c in cycles:
            for i in range(len(c)):

                    c_rot = mac.util.rotate_list(c,i)

                    c_rot_comp = c_rot.copy()
                    c_rot_comp.append(c_rot[0])

                    cycles_sorted[c_rot[0]].append(c_rot)
                    cycles_sorted[c_rot[0]].append(c_rot_comp)

        pickle.dump(cycles_sorted, open(graph_path, 'wb'))

    return cycles_sorted

def Reciprocity_Cost(X, RTF_bank, path_order, device, domain='freq'):

    N_batch, N_rtfs, L_rtf, N_windows = RTF_bank.size()
    _, Nch, _, _ = X.size()

    cycles = RTF_graph_cycles(Nch, path_order)

    J = torch.zeros((N_batch,), dtype=X.dtype, device=device)

    if domain=='freq':
        for b in range(N_batch):
            for ch in range(Nch):
                paths = cycles[ch]

                for p in paths:
                    N_nodes = len(p)
                    N_edges = N_nodes - 1

                    start_ch = p[0]
                    end_ch = p[-1]

                    H_comb = torch.full((L_rtf, N_windows), 1 + 0j, device=device)

                    for e in range(N_edges):
                        bank_ind = Nch*p[e] + p[e+1]
                        H_comb = H_comb * RTF_bank[b, bank_ind, :, :] 

                    J[b] += ((X[b,end_ch, :,:] - H_comb*X[b,start_ch, :,:])**2).mean()

        return

    if domain=='time':
        raise ValueError('time domain implementation not ready yet')

    return J

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(pad) is not tuple:
        pad = (pad, pad)
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    return h, w

def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0, output_pad=0, dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(pad) is not tuple:
        pad = (pad, pad) 
    if type(output_pad) is not tuple:
        output_pad = (output_pad, output_pad) 
    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0] + output_pad[0]
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1] + output_pad[1]
    return h, w

def gen_array_source_locs(room_dim, array_spacing, Nch):
    xyz_source = room_dim*np.random.uniform(size=(3,))
    xyz_array = np.zeros((3,Nch))
    xyz_array[:,0] = room_dim*np.random.uniform(size=(3,))
    dvec = np.random.uniform(size=(3,))
    dvec = dvec/np.linalg.norm(dvec)    
    for m in range(1,Nch):
        xyz_array[:,m] = xyz_array[:,0] + m*array_spacing*dvec
    return xyz_source, xyz_array

def gen_libriconv_dataset(signals_dir, dataset_dir, N_sigs, T_chunk, N_impulses, array_spacing, Nch):

    print('Generating Training Signal')
    rt60 = 0.5  # seconds
    room_dim = np.array([10, 6, 5])  # meters
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    xyz_source, xyz_array = gen_array_source_locs(room_dim, array_spacing, Nch)

    files = [signals_dir+f for f in os.listdir(signals_dir) if f.endswith('.wav') and os.path.isfile(os.path.join(signals_dir, f))]
    files = files[0:N_sigs]     

    for n in range(0,N_impulses):
        room = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order)
        room.add_source(xyz_source) 
        room.add_microphone_array(xyz_array)
        room.compute_rir()

        h = mac.util.get_im_rirs(room)
        h = h/np.max(abs(h))

        for f in files:
            in_audio, fs = sf.read(f)
            L_chunk = T_chunk*fs
            len_conv = len(in_audio) + len(h) - 1
            conv_audio = np.zeros((len_conv, Nch))
            for ch in range(Nch):
                conv_audio[:,ch] = np.convolve(in_audio, h[:,ch])

            N_chunk = int(np.floor(len_conv/L_chunk))

            for c in range(N_chunk):
                out_audio = conv_audio[c*L_chunk + c: (c+1)*L_chunk + c, :]
                out_audio = out_audio/np.max(out_audio)
                sf.write(dataset_dir + os.path.basename(f)[:-4] + '_' + str(n+1) + '_' + str(c) + 's.wav', data=out_audio, samplerate=fs)

def reduce_libriconv_dataset(dataset_dir, output_dir, Nch=4):
    
    print('shortening libriconv dataset')

    for f in os.listdir(dataset_dir):
        if f.endswith('.wav') and os.path.isfile(os.path.join(dataset_dir, f)):
            in_file = dataset_dir+f
            x, fs = sf.read(in_file)
            out_file = output_dir+f 
            sf.write(out_file, data=x[:,0:Nch], samplerate=fs)

class STFT_torch(object):
    def __init__(self, len_w, overlap=None):
        self.len_w = len_w
        self.overlap = overlap 
        if self.overlap == None:
            self.overlap = len_w // 2

    def __call__(self, x):
        N_batch, L, Nch = x.size()
        X0 = torch.stft(x[:,:,0], n_fft=self.len_w, hop_length=self.overlap, window=torch.hann_window(self.len_w, device=x.device), return_complex=False, onesided=True)
        _, N_freq, N_windows, _ = X0.size()
        X = torch.zeros((N_batch, Nch, N_freq, N_windows), dtype=torch.cfloat, device=x.device)
        for ch in range(Nch):
            X[:, ch, :, :] = torch.stft(x[:, :, ch], n_fft=self.len_w, hop_length=self.overlap, window=torch.hann_window(self.len_w, device=x.device), return_complex=True, onesided=True)
        return X

class ISTFT_torch(object):
    def __init__(self, len_w, overlap=None):
        self.len_w = len_w
        self.overlap = overlap 
        if self.overlap == None:
            self.overlap = len_w // 2

    def __call__(self, X, l0):
        N_batch, Nch, N_freqs, N_segs = X.size()
        x = torch.zeros((N_batch, l0, Nch), dtype=torch.float64, device=X.device)
        for b in range(N_batch):
            for ch in range(Nch):
                x[b,:, ch] = torch.istft(X[b,ch,:,:], n_fft=self.len_w, hop_length=self.overlap, window=torch.hann_window(self.len_w, device=X.device), onesided=True, length=l0)
        return x

class LibriConvDataset(Dataset):
    def __init__(self, dataset_dir, N_rtf):
        self.dataset_dir = dataset_dir

    def __len__(self):
        files = [self.dataset_dir+f for f in os.listdir(self.dataset_dir) if f.endswith('.wav') and os.path.isfile(os.path.join(self.dataset_dir, f))]
        return len(files)

    def __getitem__(self, idx):
        files = [self.dataset_dir+f for f in os.listdir(self.dataset_dir) if f.endswith('.wav') and os.path.isfile(os.path.join(self.dataset_dir, f))]
        sample = torch.tensor(sf.read(files[idx])[0])
        return sample

class Split_STFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        N_batch, Nch, N_rtf, N_segs = X.size()
        D = 2*Nch
        device = X.device
        X_split = torch.zeros((N_batch, D, N_rtf, N_segs),device=device)

        for b in range(N_batch):
            for ch in range(Nch):
                X_split[b,2*ch,:,:] = X[b,ch, :, :].real
                X_split[b,2*ch+1,:,:] = X[b,ch, :, :].imag
        return X_split

class Combine_STFT(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X):
        N_batch, D, N_rtf, N_segs = X.size()
        Nch = int(D//2)
        X_comb = torch.zeros((N_batch, Nch, N_rtf, N_segs), dtype=torch.cfloat, device=X.device)

        for b in range(N_batch):
            for ch in range(Nch):
                X_comb[b,ch,:,:].real = X[b,2*ch, :, :]
                X_comb[b,ch,:,:].imag = X[b,2*ch + 1, :, :]
        return X_comb

class NRTF_SC(nn.Module):
    def __init__(self, N_rtf, ref_ch):
        super().__init__()

        self.ref_ch = ref_ch
        self.ref_ind = ref_ch - 1

        self.stft = STFT_torch(len_w = N_rtf)
        self.istft = ISTFT_torch(len_w = N_rtf)

        self.split_stft = Split_STFT()
        self.combine_stft = Combine_STFT()

        # CNN Encoder section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(4, 6, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(6, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 12, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(12, 16, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        # CNN Decoder section
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(16, 12, 3, stride=2, padding=0, output_padding=(1,0)),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 8, 3, stride=2, padding=0, output_padding=(0,0)),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=(0,1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding=0)
        )

        # Flatten layer 
        self.flatten = nn.Flatten(start_dim=1)

        # Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(16,15,15))

        # Linear Encoder Section
        self.encoder_lin = nn.Sequential(
            nn.Linear(16 * 15 * 15, 1024),
            nn.ReLU(True),
            nn.Linear(1024, N_rtf),
            nn.ReLU(True),
            nn.Linear(N_rtf, N_rtf//2 + 1)
        )

        # Linear Decoder Section
        self.decoder_lin = nn.Sequential(
            nn.Linear(N_rtf//2 + 1, N_rtf),
            nn.ReLU(True),
            nn.Linear(N_rtf, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 16 * 15 * 15)
        )

    ### Training function
    def train_epoch(self, device, dataloader, loss_fn, optimizer):
        # Set train mode for both the encoder and the decoder
        self.train()
        train_loss = []
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for input_batch in dataloader: 
            # Move tensor to the proper device
            input_batch = input_batch.to(device)
            output_batch = self(input_batch)
            # Evaluate loss
            loss = loss_fn(output_batch, input_batch)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print batch loss
            # print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def test_epoch(self, device, dataloader, loss_fn, optimizer):
        # Set train mode for both the encoder and the decoder
        self.eval()
        test_loss = []

        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for input_batch in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
            # Move tensor to the proper device
            gpu_input_batch = input_batch.to(device)
            # Encode data
            output_batch = self(gpu_input_batch)
            # Evaluate loss
            loss = loss_fn(output_batch, gpu_input_batch)
            # Print batch loss
            # print('\t partial test loss (single batch): %f' % (loss.data))
            test_loss.append(loss.detach().cpu().numpy())

        return np.mean(test_loss)

    def forward(self, x):
        
        _, l0, _ = x.size()

        X = self.stft(x)

        N_batch, Nch, N_freq, N_times = X.size()

        X_hat = torch.zeros_like(X)
        X_hat[:,self.ref_ind,:,:] = X[:,self.ref_ind,:,:]

        enc_channels = list(range(Nch))
        enc_channels.remove(self.ref_ind)

        # Encoder section
        X_split = self.split_stft(X)
        h_1 = self.encoder_cnn(X_split)
        h_2 = self.flatten(h_1)
        h_3 = self.encoder_lin(h_2)

        # Decoder section
        h_4 = self.decoder_lin(h_3)
        h_5 = self.unflatten(h_4)
        h_6 = self.decoder_cnn(h_5)
        H = self.combine_stft(h_6)

        for ch in enc_channels:
            X_hat[:,ch,:,:] = X[:,self.ref_ind,:,:] * H[:,0,:,:]

        x_hat = self.istft(X_hat, l0)

        return x_hat

def NRTF_sc_train():

    dataset_dir = home + '/mac/datasets/libriconv/'

    N_rtf = 512

    dataset = LibriConvDataset(dataset_dir, N_rtf)
    m = len(dataset)
    train_data, test_data = random_split(dataset, [int(m-m*0.2), int(m*0.2)])
    batch_size=128

    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    print('\ndataset loaded\n')

    loss_fn = torch.nn.MSELoss()

    lr = 0.0005
    torch.manual_seed(0)

    network = NRTF_SC(N_rtf=N_rtf, ref_ch=1)
    params_to_optimise = [
        {'params': network.parameters()},
    ]

    optim = torch.optim.Adam(params_to_optimise, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    print('')
    # Move both the encoder and the decoder to the selected device
    network.to(device)
    start = time.time()
    num_epochs = 200
    diz_loss = {'train_loss':[],'test_loss':[]}
    for epoch in range(num_epochs):
        train_loss = network.train_epoch(device,train_loader,loss_fn,optim)
        test_loss = network.test_epoch(device,test_loader,loss_fn,optim)
        print('\n EPOCH {}/{} \t train loss {} \t test loss {}'.format(epoch + 1, num_epochs, 10*np.log10(train_loss), 10*np.log10(test_loss)))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['test_loss'].append(test_loss)

    stop = time.time()

    ex_time = stop-start

    print('training time = ' + str(ex_time))

    # Plot losses
    plt.figure()
    plt.plot(10*np.log10(np.array(diz_loss['train_loss'])), label='Train')
    plt.plot(10*np.log10(np.array(diz_loss['test_loss'])), label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('dB Loss')
    plt.legend()
    plt.savefig(home + '/mac/hpc/train_nrtf/loss.png')

    torch.save(network, home+'/mac/hpc/train_nrtf/nrtf_model.torch')

class NRTF_MC(nn.Module):
    def __init__(self, N_rtf, Nch):
        super().__init__()

        self.Nch = Nch 
        self.N_rtf = N_rtf

        N_layers = 4
        N_rtfs = 4*(Nch**2) # 2*Nch**2 paths because bi directional, x2 because its split real imag
        delta1 = int(np.floor((N_rtfs - Nch)/N_layers))
        delta2 = int(N_rtfs - (Nch + 3*delta1))

        N0 = 2*Nch
        N1 = N0 + delta1
        N2 = N1 + delta1
        N3 = N2 + delta1
        N4 = N3 + delta2

        print(N0)
        print(N1)
        print(N2)
        print(N3)
        print(N4)

        self.stft = STFT_torch(len_w = N_rtf)
        self.istft = ISTFT_torch(len_w = N_rtf)

        self.split_stft = Split_STFT()
        self.combine_stft = Combine_STFT()

        # CNN Encoder section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(N0, N1, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(N1, N2, 3, stride=2, padding=1),
            nn.BatchNorm2d(N2),
            nn.ReLU(True),
            nn.Conv2d(N2, N3, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(N3, N4, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        # CNN Decoder section
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(N4, N4, 3, stride=2, padding=0, output_padding=(1,0)),
            nn.ReLU(True),
            nn.ConvTranspose2d(N4, N4, 3, stride=2, padding=0, output_padding=(0,0)),
            nn.BatchNorm2d(N4),
            nn.ReLU(True),
            nn.ConvTranspose2d(N4, N4, 3, stride=2, padding=1, output_padding=(0,1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(N4, N4, 3, stride=2, padding=1, output_padding=0)
        )

        # Flatten layer 
        self.flatten = nn.Flatten(start_dim=1)

        # Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(N4,15,15))

        # Linear Encoder Section
        self.encoder_lin = nn.Sequential(
            nn.Linear(N4 * 15 * 15, 1024),
            nn.ReLU(True),
            nn.Linear(1024, N_rtf),
            nn.ReLU(True),
            nn.Linear(N_rtf, N_rtf//2 + 1)
        )

        # Linear Decoder Section
        self.decoder_lin = nn.Sequential(
            nn.Linear(N_rtf//2 + 1, N_rtf),
            nn.ReLU(True),
            nn.Linear(N_rtf, 1024),
            nn.ReLU(True),
            nn.Linear(1024, N4 * 15 * 15)
        )


    def forward(self, x):
        
        _, l0, _ = x.size()

        X = self.stft(x)

        N_batch, Nch, N_freq, N_times = X.size()

        if self.Nch != Nch:
            raise ValueError('channel numbers dont match')

        # Encoder section
        X_split = self.split_stft(X)
        h_1 = self.encoder_cnn(X_split)
        h_2 = self.flatten(h_1)
        h_3 = self.encoder_lin(h_2)

        # Decoder section
        h_4 = self.decoder_lin(h_3)
        h_5 = self.unflatten(h_4)
        h_6 = self.decoder_cnn(h_5)
        H = self.combine_stft(h_6)

        return H

    ### Training function
    def train_epoch(self, device, dataloader, optimizer):
        # Set train mode for both the encoder and the decoder
        self.train()
        train_loss = []
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for x in dataloader: 
            # Move tensor to the proper device
            x = x.to(device)
            H = self(x)
            Nch = (x.size())[-1]
            X = self.stft(x)
            # Evaluate loss
            loss = Reciprocity_Cost(X, H, path_order=Nch, device=device)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print batch loss
            # print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def test_epoch(self, device, dataloader):
        # Set train mode for both the encoder and the decoder
        self.eval()
        test_loss = []

        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for x in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
            # Move tensor to the proper device
            x = x.to(device)
            # Encode data
            H = self(x)
            X = self.stft(x)
            Nch = (x.size())[-1]
            # Evaluate loss
            loss = Reciprocity_Cost(X, H, path_order=Nch, device=device)
            # Print batch loss
            # print('\t partial test loss (single batch): %f' % (loss.data))
            test_loss.append(loss.detach().cpu().numpy())

        return np.mean(test_loss)

def NRTF_mc_train():

    dataset_dir = home + '/mac/datasets/libriconv_4ch/'

    N_rtf = 512
    Nch = 4

    dataset = LibriConvDataset(dataset_dir, N_rtf)
    m = len(dataset)
    train_data, test_data = random_split(dataset, [int(m-m*0.2), int(m*0.2)])
    batch_size = 64

    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    print('\ndataset loaded\n')

    lr = 0.005
    torch.manual_seed(0)

    network = NRTF_MC(N_rtf=N_rtf, Nch=Nch)
    params_to_optimise = [{'params': network.parameters()},]

    optim = torch.optim.Adam(params_to_optimise, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    print('')
    # Move both the encoder and the decoder to the selected device
    network.to(device)
    start = time.time()
    num_epochs = 50
    diz_loss = {'train_loss':[],'test_loss':[]}
    for epoch in range(num_epochs):
        train_loss = network.train_epoch(device, train_loader, optim)
        test_loss = network.test_epoch(device, test_loader, optim)
        print('\n EPOCH {}/{} \t train loss {} \t test loss {}'.format(epoch + 1, num_epochs, 10*np.log10(train_loss), 10*np.log10(test_loss)))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['test_loss'].append(test_loss)

    stop = time.time()

    ex_time = stop-start

    print('training time = ' + str(ex_time))

    torch.save(network, home+'/mac/hpc/train_nrtf/nrtf_model.torch')

    # Plot losses
    plt.figure()
    plt.plot(10*np.log10(np.array(diz_loss['train_loss'])), label='Train')
    plt.plot(10*np.log10(np.array(diz_loss['test_loss'])), label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('dB Loss')
    plt.legend()
    plt.savefig(home + '/mac/hpc/train_nrtf/loss.png')


def nrtf_example():

    N_rtf = 512

    home = os.path.expanduser('~')
    dataset_dir = home + '/mac/datasets/libriconv/'

    dataset = LibriConvDataset(dataset_dir, N_rtf)
    m = len(dataset)
    train_data, test_data = random_split(dataset, [int(m-m*0.2), int(m*0.2)])

    home = os.path.expanduser('~')
    dataset_dir = home + '/mac/datasets/libriconv/'
    network = torch.load(home+'/mac/hpc/train_nrtf/nrtf_model.torch', map_location=torch.device('cpu'))

    x_example = torch.unsqueeze(test_data[0].detach(), 0)
    x_example_out = network(x_example).detach()

    xe = torch.squeeze(x_example).numpy()
    xe_out = torch.squeeze(x_example_out).numpy()

    plt.figure()
    plt.plot(xe[:,0])
    plt.plot(xe_out[:,0], '--')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.savefig(home + '/mac/hpc/train_nrtf/example_ch1.png')

    plt.figure()
    plt.plot(xe[:,1])
    plt.plot(xe_out[:,1], '--')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.savefig(home + '/mac/hpc/train_nrtf/example_ch2.png')

    sf.write(home + '/mac/hpc/train_nrtf/input_ch1.wav', data=xe[:,0], samplerate=16000)
    sf.write(home + '/mac/hpc/train_nrtf/input_ch2.wav', data=xe[:,1], samplerate=16000)
    sf.write(home + '/mac/hpc/train_nrtf/output_ch1.wav', data=xe_out[:,0], samplerate=16000)
    sf.write(home + '/mac/hpc/train_nrtf/output_ch2.wav', data=xe_out[:,1], samplerate=16000)

if __name__ == "__main__":

    NRTF_mc_train()
    # NRTF_sc_train()
    # nrtf_example()