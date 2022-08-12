import mac
import os 
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import networkx as nx

# N_rtf = 512

# home = os.path.expanduser('~')
# dataset_dir = home + '/mac/datasets/libriconv/'
# dataset = mac.nrtf.LibriConvDataset(dataset_dir, N_rtf)
# m = len(dataset)
# train_data, test_data = random_split(dataset, [int(m-m*0.2), int(m*0.2)])

# X_example = torch.unsqueeze(test_data[0].detach(), 0)
# Xe = torch.squeeze(X_example).numpy()

def test_nrtf_graph():
    cycles = mac.nrtf.RTF_graph_cycles(4,4)

if __name__ == '__main__':
    test_nrtf_graph()

    