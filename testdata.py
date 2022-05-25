from DCNN.datasets import base_dataset 


clean = '/Users/vtokala/Documents/Research/di_nn/Dataset/clean_trainset_1f'
noisy = '/Users/vtokala/Documents/Research/di_nn/Dataset/noisy_trainset_1f'

train_data = base_dataset.BaseDataset(clean,noisy)
print(train_data[0])