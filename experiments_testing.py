import yaml
# !pip install tensorboardX
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
from train import train
import torch
from DCNN.trainer import DCNNLightningModule
from DCNN.utils import evalFunction
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from DCNN.matFileGen import writeMatFile
from DCNN.datasets.test_dataset import BaseDataset
from glob import glob
import soundfile as sf
import os

config = {
    "defaults": [
        "model",
        "training",
        {"dataset": "speech_dataset"}
    ],
    "dataset":{
        "noisy_training_dataset_dir": "/Users/vtokala/Documents/Research/di_nn/Dataset/noisy_trainset_1f",
        "noisy_validation_dataset_dir": "/Users/vtokala/Documents/Research/di_nn/Dataset/noisy_valset_1f",
        "noisy_test_dataset_dir": "/Users/vtokala/Documents/Research/di_nn/Dataset/noisy_testset_1f",
        "target_training_dataset_dir": "/Users/vtokala/Documents/Research/di_nn/Dataset/clean_trainset_1f",
        "target_validation_dataset_dir": "/Users/vtokala/Documents/Research/di_nn/Dataset/clean_valset_1f",
        "target_test_dataset_dir": "/Users/vtokala/Documents/Research/di_nn/Dataset/clean_testset_1f"
#         "target_training_dataset_dir": "/kaggle/input/binauralspeech30db/Clean_trainset_30dB",
#         "target_validation_dataset_dir": "/kaggle/input/binauralspeech30db/Clean_valset_30dB",
#         "target_test_dataset_dir": "/kaggle/input/binauralspeech30db/Clean_testset_30dB"
    },
    "training":{
        "batch_size": 32,
        "n_epochs": 40,
        "n_workers": 4,
        "learning_rate":0.0001,
        "train_checkpoint_path": None, #"/kaggle/working/SE_DCNN/DCNN/checkpoints/weights-epoch=19-validation_loss=-17.90.ckpt",
        "strategy": "ddp",
        "pin_memory": True,
        "accelerator": "cuda"
    },
    "model":{
        "rtf_weight" : 0,
        "snr_weight" : 1,
        "ild_weight": 1, #0.1,
        "ipd_weight": 10, #10,
        "stoi_weight": 10, #10
        "kurt_weight": 0,
        "avg_mode": "time",
        "attention": True,
        'sdr_weight': 0,
        'mse_weight': 0,
        'si_sdr_weight':0,
        'si_snr_weight':0,
        'comp_loss_weight':0,
        'msc_weight':0
        
    
    }
}

with open('config/config.yaml', 'w') as f_yaml:
    yaml.dump(config, f_yaml)
GlobalHydra.instance().clear()
initialize(config_path="./config")
config = compose("config")

test_name = 'PL_RVRB'
evalMet = evalFunction.EvalMetrics()
MODEL_CHECKPOINT_PATH = "/Users/vtokala/Documents/Research/di_nn/DCNN/checkpoints/BCCRN_HPC_PL.ckpt"
# MODEL_CHECKPOINT_PATH = "/kaggle/input/lss-resources/code/se/demo/last.ckpt"
model = DCNNLightningModule(config)
model.eval()
torch.set_grad_enabled(False)
device = torch.device('cpu')
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
# checkpoint = torch.load(MODEL_CHECKPOINT_PATH)
model.load_state_dict(checkpoint["state_dict"], strict=False)

paths=glob("/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/ICASSP_Testset/audio_files/*/", recursive = True)
pathsEn=glob("/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/ICASSP_Testset/audio_files/*/", recursive = True)



for j in range(len(paths)):
    
    paths = sorted(paths)
    pathsEn = sorted(pathsEn)
    NOISY_DATASET_PATH = os.path.join(paths[j],"Noisy_testset/")
    print(NOISY_DATASET_PATH)
    CLEAN_DATASET_PATH = os.path.join(paths[j],"Clean_testset/")
    ENHANCED_DATASET_PATH = os.path.join(pathsEn[j],"BCCRN_"+test_name+"/")
    dataset = BaseDataset(NOISY_DATASET_PATH, CLEAN_DATASET_PATH, mono=False)
    
    if os.path.isdir(ENHANCED_DATASET_PATH):
        print("Folder for Enhanced Signals Exists!")
        print(ENHANCED_DATASET_PATH)
        
        if os.path.exists(ENHANCED_DATASET_PATH):
    # Get a list of files and subdirectories in the folder
            contents = os.listdir(ENHANCED_DATASET_PATH)
            print(ENHANCED_DATASET_PATH)
            
    
        # Iterate over the contents and delete files or subdirectories
        for item in contents:
            item_path = os.path.join(ENHANCED_DATASET_PATH, item)
            if os.path.isfile(item_path):
                os.remove(item_path)  # Delete files
            elif os.path.isdir(item_path):
                # Delete subdirectories' contents recursively
                for root, dirs, files in os.walk(item_path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir in dirs:
                        os.rmdir(os.path.join(root, dir))
                # Remove the empty subdirectory
                os.rmdir(item_path)

            print(f"Contents of folder '{ENHANCED_DATASET_PATH}' have been deleted.")
    else:
        print("Folder for Enhanced Signals does not exist! - Creating Folder!!")
        os.mkdir(ENHANCED_DATASET_PATH)
    
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False
        

    )


    dataloader = iter(dataloader)
    # k = len(dataloader)

    for i in range (len(dataloader)): # Enhance 10 samples
        try:
            batch = next(dataloader)

        except StopIteration:
            break
    #     print(os.path.basename(batchEn[2][0]))

        noisy_samples = (batch[0])
        clean_samples = (batch[1])[0]
        model_output = model(noisy_samples)[0].detach().cpu()
        model_output = model_output/torch.max(model_output)
        # print(model_output.shape)

        # breakpoint()
    #     torchaudio.save(path, waveform, sample_rate)
        sf.write(ENHANCED_DATASET_PATH + os.path.basename(batch[2][0])[:len(os.path.basename(batch[2][0]))-4] + "_" + test_name + ".wav", model_output.numpy().transpose(), 16000) 
        # print(ENHANCED_DATASET_PATH + os.path.basename(batch[2][0])[:len(os.path.basename(batch[2][0]))-4] + "_DCCTN.wav")
        print(f"===== Computing Signal {i+1} of ", len(dataloader),"=====")