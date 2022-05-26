import torch
import torchaudio
import os

SR = 16000
# N_MICROPHONE_SECONDS = 1
# N_MICS = 4



class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 noisy_dataset_dir,
                 target_dataset_dir,
                 sr=SR,
                #  n_microphone_seconds=N_MICROPHONE_SECONDS,
                #  n_mics=N_MICS,
                #  metadata_dir=None
                ):

        self.sr = sr
        # self.n_microphone_seconds = n_microphone_seconds
        # self.sample_duration = self.sr*self.n_microphone_seconds
        # self.n_mics = n_mics
        # self.metadata_dir = metadata_dir
        self.target_dataset_dir = target_dataset_dir
        self.noisy_dataset_dir = noisy_dataset_dir

        # self.df = _load_dataframe(dataset_dir, metadata_dir)

    def __len__(self):
        audio_files = os.listdir(self.target_dataset_dir)
        return len(audio_files)

    def __getitem__(self, index):
        clean_audio_sample_path = self._get_audio_sample_path(index,self.target_dataset_dir)
        noisy_audio_sample_path = self._get_audio_sample_path(index,self.noisy_dataset_dir)
        #path = os.path.dirname(self.audio_dir)
        clean_signal, _ = torchaudio.load(clean_audio_sample_path)
        noisy_signal, _ = torchaudio.load(noisy_audio_sample_path)

        return (noisy_signal, clean_signal)

    def _get_audio_sample_path(self, index,dataset_dir):
        audio_files = sorted(os.listdir(dataset_dir))
        path = os.path.join(dataset_dir, audio_files[index])

        return path

    

