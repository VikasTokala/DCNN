from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from DCNN.loss import si_snr
from DCNN.utils.eval_utils import ild_db, ipd_rad, speechMask
from DCNN.feature_extractors import Stft, IStft
from DCNN.datasets.base_dataset import BaseDataset
import torch.nn as nn
import torch
from mbstoi import mbstoi
import warnings
warnings.simplefilter('ignore')

SR=16000

class EvalMetrics(nn.Module):
    def __init__(self, win_len=400, win_inc=100, fft_len=512) -> None:
        super().__init__()

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.fbins = int(fft_len/2 + 1)
        self.stft = Stft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
        self.istft = IStft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
        self.stoi = ShortTimeObjectiveIntelligibility(fs=16000)

    def forward(self,NOISY_DATASET_PATH, CLEAN_DATASET_PATH, model, testset_len=5):

        dataset = BaseDataset(NOISY_DATASET_PATH,
                              CLEAN_DATASET_PATH, mono=False)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        dataloader = iter(dataloader)
        # testset_len = len(dataloader)
        



        masked_ild_error = torch.zeros((testset_len, self.fbins))
        masked_ipd_error = torch.zeros((testset_len, self.fbins))


        
        for i in range(testset_len):  # Enhance 10 samples
            try:
                batch = next(dataloader)
            except StopIteration:
                break


            noisy_samples = (batch[0])
            clean_samples = (batch[1])[0]
            model_output = model(noisy_samples)[0]
            clean_samples=(clean_samples)/(torch.max(clean_samples))
            model_output=(model_output)/(torch.max(model_output))


            noisy_stft_l = self.stft(noisy_samples[0][0, :])
            noisy_stft_r = self.stft(noisy_samples[0][1, :])

            enhanced_stft_l = self.stft(model_output[0, :])
            enhanced_stft_r = self.stft(model_output[1, :])

            target_stft_l = self.stft(clean_samples[0, :])
            target_stft_r = self.stft(clean_samples[1, :])
            

            

            mask = speechMask(target_stft_l,target_stft_r).squeeze(0)
            target_ild = ild_db(target_stft_l.abs(), target_stft_r.abs())
            enhanced_ild = ild_db(enhanced_stft_l.abs(), enhanced_stft_r.abs())

            target_ipd = ipd_rad(target_stft_l, target_stft_r)
            enhanced_ipd = ipd_rad(enhanced_stft_l, enhanced_stft_r)

            ild_loss = (target_ild - enhanced_ild).abs()

            ipd_loss = (target_ipd - enhanced_ipd).abs()
            
            mask_sum=mask.sum(dim=1)
            mask_sum[mask_sum==0]=1
        
            masked_ild_error[i,:] = (ild_loss*mask).sum(dim=1)/ mask_sum
            masked_ipd_error[i,:] = (ipd_loss*mask).sum(dim=1)/ mask_sum
            
            
        
            print('Processed Signal ', i+1 , ' of ', testset_len)

    

       

        return masked_ild_error, masked_ipd_error