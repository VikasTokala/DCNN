import matplotlib.pyplot as plt
from pathlib import Path
from DCNN.trainer import DCNNLightniningModule
from DCNN.datasets import create_torch_dataloader
from tqdm import tqdm
from omegaconf import DictConfig
from joblib import Parallel, delayed
import hydra
import numpy as np
import soundfile
import torch
import sys
sys.path.append("/Users/vtokala/Documents/Research")
import mac

SR = 16000


def rtf_function(sig_in, ref_ch=1, N_rtf=1024):
    rtf_f, e, y_hat = mac.est.fblms_est(sig_in, ref_ch, N_rtf)
    _, _, N_segs = np.shape(rtf_f)
    M = N_rtf  # length of the rtf
    Nch = 2
    rtf_td = np.zeros((M, Nch, N_segs))

    for n in range(N_segs):
        rtf_td[:, :, n] = np.real(np.fft.ifft(rtf_f[:, :, n], axis=0))[:M, :]

    # breakpoint()
    rtf_td = rtf_td[:, 1, 1]
    # rtf_td = np.mean(rtf, axis=1)
    plt.figure()
    plt.plot(rtf_td)
    plt.show
    return rtf_td, y_hat


def load_model(config, MODEL_CHECKPOINT_PATH):
    model = DCNNLightniningModule(config)
    model.eval()

    torch.set_grad_enabled(False)
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH,
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    return model


def save_enhanced_binaural_signals(batch, batch_idx, batch_size, output_dir, model):
    noisy, clean = batch

    start_idx = batch_idx*batch_size
    n_signals = clean.shape[0]

    enhanced_0 = model(noisy[:, 0]).numpy()

    for i in range(n_signals):
        output_idx = start_idx + i
        # breakpoint()
        rtf = rtf_function(noisy[i].transpose(1, 0), ref_ch=1)
        # enhanced_0 = model(noisy[i, 0].unsqueeze(0))[0].numpy()
        enhanced_1 = np.convolve(enhanced_0[i], rtf[0], "same")
        output = np.stack([enhanced_0[i], enhanced_1])
        # breakpoint()
        soundfile.write(
            output_dir / f"{output_idx}.wav", output.transpose(1, 0), SR)
        print("Processed file number ", output_idx)

        

@hydra.main(config_path="config", config_name="config")
def test(config: DictConfig):
    """Runs the training procedure using Pytorch lightning
    And tests the model with the best validation score against the test dataset. 
    Args:
        config (DictConfig): Configuration automatically loaded by Hydra.
                                        See the config/ directory for the configuration
    """

    dataset_test = create_torch_dataloader(config, "test_rtf")
    model = load_model(config, config["test_rtf"]["checkpoint_path"])
    output_dir = Path(config["test_rtf"]["output_dir"])
    results = Parallel(n_jobs=config["test_rtf"]["n_workers"])(
        delayed(save_enhanced_binaural_signals)(
            batch, i, config["test_rtf"]["batch_size"], output_dir, model)
        for i, batch in tqdm(enumerate(dataset_test))
    )


if __name__ == "__main__":
    test()
