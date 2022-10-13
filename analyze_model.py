import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

from omegaconf import DictConfig
from tqdm import tqdm
from mbstoi import mbstoi

from DCNN.trainer import DCNNLightniningModule
from DCNN.datasets.base_dataset import BaseDataset
from DCNN.loss_old import si_snr

SR = 16000


def load_model(config, model_checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DCNNLightniningModule(config)
    model.eval()

    torch.set_grad_enabled(False)
    checkpoint = torch.load(model_checkpoint_path,
                            map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    if device == "cuda":
        model.cuda()
    return model


def analyze_batch(batch, model):
    noisy_signals, clean_signals = batch

    if torch.cuda.is_available():
        noisy_signals, clean_signals = noisy_signals.cuda(), clean_signals.cuda()

    batch_size = noisy_signals.shape[0]

    enhanced_signals = model(noisy_signals)

    # 1. Compute SNR for noisy and enhanced signal
    snr_l_n = si_snr(noisy_signals[:, 0],
                     clean_signals[:, 0], reduce_mean=False).detach().cpu().numpy()
    snr_r_n = si_snr(noisy_signals[:, 1],
                     clean_signals[:, 1], reduce_mean=False).detach().cpu().numpy()
    snr_noisy = (snr_l_n + snr_r_n)/2

    snr_l_n = si_snr(enhanced_signals[:, 0],
                     clean_signals[:, 0], reduce_mean=False).detach().cpu().numpy()
    snr_r_n = si_snr(enhanced_signals[:, 1],
                     clean_signals[:, 1], reduce_mean=False).detach().cpu().numpy()
    snr_enhanced = (snr_l_n + snr_r_n)/2

    # 2. Compute MBSTOI for noisy and enhanced signals

    noisy_signals = noisy_signals.detach().cpu().numpy()
    clean_signals = clean_signals.detach().cpu().numpy()
    enhanced_signals = enhanced_signals.detach().cpu().numpy()
    mbstoi_noisy = np.array([
        mbstoi(clean_signals[i, 0], clean_signals[i, 1],
               noisy_signals[i, 0], noisy_signals[i, 1], fsi=SR)
        for i in range(batch_size)
    ])
    mbstoi_enhanced = np.array([
        # stoi(clean_signals[i], enhanced_signals[i], SR
        mbstoi(clean_signals[i, 0], clean_signals[i, 1],
               enhanced_signals[i, 0], enhanced_signals[i, 1], fsi=SR)
        for i in range(batch_size)
    ])

    return {
        "snr_noisy": snr_noisy,
        "snr_enhanced": snr_enhanced,
        "mbstoi_noisy": mbstoi_noisy,
        "mbstoi_enhanced": mbstoi_enhanced
    }


@hydra.main(config_path="config", config_name="config")
def analyze_dataset(config: DictConfig):
    """Runs the training procedure using Pytorch lightning
    And tests the model with the best validation score against the test dataset. 
    Args:
        config (DictConfig): Configuration automatically loaded by Hydra.
                                        See the config/ directory for the configuration
    """

    dataset = BaseDataset(config["dataset"]["noisy_test_dataset_dir"],
                          config["dataset"]["target_test_dataset_dir"], SR)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config["training"]["n_workers"]
    )

    model = load_model(config, config["training"]["checkpoint_path"])
    results = [
        analyze_batch(batch, model)
        for i, batch in tqdm(enumerate(dataloader))
    ]

    results = _merge_dicts(results)

    _plot_scatter_graphs(results)


def _merge_dicts(dicts):
    output_dict = {}
    for key in dicts[0].keys():
        output_dict[key] = np.concatenate([d[key] for d in dicts], axis=None)

    return output_dict


def _plot_scatter_graphs(results, savefig=True):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    n_points = len(results["snr_enhanced"])

    # 2. Plot original SNR versus STOI of noisy signals
    axs[0, 0].scatter(results["snr_noisy"],
                      results["mbstoi_enhanced"] - results["mbstoi_noisy"])
    axs[0, 0].set_title("MBSTOI improvement for every SNR")
    axs[0, 0].set_xlabel("SNR")
    axs[0, 0].set_ylabel("MBSTOI improvement")

    # 1. Plot SNR versus STOI of enhanced signals
    axs[0, 1].scatter(results["snr_enhanced"], results["mbstoi_enhanced"])
    axs[0, 1].set_title("MBSTOI of enhanced signals versus SNR")
    axs[0, 1].set_xlabel("SNR")
    axs[0, 1].set_ylabel("MBSTOI")

    # 3. Plot Noisy SNR versus enhanced SNR
    axs[1, 0].scatter(results["snr_noisy"], results["snr_enhanced"])
    axs[1, 0].set_title("SNR before and after enhancement")
    axs[1, 0].set_xlabel("Input SNR (noisy)")
    axs[1, 0].set_ylabel("Output SNR (enhanced)")
    expected_snr = np.linspace(-30, 30, n_points)
    axs[1, 0].plot(expected_snr, expected_snr, color="red")

    # 3. Plot Noisy STOI versus enhanced STOI
    axs[1, 1].scatter(results["mbstoi_noisy"], results["mbstoi_enhanced"])
    axs[1, 1].set_title("MBSTOI before and after enhancement")
    axs[1, 1].set_xlabel("Input MBSTOI (noisy)")
    axs[1, 1].set_ylabel("Output MBSTOI (enhanced)")
    expected_mbstoi = np.linspace(0, 1, n_points)

    axs[1, 1].plot(expected_mbstoi, expected_mbstoi, color="red")

    plt.tight_layout()
    if savefig:
        plt.savefig("bla.png")
    else:
        plt.show()


if __name__ == "__main__":
    analyze_dataset()
