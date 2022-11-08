import torch    
import torch.nn.functional as F


def apply_mask(x, specs, fft_len, masking_mode="E"):
    real = specs[:, :fft_len // 2 + 1]
    imag = specs[:, fft_len // 2 + 1:]
    spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
    spec_phase = torch.atan2(imag, real)

    mask_real, mask_imag = x[:, 0], x[:, 1]
    mask_real = F.pad(mask_real, [0, 0, 1, 0])
    mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

    if masking_mode == "E":
        mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        real_phase = mask_real / (mask_mags + 1e-8)
        imag_phase = mask_imag / (mask_mags + 1e-8)
        mask_phase = torch.atan2(
            imag_phase,
            real_phase
        )

        # mask_mags = torch.clamp_(mask_mags,0,100)
        mask_mags = torch.tanh(mask_mags)
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        real = est_mags * torch.cos(est_phase)
        imag = est_mags * torch.sin(est_phase)
    elif masking_mode == "C":
        real, imag = real * mask_real - imag * \
            mask_imag, real * mask_imag + imag * mask_real
    elif masking_mode == "R":
        real, imag = real * mask_real, imag * mask_imag

    # Generate output signal
    out_spec = torch.cat([real, imag], 1)

    return out_spec
