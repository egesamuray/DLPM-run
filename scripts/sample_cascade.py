import argparse, os
import torch as th
import numpy as np
from improved_diffusion import dist_util, logger
from improved_diffusion.wavelet_datasets import wavelet_to_image, wavelet_stats
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def create_argparser():
    defaults = model_and_diffusion_defaults(task="wavelet")
    defaults.update(
        dict(
            model_root="logs/cascade",
            j=3,
            num_samples=1,
            batch_size=1,
            wavelet="db4",
            border_condition="periodization",
            final_size=256,
        )
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    models, diffusions, means, stds = [], [], [], []
    for scale in range(args.j, 0, -1):
        ckpt = os.path.join(args.model_root, f"scale{scale}", f"model{args.max_training_steps:06d}.pt")
        model, diffusion = create_model_and_diffusion(
            task="wavelet",
            conditioning_channels=args.conditioning_channels,
            **args_to_dict(args, model_and_diffusion_defaults(task="wavelet").keys(), j=scale),
        )
        model.load_state_dict(dist_util.load_state_dict(ckpt, map_location="cpu"))
        model.to(dist_util.dev())
        model.eval()
        models.append(model)
        diffusions.append(diffusion)
        mean, std = wavelet_stats(scale, args.model_root)
        means.append(mean.to(dist_util.dev()))
        stds.append(std.to(dist_util.dev()))

    low_freq = None
    for scale in range(args.j, 0, -1):
        diffusion = diffusions[args.j - scale]
        model = models[args.j - scale]
        shape = (args.batch_size, 9 if scale < args.j or args.conditional else 3, args.large_size, args.large_size)
        sample_fn = diffusion.p_sample_loop
        model_kwargs = {}
        if low_freq is not None:
            model_kwargs["conditional"] = low_freq
        sample = sample_fn(model, shape, model_kwargs=model_kwargs)
        if scale > 1:
            wave = th.cat((sample, low_freq), dim=1) if low_freq is not None else sample
            wave = wave * stds[args.j - scale][:, None, None] + means[args.j - scale][:, None, None]
            low_freq = th.from_numpy(wavelet_to_image(wave.cpu().numpy(), args.border_condition, args.wavelet, output_size=args.large_size)).to(dist_util.dev())
        else:
            final = (sample + 1) * 127.5
            arr = final.clamp(0,255).to(th.uint8).permute(0,2,3,1).contiguous()
            np.savez(os.path.join(args.model_root, "samples.npz"), arr.cpu().numpy())


if __name__ == "__main__":
    main()
