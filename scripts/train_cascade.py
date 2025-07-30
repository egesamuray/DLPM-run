import argparse, os
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.wavelet_datasets import load_data_wavelet


def create_argparser():
    defaults = model_and_diffusion_defaults(task="wavelet")
    defaults.update(
        dict(
            data_dir="",
            j=3,
            batch_size=1,
            log_root="logs/cascade",
            max_training_steps=50000,
        )
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()

    for scale in range(args.j, 0, -1):
        dist_util.setup_dist()
        logdir = os.path.join(args.log_root, f"scale{scale}")
        logger.configure(dir=logdir)
        logger.log(f"Training scale {scale}")

        model, diffusion = create_model_and_diffusion(
            task="wavelet",
            conditioning_channels=args.conditioning_channels,
            **args_to_dict(args, model_and_diffusion_defaults(task="wavelet").keys(), j=scale),
        )
        model.to(dist_util.dev())

        data = load_data_wavelet(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            j=scale,
            conditional=args.conditional,
        )

        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint="",
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=diffusion.schedule,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            max_training_steps=args.max_training_steps,
        ).run_loop()


if __name__ == "__main__":
    main()
