import argparse
import logging
import os
import time
import warnings

import torch
import torch.optim as optim
import numpy as np
import wandb
from omegaconf import OmegaConf
from accelerate import Accelerator

from src.models.model_factory import ModelFactory
from src.trainers.get_optim_n_scheduler import get_optimizer
from src.trainers.generic_trainer import GenericTrainer
from src.trainers.utils import set_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logging.getLogger("torch").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")


DATASET_MODULE_MAP = {
    "ADHD": "src.datasets.adhd",
    "HCP": "src.datasets.hcp",
    "UKB": "src.datasets.ukb",
    "HBN_movie_clf": "src.datasets.hbn_movie_clf",
}


def parse_args():
    parser = argparse.ArgumentParser(description="TABLeT Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--test-set-id", type=int, default=0, help="Test set ID for the dataset.")
    parser.add_argument("--num-axis", type=int, default=1, help="Number of axes for the model.")
    parser.add_argument("--wandb-name", type=str, default="default_run", help="Name for the wandb run.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory for epoch checkpoints. Leave None to disable checkpointing.")
    parser.add_argument("--resume_checkpoint_dir", type=str, default=None,
                        help="Path to a checkpoint file to resume training from.")
    parser.add_argument("--logit_save_dir", type=str, default=None,
                        help="Directory to save intermediate validation logits for debugging.")
    parser.add_argument("--tff_phase1_weight_path", type=str, default=None,
                        help="Path to the pretrained weights for TFF Phase 1.")
    parser.add_argument("--tff_phase2_weight_path", type=str, default=None,
                        help="Path to the pretrained weights for TFF Phase 2.")
    parser.add_argument("--tablet_pretrained_weight_path", type=str, default=None,
                        help="Path to the pretrained weights for TABLeT.")
    return parser.parse_args()


def get_dataloaders(config, args):
    """Import and call the appropriate dataset's get_dataloaders function."""
    import importlib
    module_path = DATASET_MODULE_MAP[config.data.dataset]
    dataset_module = importlib.import_module(module_path)
    get_dataloaders_fn = dataset_module.get_dataloaders

    dataloader_kwargs = dict(
        data_type=config.data.data_type,
        root_dir=config.data.root_dir,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        seed=args.seed,
        test_set_id=args.test_set_id,
        batch_size=args.batch_size,
        num_workers=config.data.num_workers,
        num_frames=config.data.num_frames,
        task=config.data.task,
        frame_iid=config.data.frame_iid,
        num_input_frames=config.data.num_input_frames,
        num_axis=args.num_axis,
        slice_axis=config.data.slice_axis,
        class_balanced=config.data.class_balanced,
        FD=config.data.FD,
        random_sample_frames=config.data.random_sample_frames,
    )

    if config.data.dataset in ("HCP", "UKB"):
        dataloader_kwargs["img_dir"] = config.data.img_dir

    if config.data.dataset == "HBN_movie_clf":
        dataloader_kwargs["first_consecutive_frames"] = config.data.first_consecutive_frames

    return get_dataloaders_fn(**dataloader_kwargs)


def build_trainer(config, args, model, train_loader, val_loader, test_loader,
                  optimizer, scheduler, accelerator):
    """Instantiate the appropriate trainer based on task/model config."""
    if config.data.task == 'tff_phase_1':
        from src.trainers.tff_phase_1_trainer import TFFPhase1Trainer
        return TFFPhase1Trainer(
            model, train_loader, val_loader, test_loader,
            optimizer, scheduler, config, args, accelerator, logger, None
        )
    elif config.data.task == 'tff_phase_2':
        from src.trainers.tff_phase_2_trainer import TFFPhase2Trainer
        return TFFPhase2Trainer(
            model, train_loader, val_loader, test_loader,
            optimizer, scheduler, config, args, accelerator, logger, None,
            phase1_weight_path=args.tff_phase1_weight_path
        )
    elif config.model.name == 'Encoder_Transformer_finetune':
        from src.trainers.tff_phase_3_trainer import TFFPhase3Trainer
        return TFFPhase3Trainer(
            model, train_loader, val_loader, test_loader,
            optimizer, scheduler, config, args, accelerator, logger, None,
            phase2_weight_path=args.tff_phase2_weight_path
        )
    else:
        return GenericTrainer(
            model, train_loader, val_loader, test_loader,
            optimizer, scheduler, config, args, accelerator, logger, None
        )


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    mixed_precision_mapping = {
        "float32": "no",
        "float16": "fp16",
        "bfloat16": "bf16",
        "float8": "fp8",
    }
    precision = mixed_precision_mapping.get(config.train.mixed_precision, "no")
    logger.info(f"Precision: {precision}")
    accelerator = Accelerator(mixed_precision=precision)

    do_checkpointing = args.checkpoint_dir is not None

    if accelerator.is_main_process:
        logger.info(f"[Config]\n{OmegaConf.to_yaml(config)}\n")
        logger.info(f"[Accelerator state]\n{accelerator.state}\n")
        if do_checkpointing:
            os.makedirs(args.checkpoint_dir, exist_ok=True)

    set_seed(42)

    # Prepare dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config, args)

    # Initialize the model
    model_factory = ModelFactory(config, args)
    model = model_factory.create_model()

    # Set up optimizer and scheduler
    optimizer = get_optimizer(config, args, model)

        model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader
        )

        total_steps = args.epochs * len(train_loader)
        if config.train.use_warmup:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=total_steps // 10
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[total_steps // 10]
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

    if accelerator.is_main_process:
        logger.info(f"Train dataloader batches: {len(train_loader)}")
        logger.info(f"Val dataloader batches: {len(val_loader)}")
        logger.info(f"Test dataloader batches: {len(test_loader)}")

    # Build trainer
    trainer = build_trainer(config, args, model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, accelerator)

    # Resume from checkpoint if specified
    start_epoch = 0
    wandb_id = None
    if args.resume_checkpoint_dir is not None:
        logger.info(f"Resuming from checkpoint {args.resume_checkpoint_dir}")
        ckpt = torch.load(args.resume_checkpoint_dir, map_location="cpu")

        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        trainer.val_best_epoch = ckpt["val_best_epoch"]
        trainer.val_best_metric = ckpt["val_best_metric"]
        trainer.test_metrics = ckpt["test_metrics"]

        start_epoch = ckpt["epoch"] + 1
        wandb_id = ckpt["wandb_run_id"]
        logger.info(f"Continuing at epoch {start_epoch}")

    # Initialize wandb (main process only)
    if accelerator.is_main_process:
        wandb.init(
            **config.wandb,
            name=args.wandb_name,
            config=OmegaConf.to_container(config, resolve=True),
            id=wandb_id,
            resume="allow",
        )
        wandb.config.update(args)
        logger.info("wandb init done, start training...")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_start_time = time.time()
        trainer.train_epoch(epoch, args.epochs, config.train.verbose, config.train.save_dir)
        train_end_time = time.time()

        val_start_time = time.time()
        trainer.val_epoch(epoch, args.epochs)
        val_end_time = time.time()

        if accelerator.is_main_process:
            logger.info(
                f"Epoch [{epoch + 1:02d}/{args.epochs}] - "
                f"Train Time: {train_end_time - train_start_time:.2f}s - "
                f"Val Time: {val_end_time - val_start_time:.2f}s"
            )

    # Final test
    trainer.test(args.epochs - 1, args.epochs)

    # Save checkpoint
    if accelerator.is_main_process and do_checkpointing:
        ckpt_dir = os.path.join(args.checkpoint_dir, f"{os.path.basename(args.config).split('.')[0]}_lr_{args.lr}")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt = {
            "epoch": args.epochs - 1,
            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_best_epoch": trainer.val_best_epoch,
            "val_best_metric": trainer.val_best_metric,
            "test_metrics": trainer.test_metrics,
            "wandb_run_id": wandb.run.id,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"checkpoint_epoch_{args.epochs - 1:02d}.pt"))
        logger.info(f"Saved checkpoint to {ckpt_dir}")

    wandb.finish()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
