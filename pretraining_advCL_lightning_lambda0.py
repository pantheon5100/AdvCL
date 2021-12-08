from __future__ import print_function


import os
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger

from zk_utils.get_args import get_args
from zk_utils.get_data_loader import get_data_loader
from zk_utils.checkpointer import Checkpointer
from zk_utils.file_backup import file_backup

from method.adssl_model import ADSSL

def main():
    # Arguments
    args = get_args()

    train_loader, val_loader = get_data_loader(args)
    
    if args.wandb:
        wandb_logger = WandbLogger(
                # name="advssl-eval_pgd_bn-1_step_train_att-trainattack",
                name=args.name,

                project="officialcode_lightning",
                entity="kaistssl",
            )
        wandb_logger.log_hyperparams(args)

    
    callbacks = []
    ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, args.name),
            frequency=args.checkpoint_frequency,
        )
    callbacks.append(ckpt)

    adssl_model = ADSSL(**args.__dict__)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        plugins=DDPPlugin(find_unused_parameters=True) if args.accelerator == "ddp" else None,
        callbacks=callbacks,
    )

    if args.wandb:
        args.train_version = trainer.logger.version
    file_backup(args)

    trainer.fit(adssl_model, train_loader, val_loader)

if __name__ == "__main__":
    main()


