import os
import shutil
import yaml
import argparse
import numpy as np
from pathlib import Path
from autoencoder.PyTorch_VAE.models import *
from autoencoder.PyTorch_VAE.experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from autoencoder.PyTorch_VAE.dataset import VAEDataset
# from pytorch_lightning.plugins import DDPPlugin

def train_vae(filename_yaml, pathtoimg, images, annotations, max_epochs):
    with open(filename_yaml, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    PATH_LOGS = './logs'
    if os.path.exists(PATH_LOGS):
        shutil.rmtree(PATH_LOGS)

    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['model_params']['name'],
                                  version=0)

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    model = vae_models[config['model_params']['name']](**config['model_params'])

    experiment = VAEXperiment(model,
                              config['exp_params'])

    config["data_params"]['images'] = images
    config["data_params"]['annotations'] = annotations
    config["data_params"]['pathtoimg'] = pathtoimg

    config["trainer_params"]['max_epochs'] = max_epochs


    data = VAEDataset(**config["data_params"])

    data.setup()
    runner = Trainer(logger=tb_logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                     ],
                     # strategy="ddp",
                     limit_val_batches=1,
                     **config['trainer_params'])


    # Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    # Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
    # Path(f"{tb_logger.log_dir}/Reconstructions_origin").mkdir(exist_ok=True, parents=True)


    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)

    return os.path.join(tb_logger.log_dir, "checkpoints", "last.ckpt")
