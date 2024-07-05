import os

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import lightning
from lightning.fabric import Fabric

from atm.model import *
from atm.dataloader import ATMPretrainDataset, get_dataloader
from atm.utils.log_utils import BestAvgLoss, MetricLogger
from atm.utils.train_utils import init_wandb, setup_lr_scheduler, setup_optimizer

@hydra.main(config_path="../conf/train_track_transformer", version_base="1.3")
def main(cfg: DictConfig):
    work_dir = HydraConfig.get().runtime.output_dir
    setup(cfg)
    
    val_vis_dataset = ATMPretrainDataset(dataset_dir=cfg.val_dataset, vis=True, **cfg.dataset_cfg, aug_prob=0.)
    val_vis_dataloader = get_dataloader(val_vis_dataset, mode="val", num_workers=1, batch_size=1)

    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)

    def vis_and_log(model, vis_dataloader, mode="train"):
        vis_dict = visualize(model, vis_dataloader, mix_precision=cfg.mix_precision)

        vis_dict["combined_track_vid"]


    vis_and_log(model, val_vis_dataloader, mode="val")

def visualize(model, dataloader, mix_precision=False):
    model.eval()
    keep_eval_dict = None

    for i, (vid, track, vis, task_emb) in enumerate(dataloader):
        vid, track, task_emb = vid.cuda(), track.cuda(), task_emb.cuda()
        if mix_precision:
            vid, track, task_emb = vid.bfloat16(), track.bfloat16(), task_emb.bfloat16()
        _, eval_dict = model.forward_vis(vid, track, task_emb, p_img=0)
        if keep_eval_dict is None or torch.rand(1) < 0.1:
            keep_eval_dict = eval_dict

        if i == 10:
            break
    return keep_eval_dict

def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)