import wandb
import numpy as np
import random
from omegaconf import DictConfig, OmegaConf


def init_wandb(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    pretty_print_cfg(cfg)
    wandb_cfg = prepare_wandb_cfg(cfg)

    wandb.init(
        config=wandb_cfg,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        group=cfg.wandb.group,
    )
    OmegaConf.save(cfg, f"{wandb.run.dir}/config.yaml")

def pretty_print_cfg(cfg):
    """
    Pretty print the config as cascading bullet points.
    """
    print("Config:")
    for key, value in cfg.items():
        print(f"- {key}:")
        if isinstance(value, DictConfig) or isinstance(value, dict):
            for k, v in value.items():
                print(f"  - {k}: {v}")
        else:
            print(f"  - {value}")


def prepare_wandb_cfg(cfg):
    wandb_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            wandb_cfg[key] = prepare_wandb_cfg(value)
        else:
            wandb_cfg[key] = value

    return wandb_cfg


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

def setup_optimizer(optim_cfg, model):
    """
    Setup the optimizer. Return the optimizer.
    """
    from torch import optim

    optimizer = eval(optim_cfg.type)
    model_trainable_params = get_named_trainable_params(model)
    # Print size of trainable parameters
    print(
        "Trainable parameters:",
        sum(p.numel() for (name, p) in model_trainable_params) / 1e6,
        "M",
    )
    return optimizer(list(model.parameters()), **optim_cfg.params)


def setup_lr_scheduler(optimizer, scheduler_cfg):
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_scheduler
    from .lr_scheduler import CosineAnnealingLRWithWarmup
    from torch.optim.lr_scheduler import CosineAnnealingLR

    sched = eval(scheduler_cfg.type)
    if sched is None:
        return None
    return sched(optimizer, **scheduler_cfg.params)

def get_named_trainable_params(model):
    return [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]