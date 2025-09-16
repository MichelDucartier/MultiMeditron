from multimeditron.cli.config import VerlTrainConfig, ModelStrategy, RewardManager
from omegaconf import OmegaConf

def make_cfg(cfg: VerlTrainConfig):
    log_backend = []
    if cfg.trainer.use_console_logging:
        log_backend.append("console")
    if cfg.trainer.use_wandb_logging:
        log_backend.append("wandb")
