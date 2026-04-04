from src.trainers.generic_trainer import GenericTrainer
from src.trainers.utils import Percept_Loss
import torch.nn as nn
from src.trainers.utils import get_intense_voxels
import wandb
import torch

class TFFPhase3Trainer(GenericTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, config, args, accelerator, logger, run, phase2_weight_path=None):
        super().__init__(model, train_loader, val_loader, test_loader, optimizer, scheduler, config, args, accelerator, logger, run)
        self.scheduler = scheduler
        assert phase2_weight_path is not None
        state_dict = torch.load(phase2_weight_path)
        self.model.load_partial_state_dict(state_dict['model_state_dict'], load_cls_embedding=False)
        self.model.loaded_model_weights_path = phase2_weight_path
