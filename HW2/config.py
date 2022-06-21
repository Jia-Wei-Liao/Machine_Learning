import torch
import torch.nn as nn

from src.losses import FL, MCCE, TSCE
from src.optimizer import ranger21
from src.models.swin import SwinTransformer, load_pretrained
from model import EfficientNet_b4, TeacherStudentModel


def get_model(args):
    Model = {
        'EfficientB4': EfficientNet_b4,
        'Swin': SwinTransformer,
        'TSM': TeacherStudentModel
    }
    model = Model[args.model](
        num_classes=args.num_classes,
        teacher_ckpt_path=args.teacher_ckpt)

    return model


def get_pretrain(model, args):
    if args.pretrain:
        Weight = {
            'Swin': load_pretrained(model)
        }

    return model


def get_criterion(args, device):
    Losses = {
        'CE':   nn.CrossEntropyLoss,
        'MCCE': MCCE.MCCE_Loss,
        'FL':   FL.FocalLoss,
        'FLSD': FL.FocalLossAdaptive,
        'TSCE': TSCE.TSCE_Loss
    }
    criterion = Losses[args.loss]()

    return criterion


def get_optimizer(args, model):
    Optimizer = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'Ranger': ranger21.Ranger21
    }
    optimizer = Optimizer[args.optim](
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    return optimizer


def get_scheduler(args, optimizer):
    Scheduler = {
        'step': torch.optim.lr_scheduler.StepLR(
              optimizer=optimizer,
              step_size=args.step_size,
              gamma=args.gamma
        ),
        'cos': torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.epoch
        )
    }
    scheduler = Scheduler[args.scheduler]

    return scheduler
