import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import time
from typing import List, Dict, Callable
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import torchmetrics
import warnings
import os
import shutil
from tqdm.autonotebook import tqdm

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Returns number of trainable parameters in torch model. This
    does not account for shared parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def parse_tensorboard(path: str, scalars: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    returns a dictionary of pandas dataframes for each requested scalar
    """
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    if scalars is not None:
        assert all(
            s in ea.Tags()["scalars"] for s in scalars
        ), "some scalars were not found in the event accumulator"
    else:
        scalars = [s for s in ea.Tags()["scalars"]]
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

def get_final_metrics(path: str, scalars: List[str] | None = None) -> Dict[str, float]:
    metrics = parse_tensorboard(path, scalars)

    return {k: float(metrics[k]["value"].iloc[-1]) for k in metrics}

def get_binary_metrics() -> list[torchmetrics.Metric]:
    '''
    Instantiate commonly-used metrics for binary classification tasks.
    '''
    return [
        torchmetrics.classification.BinaryAUROC(),
        torchmetrics.classification.BinaryAccuracy(),
        torchmetrics.classification.BinaryPrecision(),
        torchmetrics.classification.BinaryRecall()
    ]

def get_regr_metrics() -> list[torchmetrics.Metric]:
    return [
        torchmetrics.NormalizedRootMeanSquaredError(),
        torchmetrics.PearsonCorrCoef(),
        torchmetrics.MeanAbsoluteError(),
        torchmetrics.MeanSquaredError()
    ]

def get_classif_metrics(task: str, num_classes: int) -> list[torchmetrics.Metric]:
    return [
        torchmetrics.Accuracy(task, num_classes=num_classes),
        torchmetrics.CohenKappa(task, num_classes=num_classes),
        torchmetrics.classification.MulticlassAveragePrecision(num_classes=num_classes)
    ]

def remove_log(logdir):
    '''
    This is a safer verison of using rm to remove a log directory. It is
    assumed that the log contains a history directory, model.pth,
    model_definition.txt and possibly a .ipynb_checkpoints directory.
    This function only removes those things and then clears
    the directory. Anything else in the directory is ignored.
    '''
    assert os.path.exists(logdir)
    shutil.rmtree(os.path.join(logdir, "history"), ignore_errors=True)
    shutil.rmtree(os.path.join(logdir, ".ipynb_checkpoints"), ignore_errors=True)
    for f in ["model.pth", "model_definition.txt"]:
        try:
            os.remove(os.path.join(logdir, f))
        except FileNotFoundError:
            # File already doesn't exist for some reason. E.g.
            # from a run that errored out.
            warnings.warn(f"{f} not in log directory")
            continue
    
    
    os.rmdir(logdir)


class AsymmetricLoss(torch.nn.Module):
    '''
    Returns L1 or L2 loss depending on the sign of the residual and parameter 
    ``l2``. When ``l2 == "overprediction"``, L2 loss applies to overpredictions 
    (residual < 0) and L1 loss applies to underpredictions (residual > 0). Otherwise,
    L2 loss applies to overpredictions and L1 loss applies to underpredictions.
    '''
    def __init__(self, l2="overprediction"):
        super(AsymmetricLoss, self).__init__()
        if l2 == "overprediction":
            self.invert = 1
        else:
            self.invert = -1
            
    def forward(self, inputs, targets):
        residual = inputs - targets
        l1 = torch.abs(residual)
        l2 = residual ** 2
        sign = (torch.sign(residual) * self.invert) < 0
        loss = torch.where(sign, l2, l1)
        return loss.mean()
        

class WeightedMSELoss(torch.nn.Module):
    '''
    Multiplies overpredictions by a constant value.
    '''
    def __init__(self, constant=1):
        super(WeightedMSELoss, self).__init__()
        self.constant = constant

    def forward(self, inputs, targets):
        residual = inputs - targets
        l2 = residual ** 2
        is_over = torch.sign(residual) < 0
        mult = torch.where(is_over, self.constant, 1)
        return (l2 * mult).mean()
