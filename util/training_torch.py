import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Callable
import numpy as np
import xbatcher
import json
import time
from typing import List, Dict
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import torchmetrics


def parse_tensorboard(path: str, scalars: List[str]=None) -> Dict[str, pd.DataFrame]:
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

class BaseTrainer():
    '''
    Class implementing a Torch training loop.
    '''      
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: Callable,
                 train_loader: DataLoader, valid_loader: DataLoader,
                 metrics: list[torchmetrics.Metric]=[],
                 n_epochs: int=10, n_batches: int=None, verbose: bool=True, timing_log: str=None):

        self._verbose = verbose
        
        self._model = model
        self._optim = optimizer
        self._loss = loss

        self._metrics = metrics

        # Initialize a dict for storing metric results
        self.history = {
            "Train": {"Loss": []},
            "Valid": {"Loss": []}
        }
        for m in self._metrics:
            self.history["Train"][str(m)] = list()
            self.history["Valid"][str(m)] = list()

        # Initialize dataloaders
        self._train_loader = train_loader
        self._train_iter = iter(self._train_loader)
        
        self._valid_loader = valid_loader
        self._valid_iter = iter(self._valid_loader)

        # If n_batches is not provided, iterate through
        # the entire training set on each epoch.
        self._n_epochs = n_epochs
        if n_batches is None:
            self._n_batches = len(self._train_loader)
        else:
            self._n_batches = n_batches

        # Decorate events we want to log
        self._timing_log_handle = None
        if timing_log is not None:
            self._timing_log_handle = open(timing_log, "w", buffering=1)
            self.get_next_training_batch = self._with_logging(self.get_next_training_batch, "get-training-batch")
            self.get_next_validation_batch = self._with_logging(self.get_next_validation_batch, "get-validation-batch")
            self.get_validation_loss = self._with_logging(self.get_validation_loss, "get-validation-loss")
            self.train_one_epoch = self._with_logging(self.train_one_epoch, "epoch")
            self.train = self._with_logging(self.train, "run")
            
    def _with_logging(self, f: Callable, event_name: str):
        def add_timing(*args, **kwargs):
            # Log start of event
            t0 = time.time()
            self._timing_log_handle.write(json.dumps(dict(
               event = event_name + " start",
                time=t0
            )) + "\n")

            # Run the event
            ret = f(*args, **kwargs)

            # Log how long it took
            t1 = time.time()
            self._timing_log_handle.write(json.dumps(dict(
                event = event_name + " end",
                time=t1,
                duration = t1 - t0
            )) + "\n")

            return ret

        return add_timing
    
    def _reset_training_iter(self):
        self._train_iter = iter(self._train_loader)

    def _reset_validation_iter(self):
        self._valid_iter = iter(self._valid_loader)
        
    def get_next_training_batch(self):
        try:
            return next(self._train_iter)
        except StopIteration:
            self._reset_training_iter()
            return None

    def get_next_validation_batch(self):
        try:
            return next(self._valid_iter)
        except StopIteration:
            self._reset_validation_iter()
            return None
    
    def get_validation_loss(self):
        valid_loss = 0
        with torch.no_grad():
            batch = self.get_next_validation_batch()
            n_batches = 0
            while batch is not None:
                n_batches += 1
                X, y = batch
                output = self._model(X)

                # Update loss
                valid_loss += self._loss(output, y)

                # Update metrics
                for m in self._metrics:
                    m(output, y)
                    
                batch = self.get_next_validation_batch()

        # Append metrics to history and reset
        for m in self._metrics:
            self.history["Valid"][str(m)].append(m.compute())
            m.reset()
            
        return valid_loss / n_batches
    
    def train_one_epoch(self):
        train_loss = 0
        for i_batch in range(self._n_batches):
            # If self._n_batches does not equal the length of the training
            # DataLoader (e.g. when the training dataset is huge) we might
            # exhaust the iterator in the middle of an epoch.
            batch = self.get_next_training_batch()
            if batch is None: 
                batch = next(self._train_iter)

            X, y = batch

            self._optim.zero_grad()
            outputs = self._model(X)
            batch_loss = self._loss(outputs, y)

            # Update weights
            batch_loss.backward()
            self._optim.step()

            # Update metrics
            for m in self._metrics:
                m(outputs, y)

            # Update loss
            train_loss += batch_loss.item() / self._n_batches

        # Append metrics to history and reset
        for m in self._metrics:
            self.history["Train"][str(m)].append(m.compute())
            m.reset()
            
        return train_loss

    def train(self):
        for i_epoch in range(self._n_epochs):
            train_loss = self.train_one_epoch()
            valid_loss = self.get_validation_loss()
            # Update history
            self.history["Valid"]["Loss"].append(valid_loss)
            self.history["Train"]["Loss"].append(train_loss)
            
            if self._verbose:
                # Convert history to a table so we can print it nicely
                print(f"Epoch {i_epoch+1}/{self._n_epochs}")
                table = pd.DataFrame(
                    data=[
                        [self.history["Train"][key][-1], self.history["Valid"][key][-1]]
                        for key in sorted(list(self.history["Train"].keys()))
                    ],
                    columns=["Train", "Valid"],
                    index=sorted(list(self.history["Train"].keys()))
                )
                print(table)

# Based on
# https://github.com/earth-mover/dataloader-demo/blob/main/main.py
class XBatcherPyTorchDataset(Dataset):
    def __init__(self, batch_generator: xbatcher.BatchGenerator, reshaper: Callable):
        self.bgen = batch_generator
        self.reshaper = reshaper

    def __len__(self):
        return len(self.bgen)

    def __getitem__(self, idx):
        # load before stacking
        batch = self.bgen[idx].load()
        X, y = self.reshaper(batch)

        return X, y

class DamageConv3D(torch.nn.Module):
    '''
    3D ConvNet. Hardcoded for inputs of shape (5, 5, 4).
    '''
    def __init__(self):
        super(DamageConv3D, self).__init__()
        self.conv1 = self._conv_layer_set(1, 8)
        self.conv2 = self._conv_layer_set(8, 16)
        self.bn = torch.nn.BatchNorm3d(16)
        self.flat = torch.nn.Flatten()
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.fc1 = torch.nn.Linear(288, 16)
        self.fc2 = torch.nn.Linear(16, 1)

    @staticmethod
    def _conv_layer_set(in_channels, out_channels):
        conv_layer = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=(2, 2, 2), 
                stride=1,
                padding=0,
                ),
            torch.nn.LeakyReLU(),
            )
        return conv_layer

    def forward(self, x):
        # Add a channel axis to make 3d conv layers happy
        x = torch.unsqueeze(x, 1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
        
        

        