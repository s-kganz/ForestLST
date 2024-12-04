import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Callable
import numpy as np
import xbatcher
import json
import time

class DummyWriter():
    '''
    Fake TensorBoard writer for when perf_log is None.
    '''
    def add_scalar(self, *args, **kwargs):
        pass

class Trainer():
    '''
    Class implementing a Torch training loop and logging.
    '''
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: Callable,
                 train_loader: DataLoader, valid_loader: DataLoader,
                 n_epochs: int=10, n_batches: int=None,
                 model_log: str=None, perf_log: str=None, timing_log: str=None):

        self._model = model
        self._optim = optimizer
        self._loss = loss
        
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        # Load the first batch so both loaders are "ready"
        next(iter(self._train_loader))
        next(iter(self._valid_loader))
        
        self._n_epochs = n_epochs
        self._n_batches = n_batches
        self._model_log = model_log
        
        if timing_log is None:
            self._log_handle = None
        else:
            self._log_handle = open(timing_log, "w")

        if perf_log is None:
            self._writer = DummyWriter()
        else:
            self._writer = SummaryWriter(perf_log)

    def log_event(self, obj):
        if self._log_handle:
            self._log_handle.write(
                json.dumps(obj) + "\n"
            )

    def train_one_epoch(self):
        '''
        Train on one epoch of data.
        '''
        t0, t1 = None, None
        running_loss = 0
        n_batches = self._n_batches
        if not n_batches:
            n_batches = len(self._train_loader)

        t0 = time.time()
        self.log_event(dict(
                event="get-batch start",
                time=t0,
            ))
        
        for i, data in enumerate(self._train_loader):
            t1 = time.time()
            self.log_event(dict(
                event="get-batch end",
                time=t1,
                duration=t1-t0
            ))
            
            # Every data instance is an input + label pair
            inputs, labels = data

            t0 = time.time()
            self.log_event(dict(
                event="training start",
                time=t0
            ))
            
            # Zero your gradients for every batch!
            self._optim.zero_grad()
            
            # Make predictions for this batch
            outputs = self._model(inputs)
    
            # Compute the loss and its gradients
            loss = self._loss(outputs, labels)
            loss.backward()
    
            # Adjust learning weights
            self._optim.step()
    
            # Gather data
            running_loss += loss.item() / n_batches

            t1 = time.time()
            self.log_event(dict(
                event="training end",
                time=t1,
                duration=t1-t0
            ))

            # Stop epoch if n_batches is set
            if self._n_batches and i == self._n_batches:
                break

            t0 = time.time()

        # Calculate validation loss
        running_vloss = 0
        with torch.no_grad():
            for i, vdata in enumerate(self._valid_loader):
                vinputs, vlabels = vdata
                voutputs = self._model(vinputs)
                vloss = self._loss(voutputs, vlabels)
                running_vloss += vloss / len(self._valid_loader)        
    
        return running_loss, running_vloss

    def train(self):
        tstart = time.time()
        self.log_event(dict(
            event="run start",
            time=tstart,
            locals=dict()
        ))
        
        best_vloss = np.inf
        t0 = time.time()
        self.log_event(dict(
            event="epoch start",
            time=t0
        ))
        for i in range(self._n_epochs):
            tloss, vloss = self.train_one_epoch()

            t1 = time.time()
            self.log_event(dict(
                event="epoch end",
                time=t1,
                duration=t1-t0
            ))
            
            print(f"Epoch {i+1}\t Training loss: {np.round(tloss, 2)}\t Validation loss: {np.round(vloss, 2)}")
            
            # Save model if it is better
            if self._model_log is not None and vloss < best_vloss:
                best_vloss = vloss
                torch.save(self._model.state_dict(), self._model_log)
            
            # Save results
            self._writer.add_scalar("Loss/train", tloss, i)
            self._writer.add_scalar("Loss/valid", vloss, i)

        tend = time.time()
        self.log_event(dict(
            event="run end",
            time=tend,
            duration=tend-tstart
        ))
        
        if self._log_handle is not None:
            self._log_handle.flush()

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

class SinglePixelFC(torch.nn.Module):
    '''
    Simple fully-connected NN for use with single pixels.
    '''
    def __init__(self):
        super(SinglePixelFC, self).__init__()
        self.bn = torch.nn.BatchNorm1d()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.drop = torch.nn.Dropout(p=0.2)
        self.fc1 = torch.nn.Linear(128)
        self.fc2 = torch.nn.Linear(64)
        self.fc3 = torch.nn.Linear(16)
        self.out = torch.nn.Linear(1)

    def forward(self, x):
        x = self.bn(x)
        x = self.drop(x)
        
        

        