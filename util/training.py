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

class BaseTrainer:
    """
    Class implementing a Torch training loop.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss: Callable,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        metrics: list[torchmetrics.Metric] = [],
        n_epochs: int = 10,
        n_batches: int = None,
        verbose: bool = True,
        timing_log: str = None,
        tensorboard_log: str = None,
        model_log: str = None,
        defn_log: str = None
    ):

        self._verbose = verbose

        self._model = model
        self._optim = optimizer
        self._scheduler = scheduler
        self._loss = loss

        self._metrics = metrics

        # Initialize a dict for storing metric results
        self.history = dict()

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
            self._timing_log_handle = open(timing_log, "w+", buffering=1)
            self.get_next_training_batch = self._with_logging(
                self.get_next_training_batch, "get-training-batch"
            )
            self.get_next_validation_batch = self._with_logging(
                self.get_next_validation_batch, "get-validation-batch"
            )
            self.get_validation_loss = self._with_logging(
                self.get_validation_loss, "get-validation-loss"
            )
            self.train_one_epoch = self._with_logging(self.train_one_epoch, "epoch")
            self.train_one_batch = self._with_logging(self.train_one_batch, "training")
            self.train = self._with_logging(self.train, "run")

        # Create handle for tensorboard log
        if tensorboard_log is not None:
            if os.path.exists(tensorboard_log):
                warnings.warn(f"Log already exists at {tensorboard_log}")
            self.writer = SummaryWriter(tensorboard_log)
        else:
            self.writer = None

        # Hang on to the model path
        self._model_log = model_log
        if model_log is not None:
            if os.path.exists(model_log):
                warnings.warn(
                    f"Saved model already exists at {model_log}. "
                    "This will be overwritten on first epoch!"
                )

        # Write model definition to text file
        with open(defn_log, mode='w+') as f:
            f.write(str(model))

    def _log_scalar(self, key, value, step):
        """
        Adds a key/value pair to the tensorboard log (if set) and to the
        local history dict.
        """
        if key in self.history:
            self.history[key].append(value)
        else:
            self.history[key] = [value]

        if self.writer is not None:
            self.writer.add_scalar(key, value, step)

    def _log_model(self):
        """
        Save the model state dict.
        """
        if self._model_log is not None:
            torch.save(self._model.state_dict(), self._model_log)

    def _with_logging(self, f: Callable, event_name: str):
        def add_timing(*args, **kwargs):
            # Log start of event
            t0 = time.time()
            self._timing_log_handle.write(
                json.dumps(dict(event=event_name + " start", time=t0)) + "\n"
            )

            # Run the event
            ret = f(*args, **kwargs)

            # Log how long it took
            t1 = time.time()
            self._timing_log_handle.write(
                json.dumps(dict(event=event_name + " end", time=t1, duration=t1 - t0))
                + "\n"
            )

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

    def update_metrics(self, output: torch.Tensor, target: torch.Tensor) -> None:
        for m in self._metrics:
            if hasattr(m, "num_classes"):
                # this is classification - don't modify shapes
                m(output, target)
            else:
                # this is regression, flatten
                m(output.view(-1), target.view(-1))

    def log_metrics(self, suffix: str, epoch: int):
        for m in self._metrics:
            self._log_scalar(str(m) + f"/{suffix}", m.compute().cpu().numpy(), epoch)
            m.reset()

    def get_batch_loss(self, output: torch.Tensor, target: torch.Tensor):
        '''
        Calculate loss for a single batch. This function is called in the training
        and validation loops. Overload this function with a mixin to implement
        more elaborate loss behavior.
        '''
        return self._loss(output, target)
    
    def get_validation_loss(self, epoch: int):
        valid_loss = 0
        with torch.no_grad():
            n_batches = len(self._valid_loader)
            self._reset_validation_iter()
            for _ in tqdm(
                range(n_batches),
                disable=not self._verbose,
                leave=False,
                desc="Validation loss",
            ):
                batch = self.get_next_validation_batch()
                X, y = batch
                output = self._model(X)

                # Update loss
                valid_loss += self.get_batch_loss(output, y)

                # Update metrics
                self.update_metrics(output, y)

        # Append metrics to history and reset
        self.log_metrics("valid", epoch)

        return valid_loss.cpu() / n_batches

    def train_one_batch(self, batch):
        X, y = batch

        self._optim.zero_grad()
        outputs = self._model(X)
        batch_loss = self.get_batch_loss(outputs, y)

        # If loss is zero, skip optimization step. This usually occurs
        # when a batch is entirely NA.
        if batch_loss.item() > 0:
            batch_loss.backward()
            self._optim.step()

            # Update metrics
            self.update_metrics(outputs, y)

        return batch_loss.item()

    def train_one_epoch(self, epoch: int):
        train_loss = 0
        for i_batch in tqdm(
            range(self._n_batches),
            disable=not self._verbose,
            leave=False,
            desc="Training loss",
        ):
            # If self._n_batches does not equal the length of the training
            # DataLoader (e.g. when the training dataset is huge) we might
            # exhaust the iterator in the middle of an epoch.
            batch = self.get_next_training_batch()
            if batch is None:
                batch = next(self._train_iter)

            batch_loss = self.train_one_batch(batch)

            # Update loss
            train_loss += batch_loss / self._n_batches

        # Append metrics to history and reset
        self.log_metrics("train", epoch)

        return train_loss

    def status_table(self):
        # Compiles all the most recent values in the history object
        data = [[key, self.history[key][-1]] for key in self.history]
        data.sort(key=lambda x: x[0])
        table = pd.DataFrame(
            data=data,
            columns=["Key", "Value"],
        )
        return table

    def update_lr(self, train_loss: float, valid_loss: float) -> None:
        '''
        Update scheduler state. Mixins can overwrite this function
        to implement more complex LR scheduling.
        '''
        self._scheduler.step()

    def stop_condition(self, train_loss: float, valid_loss: float, i_epoch: int) -> bool:
        '''
        Stop training if a stop condition is met. Mixins can overwrite
        this function to implement early stopping behavior.
        '''
        return False
    
    def train(self):
        best_loss = float("Inf")
        for i_epoch in tqdm(
            range(self._n_epochs), disable=not self._verbose, leave=False, desc="Epoch"
        ):
            train_loss = self.train_one_epoch(i_epoch)
            self._model.eval()
            valid_loss = self.get_validation_loss(i_epoch)
            self._model.train()

            # Update history
            self._log_scalar("Loss/train", train_loss, i_epoch)
            self._log_scalar("Loss/valid", valid_loss, i_epoch)
            # Update scheduler
            self.update_lr(train_loss, valid_loss)
            self._log_scalar("LearningRate", self._scheduler.get_last_lr()[0], i_epoch)

            # Maybe log the model
            if valid_loss < best_loss:
                best_loss = valid_loss
                self._log_model()

            # Maybe print current results
            if self._verbose:
                status = self.status_table()
                print(f"Epoch {i_epoch+1} of {self._n_epochs}")
                print(status)
                print()

            # Maybe quit early
            if self.stop_condition(train_loss, valid_loss, i_epoch):
                break


class ReduceLRMixin(object):
    def __init__(self, *args, **kwargs):
        super(ReduceLRMixin, self).__init__(*args, **kwargs)
    
    def update_lr(self, train_loss, valid_loss):
        self._scheduler.step(valid_loss)

class MaskedLossMixin(object):
    '''
    Allow for nans in loss calculation. This mixin:
     - checks for no reduction in the loss function during init.
     - calls .nanmean() on the loss
    '''
    def __init__(self, *args, **kwargs):
        super(MaskedLossMixin, self).__init__(*args, **kwargs)
        for m in self._metrics:
            if hasattr(m, "num_classes"):
                raise ValueError("NaNs not supported for classification metrics.")

    def mask_output_target(self, output, target):
        mask = ~torch.isnan(target.view(-1))
        return output.view(-1)[mask], target.view(-1)[mask]
        
    def get_batch_loss(self, output, target):
        m_output, m_target = self.mask_output_target(output, target)
        return super(MaskedLossMixin, self).get_batch_loss(m_output, m_target)

    def update_metrics(self, output: torch.Tensor, target: torch.Tensor) -> None:
        m_output, m_target = self.mask_output_target(output, target)
        super(MaskedLossMixin, self).update_metrics(m_output, m_target)
        
class EarlyStopMixin:
    '''
    Stops training early if the (lightly smoothed) validation loss
    does not improve after some number of epochs.
    '''
    def __init__(self, *args, stop_patience=5, rel_improve=0.01, **kwargs):
        self.stop_patience = stop_patience
        self.rel_improve = rel_improve
        self.running_loss = None
        self.best_epoch = None
        super(EarlyStopMixin, self).__init__(*args, **kwargs)

    def stop_condition(self, train_loss, valid_loss, i_epoch):
        # If this is the first run just continue
        if self.running_loss is None:
            self.running_loss = valid_loss
            self.best_epoch = i_epoch
            return False

        # Combine this result with most recent loss, update counter
        # if there is improvement
        new_loss = (self.running_loss + valid_loss) / 2
        if 1 - (new_loss / self.running_loss) > self.rel_improve:
            self.best_epoch = i_epoch
        self.running_loss = new_loss

        # If we have waited too long for an improvement, stop training
        if i_epoch - self.best_epoch > self.stop_patience:
            return True
        else:
            return False
        
        
        
        
