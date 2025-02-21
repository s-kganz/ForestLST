from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import xbatcher
from imblearn.under_sampling import RandomUnderSampler
from typing import Callable


def theta_adjustment(p: np.ndarray, theta: float):
    """
    Applies theta adjustment to the empirical class distribution in
    the 1D array p. The k-th element of p is the proportion of a dataset
    belonging to the k-th class. Theta takes on a value from [0, 1]. This
    function returns a modified class distribution with:

    q = theta * u + (1 - theta) * p

    where u is a uniform distribution (i.e. np.ones(p.shape) / p.shape[0]).

    When theta = 0, the empirical distribution is returned. When theta = 1,
    the uniform distribution is returned. Modifying theta yields varying
    levels of class balance.
    """
    u = np.ones(p.shape) / p.shape[0]
    q = theta * u + (1 - theta) * p
    return q


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


class WindowXarrayDataset(Dataset):
    """
    A class providing an interface for a torch Dataset to pull examples from
    arbitrary windows of a xr.Dataset or xr.DataArray. Determines the indices
    of valid windows and then extracting those windows through __get__.

    Provide either a xr.DataArray or xr.Dataset to the data argument. If xr.DataArray,
    valid windows are derived from that array. Otherwise, `mask` must be set. If a
    string, that variable will be extracted from `data` and used to find windows. If a
    xr.DataArray, valid windows are derived from the array and applied to `data`.

    `window_size` defines how windowing is performed. Each key of window_size should
    be a coordinate in array. Values should be tuple[int, bool]. The first element
    defines the window size, while the center defines whether the window
    is centered (True) or right-padded (False).

    Dimensions that are not keys in `window` are not sliced. For example, if you have
    a satellite image with dimensions (x, y, band) and you window on x and y, then all
    bands will be present in the resulting patches.

    Note that this class outputs slices of the underlying data, but does not
    separate them into X, y or convert them to tensors. That logic should be
    implemented in the collate_fn passed to the DataLoader.
    """

    def __init__(
        self,
        data: xr.DataArray | xr.Dataset,
        window: dict[str, tuple[int, bool]],
        mask: xr.DataArray | str | None = None,
        downsample_step: int = 1,
    ):
        self.downsample_step = downsample_step
        self.data = data

        # Check input
        assert set(data.dims).issuperset(
            window.keys()
        ), "All keys in window must be coordinates in the array"

        if isinstance(data, xr.Dataset):
            assert mask in [
                var for var in data.data_vars
            ], "Mask variable must be present in the dataset"

        nonwindow_dims = set(data.dims) - window.keys()

        # Define offsets and window sizes
        self.offset = dict()
        self.window_size = dict()
        self.is_centered = dict()
        for coord, (size, center) in window.items():
            self.window_size[coord] = size
            self.is_centered[coord] = center
            if center:
                self.offset[coord] = (-(size // 2), size // 2 + 1)
            else:
                self.offset[coord] = (-size + 1, 1)

        # Dimensions not specified always have size 1
        for nwd in nonwindow_dims:
            self.offset[nwd] = (0, 1)

        # Determine indices where there are valid windows. Note that all
        # array dimensions, not just the window ones, are included here.
        if isinstance(data, xr.DataArray) and mask is None:
            array = data
        elif isinstance(mask, xr.DataArray):
            array = mask
            assert set(mask.dims).issubset(
                data.dims
            ), "Mask cannot have dimensions not in data"
        elif isinstance(data, xr.Dataset) and isinstance(mask, str):
            array = data[mask]
        else:
            raise ValueError("Input data must be either xr.Dataset" "or xr.DataArray")

        accepted_indices = self._get_valid_indices(array)
        self.valid_indices = dict()
        for i in range(len(array.dims)):
            if array.dims[i] in self.window_size:
                self.valid_indices[array.dims[i]] = accepted_indices[:, i]

    def _get_valid_indices(self, array) -> np.array:
        """
        Determine which indices contain non-na windows in the array. Overloads
        of this function must return a 2-D numpy array of shape (n_samples, n_axes).
        """
        return np.stack(
            np.where(
                ~np.isnan(
                    array.rolling(dim=self.window_size, center=self.is_centered).mean()
                )
            )
        ).T[:: self.downsample_step, :]

    def __len__(self) -> int:
        return next(iter(self.valid_indices.values())).shape[0]

    def __getitem__(self, idx: int) -> xr.DataArray | xr.Dataset:
        # Get coordinates of the window
        slicers = {
            k: slice(
                self.valid_indices[k][idx] + self.offset[k][0],
                self.valid_indices[k][idx] + self.offset[k][1],
            )
            for k in self.valid_indices
        }
        # Extract
        return self.data.isel(**slicers)


class ResampleXarrayDataset(WindowXarrayDataset):
    """
    This class overwrites `WindowXarrayDataset._get_valid_indices` to
    resample the response variable. This adds a new parameter `theta` that
    controls how balanced the resulting dataset is. See `util.data.theta_adjustment`
    for more details. Data balancing only occurs by undersampling majority quantiles.

    It is assumed that the mask array in this dataset is the response variable. It is
    therefore recommended that you set `mask` to your response variable
    if you have other covariates in `data`.
    """

    def __init__(self, *args, cutoff=3, ratio=1, **kwargs):
        self.cutoff = cutoff
        self.ratio = ratio
        super(ResampleXarrayDataset, self).__init__(*args, **kwargs)

    def _get_valid_indices(self, array):
        # Resample on the window mean
        window_mean = array.rolling(
            dim=self.window_size, center=self.is_centered
        ).mean()

        positive_inds = np.where(window_mean >= self.cutoff)
        negative_inds = np.where(window_mean < self.cutoff)

        # Only pull some of the samples from the negatives
        n_negatives = int(positive_inds[0].shape[0] * self.ratio)
        assert n_negatives <= negative_inds[0].shape[0], (
            f"Cannot sample {n_negatives} samples "
            f"from {negative_inds[0].shape[0]} negative samples!"
        )

        inds = np.random.choice(negative_inds[0].shape[0], n_negatives, replace=False)
        negative_inds_sample = tuple(a[inds] for a in negative_inds)

        return np.concat(
            (np.stack(positive_inds), np.stack(negative_inds_sample)), axis=1
        ).T
