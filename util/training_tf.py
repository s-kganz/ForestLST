'''
This file has functions to help with training models.
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil

def build_rao_lstm(input_shape: tuple[int],
                   activation: str="tanh",
                   dropout: float=0.05,
                   bias_regularizer: keras.regularizers.Regularizer=keras.regularizers.L2(0.001),
                   output_bias_init="zeros"
                   ) -> keras.Model:
    ''' 
    Returns an LSTM with the same architecture as that used in Rao et al. (2020) but
    with an arbitrary input shape.
    '''
    inp = keras.layers.Input(shape=input_shape)
    norm = keras.layers.BatchNormalization(axis=-1)(inp)
    lstm1 = keras.layers.LSTM(
        units=10, 
        activation=activation,
        dropout=dropout,
        recurrent_dropout=dropout,
        bias_regularizer=bias_regularizer,
        return_sequences=True
    )(norm)

    lstm2 = keras.layers.LSTM(
        units=10, 
        activation=activation,
        dropout=dropout,
        recurrent_dropout=dropout,
        bias_regularizer=bias_regularizer,
        return_sequences=True
    )(lstm1)

    lstm3 = keras.layers.LSTM(
        units=10, 
        activation=activation,
        dropout=dropout,
        recurrent_dropout=dropout,
        bias_regularizer=bias_regularizer,
        return_sequences=True
    )(lstm2)

    lstm4 = keras.layers.LSTM(
        units=10, 
        activation=activation,
        dropout=dropout,
        recurrent_dropout=dropout,
        bias_regularizer=bias_regularizer,
        return_sequences=False
    )(lstm3)

    # Connect to output
    out = keras.layers.Dense(
        units=1, activation="sigmoid",
        name="mort",
        bias_initializer=output_bias_init
    )(lstm4)

    return keras.models.Model(inputs=inp, outputs=out)


def build_dense(input_shape: tuple[int], 
                bias_init="str",
                dropout: float=0.2, activation: str="tanh"
                ) -> keras.Model:
    '''
    Returns a 3-layer, densely-connected neural network with the given parameters
    and input shape.
    '''
    inp = keras.layers.Input(shape=input_shape)
    norm = keras.layers.BatchNormalization(axis=-1)(inp)
    
    d1 = keras.layers.Dense(128, activation=activation)(norm)
    drop1 = keras.layers.Dropout(dropout)(d1)
    d2 = keras.layers.Dense(16, activation=activation)(drop1)
    drop2 = keras.layers.Dropout(dropout)(d2)
    d3 = keras.layers.Dense(8, activation=activation)(drop2)

    out = keras.layers.Dense(
        1, activation="sigmoid", 
        name="mort",
        bias_initializer="zeros" if bias_init is None else bias_init
    )(d3)

    return keras.models.Model(inputs=inp, outputs=out)

def bias_init_classification(y, as_log=True):
    '''
    Calculate the initial bias for a binary classification problem with possibly
    imbalance data. It is expected that y only takes on values 0, 1.
    '''
    imbalance_ratio = np.sum(y == 1) / np.sum(y == 0)
    if as_log:
        return np.log(imbalance_ratio)
    else:
        return imbalance_ratio


def bias_init_regression(y, as_log=True):
    '''
    Calculate the initial bias for a regression problem. This is simply
    the mean of the target vector.
    '''
    return np.log(np.mean(y)) if as_log else np.mean(y)

def regression_metrics():
    return [
        keras.metrics.RootMeanSquaredError(name="rmse"),
        keras.metrics.R2Score(name="r2_score")
    ]

def binary_classification_metrics():
    return [
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc")
    ]

def _delete_folder_contents(dir: str) -> None:
    for fname in os.listdir(dir):
        file_path = os.path.join(dir, fname)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except OSError as e:
            print("Failed to delete", file_path)

def make_model_log_directory(model_name: str, log_dir: str="logs", 
                             model_dir: str=os.path.join("data_working", "models")) -> tuple[str]:
    log_output = os.path.join(log_dir, model_name)
    model_output = os.path.join(model_dir, model_name) + ".keras"

    os.makedirs(log_output, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    _delete_folder_contents(log_output)

    return log_output, model_output

def make_train_valid_split(input_arr: np.ndarray, target_arr: np.ndarray, 
                           prop_valid=0.20, **kwargs) -> tuple[tf.data.Dataset]:
    input_train, input_valid, target_train, target_valid = train_test_split(
        input_arr, target_arr, test_size=prop_valid, **kwargs
    )

    # Add another axis to the target arrays
    target_train = target_train[:, np.newaxis]
    target_valid = target_valid[:, np.newaxis]

    train_tfdata = tf.data.Dataset.from_tensor_slices((input_train, target_train))
    valid_tfdata = tf.data.Dataset.from_tensor_slices((input_valid, target_valid))

    return (train_tfdata, valid_tfdata)

