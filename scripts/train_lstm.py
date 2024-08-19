'''
This file trains the LSTM version of the forest mortality model at different time
series lengths. Models and logs are saved as well.
'''
import datetime
import os

import sys
sys.path.append(os.getcwd())
import util

from tensorflow import keras
from google.cloud.storage import Client

EXPT_NAME = "preisler_rectangular_v2_lstm"

LOG_DIR = os.path.join("logs", EXPT_NAME)
MODEL_DIR = os.path.join("data_working", "models", EXPT_NAME)

# Size of time series in years
TS_BEGIN =  1
TS_END   = 15

# Training parameters
EPOCHS = 50
BATCH_SIZE = 512
LOSS = "mean_squared_error"
OPTIMIZER = "Nadam"

# Download background data
c = Client(project="forest-lst")
alldata = util.gcs.read_gcs_csv(c, "preisler_tfdata", "preisler-rectangular-v2")

# Copy longitude and latitude columns so they don't get lost during windowing
alldata["lat"] = alldata["latitude"]
alldata["lon"] = alldata["longitude"]

def do_training_run(ts_length):
    # Build windowed dataset
    timeseries_data = util.data.csv_to_timeseries_dataset(alldata, timeseries_length=ts_length)
    ds_train, ds_valid = util.training.make_train_valid_split(
        input_arr=timeseries_data.input.values,
        target_arr=timeseries_data.mort.values
    )

    # Build the model
    input_shape = timeseries_data.input.values.shape[1], timeseries_data.input.values.shape[2]
    init_bias = util.training.bias_init_regression(timeseries_data.mort.values)
    this_model = util.training.build_rao_lstm(
        input_shape=input_shape,
        output_bias_init=keras.initializers.Constant(init_bias)
    )

    # Prepare log and model directories
    model_name = f"lstm_{ts_length}"
    log_output, model_output = util.training.make_model_log_directory(model_name, LOG_DIR, MODEL_DIR)

    # Do the training
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5),
        keras.callbacks.ModelCheckpoint(model_output, save_best_only=True),
        keras.callbacks.TensorBoard(log_output, histogram_freq=1)
    ]

    metrics = util.training.regression_metrics()

    this_model.compile(
        OPTIMIZER,
        loss=LOSS,
        metrics=metrics
    )

    ds_train_batch = ds_train.batch(BATCH_SIZE)
    ds_valid_batch = ds_valid.batch(BATCH_SIZE)

    history = this_model.fit(
        x=ds_train_batch,
        validation_data=ds_valid_batch,
        epochs=EPOCHS,
        verbose=0,
        callbacks=callbacks
    )

    return history

def print_with_timestamp(*args, **kwargs):
    print(f"[{datetime.datetime.now()}]", *args, **kwargs)

if __name__ == "__main__":
    print("Starting experiment with following parameters:")
    print("Log dir:", LOG_DIR)
    print("Model dir", MODEL_DIR)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    
    print_with_timestamp("Starting training")
    for i in range(TS_BEGIN, TS_END):
        print_with_timestamp(f"Length {i} years... ", end="")
        history = do_training_run(i)
        print("validation R2: {:.2f}".format(history.history["val_r2_score"][-1]))


