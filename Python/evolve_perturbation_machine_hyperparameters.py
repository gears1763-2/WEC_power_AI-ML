"""
    WEC Power AI/ML
    Copyright 2024 (C)

    Anthony Truelove MASc, P.Eng.
    email:  gears1763@tutanota.com
    github: gears1763-2

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors
       may be used to endorse or promote products derived from this software without
       specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    CONTINUED USE OF THIS SOFTWARE CONSTITUTES ACCEPTANCE OF THESE TERMS.
"""


"""
    A script for evolving hyperparameters for the perturbation machine
    (tensorflow.keras.Sequential).
"""


import gc

import math

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as nptype

import os

import scipy.optimize as spo

import sklearn.model_selection as skl_ms
import sklearn.preprocessing as skl_pp

import tensorflow.keras as tfk

import time



#### ============================================================================== ####

#   CONSTANTS

START_TIME = time.time()
TIME_LIMIT_S = 48 * 3600    # 48 hours

GENERATION = 0
EVOLUTION_LOG = []

N_HIDDEN_LAYERS = 6
MAX_NEURONS = 512

LINEAR_ACTIVATION_DICT = {    # ref: <https://www.tensorflow.org/api_docs/python/tf/keras/activations>
    0: "elu",          # exponential linear unit
    1: "gelu",         # Gaussian error linear unit
    2: "hard_silu",    # piecewise linear approximation of silu
    3: "leaky_relu",   # relu with some response below the threshold value
    4: "linear",       # linear unit, a.k.a. identity unit
    5: "relu",         # rectified linear unit
    6: "selu",         # scaled exponential linear unit
    7: "silu"          # sigmoid linear unit
}

MAX_ACTIVATION = len(LINEAR_ACTIVATION_DICT) - 1

MAX_TRAINING_EPOCHS = 32

#### ============================================================================== ####



#### ============================================================================== ####

def evolutionCallback(intermediate_result : spo.OptimizeResult) -> None:
    """
    A callback function used to track the change in best objective value from one 
    GENERATION to the next (for plotting later). Also enforces evolution early stopping,
    based on no change in evolution log and/or timeout.
    
    Parameters
    ----------
    
    intermediate_result : spo.OptimizeResult
        The intermediate optimization result (current generational best).
    
    Returns
    -------
    
    None
    """
    
    quit_flag = False
    
    
    #   log intermediate results
    global GENERATION
    global EVOLUTION_LOG
    
    GENERATION += 1
    EVOLUTION_LOG.append(intermediate_result.fun)
    
    global LINEAR_ACTIVATION_DICT
    
    hidden_layers = [round(x) for x in intermediate_result.x[0:6]]
    activation = LINEAR_ACTIVATION_DICT[round(intermediate_result.x[6])]
    
    print()
    print(
        "sequential_regressor GENERATION:",
        GENERATION,
        "\n\thyperparams: ",
        hidden_layers,
        "   ",
        activation,
        "\n\tobjective: ",
        round(intermediate_result.fun, 6)
    )

    np.save(
        "data/perturbation_machine_hyperparams.npy",
        np.array(hidden_layers + [activation])
    )
    
    
    #   plot intermediate results
    if len(EVOLUTION_LOG) >= 2:
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7")
        plt.plot(
            [i for i in range(0, len(EVOLUTION_LOG))],
            EVOLUTION_LOG,
            zorder=2
        )
        plt.xlim(0, len(EVOLUTION_LOG) - 1)
        plt.xlabel("Generation [ ]")
        plt.ylabel("Objective Value [ ]")
        plt.savefig(
            "../LaTeX/images/perturbation_machine/perturbation_machine_evolution.png",
            format="png",
            dpi=128,
            bbox_inches="tight"
        )
        plt.close()
    
    
    #   clean up (clear keras backend and collect garbage, once per generation)
    """
        tensorflow.keras seems pretty leaky when it comes building a large number of
        models in either a loop or a pool of multiprocesses. This is a known issue.
        For instance, see
        
        <https://stackoverflow.com/questions/76527878/memory-leak-in-tensorflow>
        <https://github.com/tensorflow/tensorflow/issues/44711>
        
        Perhaps using __call__() instead of predict would solve the memory leak problem,
        but it's also a fair bit slower it seems. For optimization, maybe it's better 
        to just accept the leakiness for the sake of getting run times down. Luckily, I
        have a lot of RAM to play with. =P
    """
    tfk.backend.clear_session()     # <-- trying to mitigate memory leaks in tensorflow.keras, but not working as expected
    gc.collect()                    # <-- trying to mitigate memory leaks in tensorflow.keras, but not working as expected
    
    
    #   check apparent convergence condition
    if len(EVOLUTION_LOG) > 10:
        if np.std(EVOLUTION_LOG[-10 :]) == 0:
            print("**** STOPPING EARLY : CONVERGENCE ****")
            quit_flag = True
    
    
    #   check timeout condition
    global START_TIME
    global TIME_LIMIT_S
    
    end_time = time.time()
    
    time_elapsed = end_time - START_TIME
    print()
    print("time elapsed: ", time_elapsed, "s")
    print()
    
    if time_elapsed > TIME_LIMIT_S:
        print("**** STOPPING EARLY : TIMEOUT ****")
        quit_flag = True
    
    
    #   handle early stopping
    if quit_flag:
        quit()
    
    return

#### ============================================================================== ####



#### ============================================================================== ####

def blendedLossFunction(y_true : nptype.ArrayLike, y_pred : nptype.ArrayLike) -> float:
    """
    A blended loss function which returns a weighted sum of the mean squared, mean 
    absolute, and mean squared log errors.
    
    Ref: <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError>
    Ref: <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanAbsoluteError>
    Ref: <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredLogarithmicError>
    Ref: <https://stats.stackexchange.com/questions/261704/training-a-neural-network-for-regression-always-predicts-the-mean>
    
    Parameters
    ----------
    
    y_true : nptype.ArrayLike
        An array like of true target values.
    
    y_pred : nptype.ArrayLike
        An array like of predicted target values.
    
    Returns
    -------
    
    float
        The loss value.
    """
    
    mean_squared_error_weight = 1
    mean_squared_error_term = tfk.losses.MeanSquaredError().call(y_true, y_pred)
    
    mean_absolute_error_weight = 1
    mean_absolute_error_term = tfk.losses.MeanAbsoluteError().call(y_true, y_pred)
    
    mean_squared_log_error_weight = 5
    mean_squared_log_error_term = tfk.losses.MeanSquaredLogarithmicError().call(y_true, y_pred)
        
    blended_loss = (
        (mean_squared_error_weight * mean_squared_error_term) +
        (mean_absolute_error_weight * mean_absolute_error_term) +
        (mean_squared_log_error_weight * mean_squared_log_error_term)
    )
    
    return blended_loss


#### ============================================================================== ####



#### ============================================================================== ####

def buildSequentialRegressor(
    input_dimension : int,
    output_dimension : int,
    hidden_layers : list[int],
    activation : str,
    dropout_rate : float = 0
) -> tfk.Sequential:
    """
    A helper function which initializes and compiles a tensorflow.keras.Sequential model
    for use in regression.
    
    Ref: <https://www.geeksforgeeks.org/vanishing-and-exploding-gradients-problems-in-deep-learning/>
    Ref: <https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-neural-networks-with-weight-constraints-in-keras/>
    Ref: <https://stats.stackexchange.com/questions/299292/dropout-makes-performance-worse>
    
    Parameters
    ----------
    
    input_dimension : int
        The input dimension, or number of neurons in the input layer.
    
    output_dimension : int
        The output (target) dimension, or number of neurons in the output layer.
    
    hidden_layers : list[int]
        A list of integers, where the len of the list is the number of hidden layers, 
        and each element of the list is the number of neurons in each layer.
    
    activation : str
        The activation function to use in each hidden layer neuron.
    
    dropout_rate : float = 0
        The dropout rate to use for dropout layers. Defaults to 0 (i.e., no dropout 
        layers by default).
    
    Returns
    -------
    
    tfk.Sequential
        A compiled sequential regressor, namely tensorflow.keras.Sequential.
    """
    
    #   init sequential regressor, add input layer
    sequential_regressor = tfk.Sequential()
    
    sequential_regressor.add(
        tfk.layers.InputLayer(
            shape=(input_dimension,),
            name="input_layer"
        )
    )
    
    
    #   build out hidden layers
    layer_count = 1
    
    for n_neurons in hidden_layers:
        if n_neurons >= 1:
            sequential_regressor.add(
                tfk.layers.Dense(
                    n_neurons,
                    activation=activation,
                    kernel_initializer=tfk.initializers.GlorotUniform(),
                    kernel_constraint=tfk.constraints.MaxNorm(max_value=2.0),
                    name="hidden_layer_{}".format(layer_count)
                )
            )
            
            if dropout_rate > 0 and layer_count < len(hidden_layers):
                sequential_regressor.add(
                    tfk.layers.Dropout(
                        dropout_rate,
                        name="dropout_layer_{}".format(layer_count)
                    )
                )
        
        layer_count += 1
    
    
    #   add output layer
    sequential_regressor.add(
        tfk.layers.Dense(
            output_dimension,
            activation="linear",    # since it's a regressor
            kernel_initializer=tfk.initializers.GlorotUniform(),
            kernel_constraint=tfk.constraints.MaxNorm(max_value=2.0),
            name="output_layer"
        )
    )
    
    
    #   compile sequential regressor
    sequential_regressor.compile(
        optimizer=tfk.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        ),
        loss=blendedLossFunction
    )
    
    return sequential_regressor

#### ============================================================================== ####



#### ============================================================================== ####

def hyperparamObjective(
    hyperparam_array : nptype.ArrayLike,
    input_data : nptype.ArrayLike,
    target_data : nptype.ArrayLike
) -> float:
    """
    Function which takes in a candidate hyperparameter array along with input and target
    data (not normalized), randomly splits the data into training and test sets,
    normalizes the data, trains a sequential regressor, computes various
    training and test set performance metrics, and then computes and returns a
    multiobjective which seeks to balance overfitting against training and test set
    performance.
    
    Parameters
    ----------
    
    hyperparam_array : nptype.ArrayLike
        An array of floating point values for the hyperparameters. The first three 
        parameters are the number of neurons in each hidden layer (if 0, then layer is 
        omitted), and the fourth parameter is mapped to the activation function to be 
        used.
    
    input_data : nptype.ArrayLike
        An array-like of input data (not normalized).
    
    target_data : nptype.ArrayLike
        An array-like of target data.
    
    Returns
    -------
    
    float
        A multiobjective which seeks to balance overfitting against training and test 
        set performance.
    """
    
    #   get input and output dimensions
    try:
        input_dimension = input_data.shape[1]
    except:
        input_dimension = 1
    
    try:
        output_dimension = target_data.shape[1]
    except:
        output_dimension = 1
    
    
    #   unpack hyperparameter array
    global LINEAR_ACTIVATION_DICT
    
    hidden_layers = [round(x) for x in hyperparam_array[0:6]]
    activation = LINEAR_ACTIVATION_DICT[round(hyperparam_array[6])]
    
    if len(hidden_layers) == 0:
        hidden_layers = [1]
    
    
    #   reduce amount of data used for hyperparameter optimization (to help with runtime here)
    #   only use up to 35,000 rows of data
    opt_size = 35000 / input_data.shape[0]
    
    if opt_size < 1:
        (
            input_data,
            __,
            target_data,
            __
        ) = skl_ms.train_test_split(
            input_data,
            target_data,
            train_size=opt_size
        )
    
    
    #   split into training and test data (random 80/20 split)
    (
        input_train,
        input_test,
        target_train,
        target_test
    ) = skl_ms.train_test_split(
        input_data,
        target_data,
        test_size=0.20
    )
    
    
    #   normalize using a standard scaler (calibrated using training data)
    standard_scaler= skl_pp.StandardScaler()
    standard_scaler.fit(input_train)
    
    input_train_norm = standard_scaler.transform(input_train)
    input_test_norm = standard_scaler.transform(input_test)
    
    
    #   build sequential regressor (tensorflow.keras.Sequential)
    sequential_regressor = buildSequentialRegressor(
        input_dimension,
        output_dimension,
        hidden_layers,
        activation
    )
    
    
    #   train sequential regressor
    sequential_regressor.fit(
        input_train_norm,
        target_train,
        epochs=MAX_TRAINING_EPOCHS,
        verbose=0,
        callbacks=[
            tfk.callbacks.EarlyStopping(
                patience=MAX_TRAINING_EPOCHS,
                start_from_epoch=0,
                restore_best_weights=True
            )
        ],
        validation_data=(input_test_norm, target_test)
    )
    
    
    #   compute regression performance metrics
    predict_train = sequential_regressor.predict(
        input_train_norm,
        verbose=0
    ).flatten()
    
    predict_test = sequential_regressor.predict(
        input_test_norm,
        verbose=0
    ).flatten()
    
    train_diffs = target_train - predict_train
    test_diffs = target_test - predict_test
    
    mu_train = np.mean(train_diffs)
    sigma_train = np.std(train_diffs)
    median_train = np.median(train_diffs)
    
    mu_test = np.mean(test_diffs)
    sigma_test = np.std(test_diffs)
    median_test = np.median(test_diffs)
    
    
    #   compute objective (multiobjective which balances overfitting against training 
    #   and test set performance)
    objective = 0
    
    objective += abs(mu_test - mu_train)            # minimize overfitting (abs difference of mean errors)
    objective += abs(sigma_test - sigma_train)      # minimize overfitting (abs difference of error standard deviations)
    objective += abs(median_test - median_train)    # minimize overfitting (abs difference of median errors)
    
    objective += abs(mu_train)                      # minimize training set abs mean error
    objective += sigma_train                        # minimize training set error standard deviation
    objective += abs(median_train)                  # minimize training set abs median error
    
    objective += abs(mu_test)                       # minimize test set abs mean error
    objective += sigma_test                         # minimize test set error standard deviation
    objective += abs(median_test)                   # minimize test set abs median error
    
    
    #   guard against nan objectives
    if math.isnan(objective):
        objective = math.inf
    
    
    #   running print
    print(
        "hidden layers: {}, ".format(hidden_layers),
        "activation: " + activation + ", ",
        "objective: {}".format(round(objective, 6)),
        4 * " ",
        end="\r",
        flush=True
    )
    
    
    #   clean up and return
    del sequential_regressor    # <-- trying to mitigate memory leaks in tensorflow.keras, but not working as expected
    
    return objective

#### ============================================================================== ####



if __name__ == "__main__":
    #   load data
    input_regression = np.load("data/extended_feature_array_trimmed.npy")
    target_regression = np.load("data/target_ratios_array_trimmed.npy")
    
    size = input_regression.shape[0]
    dimension = input_regression.shape[1]
    
    
    
    #   seek optimal hyperparameters (by way of differential evolution)
    #   Ref: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>
    #   Ref: <https://en.wikipedia.org/wiki/Differential_evolution>
    #   Ref: <https://xloptimizer.com/features/differential-evolution/de-best-1-bin>
    #   Ref: <https://machinelearningmastery.com/differential-evolution-from-scratch-in-python/>
    print()
    print("CPU count:", os.cpu_count())
    print()
    
    print("sequential_regressor GENERATION:", GENERATION)
    
    bounds_list = N_HIDDEN_LAYERS * [(0, MAX_NEURONS)]
    bounds_list += [(0, MAX_ACTIVATION)]
    
    integrality_list = (N_HIDDEN_LAYERS + 1) * [True]
    
    opt_res = spo.differential_evolution(
        hyperparamObjective,
        bounds=bounds_list,
        integrality=integrality_list,
        args=(input_regression, target_regression.flatten()),
        popsize=64,
        init="sobol",
        mutation=(0.65, 0.98),  # differential weight (typical ~ 0.8)
        recombination=0.9,      # crossover probability (typical)
        polish=False,
        workers=max([1, os.cpu_count() - 1]),
        updating="deferred",
        callback=evolutionCallback
    )
    
    end_time = time.time()
    
    print()
    print("\nTime to optimize hyperparameters: {} s".format(round(end_time - START_TIME)))
    
    
    
    #   unpack, save, and print apparent optimal hyperparameter array
    hidden_layers = [round(x) for x in opt_res.x[0:6]]
    activation = LINEAR_ACTIVATION_DICT[round(opt_res.x[6])]
    
    np.save(
        "data/perturbation_machine_hyperparams.npy",
        np.array(hidden_layers + [activation])
    )
    
    print(
        "\napparent optimal hyperparameters (tensorflow.keras.Sequential):\n\n",
        "\thidden layers: {}\n".format(hidden_layers),
        "\tactivation: " + activation
    )
