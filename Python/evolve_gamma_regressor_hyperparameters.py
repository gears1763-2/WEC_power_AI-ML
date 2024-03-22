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
    A script for evolving hyperparameters for a gamma regressor (MLPRegressor).
"""

import math

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as nptype

import os

import scipy.optimize as spo

import sklearn.metrics as skl_me
import sklearn.model_selection as skl_ms
import sklearn.neural_network as skl_nn
import sklearn.preprocessing as skl_pp

import time



#### ============================================================================== ####

start_time = time.time()
time_limit_s = 48 * 3600    # 48 hours

generation = 0
evolution_log = []

activation_dict = {
    0: "identity",
    1: "logistic",
    2: "tanh",
    3: "relu"
}

#### ============================================================================== ####



#### ============================================================================== ####

def evolutionCallback(intermediate_result : spo.OptimizeResult) -> None:
    """
    A callback function used to track the change in best objective value from one 
    generation to the next (for plotting later). Also enforces evolution timeout.
    
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
    global generation
    global evolution_log
    
    generation += 1
    evolution_log.append(intermediate_result.fun)
    
    print()
    print(
        "ANN generation:",
        generation,
        "\n\thyperparams: ",
        intermediate_result.x,
        "\n\tobjective: ",
        round(intermediate_result.fun, 6)
    )
    
    global activation_dict
    
    hidden_layers = [round(x) for x in intermediate_result.x[0:4] if round(x) > 0]
    activation = activation_dict[round(intermediate_result.x[4])]

    np.save(
        "data/gamma_regressor_hyperparams.npy",
        np.array(hidden_layers + [activation])
    )
    
    
    #   check apparent convergence condition
    if len(evolution_log) > 10:
        if np.std(evolution_log[-10 :]) == 0:
            quit_flag = True
    
    
    #   check timeout condition
    global start_time
    global time_limit_s
    
    end_time = time.time()
    
    time_elapsed = end_time - start_time
    print()
    print("time elapsed: ", time_elapsed, "s")
    print()
    
    if time_elapsed > time_limit_s:
        quit_flag = True
    
    
    #   handle early stopping
    if quit_flag:
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7")
        plt.plot(
            [i for i in range(0, len(evolution_log))],
            evolution_log,
            zorder=2
        )
        plt.xlim(0, len(evolution_log) - 1)
        plt.xlabel("Generation [ ]")
        plt.ylabel("Objective Value [ ]")
        plt.savefig(
            "../LaTeX/images/regressor/gamma_regressor_evolution.png",
            format="png",
            dpi=128,
            bbox_inches="tight"
        )
        
        print("**** STOPPING EARLY ****")
        quit()
    
    return

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
    normalizes the data, trains an ANN regressor, computes various training and test set
    performance metrics, and then computes and returns a multiobjective which seeks to
    balance overfitting against training and test set performance.
    
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
    
    #   unpack hyperparameter array
    global activation_dict
    
    hidden_layers = [round(x) for x in hyperparam_array[0:4] if round(x) > 0]
    activation = activation_dict[round(hyperparam_array[4])]
    
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
    
    
    #   normalize using min-max scaler (calibrated using training data)
    standard_scaler = skl_pp.MinMaxScaler()
    standard_scaler.fit(input_train)
    
    input_train_norm = standard_scaler.transform(input_train)
    input_test_norm = standard_scaler.transform(input_test)
    
    
    #   init ANN (multi-layer perceptron) regressor
    ANN = skl_nn.MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        max_iter=1000,
        early_stopping=True
    )
    
    
    #   train ANN regressor
    ANN.fit(input_train_norm, target_train)
    
    
    #   compute regression performance metrics
    predict_train = ANN.predict(input_train_norm)
    predict_test = ANN.predict(input_test_norm)
    
    mu_train = np.mean(target_train - predict_train)
    sigma_train = np.std(target_train - predict_train)
    
    mu_test = np.mean(target_test - predict_test)
    sigma_test = np.std(target_test - predict_test)
    
    
    #   compute objective (multiobjective which balances overfitting against training 
    #   and test set performance)
    objective = 0
    
    objective += abs(mu_test - mu_train)            # minimize overfitting (abs difference of mean errors)
    objective += abs(sigma_test - sigma_train)      # minimize overfitting (abs difference of error standard deviations)
    
    objective += abs(mu_train)                      # minimize training set abs mean error
    objective += sigma_train                        # minimize training set error standard deviation
    
    objective += abs(mu_test)                       # minimize test set abs mean error
    objective += sigma_test                         # minimize test set error standard deviation
    
    
    #   running print and return
    print(
        "hidden layers: {}, ".format(hidden_layers),
        "activation: " + activation + ", ",
        "objective: {}".format(round(objective, 6)),
        4 * " ",
        end="\r",
        flush=True
    )
    
    return objective

#### ============================================================================== ####



if __name__ == "__main__":
    #   load data
    input_regression = np.load("data/feature_array_no_sentinel.npy")
    target_regression = np.load("data/gamma_array_no_sentinel.npy")
    
    size = input_regression.shape[0]
    dimension = input_regression.shape[1]
    
    
    
    #   drop Pi_4 (since exactly correlated with peak period)
    input_regression = np.delete(input_regression, dimension - 1, axis=1)
    
    
    
    #   seek optimal hyperparameters (by way of differential evolution)
    #   Ref: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>
    #   Ref: <https://en.wikipedia.org/wiki/Differential_evolution>
    #   Ref: <https://xloptimizer.com/features/differential-evolution/de-best-1-bin>
    #   Ref: <https://machinelearningmastery.com/differential-evolution-from-scratch-in-python/>
    print("CPU count:", os.cpu_count())
    print()
    
    print("ANN generation:", generation)
    
    opt_res = spo.differential_evolution(
        hyperparamObjective,
        bounds=[
            (0, 256),
            (0, 256),
            (0, 256),
            (0, 256),
            (0, 3)
        ],
        args=(
            input_regression,
            target_regression.flatten()
        ),
        popsize=64,
        init="sobol",
        mutation=(0.65, 0.95),  # differential weight (typical ~ 0.8)
        recombination=0.9,      # crossover probability (typical)
        polish=False,
        workers=max([1, os.cpu_count() - 1]),
        updating="deferred",
        callback=evolutionCallback
    )
    
    end_time = time.time()
    
    print()
    print("\nTime to optimize hyperparameters: {} s".format(round(end_time - start_time)))
    
    
    
    #   unpack, save, and print apparent optimal hyperparameter array
    hidden_layers = [round(x) for x in opt_res.x[0:4] if round(x) > 0]
    activation = activation_dict[round(opt_res.x[4])]

    np.save(
        "data/gamma_regressor_hyperparams.npy",
        np.array(hidden_layers + [activation])
    )
    
    print(
        "\napparent optimal hyperparameters (sklearn.neural_network.MLPRegressor):\n\n",
        "\thidden layers: {}\n".format(hidden_layers),
        "\tactivation: " + activation
    )
    
    
    
    #   plot and save evolution curve
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7")
    plt.plot(
        [i for i in range(0, len(evolution_log))],
        evolution_log,
        zorder=2
    )
    plt.xlim(0, len(evolution_log) - 1)
    plt.xlabel("Generation [ ]")
    plt.ylabel("Objective Value [ ]")
    plt.savefig(
        "../LaTeX/images/regressor/gamma_regressor_evolution.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
