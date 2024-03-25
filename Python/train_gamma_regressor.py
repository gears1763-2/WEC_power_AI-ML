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
    A script for training a gamma regressor (tensorflow.keras.Sequential).
"""


import math

import matplotlib.pyplot as plt

import numpy as np

import sklearn.metrics as skl_me
import sklearn.model_selection as skl_ms
import sklearn.preprocessing as skl_pp

import tensorflow.keras as tfk



#### ============================================================================== ####

def buildSequentialRegressor(
    input_dimension : int,
    output_dimension : int,
    hidden_layers : list[int],
    activation : str,
    dropout_rate : float = 0.1
) -> tfk.Sequential:
    """
    A helper function which initializes and compiles a tensorflow.keras.Sequential model
    for use in regression.
    
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
        The activation function to use in each neuron.
    
    dropout_rate : float = 0.1
        The dropout rate to use for dropout layers. Defaults to 0.1. Not presently used.
    
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
                    name="hidden_layer_{}".format(layer_count)
                )
            )
            """
            sequential_regressor.add(
                tfk.layers.Dropout(
                    dropout_rate,
                    name="dropout_layer_{}".format(layer_count)
                )
            )
            """
            layer_count += 1
    
    
    #   add output layer
    sequential_regressor.add(
        tfk.layers.Dense(
            output_dimension,
            activation=activation,
            name="output_layer"
        )
    )
    
    
    #   compile sequential regressor
    adam_learning_rate = 0.001
    
    sequential_regressor.compile(
        optimizer=tfk.optimizers.Adam(adam_learning_rate),
        loss="mean_squared_error"
    )
    
    return sequential_regressor

#### ============================================================================== ####



if __name__ == "__main__":
    #   load data
    input_regression = np.load("data/feature_array_no_sentinel.npy")
    target_regression = np.load("data/gamma_array_no_sentinel.npy")
    
    size = input_regression.shape[0]
    dimension = input_regression.shape[1]
    
    
    
    #   drop Pi_4 (since exactly correlated with peak period T_p)
    input_regression = np.delete(input_regression, dimension - 1, axis=1)
    dimension -= 1
    
    
    
    #   load optimal hyperparameters
    sequential_regressor_hyperparams = np.load("data/gamma_regressor_hyperparams.npy")
    
    hidden_layers = [int(x) for x in sequential_regressor_hyperparams[0:6]]
    activation = sequential_regressor_hyperparams[6]
    
    print(
        "\napparent optimal hyperparameters (tensorflow.keras.Sequential):\n\n",
        "\thidden layers: {}\n".format(hidden_layers),
        "\tactivation: " + activation
    )
    
    
    
    #   split into training and test data (random 80/20 split)
    (
        input_regression_train,
        input_regression_test,
        target_regression_train,
        target_regression_test
    ) = skl_ms.train_test_split(
        input_regression,
        target_regression.flatten(),
        test_size=0.20
    )
    
    
    
    #   save training and test data for later application
    np.savez(
        "data/gamma_regressor_train_test_split.npz",
        input_regression_train=input_regression_train,
        input_regression_test=input_regression_test,
        target_regression_train=target_regression_train,
        target_regression_test=target_regression_test
    )
    
    
    
    #   normalize using min-max scaler (calibrated using training data)
    min_max_scaler = skl_pp.MinMaxScaler()
    min_max_scaler.fit(input_regression_train)
    
    input_regression_train_norm = min_max_scaler.transform(input_regression_train)
    input_regression_test_norm = min_max_scaler.transform(input_regression_test)
    
    
    
    #   init and train sequential regressor (using apparent optimal hyperparameters)
    gamma_regressor = buildSequentialRegressor(
        dimension,
        1,
        hidden_layers,
        activation
    )
    
    gamma_regressor.summary()
    
    fit_history = gamma_regressor.fit(
        input_regression_train_norm,
        target_regression_train,
        epochs=512,
        callbacks=[
            tfk.callbacks.EarlyStopping(
                patience=32,
                start_from_epoch=32
            )
        ],
        validation_data=(input_regression_test_norm, target_regression_test)
    )
    
    
    
    #   plot training results
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.plot(
        fit_history.history["loss"],
        color="C0",
        zorder=2,
        label="training set"
    )
    plt.plot(
        fit_history.history["val_loss"],
        color="C3",
        linestyle="--",
        zorder=3,
        label="test set"
    )
    plt.xlabel("Training Epoch [ ]")
    plt.ylabel("Loss Metric (mean squared error) [ ]")
    plt.savefig(
        "../LaTeX/images/regressor/gamma_regressor_training_history.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    
    
    #   save trained model for later application
    gamma_regressor.save("data/gamma_regressor.keras")
    
    
    
    #   predict, plot, and print/save/show peformance metrics
    predict_regression_train = gamma_regressor.predict(
        input_regression_train_norm
    ).flatten()
    
    predict_regression_test = gamma_regressor.predict(
        input_regression_test_norm
    ).flatten()
    
    mu_test = np.mean(target_regression_test - predict_regression_test)
    sigma_test = np.std(target_regression_test - predict_regression_test)
    
    print(
        "\nperformance metrics (tensorflow.keras.Sequential):\n\n",
        "\tmu_test: {}\n".format(round(mu_test, 6)),
        "\tsigma_test: {}\n".format(round(sigma_test, 6))
    )
    
    
    plt_min = 0.90 * min(
        [np.min(target_regression_train), np.min(predict_regression_train)] +
        [np.min(target_regression_test), np.min(predict_regression_test)]
    )
    
    plt_max = 1.10 * max(
        [np.max(target_regression_train), np.max(predict_regression_train)] +
        [np.max(target_regression_test), np.max(predict_regression_test)]
    )
    
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.scatter(
        target_regression_train,
        predict_regression_train,
        color="C0",
        s=16,
        zorder=2,
        alpha=0.8,
        label="training set"
    )
    plt.scatter(
        target_regression_test,
        predict_regression_test,
        color="C3",
        marker="*",
        s=32,
        zorder=3,
        alpha=0.8,
        label="test set"
    )
    plt.plot(
        [plt_min, plt_max],
        [plt_min, plt_max],
        linestyle="--",
        color="black",
        zorder=4
    )
    plt.xlim(plt_min, plt_max)
    plt.xlabel("Target [ ]")
    plt.ylim(plt_min, plt_max)
    plt.ylabel("Prediction [ ]")
    plt.legend()
    plt.savefig(
        "../LaTeX/images/regressor/gamma_regressor_performance_1.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.hist(
        target_regression_train - predict_regression_train,
        bins=20,
        density=True,
        color="C0",
        alpha=0.666,
        zorder=2,
        label="training set"
    )
    plt.hist(
        target_regression_test - predict_regression_test,
        bins=20,
        density=True,
        color="C3",
        alpha=0.666,
        zorder=3,
        label="test set"
    )
    plt.xlabel("Target - Prediction [ ]")
    plt.ylabel("Probability Density [ ]")
    plt.legend()
    plt.savefig(
        "../LaTeX/images/regressor/gamma_regressor_performance_2.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.scatter(
        target_regression_train,
        predict_regression_train - target_regression_train,
        color="C0",
        s=16,
        zorder=2,
        alpha=0.8,
        label="training set"
    )
    plt.scatter(
        target_regression_test,
        predict_regression_test - target_regression_test,
        color="C3",
        marker="*",
        s=32,
        zorder=3,
        alpha=0.8,
        label="test set"
    )
    plt.plot(
        [plt_min, plt_max],
        [0, 0],
        linestyle="--",
        color="black",
        zorder=4
    )
    plt.xlim(plt_min, plt_max)
    plt.xlabel("Target [ ]")
    plt.ylabel("Target - Prediction [ ]")
    plt.legend()
    plt.savefig(
        "../LaTeX/images/regressor/gamma_regressor_performance_3.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    
    #plt.show()
