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
    A script for training a gamma regressor (MLPRegressor).
"""


import math

import matplotlib.pyplot as plt

import numpy as np

import sklearn.metrics as skl_me
import sklearn.model_selection as skl_ms
import sklearn.neural_network as skl_nn
import sklearn.preprocessing as skl_pp


if __name__ == "__main__":
    #   load data
    input_regression = np.load("data/feature_array_no_sentinel.npy")
    target_regression = np.load("data/gamma_array_no_sentinel.npy")
    
    size = input_regression.shape[0]
    dimension = input_regression.shape[1]
    
    
    
    #   drop Pi_4 (since exactly correlated with peak period)
    input_regression = np.delete(input_regression, dimension - 1, axis=1)
    
    
    
    #   load optimal hyperparameters
    ANN_hyperparams = np.load("data/gamma_regressor_hyperparams.npy")
    
    hidden_layers = [int(x) for x in ANN_hyperparams[0:4]]
    activation = ANN_hyperparams[4]
    
    print(
        "\napparent optimal hyperparameters (sklearn.neural_network.MLPRegressor):\n\n",
        "\thidden layers: {}\n".format(hidden_layers),
        "\tactivation: " + activation
    )
    
    
    
    #   split into training and test data (random 80/20 split)
    #   using specific RNG seed here, so as to get "apples-to-apples" metrics
    (
        input_regression_train,
        input_regression_test,
        target_regression_train,
        target_regression_test
    ) = skl_ms.train_test_split(
        input_regression,
        target_regression.flatten(),
        test_size=0.20,
        random_state=12345
    )
    
    
    
    #   normalize using min-max scaler (calibrated using training data)
    standard_scaler = skl_pp.MinMaxScaler()
    standard_scaler.fit(input_regression_train)
    
    input_regression_train_norm = standard_scaler.transform(input_regression_train)
    input_regression_test_norm = standard_scaler.transform(input_regression_test)
    
    
    
    #   init and train ANN regressor (using apparent optimal hyperparameters)
    ANN = skl_nn.MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=activation
    )
    
    ANN.fit(input_regression_train_norm, target_regression_train)
    
    
    
    #   predict, plot, and print/save/show peformance metrics
    predict_regression_train = ANN.predict(input_regression_train_norm)
    predict_regression_test = ANN.predict(input_regression_test_norm)
    
    mu_test = np.mean(target_regression_test - predict_regression_test)
    sigma_test = np.std(target_regression_test - predict_regression_test)
    
    print(
        "\nperformance metrics (sklearn.neural_network.MLPRegressor):\n\n",
        "\tmu_test: {}\n".format(round(mu_test, 6)),
        "\tsigma_test: {}\n".format(round(sigma_test, 6))
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
        [np.min(target_regression_test), np.max(target_regression_test)],
        [np.min(target_regression_test), np.max(target_regression_test)],
        linestyle="--",
        color="black",
        zorder=4
    )
    plt.xlim(np.min(target_regression_test), np.max(target_regression_test))
    plt.xlabel("Target [ ]")
    plt.ylim(np.min(target_regression_test), np.max(target_regression_test))
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
        [np.min(target_regression_test), np.max(target_regression_test)],
        [0, 0],
        linestyle="--",
        color="black",
        zorder=4
    )
    plt.xlim(np.min(target_regression_test), np.max(target_regression_test))
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
