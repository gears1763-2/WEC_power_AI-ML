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
    A script for training and assessing the perturbation machine given evolved 
    hyperparameters (tensorflow.keras.Sequential).
"""


import math

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as nptype

import sklearn.model_selection as skl_ms
import sklearn.preprocessing as skl_pp

import tensorflow.keras as tfk



#### ============================================================================== ####

@tfk.utils.register_keras_serializable()
def blendedLossFunction(y_true : nptype.ArrayLike, y_pred : nptype.ArrayLike) -> float:
    """
    A blended loss function which returns a weighted sum of the mean squared, mean 
    absolute, and mean squared log errors.
    
    Ref: <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError>
    Ref: <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanAbsoluteError>
    Ref: <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredLogarithmicError>
    
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



if __name__ == "__main__":
    use_dropout_machine = False
    
    #   load data
    input_regression = np.load("data/extended_feature_array_trimmed.npy")
    target_regression = np.load("data/target_ratios_array_trimmed.npy")
    
    size = input_regression.shape[0]
    dimension = input_regression.shape[1]
    
    
    
    #   load data and perturbation machine, build if can't
    try:
        if use_dropout_machine:
            train_test_array = np.load("data/perturbation_machine_DROPOUT_train_test_split.npz")
            
            perturbation_machine = tfk.models.load_model("data/perturbation_machine_DROPOUT.keras")
        
        else:
            train_test_array = np.load("data/perturbation_machine_train_test_split.npz")
            
            perturbation_machine = tfk.models.load_model("data/perturbation_machine.keras")
        
        input_train = train_test_array["input_train"]
        input_test = train_test_array["input_test"]
        target_train = train_test_array["target_train"]
        target_test = train_test_array["target_test"]
        
        standard_scaler = skl_pp.StandardScaler()
        standard_scaler.fit(input_train)
        
        input_train_norm = standard_scaler.transform(input_train)
        input_test_norm = standard_scaler.transform(input_test)
        
        perturbation_machine.summary()
    
    
    except:
        #   split into training and test data (random 80/20 split), save for later use
        (
            input_train,
            input_test,
            target_train,
            target_test
        ) = skl_ms.train_test_split(
            input_regression,
            target_regression.flatten(),
            test_size=0.20
        )
        
        if use_dropout_machine:
            np.savez(
                "data/perturbation_machine_DROPOUT_train_test_split.npz",
                input_train=input_train,
                input_test=input_test,
                target_train=target_train,
                target_test=target_test
            )
            
        else:
            np.savez(
                "data/perturbation_machine_train_test_split.npz",
                input_train=input_train,
                input_test=input_test,
                target_train=target_train,
                target_test=target_test
            )
        
        
        
        #   normalize using a standard scaler (calibrated using training data)
        standard_scaler = skl_pp.StandardScaler()
        standard_scaler.fit(input_train)
        
        input_train_norm = standard_scaler.transform(input_train)
        input_test_norm = standard_scaler.transform(input_test)
        
        
        
        #   load optimal hyperparameters
        if use_dropout_machine:
            hidden_layers = [512 for i in range(0, 16)]
            activation = "relu"
        
        else:
            opt_hyperparams = np.load("data/perturbation_machine_hyperparams.npy")
            
            hidden_layers = [int(x) for x in opt_hyperparams[0:6]]
            activation = opt_hyperparams[6]
        
        print("hidden_layers =", hidden_layers)
        print("activation =", activation)
        
        
        
        #   build perturbation machine (tensorflow.keras.Sequential), print summary
        if use_dropout_machine:
            perturbation_machine = buildSequentialRegressor(
                dimension,
                1,
                hidden_layers,
                activation,
                dropout_rate=0.2
            )
        
        else:
            perturbation_machine = buildSequentialRegressor(
                dimension,
                1,
                hidden_layers,
                activation
            )
        
        perturbation_machine.summary()
        
        
        
        #   train perturbation machine
        train_history = perturbation_machine.fit(
            input_train_norm,
            target_train,
            epochs=128,
            callbacks=[
                tfk.callbacks.EarlyStopping(
                    patience=128,
                    start_from_epoch=0,
                    restore_best_weights=True
                )
            ],
            validation_data=(input_test_norm, target_test)
        )
        
        
        
        #   plot training history
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, zorder=1)
        plt.plot(
            train_history.history["loss"],
            color="C0",
            zorder=2,
            label="training"
        )
        plt.plot(
            train_history.history["val_loss"],
            color="C3",
            linestyle="--",
            alpha=0.666,
            zorder=3,
            label="test"
        )
        plt.xlim(0, len(train_history.history["loss"]))
        plt.xlabel("Training Epoch [ ]")
        plt.ylim(
            0,
            1.01 * max(
                [
                    max(train_history.history["loss"]),
                    max(train_history.history["val_loss"])
                ]
            )
        )
        plt.ylabel(r"Loss Function [ ]")
        plt.legend()
        plt.savefig(
            "../LaTeX/images/perturbation_machine/perturbation_machine_{}_training.png".format(use_dropout_machine),
            format="png",
            dpi=128,
            bbox_inches="tight"
        )
        
        
        
        #   save perturbation machine
        if use_dropout_machine:
            perturbation_machine.save("data/perturbation_machine_DROPOUT.keras")
        
        else:
            perturbation_machine.save("data/perturbation_machine.keras")
        
        plt.show()
    
    
    
    #   compute regression performance metrics
    predict_train = perturbation_machine.predict(input_train_norm).flatten()
    predict_test = perturbation_machine.predict(input_test_norm).flatten()
    
    train_diffs = target_train - predict_train
    test_diffs = target_test - predict_test
    
    mu_train = np.mean(train_diffs)
    sigma_train = np.std(train_diffs)
    median_train = np.median(train_diffs)
    
    mu_test = np.mean(test_diffs)
    sigma_test = np.std(test_diffs)
    median_test = np.median(test_diffs)
    
    print("mu_train =", mu_train)
    print("sigma_train =", sigma_train)
    print("median_train =", median_train)
    print()
    
    print("mu_test =", mu_test)
    print("sigma_test =", sigma_test)
    print("median_test =", median_test)
    print()
    
    
    
    #   generate performance plots
    plt_min = 1e-2
    plt_max = 2
    
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.scatter(
        target_train,
        predict_train,
        color="C0",
        alpha=0.666,
        s=16,
        zorder=2,
        label="training set"
    )
    plt.scatter(
        target_test,
        predict_test,
        color="C3",
        alpha=0.666,
        marker="*",
        s=32,
        zorder=3,
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
    plt.xscale("log")
    plt.xlabel("Target [ ]")
    plt.ylim(plt_min, plt_max)
    plt.yscale("log")
    plt.ylabel("Prediction [ ]")
    plt.legend()
    plt.savefig(
        "../LaTeX/images/perturbation_machine/perturbation_machine_{}_performance_1.png".format(use_dropout_machine),
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.hist(
        target_train - predict_train,
        bins=20,
        density=True,
        color="C0",
        alpha=0.666,
        zorder=2,
        label="training set"
    )
    plt.hist(
        target_test - predict_test,
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
        "../LaTeX/images/perturbation_machine/perturbation_machine_{}_performance_2.png".format(use_dropout_machine),
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.scatter(
        target_train,
        target_train - predict_train,
        color="C0",
        alpha=0.666,
        s=16,
        zorder=2,
        label="training set"
    )
    plt.scatter(
        target_test,
        target_test - predict_test,
        color="C3",
        alpha=0.666,
        marker="*",
        s=32,
        zorder=3,
        label="test set"
    )
    plt.plot(
        [-0.1, plt_max],
        [0, 0],
        linestyle="--",
        color="black",
        zorder=4
    )
    plt.xlim(-0.1, 1.5)
    plt.xlabel("Target [ ]")
    plt.ylabel("Target - Prediction [ ]")
    plt.legend()
    plt.savefig(
        "../LaTeX/images/perturbation_machine/perturbation_machine_{}_performance_3.png".format(use_dropout_machine),
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
