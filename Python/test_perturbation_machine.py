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
    A script for testing the trained perturbation machine.
"""


import math

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as nptype

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



if __name__ == "__main__":
    use_dropout_machine = False
    
    #   load data
    extended_feature_array = np.load("data/extended_feature_array_trimmed.npy")
    expected_power_reduced_array_kW = np.load("data/reduced_target_array_trimmed.npy")
    expected_power_PDS_array_kW = np.load("data/target_array_trimmed.npy")
    
    
    
    #   set up normalizer
    if use_dropout_machine:
        train_test_array = np.load("data/perturbation_machine_DROPOUT_train_test_split.npz")
    
    else:
        train_test_array = np.load("data/perturbation_machine_train_test_split.npz")
    
    input_train = train_test_array["input_train"]
    
    standard_scaler = skl_pp.StandardScaler()
    standard_scaler.fit(input_train)
    
    
    
    #   load perturbation machine
    if use_dropout_machine:
        perturbation_machine = tfk.models.load_model("data/perturbation_machine_DROPOUT.keras")
    
    else:
        perturbation_machine = tfk.models.load_model("data/perturbation_machine.keras")
    
    
    
    #   generate predicted target ratios array
    predicted_target_ratios_array = perturbation_machine.predict(
        standard_scaler.transform(extended_feature_array)
    ).flatten()
    
    
    
    #   generate expected power predicted array (correct any negative to zero)
    expected_power_predicted_array_kW = np.multiply(
        predicted_target_ratios_array, expected_power_reduced_array_kW
    )
    
    for i in range(0, len(expected_power_predicted_array_kW)):
        if expected_power_predicted_array_kW[i] < 0:
            expected_power_predicted_array_kW[i] = 0
    
    
    
    #   generate errors array
    error_array_kW = expected_power_PDS_array_kW - expected_power_predicted_array_kW
    
    
    
    #   print metrics
    mean_error_kW = np.mean(error_array_kW)
    std_error_kW = np.std(error_array_kW)
    median_error_kW = np.median(error_array_kW)
    
    print("mean error =", mean_error_kW, "kW")
    print("std error =", std_error_kW, "kW")
    print("median error =", median_error_kW, "kW")
    print()
    
    
    
    #   plot metrics
    plt_min = 0
    plt_max = 1.05 * np.max(expected_power_PDS_array_kW)
    
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.scatter(
        expected_power_PDS_array_kW,
        expected_power_predicted_array_kW,
        color="C0",
        s=16,
        zorder=2
    )
    plt.plot(
        [plt_min, plt_max],
        [plt_min, plt_max],
        linestyle="--",
        color="black",
        zorder=3
    )
    plt.xlim(1e-3, 1e5)
    plt.xscale("log")
    plt.xlabel(r"E$_{PDS}\{P\}$ [kW]")
    plt.ylim(1e-3, 1e5)
    plt.yscale("log")
    plt.ylabel(r"$\mathcal{P}(\cdots)$E$_{reduced}\{P\}$ [kW]")
    plt.savefig(
        "../LaTeX/images/perturbation_machine/perturbation_machine_{}_power_performance_1.png".format(use_dropout_machine),
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.hist(
        error_array_kW,
        bins="scott",
        density=True,
        color="C0",
        zorder=2
    )
    plt.xlim(
        mean_error_kW - 2 * std_error_kW,
        mean_error_kW + 2 * std_error_kW
    )
    plt.xlabel(r"E$_{PDS}\{P\}$ - $\mathcal{P}(\cdots)$E$_{reduced}\{P\}$ [kW]")
    plt.ylabel("Probability Density [1/kW]")
    plt.savefig(
        "../LaTeX/images/perturbation_machine/perturbation_machine_{}_power_performance_2.png".format(use_dropout_machine),
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.scatter(
        expected_power_PDS_array_kW,
        error_array_kW,
        color="C0",
        s=16,
        zorder=2
    )
    plt.plot(
        [plt_min, plt_max],
        [0, 0],
        linestyle="--",
        color="black",
        zorder=3
    )
    plt.xlim(plt_min, plt_max)
    plt.xlabel(r"E$_{PDS}\{P\}$ [kW]")
    plt.ylabel(r"E$_{PDS}\{P\}$ - $\mathcal{P}(\cdots)$E$_{reduced}\{P\}$ [kW]")
    plt.savefig(
        "../LaTeX/images/perturbation_machine/perturbation_machine_{}_power_performance_3.png".format(use_dropout_machine),
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    
    perc_errors_array = np.divide(error_array_kW, expected_power_PDS_array_kW)
    
    print("mean perc error =", np.mean(perc_errors_array))
    print("std perc error =", np.std(perc_errors_array))
    print("median perc error =", np.median(perc_errors_array))
    print()
    
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.hist(
        perc_errors_array,
        bins="scott",
        density=True,
        color="C0",
        zorder=2
    )
    plt.xlim(
        np.mean(perc_errors_array) - 3 * np.std(perc_errors_array),
        np.mean(perc_errors_array) + 3 * np.std(perc_errors_array)
    )
    plt.xlabel(r"(E$_{PDS}\{P\}$ - $\mathcal{P}(\cdots)$E$_{reduced}\{P\}$) / E$_{PDS}\{P\}$ [ ]")
    plt.ylabel("Probability Density [ ]")
    plt.savefig(
        "../LaTeX/images/perturbation_machine/perturbation_machine_{}_power_performance_4.png".format(use_dropout_machine),
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    lower_95 = np.mean(perc_errors_array) - 2 * np.std(perc_errors_array)
    upper_95 = np.mean(perc_errors_array) + 2 * np.std(perc_errors_array)
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.scatter(
        expected_power_PDS_array_kW,
        perc_errors_array,
        color="C0",
        s=16,
        zorder=2
    )
    plt.plot(
        [plt_min, plt_max],
        [0, 0],
        linestyle="--",
        color="black",
        zorder=3
    )
    plt.axhline(
        y=np.mean(perc_errors_array) + 2 * np.std(perc_errors_array),
        color="C3",
        linestyle="--",
        alpha=0.66,
        zorder=3,
        label="95% range: [{}, {}]".format(
            round(lower_95, 3),
            round(upper_95, 3)
        )
    )
    plt.axhline(
        y=np.mean(perc_errors_array) - 2 * np.std(perc_errors_array),
        color="C3",
        linestyle="--",
        alpha=0.66,
        zorder=3
    )
    plt.xlim(1e-3, 1e5)
    plt.xscale("log")
    plt.xlabel(r"E$_{PDS}\{P\}$ [kW]")
    plt.ylabel(r"(E$_{PDS}\{P\}$ - $\mathcal{P}(\cdots)$E$_{reduced}\{P\}$) / E$_{PDS}\{P\}$ [ ]")
    plt.legend()
    plt.savefig(
        "../LaTeX/images/perturbation_machine/perturbation_machine_{}_power_performance_5.png".format(use_dropout_machine),
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    print()
    print(round(lower_95, 3))
    print(round(upper_95, 3))
    print(round((abs(lower_95) + upper_95) / 2, 3))
