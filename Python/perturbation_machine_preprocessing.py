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
    A script for pre-processing the features and target for the perturbation machine.
"""


import math

import matplotlib.pyplot as plt

import numpy as np


if __name__ == "__main__":
    #   load features and target (clean)
    feature_array = np.load("data/feature_array.npy")
    dimensionless_feature_array = np.load("data/dimensionless_feature_array.npy")
    target_ratios_array = np.load("data/target_ratios_array.npy")
    reduced_target_array = np.load("data/reduced_target_array.npy")
    target_array = np.load("data/target_array.npy")
    
    size = feature_array.shape[0]
    
    
    
    #   merge into extended feature array
    extended_feature_array = np.append(feature_array, dimensionless_feature_array, axis=1)
    
    
    
    #   log10 scale select columns (namely b, Pi_0, Pi_1, and Pi_3)
    log10_scale_idx = [
        5,  # b
        6,  # Pi_0
        7,  # Pi_1
        9   # Pi_3
    ]
    
    for i in log10_scale_idx:
        extended_feature_array[:, i] = np.log10(extended_feature_array[:, i])
    
    
    
    #   trim target ratios outliers, plot histogram and correlation
    outlier_threshold = np.percentile(target_ratios_array, 95)
    idx_not_outlier = target_ratios_array <= outlier_threshold
    
    extended_feature_array = extended_feature_array[idx_not_outlier, :]
    target_ratios_array = target_ratios_array[idx_not_outlier]
    reduced_target_array = reduced_target_array[idx_not_outlier]
    target_array = target_array[idx_not_outlier]
    
    size_after = extended_feature_array.shape[0]
    
    print(
        "Outliers dropped.\n",
        round(100 * (size_after / size), 2),
        "% of the data was retained."
    )
    print("size after =", size_after)
    print()
    
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.hist(
        target_ratios_array,
        bins="scott",
        zorder=2
    )
    plt.xlim(0, 1.5)
    plt.xticks([0.1 * i for i in range(0, 16)])
    plt.xlabel(r"E$_{PDS}\{P\}$ / E$_{reduced}\{P\}$ [ ]")
    #plt.ylim(1e2, 3e3)
    plt.ylabel("Count [ ]")
    plt.savefig(
        "../LaTeX/images/perturbation_machine/target_ratios_histogram_trimmed.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    plt.close()
    
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.scatter(
        target_array,
        target_ratios_array,
        s=12,
        zorder=2
    )
    plt.xlim(1e-3, 1e5)
    plt.xscale("log")
    plt.xlabel(r"E$_{PDS}\{P\}$ [kW]")
    plt.ylim(0, 1.5)
    plt.yticks([0.1 * i for i in range(0, 16)])
    plt.ylabel(r"E$_{PDS}\{P\}$ / E$_{reduced}\{P\}$ [ ]")
    plt.savefig(
        "../LaTeX/images/perturbation_machine/target_ratios_correlation_trimmed.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    plt.close()
    
    
    
    #   save extended feature array and target ratios array (trimmed)
    np.save(
        "data/extended_feature_array_trimmed.npy",
        extended_feature_array
    )
    
    np.save(
        "data/target_ratios_array_trimmed.npy",
        target_ratios_array
    )
    
    np.save(
        "data/reduced_target_array_trimmed.npy",
        reduced_target_array
    )
    
    np.save(
        "data/target_array_trimmed.npy",
        target_array
    )
