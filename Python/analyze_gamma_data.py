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
    A script for analyzing gamma data.
"""


import matplotlib.pyplot as plt

import numpy as np


if __name__ == "__main__":
    #   load data
    input_regression = np.load("data/input_gamma_regression.npy")
    dimensionless_feature_array = np.load("data/dimensionless_feature_array.npy")
    target_regression = np.load("data/target_gamma_regression.npy")
    
    dimension = input_regression.shape[1]
    size = input_regression.shape[0]
    
    
    #   build classification array (sentinel vs non-sentinel)
    target_classification = np.zeros(size, dtype=np.int32)
    
    for i in range(0, size):
        if target_regression[i] >= 0:
            target_classification[i] = 1
    
    
    #   plot sentinel vs non-sentinel values
    print("\nMaking some dimensionless sentinel scatter plots ... ", end="", flush=True)
    
    dimensionless_feature_column_list = [
        r"$\Pi_1 = \overline{k}^2(D^2 - d^2)$",
        r"$\Pi_2 = \frac{b}{\rho g H_s^2T_p}$",
        r"$\Pi_3 = \frac{b}{2\sqrt{(k + k_D)m}}$",
        r"$\Pi_4 = T_p\sqrt{\frac{k + k_D}{m}}$"
    ]
    
    n_dimensionless_features = len(dimensionless_feature_column_list)
    for i in range(0, n_dimensionless_features):
        for j in range(i + 1, n_dimensionless_features):
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, zorder=1)
            plt.scatter(
                dimensionless_feature_array[:, i],
                dimensionless_feature_array[:, j],
                s=16,
                c=target_classification,
                cmap="summer",
                zorder=3
            )
            plt.colorbar(
                label="sentinel = 0, non-sentinel = 1"
            )
            plt.xlabel(dimensionless_feature_column_list[i])
            if i < 3:
                plt.xscale("log")
            
            plt.ylabel(dimensionless_feature_column_list[j])
            if j < 3:
                plt.yscale("log")
            
            plt.savefig(
                "../LaTeX/images/mining/dimensionless_sentinel_mining_{}_{}.png".format(i, j),
                format="png",
                dpi=128,
                bbox_inches="tight"
            )
            
            plt.close()
    
    print("DONE (saved to ../LaTeX/images/mining/)")
    
    
    #   filter out sentinels
    not_sentinel_array = target_regression >= 0
    
    input_regression = input_regression[not_sentinel_array, :]
    dimensionless_feature_array = dimensionless_feature_array[not_sentinel_array, :]
    target_regression = target_regression[not_sentinel_array]
    
    size_after = input_regression.shape[0]
    
    print(
        "\nrows with sentinels dropped,",
        round(100 * (size_after / size), 2),
        "% of the data was retained"
    )
    
    
    #   print some gamma statistics
    print()
    print("gamma statistics:")
    print("\tmin:", np.min(target_regression))
    print("\tmax:", np.max(target_regression))
    print("\tmean:", np.mean(target_regression))
    print("\tmedian:", np.median(target_regression))
    print("\tstd:", np.std(target_regression))
    
    
    #   trim apparent outliers
    """
    gamma_95perc = np.percentile(target_regression, 95)
    
    not_outlier_array = target_regression <= gamma_95perc
    
    input_regression = input_regression[not_outlier_array, :]
    dimensionless_feature_array = dimensionless_feature_array[not_outlier_array, :]
    target_regression = target_regression[not_outlier_array]
    
    size_after = input_regression.shape[0]
    
    print(
        "\nrows with apparent outliers dropped,",
        round(100 * (size_after / size), 2),
        "% of the data was retained"
    )
    """
    
    
    #   make some histograms
    print("\nMaking some histograms ... ", end="", flush=True)
    
    feature_name_list = [
        "Water Depth [m]",
        "Significant Wave Height [m]",
        "Wave Peak Period [s]",
        "Float Diameter [m]",
        "Power Takeoff Stiffness [N/m]",
        "Power Takeoff Damping [N.s/m]"
    ]
    
    target_name_list = ["gamma [ ]"]
    
    for i in range(0, dimension):
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, zorder=1)
        plt.hist(
            input_regression[:, i],
            bins="scott",
            zorder=2
        )
        plt.xlim(np.min(input_regression[:, i]), np.max(input_regression[:, i]))
        plt.xlabel(feature_name_list[i])
        if i == dimension - 1:
            plt.xscale("log")
        plt.ylabel("Count [ ]")
        plt.savefig(
            "../LaTeX/images/mining/histogram_feature_{}_no_sentinel.png".format(i),
            format="png",
            dpi=128,
            bbox_inches="tight"
        )
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.hist(
        target_regression,
        bins="scott",
        zorder=2
    )
    plt.xlim(np.min(target_regression), np.max(target_regression))
    plt.xlabel(target_name_list[0])
    plt.ylabel("Count [ ]")
    plt.yscale("log")
    plt.savefig(
        "../LaTeX/images/mining/histogram_target_no_sentinel.png".format(i),
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    print("DONE (saved to ../LaTeX/images/mining/)")
    
    
    #   plot dimensionless gamma heat maps
    print("\nMaking some dimensionless gamma heat maps ... ", end="", flush=True)
    
    dimensionless_feature_column_list = [
        r"$\Pi_1 = \overline{k}^2(D^2 - d^2)$",
        r"$\Pi_2 = \frac{b}{\rho g H_s^2T_p}$",
        r"$\Pi_3 = \frac{b}{2\sqrt{(k + k_D)m}}$",
        r"$\Pi_4 = T_p\sqrt{\frac{k + k_D}{m}}$"
    ]
    
    n_dimensionless_features = len(dimensionless_feature_column_list)
    for i in range(0, n_dimensionless_features):
        for j in range(i + 1, n_dimensionless_features):
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, zorder=1)
            plt.scatter(
                dimensionless_feature_array[:, i],
                dimensionless_feature_array[:, j],
                s=16,
                c=target_regression,
                cmap="jet",
                zorder=3
            )
            plt.colorbar(
                label=target_name_list[0]
            )
            plt.xlabel(dimensionless_feature_column_list[i])
            if i < 3:
                plt.xscale("log")
            
            plt.ylabel(dimensionless_feature_column_list[j])
            if j < 3:
                plt.yscale("log")
            
            plt.savefig(
                "../LaTeX/images/mining/dimensionless_gamma_mining_{}_{}.png".format(i, j),
                format="png",
                dpi=128,
                bbox_inches="tight"
            )
            
            plt.close()
    
    print("DONE (saved to ../LaTeX/images/mining/)")
