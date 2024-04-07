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
    A script for mining the PDS simulation results.
"""


import math

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd


if __name__ == "__main__":
    #   read simulation results into pandas DataFrame
    simulation_results = pd.read_csv("data/PDS_simulation_results.csv")
    print("DataFrame shape:", simulation_results.shape)
    print()
    
    size = simulation_results.shape[0]
    
    
    
    #   define feature and target lists
    column_list = list(simulation_results)
    print("column list:\n\t", column_list)
    print()
    
    feature_column_list = column_list[1 : 7]
    print("feature column list:\n\t", feature_column_list)
    print()
    
    target_column_list = column_list[7]
    print("target column list:\n\t", target_column_list)
    print()
    
    
    
    #   drop rows with ERROR terminal condition
    filter_idx = simulation_results["Simulation Terminal Condition"] != "ERROR"
    
    simulation_results = simulation_results[filter_idx]
    
    size_after = simulation_results.shape[0]
    
    print(
        "Rows with ERROR terminal condition dropped.\n",
        round(100 * (size_after / size), 2),
        "% of the data was retained."
    )
    print()
    
    
    
    #   drop rows with NaNs
    simulation_results.dropna()
    
    size_after = simulation_results.shape[0]
    
    print(
        "Rows with NaNs dropped.\n",
        round(100 * (size_after / size), 2),
        "% of the data was retained."
    )
    print()
    
    
    
    #   drop rows with H_s > Z / 2
    filter_idx = (
        simulation_results["Significant Wave Height [m]"] <=
        0.5 * simulation_results["Water Depth [m]"]
    )
    
    simulation_results = simulation_results[filter_idx]
    
    size_after = simulation_results.shape[0]
    
    print(
        "Rows with H_s > 0.5 * Z dropped.\n",
        round(100 * (size_after / size), 2),
        "% of the data was retained."
    )
    print()
    
    
    
    #   extract feature and target arrays
    feature_array = simulation_results[feature_column_list].values
    target_array = simulation_results[target_column_list].values.flatten()
    
    size = feature_array.shape[0]
    dimension = feature_array.shape[1]
    
    
    
    #   save feature and target arrays
    np.save(
        "data/feature_array.npy",
        feature_array
    )
    
    np.save(
        "data/target_array.npy",
        target_array
    )
    
    
    
    #   print some target statistics
    print("min target =", round(np.min(target_array), 3), "kW")
    print("max target =", round(np.max(target_array), 3), "kW")
    print("mean target =", round(np.mean(target_array), 3), "kW")
    print("standard deviation of target =", round(np.std(target_array), 3), "kW")
    print("median target =", round(np.median(target_array), 3), "kW")
    
    percentiles_list = [5, 20, 40, 60, 80, 90, 95, 99]
    percentiles_array = np.percentile(target_array, percentiles_list)
    print()
    print("target percentiles:")
    for i in range(0, len(percentiles_list)):
        print(
            "\t{}-percentile:".format(percentiles_list[i]),
            round(percentiles_array[i], 3),
            "kW"
        )
    print()
    
    
    
    #   make target histogram
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.hist(
        target_array,
        bins="scott",
        zorder=2
    )
    plt.xlim(0, 80000)
    plt.xlabel(target_column_list)
    plt.ylabel("Count [ ]")
    plt.yscale("log")
    plt.savefig(
        "../LaTeX/images/PDS_results_mining/target_histogram.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    plt.close()
    
    
    
    #   get target sorted indices (for the sake of plotting)
    idx_target_sort = np.argsort(target_array)
    
    
    
    #   make some scatter plots
    for i in range(0, dimension):
        for j in range(i + 1, dimension):
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, zorder=1)
            plt.scatter(
                feature_array[idx_target_sort, i],
                feature_array[idx_target_sort, j],
                s=12,
                c=target_array[idx_target_sort],
                cmap="jet",
                alpha=0.75,
                zorder=2
            )
            plt.colorbar(
                label=target_column_list
            )
            plt.xlabel(feature_column_list[i])
            if i == dimension - 1:
                plt.xscale("log")
            plt.ylabel(feature_column_list[j])
            if j == dimension - 1:
                plt.yscale("log")
            plt.savefig(
                "../LaTeX/images/PDS_results_mining/initial_mining_{}_{}.png".format(i, j),
                format="png",
                dpi=128,
                bbox_inches="tight"
            )
            plt.close()
