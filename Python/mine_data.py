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
    A script for mining the WEC data set for some initial insight, generating some 
    dimensionless variables, and then preparing and saving the feature and power arrays.
"""


import math

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import wave_utils as wu

import wave_energy_converter as wec


if __name__ == "__main__":
    #   read in data, extract column header list
    WEC_dataframe = pd.read_csv("data/data_table.csv")
    column_header_list = list(WEC_dataframe)
    
    
    
    #   print summary info
    print("WEC_dataframe.shape:", WEC_dataframe.shape)
    print("\ncolumn_header_list:\n")
    print(column_header_list)
    print("\nWEC_dataframe.info():\n")
    print(WEC_dataframe.info())
    print("\nWEC_dataframe.describe():\n")
    print(WEC_dataframe.describe())
    
    
    
    #   drop NaN rows
    size_before = WEC_dataframe.shape[0]
    
    WEC_dataframe = WEC_dataframe.dropna()
    size_after = WEC_dataframe.shape[0]
    
    print(
        "\nrows with NaNs dropped,",
        round(100 * (size_after / size_before), 2),
        "% of the data was retained"
    )
    
    
    
    #   drop ERROR rows
    WEC_dataframe = WEC_dataframe[
        WEC_dataframe["Simulation Terminal Condition"] != "ERROR"
    ]
    size_after = WEC_dataframe.shape[0]
    
    print(
        "\nrows with \"ERROR\" dropped,",
        round(100 * (size_after / size_before), 2),
        "% of the data was retained"
    )
    
    
    
    #   drop rows for which significant wave height is > depth / 2 (deemed to be excessive)
    WEC_dataframe = WEC_dataframe[
        WEC_dataframe["Significant Wave Height [m]"] <=
        (1 / 2) * WEC_dataframe["Water Depth [m]"]
    ]
    size_after = WEC_dataframe.shape[0]
    
    print(
        "\nrows with excessively high significant wave height dropped,",
        round(100 * (size_after / size_before), 2),
        "% of the data was retained"
    )
    
    
    
    #   define feature and target column lists
    feature_column_list = column_header_list[1 : 7]
    target_column_list = [column_header_list[7]]
    
    
    
    #   sort by target value (ascending, for the sake of plotting)
    WEC_dataframe = WEC_dataframe.sort_values(by=target_column_list)
    
    
    
    #   extract feature and target arrays
    feature_array = WEC_dataframe[feature_column_list].astype(float).values
    target_array = WEC_dataframe[target_column_list].astype(float).values
    
    size = feature_array.shape[0]
    dimension = feature_array.shape[1]
    
    
    
    #   make histograms
    print("\nMaking some histograms ... ", end="", flush=True)
    
    for i in range(0, len(feature_column_list)):
        values = WEC_dataframe[feature_column_list[i]].astype(float).values
        
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, zorder=1)
        plt.hist(
            values,
            bins="scott",
            alpha=0.8,
            zorder=2
        )
        plt.xlim(np.min(values), np.max(values))
        plt.xlabel(feature_column_list[i])
        plt.ylabel("Count [ ]")
        plt.savefig(
            "../LaTeX/images/mining/feature_{}_histogram.png".format(i),
            format="png",
            dpi=128,
            bbox_inches="tight"
        )
        
        plt.close()
    
    for i in range(0, len(target_column_list)):
        values = WEC_dataframe[target_column_list[i]].astype(float).values
        
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, zorder=1)
        plt.hist(
            values,
            bins="scott",
            alpha=0.8,
            zorder=2
        )
        plt.xlim(np.min(values), np.max(values))
        plt.xlabel(target_column_list[i])
        plt.yscale("log")
        plt.ylabel("Count [ ]")
        plt.savefig(
            "../LaTeX/images/mining/target_{}_histogram.png".format(i),
            format="png",
            dpi=128,
            bbox_inches="tight"
        )
        
        plt.close()
    
    print("DONE (saved to ../LaTeX/images/mining/)")
    
    
    
    #   log10 scale the power takeoff stiffness, plot histogram
    log10_damping = np.zeros(size)
    
    for i in range(0, size):
        log10_damping[i] = math.log10(feature_array[i, dimension - 1])
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.hist(
        log10_damping,
        bins="scott",
        alpha=0.8,
        zorder=2
    )
    plt.xlim(np.min(log10_damping), np.max(log10_damping))
    plt.xlabel(r"$\log_{10}($" + feature_column_list[dimension - 1] + r"$)$")
    plt.ylabel("Count [ ]")
    plt.savefig(
        "../LaTeX/images/mining/log10_feature_{}_histogram.png".format(dimension - 1),
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    plt.close()
    
    
    
    #   make some scatter plots (initial data mining)
    print("\nMaking some initial scatter plots ... ", end="", flush=True)
    
    n_features = len(feature_column_list)
    for i in range(0, n_features):
        for j in range(i + 1, n_features):
            x_values = feature_array[:, i]
            y_values = feature_array[:, j]
            
            if i == dimension - 1:
                x_values = log10_damping
            
            if j == dimension - 1:
                y_values = log10_damping
            
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, zorder=1)
            plt.scatter(
                x_values,
                y_values,
                s=16,
                c=target_array,
                cmap="jet",
                zorder=3
            )
            plt.colorbar(
                label=target_column_list[0]
            )
            plt.xlabel(feature_column_list[i])
            if i == dimension - 1:
                plt.xlabel(r"$\log_{10}($" + feature_column_list[i] + r"$)$")
            
            plt.ylabel(feature_column_list[j])
            if j == dimension - 1:
                plt.ylabel(r"$\log_{10}($" + feature_column_list[j] + r"$)$")
            
            plt.savefig(
                "../LaTeX/images/mining/initial_data_mining_{}_{}.png".format(i, j),
                format="png",
                dpi=128,
                bbox_inches="tight"
            )
            
            plt.close()
    
    print("DONE (saved to ../LaTeX/images/mining/)")
    
    
    #   generate some dimensionless terms (or load if already done)
    try:
        dimensionless_feature_array = np.load("data/dimensionless_feature_array.npy")
    
    except:
        data_size = len(target_array)
        dimensionless_feature_array = np.zeros((data_size, 4))
        
        """
            dimensionless_feature_array[:, 0] = log_10(overline{k}^2(D^2 - d^2))
            dimensionless_feature_array[:, 1] = log_10(b / (rho * g * H_s^2 * T_p))
            dimensionless_feature_array[:, 2] = log_10(b / 2 * sqrt((k + k_D) * m))
            dimensionless_feature_array[:, 3] = T_p * sqrt((k + k_D) / m)
        """
        
        float_inner_diameter_m = 6
        float_resting_draft_m = 2.5
        
        fluid_density_kgm3 = 1025
        gravity_ms2 = 9.81
        
        fundamental_period_s = 900
        
        for i in range(0, data_size):
            print(
                "Building dimensionless terms",
                i + 1,
                "/",
                data_size,
                16 * " ",
                end="\r",
                flush=True
            )
            
            sea_depth_m = feature_array[i, 0]
            significant_wave_height_m = feature_array[i, 1]
            wave_peak_period_s = feature_array[i, 2]
            float_outer_diameter_m = feature_array[i, 3]
            power_takeoff_stiffness_Nm = feature_array[i, 4]
            power_takeoff_damping_Nsm = feature_array[i, 5]
            
            n_components = 4 * math.ceil(fundamental_period_s / wave_peak_period_s)
            
            frequency_array_Hz = np.array(
                [n / fundamental_period_s for n in range(1, n_components + 1)]
            )
            
            wave_number_array_m = wu.getWaveNumberArray(
                frequency_array_Hz,
                sea_depth_m
            )
            
            characteristic_wave_number_m = wu.getCharacteristicWaveNumber(
                frequency_array_Hz,
                wave_number_array_m,
                significant_wave_height_m,
                wave_peak_period_s,
                sea_depth_m
            )
            
            dimensionless_feature_array[i, 0] = math.log10(
                math.pow(characteristic_wave_number_m, 2) * (
                    math.pow(float_outer_diameter_m, 2) -
                    math.pow(float_inner_diameter_m, 2)
                )
            )
            
            dimensionless_feature_array[i, 1] = math.log10(
                power_takeoff_damping_Nsm / (
                    fluid_density_kgm3 * gravity_ms2 *
                    math.pow(significant_wave_height_m, 2) * wave_peak_period_s
                )
            )
            
            float_mass_kg = wec.getFloatMass(
                float_inner_diameter_m,
                float_outer_diameter_m,
                float_resting_draft_m
            )
            
            buoyancy_stiffness_Nm = wec.getBuoyancyStiffness(
                float_inner_diameter_m,
                float_outer_diameter_m
            )
            
            dimensionless_feature_array[i, 2] = math.log10(
                power_takeoff_damping_Nsm / 
                (
                    2 * math.sqrt(
                        (power_takeoff_stiffness_Nm + buoyancy_stiffness_Nm) *
                        float_mass_kg
                    )
                )
            )
            
            dimensionless_feature_array[i, 3] = (
                wave_peak_period_s *
                math.sqrt(
                    (power_takeoff_stiffness_Nm + buoyancy_stiffness_Nm) /
                    float_mass_kg
                )
            )
        
        print()
    
        np.save(
            "data/dimensionless_feature_array.npy",
            dimensionless_feature_array
        )
    
    
    
    #   make some more scatter plots (dimensionless terms)
    print("\nMaking some dimensionless scatter plots ... ", end="", flush=True)
    
    dimensionless_feature_column_list = [
        r"$\Pi_1 = \log_{10}\left(\overline{k}^2(D^2 - d^2)\right)$",
        r"$\Pi_2 = \log_{10}\left(\frac{b}{\rho g H_s^2T_p}\right)$",
        r"$\Pi_3 = \log_{10}\left(\frac{b}{2\sqrt{(k + k_D)m}}\right)$",
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
                c=target_array,
                cmap="jet",
                zorder=3
            )
            plt.colorbar(
                label=target_column_list[0]
            )
            plt.xlabel(dimensionless_feature_column_list[i])
            plt.ylabel(dimensionless_feature_column_list[j])
            plt.savefig(
                "../LaTeX/images/mining/dimensionless_data_mining_{}_{}.png".format(i, j),
                format="png",
                dpi=128,
                bbox_inches="tight"
            )
            
            plt.close()
    
    print("DONE (saved to ../LaTeX/images/mining/)")
    
    
    
    #   prepare and save feature and power arrays (overwrite damping with log10 damping)
    feature_array = np.append(feature_array, dimensionless_feature_array, axis=1)
    power_array = target_array.flatten()
    
    for i in range(0, size):
        feature_array[i, dimension - 1] = log10_damping[i]
    
    np.save(
        "data/feature_array.npy",
        feature_array
    )
    
    np.save(
        "data/power_array.npy",
        power_array
    )
