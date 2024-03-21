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
    A script for mining the WEC data set for some initial insight.
"""


import math

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as nptype

import pandas as pd

import scipy.optimize as spo

import wave_utils as wu

import wave_energy_converter as wec



#### ============================================================================== ####

def gammaObjective(
    gamma : float,
    component_amplitude_array_m : nptype.ArrayLike,
    component_wave_number_array_m : nptype.ArrayLike,
    component_phase_array : nptype.ArrayLike,
    float_inner_diameter_m : float,
    float_outer_diameter_m : float,
    float_mass_kg : float,
    power_takeoff_stiffness_Nm : float,
    power_takeoff_damping_Nsm : float,
    buoyancy_stiffness_Nm : float,
    fundamental_period_s : float,
    WEC_expected_power_true_kW : float
) -> float:
    """
    Function which takes in gamma and case-specific sea state and WEC parameters, and 
    then computes and returns the squared error between the predicted (theory) and 
    modelled (numerical) expected power outputs.
    
    Parameters
    ----------
    
    gamma : float
        The gamma value of eq'n (2.24) of the main report [ ].
    
    component_amplitude_array_m : nptype.ArrayLike
        The component amplitude array [m] for modelling the sea surface.
    
    component_wave_number_array_m : nptype.ArrayLike
        The component wave number array [1/m] for modelling the sea surface.
    
    component_phase_array : nptype.ArrayLike
        The component phase array [ ] for modelling the sea surface.
    
    float_inner_diameter_m : float
        The inner diameter of the cylindrical float [m].
    
    float_outer_diameter_m : float
        The outer diameter of the cylindrical float [m].
    
    float_mass_kg : float
        The mass of the cylindrical float [kg].
    
    power_takeoff_stiffness_Nm : float
        The stiffness of the power takeoff assembly [N/m].
    
    power_takeoff_damping_Nsm : float
        The damping of the power takeoff assembly [N.s/m]
    
    buoyancy_stiffness_Nm : float
        The buoyancy stiffness of the float [N/m].
    
    fundamental_period_s : float
        The fundamental modelling period for the WEC [s].
    
    WEC_expected_power_true_kW : float
        The true (modelled, numerical) expected power output of the WEC [kW].
    
    Returns
    -------
    
    float
        The squared error between the predicted (theory) and modelled (numerical)
        expected power outputs [kW^2].
    """
    
    #   unpack gamma
    gamma = gamma[0]
    
    
    #   compute alpha beta array
    alpha_beta_array_m = wec.getAlphaBetaArray(
        component_amplitude_array_m,
        component_wave_number_array_m,
        component_phase_array,
        float_inner_diameter_m,
        float_outer_diameter_m,
        gamma
    )
    
    
    #   compute A B array
    A_B_array_m = wec.getABArray(
        alpha_beta_array_m,
        float_mass_kg,
        power_takeoff_stiffness_Nm,
        power_takeoff_damping_Nsm,
        buoyancy_stiffness_Nm,
        fundamental_period_s
    )
    
    
    #   get expected WEC power (predicted)
    WEC_expected_power_predicted_kW = wec.getExpectedWECPower(
        A_B_array_m,
        power_takeoff_damping_Nsm,
        fundamental_period_s
    )
    
    
    #   compute and return objective
    objective = math.pow(
        WEC_expected_power_predicted_kW - WEC_expected_power_true_kW, 2
    )
    
    return objective

#### ============================================================================== ####



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
    
    
    
    #   save feature array for later use
    np.save(
        "data/input_gamma_regression.npy",
        feature_array
    )
    
    
    
    #   generate optimal gamma array for given target array
    data_size = len(target_array)
    gamma_array = np.zeros(data_size)
    
    float_inner_diameter_m = 6
    float_resting_draft_m = 2.5

    fluid_density_kgm3 = 1025
    gravity_ms2 = 9.81

    fundamental_period_s = 900
    
    opt_lower_bound = 0
    opt_upper_bound = np.inf    # numpy float infinity
    
    gamma_incumbent = 0
    
    WEC_power_error_kW = 0
    objective_threshold = 0.10
    
    do_over_count = 0
    sentinel_count = 0
    
    
    print()
    for i in range(0, data_size):
        #   extract features and target
        sea_depth_m = feature_array[i, 0]
        significant_wave_height_m = feature_array[i, 1]
        wave_peak_period_s = feature_array[i, 2]
        float_outer_diameter_m = feature_array[i, 3]
        power_takeoff_stiffness_Nm = feature_array[i, 4]
        power_takeoff_damping_Nsm = feature_array[i, 5]
        
        WEC_expected_power_true_kW = target_array[i][0]
        
        
        #   model sea state (Pierson-Moskowitz)
        n_components = 4 * math.ceil(fundamental_period_s / wave_peak_period_s)

        frequency_array_Hz = np.array(
            [n / fundamental_period_s for n in range(1, n_components + 1)]
        )
        
        component_phase_array = wu.getRandomPhaseArray(len(frequency_array_Hz))
        
        component_wave_number_array_m = wu.getWaveNumberArray(
            frequency_array_Hz,
            sea_depth_m
        )
        
        component_amplitude_array_m = wu.getComponentAmplitudeArray(
            frequency_array_Hz,
            significant_wave_height_m,
            wave_peak_period_s
        )
        
        
        #   model WEC components
        float_mass_kg = wec.getFloatMass(
            float_inner_diameter_m,
            float_outer_diameter_m,
            float_resting_draft_m
        )
        
        buoyancy_stiffness_Nm = wec.getBuoyancyStiffness(
            float_inner_diameter_m,
            float_outer_diameter_m
        )
        
        
        #   seek optimal gamma value
        opt_res = spo.minimize(
            gammaObjective,
            x0=gamma_incumbent,
            args=(
                component_amplitude_array_m,
                component_wave_number_array_m,
                component_phase_array,
                float_inner_diameter_m,
                float_outer_diameter_m,
                float_mass_kg,
                power_takeoff_stiffness_Nm,
                power_takeoff_damping_Nsm,
                buoyancy_stiffness_Nm,
                fundamental_period_s,
                WEC_expected_power_true_kW
            ),
            bounds=[(opt_lower_bound, opt_upper_bound)]
        )
        
        
        #   extract gamma, objective, update incumbent
        gamma = opt_res.x[0]
        objective = opt_res.fun
        
        WEC_power_error_kW = math.sqrt(objective)
        
        gamma_incumbent = gamma
        
        #   reset incumbent and do over if excessively high error
        if WEC_power_error_kW / WEC_expected_power_true_kW > objective_threshold:
            do_over_count += 1
            
            gamma_incumbent = 0
            
            opt_res = spo.minimize(
                gammaObjective,
                x0=gamma_incumbent,
                args=(
                    component_amplitude_array_m,
                    component_wave_number_array_m,
                    component_phase_array,
                    float_inner_diameter_m,
                    float_outer_diameter_m,
                    float_mass_kg,
                    power_takeoff_stiffness_Nm,
                    power_takeoff_damping_Nsm,
                    buoyancy_stiffness_Nm,
                    fundamental_period_s,
                    WEC_expected_power_true_kW
                ),
                bounds=[(opt_lower_bound, opt_upper_bound)]
            )
            
            
            #   extract gamma, objective, update incumbent
            gamma = opt_res.x[0]
            objective = opt_res.fun
            
            WEC_power_error_kW = math.sqrt(objective)
            
            gamma_incumbent = gamma
        
        
        #   write results to gamma array
        gamma_array[i] = gamma
        
        
        #   overwrite with sentinel value if error is still excessively high 
        # (perturbation evidently not working well here ...)
        if WEC_power_error_kW / WEC_expected_power_true_kW > objective_threshold:
            sentinel_count += 1
            gamma_array[i] = -1
        
        
        #   running print
        print(
            "Row", i + 1, "/", data_size,
            "gamma:", round(gamma, 5),
            "power error: +\-", round(WEC_power_error_kW, 5), "kW",
            "sentinel count:", sentinel_count,
            "(", round(100 * (sentinel_count / (i + 1))), "% of rows)",
            8 * " ",
            end="\r",
            flush=True
        )
    
    
    #   save results
    np.save(
        "data/target_gamma_regression.npy",
        gamma_array
    )
