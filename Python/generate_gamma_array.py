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
    A script for generating the gamma array (for use as ML target).
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
    #   load feature and power arrays
    feature_array = np.load("data/feature_array.npy")
    power_array = np.load("data/power_array.npy")
    
    size = feature_array.shape[0]
    
    
    
    #   generate optimal gamma array for given target array (or load if already done)
    try:
        gamma_array = np.load("data/gamma_array.npy")
    
    except:
        gamma_array = np.zeros(size)
        
        float_inner_diameter_m = 6
        float_resting_draft_m = 2.5

        fluid_density_kgm3 = 1025
        gravity_ms2 = 9.81

        fundamental_period_s = 900
        
        opt_lower_bound = 0
        opt_upper_bound = np.inf    # numpy float infinity
        
        gamma_incumbent = 0
        
        WEC_power_error_kW = 0
        error_threshold = 0.10
        
        do_over_count = 0
        sentinel_count = 0
        
        
        print()
        for i in range(0, size):
            #   extract features and target
            sea_depth_m = feature_array[i, 0]
            significant_wave_height_m = feature_array[i, 1]
            wave_peak_period_s = feature_array[i, 2]
            float_outer_diameter_m = feature_array[i, 3]
            power_takeoff_stiffness_Nm = feature_array[i, 4]
            power_takeoff_damping_Nsm = math.pow(10, feature_array[i, 5])
            
            WEC_expected_power_true_kW = power_array[i]
            
            
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
            if WEC_power_error_kW / WEC_expected_power_true_kW > error_threshold:
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
            #   (perturbation evidently not working well here ...)
            if WEC_power_error_kW / WEC_expected_power_true_kW > error_threshold:
                sentinel_count += 1
                gamma_array[i] = -1
            
            
            #   running print
            print(
                "Row", i + 1, "/", size,
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
            "data/gamma_array.npy",
            gamma_array
        )
    
    
    
    #   trim sentinels and save
    idx_not_sentinel = gamma_array >= 0
    
    feature_array_no_sentinel = feature_array[idx_not_sentinel, :]
    power_array_no_sentinel = power_array[idx_not_sentinel]
    gamma_array_no_sentinel = gamma_array[idx_not_sentinel]
    
    size_after = feature_array_no_sentinel.shape[0]
    
    print(
        "\nrows with sentinels dropped,",
        round(100 * (size_after / size), 2),
        "% of the data was retained"
    )
    
    np.save(
        "data/feature_array_no_sentinel.npy",
        feature_array_no_sentinel
    )
    
    np.save(
        "data/power_array_no_sentinel.npy",
        power_array_no_sentinel
    )
    
    np.save(
        "data/gamma_array_no_sentinel.npy",
        gamma_array_no_sentinel
    )
    
    
    
    #   get gamma sorting (ascending) indices, for the sake of plotting
    idx_sort = np.argsort(gamma_array_no_sentinel)
    
    
    
    #   make some scatter plots
    print("\nMaking some initial scatter plots ... ", end="", flush=True)
    
    WEC_dataframe = pd.read_csv("data/data_table.csv")
    column_header_list = list(WEC_dataframe)[1 : 7]
    
    column_header_list[5] = r"$\log_{10}($" + column_header_list[5] + r"$)$"
    
    dimensionless_feature_column_list = [
        r"$\Pi_1 = \log_{10}\left(\overline{k}^2(D^2 - d^2)\right)$",
        r"$\Pi_2 = \log_{10}\left(\frac{b}{\rho g H_s^2T_p}\right)$",
        r"$\Pi_3 = \log_{10}\left(\frac{b}{2\sqrt{(k + k_D)m}}\right)$",
        r"$\Pi_4 = T_p\sqrt{\frac{k + k_D}{m}}$"
    ]
    
    feature_column_list = column_header_list + dimensionless_feature_column_list
    
    for i in range(0, len(feature_column_list)):
        for j in range(i + 1, len(feature_column_list)):
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, zorder=1)
            plt.scatter(
                feature_array_no_sentinel[idx_sort, i],
                feature_array_no_sentinel[idx_sort, j],
                s=16,
                c=gamma_array_no_sentinel[idx_sort],
                cmap="jet",
                zorder=3
            )
            plt.colorbar(
                label=r"$\gamma$ [ ]"
            )
            plt.xlabel(feature_column_list[i])
            plt.ylabel(feature_column_list[j])
            plt.savefig(
                "../LaTeX/images/regressor/gamma_scatter_{}_{}.png".format(i, j),
                format="png",
                dpi=128,
                bbox_inches="tight"
            )
            
            plt.close()
    
    print("DONE (saved to ../LaTeX/images/mining/)")
