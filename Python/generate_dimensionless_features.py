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
    A script for generating some additional dimensionless features from the simulation
    features.
"""


import math

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as nptype

import sklearn.manifold as skl_mf
import sklearn.preprocessing as skl_pp

import wave_energy_converter as wec

import wave_utils as wu



#### ============================================================================== ####

def getPi0(
    frequency_array_Hz : nptype.ArrayLike,
    wave_number_array_m : nptype.ArrayLike,
    significant_wave_height_m : float,
    wave_peak_period_s : float,
    sea_depth_m : float,
    float_outer_diameter_m : float,
    float_inner_diameter_m : float
) -> float:
    """
    A function which computes and returns the first dimensionless feature.
    
    Parameters
    ----------
    
    frequency_array_Hz : nptype.ArrayLike
        An array of wave component frequencies [Hz].
    
    wave_number_array_m : nptype.ArrayLike
        An array of component wave numbers [1/m].
    
    significant_wave_height_m : float
        The significant wave height [m].
    
    wave_peak_period_s : float
        The wave spectral peak period [s].
    
    sea_depth_m : float
        The depth of the sea [m].
    
    float_outer_diameter_m : float
        The outer diameter of the cylindrical float [m].
    
    float_inner_diameter_m : float
        The inner diameter of the cylindrical float [m].
    
    Returns
    -------
    
    float
        The first dimensionless feature, as describe in eq'n (4.14) of the main report.
    """
    
    #   compute characteristic wave number
    characteristic_wave_number_m = wu.getCharacteristicWaveNumber(
        frequency_array_Hz,
        wave_number_array_m,
        significant_wave_height_m,
        wave_peak_period_s,
        sea_depth_m
    )
    
    
    #   compute Pi_0
    Pi_0 = math.pow(characteristic_wave_number_m, 2)
    Pi_0 *= math.pow(float_outer_diameter_m, 2) - math.pow(float_inner_diameter_m, 2)
    
    return Pi_0

#### ============================================================================== ####



#### ============================================================================== ####

def getPi1(
    float_mass_kg : float,
    power_takeoff_stiffness_Nm : float,
    buoyancy_stiffness_Nm : float,
    power_takeoff_damping_Nsm : float
) -> float:
    """
    A function which computes and returns the second dimensionless feature.
    
    Parameters
    ----------
    
    float_mass_kg : float
        The mass of the cylindrical float [kg].
    
    power_takeoff_stiffness_Nm : float
        The stiffness of the power takeoff [N/m].
    
    buoyancy_stiffness_Nm : float
        The buoyancy stiffness of the cylindrical float [N/m].
    
    power_takeoff_damping_Nsm : float
        The damping of the power takeoff [N.s/m]
    
    Returns
    -------
    
    float
        The second dimensionless feature, as defined in eq'n (4.15) of the main report.
    """
    
    Pi_1 = power_takeoff_damping_Nsm
    Pi_1 /= 2 * math.sqrt(
        float_mass_kg * (
            power_takeoff_stiffness_Nm + buoyancy_stiffness_Nm
        )
    )
    
    return Pi_1

#### ============================================================================== ####



#### ============================================================================== ####

def getPi2(
    Pi_1 : float,
    float_mass_kg : float,
    power_takeoff_stiffness_Nm : float,
    buoyancy_stiffness_Nm : float,
    wave_peak_period_s : float
) -> float:
    """
    A function which computes and returns the third dimensionless feature.
    
    Parameters
    ----------
    
    Pi_1 : float
        The second dimensionless term.
    
    float_mass_kg : float
        The mass of the cylindrical float [kg].
    
    power_takeoff_stiffness_Nm : float
        The stiffness of the power takeoff [N/m].
    
    buoyancy_stiffness_Nm : float
        The buoyancy stiffness of the cylindrical float [N/m].
    
    wave_peak_period_s : float
        The wave spectral peak period [s].
    
    Returns
    -------
    
    float
        The third dimensionless feature, as defined in eq'n (4.17) of the main report.
    """
    
    #   compute damped natural frequency
    damped_natural_frequency_Hz = 1 / (2 * math.pi)
    damped_natural_frequency_Hz *= math.sqrt(
        (power_takeoff_stiffness_Nm + buoyancy_stiffness_Nm) /
        float_mass_kg
    )
    
    if (abs(Pi_1) < 1):
        damped_natural_frequency_Hz *= math.sqrt(1 - math.pow(Pi_1, 2))
    
    
    #   compute Pi_2
    Pi_2 = wave_peak_period_s * damped_natural_frequency_Hz
    
    return Pi_2

#### ============================================================================== ####



#### ============================================================================== ####

def getPi3(
    power_takeoff_damping_Nsm : float,
    significant_wave_height_m : float,
    wave_peak_period_s : float,
    fluid_density_kgm3 : float = 1025,
    gravity_ms2 : float = 9.81
) -> float:
    """
    A function which computes and returns the fourth dimensionless feature.
    
    Parameters
    ----------
    
    power_takeoff_damping_Nsm : float
        The damping of the power takeoff [N.s/m]
    
    significant_wave_height_m : float
        The significant wave height [m].
    
    wave_peak_period_s : float
        The wave spectral peak period [s].
    
    fluid_density_kgm3 : float = 1025
        The density of the fluid [kg/m^3]. Defaults to 1025.
    
    gravity_ms2 : float = 9.81
        Acceleration due to gravity [m/s^2]. Defaults to 9.81.
    
    Returns
    -------
    
    float
        The fourth dimensionless feature, as defined in eq'n (4.18) of the main report.
    """
    
    Pi_3 = power_takeoff_damping_Nsm
    Pi_3 /= (
        fluid_density_kgm3 * gravity_ms2 *
        math.pow(significant_wave_height_m, 2) * wave_peak_period_s
    )
    
    return Pi_3

#### ============================================================================== ####



if __name__ == "__main__":
    #   load features and target (clean)
    feature_array = np.load("data/feature_array.npy")
    target_array = np.load("data/target_array.npy")
    
    size = feature_array.shape[0]
    
    
    
    #   load dimensionless feature array, build if can't
    dimensionless_feature_list = [
        r"$\Pi_0$ [ ]",
        r"$\Pi_1$ [ ]",
        r"$\Pi_2$ [ ]",
        r"$\Pi_3$ [ ]"
    ]
    
    try:
        dimensionless_feature_array = np.load("data/dimensionless_feature_array.npy")
    
    except:
        #   init dimensionless feature array
        dimensionless_feature_array = np.zeros((size, 4))
        
        
        
        #   populate dimensionless feature array
        float_inner_diameter_m = 6
        float_resting_draft_m = 2.5
        
        fundamental_period_s = 900
        
        for i in range(0, size):
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
            
            float_mass_kg = wec.getFloatMass(
                float_inner_diameter_m,
                float_outer_diameter_m,
                float_resting_draft_m
            )
            
            buoyancy_stiffness_Nm = wec.getBuoyancyStiffness(
                float_inner_diameter_m,
                float_outer_diameter_m
            )
            
            dimensionless_feature_array[i, 0] = getPi0(
                frequency_array_Hz,
                wave_number_array_m,
                significant_wave_height_m,
                wave_peak_period_s,
                sea_depth_m,
                float_outer_diameter_m,
                float_inner_diameter_m
            )
            
            dimensionless_feature_array[i, 1] = getPi1(
                float_mass_kg,
                power_takeoff_stiffness_Nm,
                buoyancy_stiffness_Nm,
                power_takeoff_damping_Nsm
            )
            
            dimensionless_feature_array[i, 2] = getPi2(
                dimensionless_feature_array[i, 1],
                float_mass_kg,
                power_takeoff_stiffness_Nm,
                buoyancy_stiffness_Nm,
                wave_peak_period_s
            )
            
            dimensionless_feature_array[i, 3] = getPi3(
                power_takeoff_damping_Nsm,
                significant_wave_height_m,
                wave_peak_period_s
            )
            
            print(
                "Generating dimensionless features",
                i + 1,
                "/",
                size,
                4 * " ",
                end="\r",
                flush=True
            )
        
        print()
        
        
        
        #   save dimensionless feature array
        np.save(
            "data/dimensionless_feature_array.npy",
            dimensionless_feature_array
        )
    
    
    
    #   make some dimensionless feature scatter plots
    idx_target_sort = np.argsort(target_array)
    
    for i in range(0, len(dimensionless_feature_list)):
        for j in range(i + 1, len(dimensionless_feature_list)):
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, zorder=1)
            plt.scatter(
                dimensionless_feature_array[idx_target_sort, i],
                dimensionless_feature_array[idx_target_sort, j],
                s=12,
                c=target_array[idx_target_sort],
                cmap="jet",
                alpha=0.75,
                zorder=2
            )
            plt.colorbar(
                label="Expected Power Output [kW]"
            )
            plt.xlabel(dimensionless_feature_list[i])
            if i != 2:
                plt.xscale("log")
            plt.ylabel(dimensionless_feature_list[j])
            if j != 2:
                plt.yscale("log")
            plt.savefig(
                "../LaTeX/images/PDS_results_mining/dimensionless_mining_{}_{}.png".format(i, j),
                format="png",
                dpi=128,
                bbox_inches="tight"
            )
            plt.close()
    
    
    
    #   merge arrays
    merge_array = np.append(feature_array, dimensionless_feature_array, axis=1)
    merge_array = np.append(merge_array, target_array.reshape(-1, 1), axis=1)
    
    
    
    #   log10 scale select columns (namely b, Pi_0, Pi_1, and Pi_3)
    log10_scale_idx = [
        5,  # b
        6,  # Pi_0
        7,  # Pi_1
        9   # Pi_3
    ]
    
    for i in log10_scale_idx:
        merge_array[:, i] = np.log10(merge_array[:, i])
    
    
    
    #   normalize merge array
    min_max_scaler = skl_pp.MinMaxScaler()
    min_max_scaler.fit(merge_array)
    
    merge_array_norm = min_max_scaler.transform(merge_array)
    
    
    
    #   get t-SNE projection (2D) of merge array
    tSNE = skl_mf.TSNE(
        n_components=2,
        verbose=1
    )
    
    merge_array_tSNE = tSNE.fit_transform(merge_array_norm)
    
    
    
    #   make t-SNE plot
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.scatter(
        merge_array_tSNE[idx_target_sort, 0],
        merge_array_tSNE[idx_target_sort, 1],
        s=12,
        c=target_array[idx_target_sort],
        cmap="jet",
        alpha=0.75,
        zorder=2
    )
    plt.colorbar(
        label="Expected Power Output [kW]"
    )
    plt.xlabel("t-SNE Component 0")
    plt.ylabel("t-SNE Component 1")
    plt.savefig(
        "../LaTeX/images/PDS_results_mining/t-SNE_clustering.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    plt.close()
