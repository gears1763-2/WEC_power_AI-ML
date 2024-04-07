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
    A script for computing expected power output under the reduced dynamics.
"""


import math

import matplotlib.pyplot as plt

import numpy as np

import sklearn.manifold as skl_mf
import sklearn.preprocessing as skl_pp

import time

import wave_energy_converter as wec

import wave_utils as wu



#### ============================================================================== ####

def getReducedDynamicsPower(
    fundamental_period_s : float,
    significant_wave_height_m : float,
    wave_peak_period_s : float,
    sea_depth_m : float,
    float_inner_diameter_m : float,
    float_outer_diameter_m : float,
    float_resting_draft_m : float,
    power_takeoff_stiffness_Nm : float,
    power_takeoff_damping_Nsm: float
) -> float:
    """
    Function which computes the expected power output of a WEC under the reduced
    dynamics, and kernel machine, of the main report.
    
    Parameters
    ----------
    
    fundamental_period_s : float
        The fundamental (Fourier) period for modelling the WEC [s].
    
    significant_wave_height_m : float
        The significant wave height [m].
    
    wave_peak_period_s : float
        The wave spectral peak period [s].
    
    sea_depth_m : float
        The depth of the sea [m].
    
    float_inner_diameter_m : float
        The inner diameter of the cylindrical float [m].
    
    float_outer_diameter_m : float
        The outer diameter of the cylindrical float [m].
    
    float_resting_draft_m : float
        The resting draft of the cylindrical float [m].
    
    power_takeoff_stiffness_Nm : float
        The power takeoff stiffness [N/m].
    
    power_takeoff_damping_Nsm: float
        The power takeoff damping [N.s/m].
    
    Returns
    -------
    
    float
        The expected power output of the WEC under the reduced dynamics.
    """
    
    #   construct component frequency array
    n_components = 4 * math.ceil(fundamental_period_s / wave_peak_period_s)
            
    frequency_array_Hz = np.array(
        [n / fundamental_period_s for n in range(1, n_components + 1)]
    )
    
    
    #   generate random phase array
    random_phase_array = wu.getRandomPhaseArray(n_components)
    
    
    #   construct component wave number array
    component_wave_number_array_m = wu.getWaveNumberArray(
        frequency_array_Hz,
        sea_depth_m
    )
    
    
    #   construct component wave direction array
    component_wave_direction_array = np.random.normal(
        0,
        5 * (math.pi / 180),    # 95% of waves within +\- 10 deg of 000
        n_components
    )
    
    
    #   construct component wave amplitude array
    component_amplitude_array_m = wu.getComponentAmplitudeArray(
        frequency_array_Hz,
        significant_wave_height_m,
        wave_peak_period_s
    )
    
    
    #   get float mass and buoyancy stiffness
    float_mass_kg = wec.getFloatMass(
        float_inner_diameter_m,
        float_outer_diameter_m,
        float_resting_draft_m
    )
    
    buoyancy_stiffness_Nm = wec.getBuoyancyStiffness(
        float_inner_diameter_m,
        float_outer_diameter_m
    )
    
    
    #   generate alpha beta array
    alpha_beta_array_m = wec.getAlphaBetaArray(
        component_amplitude_array_m,
        component_wave_number_array_m,
        component_wave_direction_array,
        random_phase_array,
        float_inner_diameter_m,
        float_outer_diameter_m
    )
    
    
    #   generate A B array
    A_B_array_m = wec.getABArray(
        alpha_beta_array_m,
        float_mass_kg,
        power_takeoff_stiffness_Nm,
        power_takeoff_damping_Nsm,
        buoyancy_stiffness_Nm,
        fundamental_period_s
    )
    
    
    #   compute expected power output
    expected_WEC_power_kW = wec.getExpectedWECPower(
        A_B_array_m,
        power_takeoff_damping_Nsm,
        fundamental_period_s
    )
    
    return expected_WEC_power_kW

#### ============================================================================== ####



if __name__ == "__main__":
    """
    #   verify that using random phases does not result in wildy random expected power outputs!
    import random 
    
    fundamental_period_s = 900
    significant_wave_height_m = random.uniform(2, 3)
    wave_peak_period_s = random.uniform(9, 11)
    sea_depth_m = random.uniform(50, 100)
    float_inner_diameter_m = 6
    float_outer_diameter_m = random.uniform(20, 40)
    float_resting_draft_m = 2.5
    power_takeoff_stiffness_Nm = random.uniform(0, 1e3)
    power_takeoff_damping_Nsm = random.uniform(1e5, 1e8)
    
    n_samples = 100
    power_array = np.zeros(n_samples)
    
    for i in range(0, n_samples):
        power_kW = getReducedDynamicsPower(
            fundamental_period_s,
            significant_wave_height_m,
            wave_peak_period_s,
            sea_depth_m,
            float_inner_diameter_m,
            float_outer_diameter_m,
            float_resting_draft_m,
            power_takeoff_stiffness_Nm,
            power_takeoff_damping_Nsm
        )
        
        power_array[i] = power_kW
        
        print(power_kW)
    
    print()
    print("mean =", np.mean(power_array))
    print("std =", np.std(power_array))
    
    quit()
    """
    
    
    
    start_time = time.time()
    
    #   load features and target (clean)
    feature_array = np.load("data/feature_array.npy")
    dimensionless_feature_array = np.load("data/dimensionless_feature_array.npy")
    target_array = np.load("data/target_array.npy")
    
    size = feature_array.shape[0]
    
    
    
    #   load reduced target array, generate if can't
    try:
        reduced_target_array = np.load("data/reduced_target_array.npy")
    
    except:
        #   init reduced target array
        reduced_target_array = np.zeros(size)
        
        
        
        #   populate reduced target array
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
            
            
            
            reduced_target_array[i] = getReducedDynamicsPower(
                fundamental_period_s,
                significant_wave_height_m,
                wave_peak_period_s,
                sea_depth_m,
                float_inner_diameter_m,
                float_outer_diameter_m,
                float_resting_draft_m,
                power_takeoff_stiffness_Nm,
                power_takeoff_damping_Nsm
            )
            
            print(
                "Generating reduced dynamics power output",
                i + 1,
                "/",
                size,
                4 * " ",
                end="\r",
                flush=True
            )
        
        print()
        
        
        
        #   save reduced target array
        np.save(
            "data/reduced_target_array.npy",
            reduced_target_array
        )
    
    
    
    #   make superposition of target histograms
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.hist(
        target_array, 
        bins="scott",
        color="C0",
        alpha=0.5,
        zorder=2,
        label="from PDS sims"
    )
    plt.hist(
        reduced_target_array, 
        bins="scott",
        color="C3",
        alpha=0.5,
        zorder=3,
        label="from reduced dynamics"
    )
    plt.xlim(0, 350000)
    plt.xlabel("Expected Power Output [kW]")
    plt.ylabel("Count [ ]")
    plt.yscale("log")
    plt.legend()
    plt.savefig(
        "../LaTeX/images/perturbation_machine/target_histogram_superposition.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    plt.close()
    
    
    
    #   make histogram of differences
    target_differences_array = target_array - reduced_target_array
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.hist(
        target_differences_array,
        bins="scott",
        zorder=2
    )
    plt.xlim(-325000, 0)
    plt.xlabel(r"E$_{PDS}\{P\} - $E$_{reduced}\{P\}$ [kW]")
    plt.ylabel("Count [ ]")
    plt.yscale("log")
    plt.savefig(
        "../LaTeX/images/perturbation_machine/target_differences_histogram.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    plt.close()
    
    
    
    #   merge arrays
    merge_array = np.append(feature_array, dimensionless_feature_array, axis=1)
    merge_array = np.append(merge_array, target_differences_array.reshape(-1, 1), axis=1)
    
    
    
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
    idx_target_sort = np.argsort(target_differences_array)
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.scatter(
        merge_array_tSNE[idx_target_sort, 0],
        merge_array_tSNE[idx_target_sort, 1],
        s=12,
        c=target_differences_array[idx_target_sort],
        cmap="jet",
        alpha=0.75,
        zorder=2
    )
    plt.colorbar(
        label=r"E$_{PDS}\{P\} - $E$_{reduced}\{P\}$ [kW]"
    )
    plt.xlabel("t-SNE Component 0")
    plt.ylabel("t-SNE Component 1")
    plt.savefig(
        "../LaTeX/images/perturbation_machine/t-SNE_clustering_differences.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    plt.close()
    
    
    
    #   make histogram of ratios
    target_ratios_array = np.divide(target_array, reduced_target_array)
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.hist(
        target_ratios_array,
        bins="scott",
        zorder=2
    )
    plt.xlim(0, 10)
    plt.xlabel(r"E$_{PDS}\{P\}$ / E$_{reduced}\{P\}$ [ ]")
    plt.ylabel("Count [ ]")
    plt.yscale("log")
    plt.savefig(
        "../LaTeX/images/perturbation_machine/target_ratios_histogram.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    plt.close()
    
    
    
    #   save target ratios array
    np.save(
        "data/target_ratios_array.npy",
        target_ratios_array
    )
    
    
    
    #   merge arrays
    merge_array = np.append(feature_array, dimensionless_feature_array, axis=1)
    merge_array = np.append(merge_array, target_ratios_array.reshape(-1, 1), axis=1)
    
    
    
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
    idx_target_sort = np.argsort(target_ratios_array)
    
    plt.figure(figsize=(8, 6))
    plt.grid(color="C7", alpha=0.5, zorder=1)
    plt.scatter(
        merge_array_tSNE[idx_target_sort, 0],
        merge_array_tSNE[idx_target_sort, 1],
        s=12,
        c=target_ratios_array[idx_target_sort],
        cmap="jet",
        alpha=0.75,
        zorder=2
    )
    plt.colorbar(
        label=r"E$_{PDS}\{P\}$ / E$_{reduced}\{P\}$ [ ]"
    )
    plt.xlabel("t-SNE Component 0")
    plt.ylabel("t-SNE Component 1")
    plt.savefig(
        "../LaTeX/images/perturbation_machine/t-SNE_clustering_ratios.png",
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    
    plt.close()
    
    
    end_time = time.time()
    print("Time Elapsed:", end_time - start_time, "s")
