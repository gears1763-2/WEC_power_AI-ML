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
    A script for testing the various C extensions generated from files in pyx/.
"""


import math

import matplotlib.pyplot as plt

import numpy as np

import random

import wave_energy_converter as wec

import wave_utils as wu


if __name__ == "__main__":
    print("\n**** TESTING C EXTENSIONS ****")
    
    try:
        #### ======== test wave_utils ============================================= ####
        
        #   wave_utils 1: generate a test S(f) plot
        significant_wave_height_m = random.uniform(1, 3)
        wave_peak_period_s = random.uniform(8, 12)
        
        fundamental_period_s = 900
        
        n_components = 4 * math.ceil(fundamental_period_s / wave_peak_period_s)
        
        frequency_array_Hz = np.array(
            [n / fundamental_period_s for n in range(1, n_components + 1)]
        )
        
        S_array_m2Hz = np.zeros(len(frequency_array_Hz))
        
        for i in range(0, len(frequency_array_Hz)):
            S_array_m2Hz[i] = wu.getPiersonMoskowitzS(
                frequency_array_Hz[i],
                significant_wave_height_m,
                wave_peak_period_s
            )
        
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, zorder=1)
        plt.plot(
            frequency_array_Hz,
            S_array_m2Hz,
            zorder=2,
            label=r"continuous ($H_s = {}$ m, $T_p = {}$ s)".format(
                round(significant_wave_height_m, 2),
                round(wave_peak_period_s, 2)
            )
        )
        plt.xlim(frequency_array_Hz[0], frequency_array_Hz[-1])
        plt.xlabel(r"$f$ [Hz]")
        plt.ylim(0, 1.02 * np.max(S_array_m2Hz))
        plt.ylabel(r"$S(f)$ [m$^2$/Hz]")
        
        
        
        #   wave_utils 2: generate a test random phase array
        random_phase_array = wu.getRandomPhaseArray(n_components)
        
        assert(len(random_phase_array) == n_components)
        
        for i in range(0, n_components):
            assert(random_phase_array[i] <= math.pi)
            assert(random_phase_array[i] >= -1 * math.pi)
        
        
        
        #   wave_utils 3: generate a test wave number array
        sea_depth_m = random.uniform(20, 100)
        
        wave_number_array_m = wu.getWaveNumberArray(
            frequency_array_Hz,
            sea_depth_m
        )
        
        assert(len(wave_number_array_m) == n_components)
        
        for i in range(0, n_components):
            frequency_Hz = frequency_array_Hz[i]
            wave_number_m = wave_number_array_m[i]
            
            assert(
                abs(
                    math.pow(2 * math.pi * frequency_Hz, 2) - 
                    9.81 * wave_number_m * math.tanh(wave_number_m * sea_depth_m)
                ) <= 1e-4
            )
        
        
        
        #   wave_utils 4: generate a test frequency bounds array
        frequency_bounds_array_Hz = wu.getFrequencyBoundsArray(frequency_array_Hz)
        
        assert(len(frequency_bounds_array_Hz) == n_components + 1)
        
        
        
        #   wave_utils 5: generate a test component amplitude array, plot upon S(f)
        component_amplitude_array_m = wu.getComponentAmplitudeArray(
            frequency_array_Hz,
            significant_wave_height_m,
            wave_peak_period_s
        )
        
        assert(len(component_amplitude_array_m) == n_components)
        
        frequency_bounds_array_Hz = np.array(frequency_bounds_array_Hz)
        delta_frequency_array_Hz = np.diff(frequency_bounds_array_Hz)
        component_amplitude_array_m = np.array(component_amplitude_array_m)
        
        S_bar_array_m2Hz = np.zeros(n_components)
        
        for i in range(0, n_components):
            S_bar_array_m2Hz[i] = math.pow(component_amplitude_array_m[i], 2)
            S_bar_array_m2Hz[i] /= 2 * delta_frequency_array_Hz[i]
        
        plt.bar(
            frequency_bounds_array_Hz[0 : -1],
            S_bar_array_m2Hz,
            width=delta_frequency_array_Hz,
            align="edge",
            color="C1",
            alpha=0.5,
            zorder=3,
            label=r"discrete (binned, $N = {}$)".format(n_components)
        )
        
        
        
        #   wave_utils 6: generate test characteristic wave number
        weighted_average_wave_number_m = wu.getCharacteristicWaveNumber(
            frequency_array_Hz,
            wave_number_array_m,
            significant_wave_height_m,
            wave_peak_period_s,
            sea_depth_m
        )
        
        assert(weighted_average_wave_number_m > 0)
        assert(weighted_average_wave_number_m <= 1)
        
        #### ======== end test wave_utils ========================================= ####
        
        
        
        #### ======== test wave_energy_converter ================================== ####
        
        #   wave_energy_converter 1: test float mass function
        float_inner_diameter_m = 6
        float_outer_diameter_m = 30
        float_resting_draft_m = 2.5
        
        float_mass_kg = wec.getFloatMass(
            float_inner_diameter_m,
            float_outer_diameter_m,
            float_resting_draft_m
        )
        
        assert(abs(float_mass_kg - 1738871.53376195) <= 1e-4)
        
        
        
        #   wave_energy_converter 2: test buoyancy stiffness function
        buoyancy_stiffness_Nm = wec.getBuoyancyStiffness(
            float_inner_diameter_m,
            float_outer_diameter_m
        )
        
        assert(buoyancy_stiffness_Nm >= 0)
        
        
        
        #   wave_energy_converter 3: compute I_cos values
        float_outer_diameter_list_m = [
            17.73667733,
            42.66128779,
            49.35739945,
            17.3639375,
            49.74201534
        ]
        
        wave_number_list_m = [
            0.04627764488,
            0.1303163431,
            0.01495962477,
            0.1109090815,
            0.05898573461
        ]
        
        wave_direction_list = [
            4.871182714,
            3.956143591,
            1.906647693,
            4.933701253,
            0.5193229516
        ]
        
        expected_Icos_list = [
            213.7062906,
            400.4859664,
            1852.669425,
            182.5124656,
            1437.166192
        ]
        
        expected_Icos_list = [
            (4.0 / (
                math.pi * (
                    math.pow(float_outer_diameter_list_m[i], 2) -
                    math.pow(float_inner_diameter_m, 2)
                )
            )) * expected_Icos_list[i] for i in range(0, len(expected_Icos_list))
        ]
        
        
        for i in range(0, len(expected_Icos_list)):
            I_cos = wec.getIcos(
                float_inner_diameter_m,
                float_outer_diameter_list_m[i],
                wave_number_list_m[i],
                wave_direction_list[i]
            )
            
            assert(abs(I_cos - expected_Icos_list[i]) <= 1e-4)
        
        
        
        #   wave_energy_converter 4: generate a test alpha beta array
        component_wave_number_array_m = wu.getWaveNumberArray(
            frequency_array_Hz,
            sea_depth_m
        )
        
        component_wave_direction_array = np.random.normal(
            0,
            5 * (math.pi / 180),    # 95% of waves within +\- 10 deg of 000
            len(frequency_array_Hz)
        )
        
        alpha_beta_array_m = wec.getAlphaBetaArray(
            component_amplitude_array_m,
            component_wave_number_array_m,
            component_wave_direction_array,
            random_phase_array,
            float_inner_diameter_m,
            float_outer_diameter_m
        )
        
        alpha_beta_array_m = np.array(alpha_beta_array_m)
        
        assert(alpha_beta_array_m.shape[0] == n_components)
        assert(alpha_beta_array_m.shape[1] == 2)
        
        
        
        #   wave_energy_converter 5: generate a test A B array
        power_takeoff_stiffness_Nm = 100
        power_takeoff_damping_Nsm = int(1e5)
        
        A_B_array_m = wec.getABArray(
            alpha_beta_array_m,
            float_mass_kg,
            power_takeoff_stiffness_Nm,
            power_takeoff_damping_Nsm,
            buoyancy_stiffness_Nm,
            fundamental_period_s
        )
        
        A_B_array_m = np.array(A_B_array_m)
        
        assert(A_B_array_m.shape[0] == n_components)
        assert(A_B_array_m.shape[1] == 2)
        
        
        
        #   wave_energy_converter 6: compute expected WEC power
        expected_WEC_power_kW = wec.getExpectedWECPower(
            A_B_array_m,
            power_takeoff_damping_Nsm,
            fundamental_period_s
        )
        
        assert(expected_WEC_power_kW >= 0)
        assert(expected_WEC_power_kW <= 100)
        
        print("\n\tSig. Wave Height =", round(significant_wave_height_m, 2), "m")
        print("\tWave Peak Period =", round(wave_peak_period_s, 2), "s")
        print("\tCharacteristic Wave Number =", round(weighted_average_wave_number_m, 5), "1/m")
        print("\tCharacteristic Wave Length =", round((2 * math.pi) / weighted_average_wave_number_m, 2), "m")
        print("\tFloat Diameter =", float_outer_diameter_m, "m")
        print("\tPTO Stiffness =", power_takeoff_stiffness_Nm, "N/m")
        print("\tPTO damping =", power_takeoff_damping_Nsm, "N.s/m")
        print("\tExpected WEC power =", round(expected_WEC_power_kW, 2), "kW")
        
        #### ======== end test wave_energy_converter ============================== ####
    
    
    
    except:
        print("\n**** TESTING INCOMPLETE: A TEST FAILED ****\n")
        raise
    
    
    
    else:
        print("\n**** TESTING COMPLETE: ALL PASSED ****\n")
        plt.legend()
        plt.show()
