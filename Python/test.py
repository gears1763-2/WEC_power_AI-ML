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
    A script for testing the various C extensions.
"""


import math
import matplotlib.pyplot as plt
import numpy as np
import random

import wave_utils as wu


if __name__ == "__main__":
    print("\n**** TESTING C EXTENSIONS ****")
    
    try:
        #   wave_utils 1: generate a test S(f) plot
        significant_wave_height_m = 3
        wave_peak_period_s = 10
        wave_peak_frequency_Hz = 1 / wave_peak_period_s
        
        n_components = random.randint(500, 1000)
        
        frequency_array_Hz = np.linspace(
            0.5 * wave_peak_frequency_Hz,
            4 * wave_peak_frequency_Hz,
            n_components
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
            label="continuous"
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
            label="discrete (binned)"
        )
    
    
    
    except:
        print("\n**** TESTING INCOMPLETE: A TEST FAILED ****\n")
        raise
    
    
    
    else:
        print("\n**** TESTING COMPLETE: ALL PASSED ****\n")
        plt.legend()
        plt.show()
