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
    A collection of utility functions for modelling an irregular sea surface. Assumes 
    a Pierson-Moskowitz sea.
"""


import cython

import math

import numpy as np


#### ============================================================================== ####

cpdef double getPiersonMoskowitzS(
    double frequency_Hz,
    double significant_wave_height_m,
    double wave_peak_period_s
):
    """
    Function which computes and returns the variance density S [m2/hZ] about the given 
    frequency [Hz] for a Pierson-Moskowitz sea with the given significant wave height 
    [m] and wave peak period [s].
    
    Parameters
    ----------
    
    double frequency_Hz
        The frequency [Hz] about which the variance density [m2/Hz] is desired.
    
    double significant_wave_height_m
        The significant wave height [m] of the sea.
    
    double wave_peak_period_s
        The wave peak period [s] of the sea.
    
    Returns
    -------
    
    double
        The variance density [m2/Hz] about the given frequency [Hz].
    """
    
    cdef double S_m2Hz = (5.0 / 16.0) * math.pow(significant_wave_height_m, 2)
    S_m2Hz *= math.pow(wave_peak_period_s, -4) * math.pow(frequency_Hz, -5)
    S_m2Hz *= math.exp((-5.0 / 4.0) * math.pow(1.0 / (wave_peak_period_s * frequency_Hz), 4))
    
    return S_m2Hz

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double[:] getRandomPhaseArray(int n_components):
    """
    Function which generates and returns an array of N > 0 random phases, with each 
    phase uniformly distributed over the closed interval [-pi, pi].
    
    Parameters
    ----------
    
    int n_components
        The number of components in the array.
    
    Returns
    -------
    
    double[:]
        An array of random phases, with each phase uniformly distributed over the closed
        interval [-pi, pi].
    """
    
    cdef double[:] random_phase_array = 2 * math.pi * np.random.rand(n_components) - math.pi
    
    return random_phase_array

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double dispersionRelationObjective(
    double wave_number_m,
    double frequency_Hz,
    double sea_depth_m,
    double gravity_ms2 = 9.81
):
    """
    Function which acts as the objective function in seeking wave numbers [1/m] 
    corresponding to the given frequency [Hz] and sea depth [m] under the Airy
    dispersion relation.
    
    Parameters
    ----------
    
    double wave_number_m
        The candidate wave number [1/m].
    
    double frequency_Hz
        The given frequency [Hz].
    
    double sea_depth_m
        The given sea depth [m].
    
    double gravity_ms2
        Acceleration due to gravity [m/s2]. Defaults to 9.81.
    
    Returns
    -------
    
    double
        An error metric, to be minimized in mathching wave numbers with frequencies.
    """

    cdef double left_hand_side = math.pow(2 * math.pi * frequency_Hz, 2)
    cdef double right_hand_side = gravity_ms2 * wave_number_m * math.tanh(
        wave_number_m * sea_depth_m
    )
    
    return left_hand_side - right_hand_side

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double getWaveNumber(
    double frequency_Hz,
    double sea_depth_m,
    double gravity_ms2 = 9.81
):
    """
    Function which probes for suitable Airy wave number [1/m] using a very simple
    higher/lower approach. Initial guess is the deep water wave number.
    
    Parameters
    ----------
    
    double frequency_Hz
        The given frequency [Hz].
    
    double sea_depth_m
        The given sea depth [m].
    
    double gravity_ms2
        Acceleration due to gravity [m/s2]. Defaults to 9.81.
    
    Returns
    -------
    
    double
        Suitable wave number [1/m].
    """
    
    cdef:
        double wave_number_m = math.pow(2 * math.pi * frequency_Hz, 2) / gravity_ms2
        double step_ratio = 0.05
        double error = dispersionRelationObjective(
            wave_number_m,
            frequency_Hz,
            sea_depth_m,
            gravity_ms2
        )
    
    while abs(error) > 1e-4:
        wave_number_m += step_ratio * error
        
        error = dispersionRelationObjective(
            wave_number_m,
            frequency_Hz,
            sea_depth_m,
            gravity_ms2
        )
    
    return wave_number_m

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double[:] getWaveNumberArray(
    double[:] frequency_array_Hz,
    double sea_depth_m,
    double gravity_ms2 = 9.81
):
    """
    Function which takes in an array of frequencies [Hz] and a given sea depth [m], and 
    then generates the corresponding array of Airy (i.e. linear) wave numbers [1/m] by 
    way of the dispersion relation.
    
    Parameters
    ----------
    
    double[:] frequency_array_Hz
        An array of component frequencies [Hz].
    
    double sea_depth_m
        The sea depth [m].
    
    double gravity_ms2
        Acceleration due to gravity [m/s2]. Defaults to 9.81.
    
    Returns
    -------
    
    double[:]
        An array of Airy wave numbers [1/m].
    """
    
    cdef:
        int i = 0
        int N = len(frequency_array_Hz)
        double[:] wave_number_array_m = np.zeros(N)
    
    while i < N:
        wave_number_array_m[i] = getWaveNumber(
            frequency_array_Hz[i],
            sea_depth_m,
            gravity_ms2
        )
        
        i += 1
    
    return wave_number_array_m

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double[:] getFrequencyBoundsArray(double[:] frequency_array_Hz):
    """
    Function which generates a frequency bounds array [Hz] from the given frequency 
    array [Hz]. This defines the "binning".
    
    Parameters
    ----------
    
    double[:] frequency_array_Hz
        An array of component frequencies [Hz].
    
    Returns
    -------
    
    double[:]
        A frequency bounds array [Hz]
    """
    
    cdef:
        int i = 1
        int N = len(frequency_array_Hz)
        double[:] frequency_bounds_array_Hz = np.zeros(N + 1)
    
    while i < N:
        frequency_bounds_array_Hz[i] = 0.5 * (
            frequency_array_Hz[i - 1] + 
            frequency_array_Hz[i]
        )
        
        i += 1
    
    cdef double lower_delta_Hz = frequency_bounds_array_Hz[1] - frequency_array_Hz[0]
    frequency_bounds_array_Hz[0] = frequency_array_Hz[0] - lower_delta_Hz
    
    cdef double upper_delta_Hz = frequency_array_Hz[N - 1] - frequency_bounds_array_Hz[N - 1]
    frequency_bounds_array_Hz[N] = frequency_array_Hz[N - 1] + upper_delta_Hz
    
    return frequency_bounds_array_Hz

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double getVariancePreservingPiersonMoskowitzS(
    double lower_frequency_Hz,
    double upper_frequency_Hz,
    double significant_wave_height_m,
    double wave_peak_period_s
):
    """
    Function which computes and returns the single variance preserving S value [m2/Hz],
    between the given lower and upper frequency bounds [Hz], assuming a Pierson-Moskowitz
    sea with the given state parameters significant wave height [m] and wave peak period 
    [s]. This is simply the average S value between the bounds, obtained by way of a 
    discrete mean.
    
    Parameters
    ----------
    
    double lower_frequency_Hz
        The lower frequency bound [Hz].
    
    double upper_frequency_Hz
        The upper frequency bound [Hz].
    
    double significant_wave_height_m
        The significant wave height [m] of the sea.
    
    double wave_peak_period_s
        The wave peak period [s] of the sea.
    
    Returns
    -------
    
    double
        The single variance preserving S value [m2/Hz] between the given lower and upper 
        frequency bounds [Hz].
    """
    
    cdef:
        int i = 0
        int span = 100
        double[:] frequency_array_Hz = np.linspace(
            lower_frequency_Hz,
            upper_frequency_Hz,
            span
        )
        double S_sum_m2Hz = 0
    
    while i < span:
        S_sum_m2Hz += getPiersonMoskowitzS(
            frequency_array_Hz[i],
            significant_wave_height_m,
            wave_peak_period_s
        )
        
        i += 1
    
    return S_sum_m2Hz / span

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double[:] getComponentAmplitudeArray(
    double[:] frequency_array_Hz,
    double significant_wave_height_m,
    double wave_peak_period_s
):
    """
    Function which takes in an array of frequencies [Hz] and given sea state parameters 
    significant wave height [m] and wave peak period [s], and then generates the 
    corresponding array of component amplitudes [m] assuming a Pierson-Moskowitz sea.
    
    Parameters
    ----------
    
    double[:] frequency_array_Hz
        An array of component frequencies [Hz].
    
    double significant_wave_height_m
        The significant wave height [m] of the sea.
    
    double wave_peak_period_s
        The wave peak period [s] of the sea.
    
    Returns
    -------
    
    double[:]
        An array of component amplitudes [m] assuming a Pierson-Moskowitz sea.
    """
    
    cdef:
        int i = 0
        int N = len(frequency_array_Hz)
        double[:] frequency_bounds_array_Hz = getFrequencyBoundsArray(frequency_array_Hz)
        double lower_frequency_Hz = 0
        double upper_frequency_Hz = 0
        double S_m2Hz = 0
        double[:] component_amplitude_array_m = np.zeros(N)
    
    while i < N:
        lower_frequency_Hz = frequency_bounds_array_Hz[i]
        upper_frequency_Hz = frequency_bounds_array_Hz[i + 1]
        
        S_m2Hz = getVariancePreservingPiersonMoskowitzS(
            lower_frequency_Hz,
            upper_frequency_Hz,
            significant_wave_height_m,
            wave_peak_period_s
        )
        
        component_amplitude_array_m[i] = math.sqrt(
            2 * S_m2Hz * 
            (upper_frequency_Hz - lower_frequency_Hz)
        )
        
        i += 1
    
    return component_amplitude_array_m

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double getCharacteristicWaveNumber(
    double[:] frequency_array_Hz,
    double[:] component_wave_number_array_m,
    double significant_wave_height_m,
    double wave_peak_period_s,
    double sea_depth_m,
    double gravity_ms2 = 9.81
):
    """
    Function which takes in an array of frequencies [Hz] and wave numbers [1/m] and
    given sea state parameters significant wave height [m] and wave peak period [s], and
    then generates the characteristic (i.e., variance density weighted average) wave
    number [1/m].
    
    Parameters
    ----------
    
    double[:] frequency_array_Hz
        An array of component frequencies [Hz].
    
    double[:] component_wave_number_array_m
        An array of component wave numbers [1/m].
    
    double significant_wave_height_m
        The significant wave height [m] of the sea.
    
    double wave_peak_period_s
        The wave peak period [s] of the sea.
    
    double sea_depth_m
        The sea depth [m].
    
    double gravity_ms2
        Acceleration due to gravity [m/s2]. Defaults to 9.81.
    
    Returns
    -------
    
    double
        The characteristic (i.e, variance density weighted average) of the component
        wave numbers [1/m].
    """
    
    cdef:
        int i = 0
        int N = len(frequency_array_Hz)
        double[:] frequency_bounds_array_Hz = getFrequencyBoundsArray(frequency_array_Hz)
        double lower_frequency_Hz = 0
        double upper_frequency_Hz = 0
        double S_m2Hz = 0
        double wave_number_m = 0
        double numerator_sum = 0
        double denominator_sum = 0
    
    while i < N:
        lower_frequency_Hz = frequency_bounds_array_Hz[i]
        upper_frequency_Hz = frequency_bounds_array_Hz[i + 1]
        
        S_m2Hz = getVariancePreservingPiersonMoskowitzS(
            lower_frequency_Hz,
            upper_frequency_Hz,
            significant_wave_height_m,
            wave_peak_period_s
        )
        
        wave_number_m = component_wave_number_array_m[i]
        
        numerator_sum += wave_number_m * S_m2Hz
        denominator_sum += S_m2Hz
        
        i += 1
    
    return numerator_sum / denominator_sum

#### ============================================================================== ####



if __name__ == "__main__":
    pass
