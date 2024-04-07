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
    A collection of utility functions for modelling a cylindrical, heave constrained, 
    point absorber wave energy converter (WEC).
"""


import cython

import math

import numpy as np


#### ============================================================================== ####

cpdef double getFloatMass(
    double float_inner_diameter_m,
    double float_outer_diameter_m,
    double float_resting_draft_m,
    double fluid_density_kgm3 = 1025
):
    """
    Function which takes in the inner and outer diameters [m] and resting draft [m] of 
    a heave constrained, cylindrical float, and then returns the mass [kg]. This is 
    based on eq'n (2.6) of the main report.
    
    Parameters
    ----------
    
    double float_inner_diameter_m
        The inner diameter of the cylindrical float [m].
    
    double float_outer_diameter_m
        The outer diameter of the cylindrical float [m].
    
    double float_resting_draft_m
        The resting draft of the cylindrical float [m].
    
    double fluid_density_kgm3
        The density of the fluid in which the cylindrical float is immersed [kg/m3]. 
        Defaults to 1025 (the nominal density of sea water).
    
    Returns
    -------
    
    double
        The mass of the cylindrical float [kg].
    """
    
    cdef double float_mass_kg = (math.pi * fluid_density_kgm3 * float_resting_draft_m) / 4
    float_mass_kg *= math.pow(float_outer_diameter_m, 2) - math.pow(float_inner_diameter_m, 2)
    
    return float_mass_kg

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double getBuoyancyStiffness(
    double float_inner_diameter_m,
    double float_outer_diameter_m,
    double fluid_density_kgm3 = 1025,
    double gravity_ms2 = 9.81
):
    """
    Function which takes in the inner and outer diameters [m] of a heave constrained, 
    cylindrical float, and then returns the buoyancy stiffness [N/m]. This is the 
    k_D term in the main report, as defined in eq'n (2.9).
    
    Parameters
    ----------
    
    double float_inner_diameter_m
        The inner diameter of the cylindrical float [m].
    
    double float_outer_diameter_m
        The outer diameter of the cylindrical float [m].
    
    double fluid_density_kgm3
        The density of the fluid in which the cylindrical float is immersed [kg/m3]. 
        Defaults to 1025 (the nominal density of sea water).
    
    double gravity_ms2
        Acceleration due to gravity [m/s2]. Defaults to 9.81.
    
    Returns
    -------
    
    double
        The buoyancy stiffness of the cylindrical float [N/m]. This is the k_D term in 
        the main report.
    """
    
    cdef double buoyancy_stiffness_Nm = (math.pi * fluid_density_kgm3 * gravity_ms2) / 4
    buoyancy_stiffness_Nm *= math.pow(float_outer_diameter_m, 2) - math.pow(float_inner_diameter_m, 2)
    
    return buoyancy_stiffness_Nm

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double __integrandIcos(
    double r_m,
    double wave_number_m,
    double wave_direction,
    double theta
):
    """
    Helper function to evaluate the I_cos integrand at a single point.
    
    Parameters
    ----------
    
    double r_m
        Radial polar coordinate [m].
    
    double wave_number_m
        Component wave number [1/m].
    
    double wave_direction
        Component wave direction [ ].
    
    double theta
        Angular polar coordinate [ ].
    
    Returns
    -------
    
    double
        An evaluation of the I_cos integrand.
    """
    
    cdef double I_cos = r_m * math.cos(wave_number_m * r_m * math.cos(wave_direction - theta))
    
    return I_cos

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double getIcos(
    double float_inner_diameter_m,
    double float_outer_diameter_m,
    double wave_number_m,
    double wave_direction,
    int discretization_level = 64
):
    """
    A function for generating a numerical approximation of I_cos.
    
    Parameters
    ----------
    
    double float_inner_diameter_m
        The inner diameter of the cylindrical float [m].
    
    double float_outer_diameter_m
        The outer diameter of the cylindrical float [m].
    
    double wave_number_m
        The component wave number [1/m].
    
    double wave_direction
        The component wave direction [ ].
    
    int discretization_level
        The discretization level to use in generating the approximation. Defaults to 
        10.
    
    Returns
    -------
    
    double
        A numerical approximation of I_cos.
    """
    
    #   build integrand array
    cdef:
        int i = 0
        int j = 0
        int N = discretization_level
        
        double[:] r_array_m = np.linspace(
            float_inner_diameter_m / 2,
            float_outer_diameter_m / 2,
            N
        )
        
        double[:] theta_array = np.linspace(0, 2 * math.pi, N)
        
        double[:, :] integrand_array = np.zeros((N, N))
    
    while i < N:      # r
        while j < N:  # theta
            integrand_array[i, j] = __integrandIcos(
                r_array_m[i],
                wave_number_m,
                wave_direction,
                theta_array[j]
            )
            
            j += 1
        
        i += 1
        j = 0
    
    
    #   build up I_cos sum (using trapezoidal approach)
    cdef:
        double average_integrand = 0
        double I_cos = 0
        double dr_m = 0
        double dtheta = 0
    
    i = 0
    j = 0
    while i < N - 1:      # r
        while j < N - 1:  # theta
            average_integrand = integrand_array[i, j]
            average_integrand += integrand_array[i + 1, j]
            average_integrand += integrand_array[i, j + 1]
            average_integrand += integrand_array[i + 1, j + 1]
            average_integrand /= 4
            
            dr_m = r_array_m[i + 1] - r_array_m[i]
            dtheta = theta_array[j + 1] - theta_array[j]
            
            I_cos += average_integrand * dr_m * dtheta
            
            j += 1
        
        i += 1
        j = 0
    
    
    #   scale I_cos sum and return
    I_cos *= (
        4.0 / (
            math.pi * (
                math.pow(float_outer_diameter_m, 2) -
                math.pow(float_inner_diameter_m, 2)
            )
        )
    )
    
    return I_cos

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double[:, :] getAlphaBetaArray(
    double[:] component_amplitude_array_m,
    double[:] component_wave_number_array_m,
    double[:] component_wave_direction_array,
    double[:] component_phase_array,
    double float_inner_diameter_m,
    double float_outer_diameter_m
):
    """
    Function which takes in a component amplitude array [m], component wave number array
    [1/m], component phase array [ ], WEC inner and outer diameters [m], and a gamma 
    value [ ], and then computes and returns the corresponding alpha beta array [m]. 
    These are the alpha and beta values that first appear in eq'ns (2.30a-c) of the main
    report.
    
    Parameters
    ----------
    
    double[:] component_amplitude_array_m
        An array of wave component amplitudes [m].
    
    double[:] component_wave_number_array_m
        An array of component wave numbers [1/m].
    
    double[:] component_wave_direction_array
        An array of component wave directions [ ].
    
    double[:] component_phase_array
        An array of wave component phases [ ].
    
    double float_inner_diameter_m
        The inner diameter of the cylindrical float [m].
    
    double float_outer_diameter_m
        The outer diameter of the cylindrical float [m].
    
    Returns
    -------
    
    double [:, :]
        The corresponding alpha beta array [m], with the alpha array being the first
        column, and beta array being the second column.
    """

    cdef:
        int i = 0
        int N = len(component_amplitude_array_m)
        double I_cos = 0
        double common_factor = 0
        double alpha = 0
        double beta = 0
        double[:, :] alpha_beta_array_m = np.zeros((N, 2))
    
    
    while i < N:
        I_cos = getIcos(
            float_inner_diameter_m,
            float_outer_diameter_m,
            component_wave_number_array_m[i],
            component_wave_direction_array[i]
        )
        
        common_factor = component_amplitude_array_m[i] * I_cos
        
        alpha = common_factor * math.cos(component_phase_array[i])
        beta = common_factor * math.sin(component_phase_array[i])
        
        alpha_beta_array_m[i, 0] = alpha
        alpha_beta_array_m[i, 1] = beta
        
        i += 1
    
    return alpha_beta_array_m

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double[:, :] getABArray(
    double [:, :] alpha_beta_array_m,
    double float_mass_kg,
    double power_takeoff_stiffness_Nm,
    double power_takeoff_damping_Nsm,
    double buoyancy_stiffness_Nm,
    double fundamental_period_s
):
    """
    Function which takes in an alpha beta array [m], WEC float mass [kg], power takeoff 
    stiffness [N/m] and damping [N.s/m], the buoyancy stiffness of the float [kg], and 
    a fundamental (Fourier) period [s], and then computes and returns the corresponding 
    A B array [m]. These are the A and B values that first appear in eq'n (2.28) of the 
    main report.
    
    Parameters
    ----------
    
    double[:, :] alpha_beta_array_m
        An array of alpha and beta values [m], with the alpha array being the first 
        column, and the beta array being the second column.
    
    double float_mass_kg
        The mass of the cylindrical float [kg].
    
    double power_takeoff_stiffness_Nm
        The power takeoff stiffness [N/m].
    
    double power_takeoff_damping_Nsm
        The power takeoff damping [N.s/m].
    
    double buoyancy_stiffness_Nm
        The buoyancy stiffness of the cylindrical float [N/m].
    
    double fundamental_period_s
        The fundamental (Fourier) period used in modelling the WEC dynamics [s].
    
    Returns
    -------
    
    double[:, :]
        The corresponding A B array [m], with the A array being the first column, and 
        the B array being the second column.
    """
    
    cdef:
        int i = 0
        int n = 0
        int N = np.array(alpha_beta_array_m).shape[0]
        double alpha = 0
        double beta = 0
        double denominator = 0
        double A_numerator = 0
        double B_numerator = 0
        double A = 0
        double B = 0
        double[:, :] A_B_array_m = np.zeros((N, 2))
    
    while i < N:
        n = i + 1
        alpha = alpha_beta_array_m[i, 0]
        beta = alpha_beta_array_m[i, 1]
        
        denominator = math.pow(
            power_takeoff_stiffness_Nm +
            buoyancy_stiffness_Nm -
            (
                float_mass_kg *
                math.pow((2 * math.pi * n) / fundamental_period_s, 2)
            ),
            2
        )
        denominator += (
            math.pow(power_takeoff_damping_Nsm, 2) *
            math.pow((2 * math.pi * n) / fundamental_period_s, 2)
        )
        
        A_numerator = buoyancy_stiffness_Nm * alpha * (
            power_takeoff_stiffness_Nm +
            buoyancy_stiffness_Nm -
            (
                float_mass_kg *
                math.pow((2 * math.pi * n) / fundamental_period_s, 2)
            )
        )
        A_numerator -= buoyancy_stiffness_Nm * beta * power_takeoff_damping_Nsm * (
            (2 * math.pi * n) / fundamental_period_s
        )
        
        B_numerator = buoyancy_stiffness_Nm * beta * (
            power_takeoff_stiffness_Nm +
            buoyancy_stiffness_Nm -
            (
                float_mass_kg *
                math.pow((2 * math.pi * n) / fundamental_period_s, 2)
            )
        )
        B_numerator += buoyancy_stiffness_Nm * alpha * power_takeoff_damping_Nsm * (
            (2 * math.pi * n) / fundamental_period_s
        )
        
        A = A_numerator / denominator
        B = B_numerator / denominator
        
        A_B_array_m[i, 0] = A
        A_B_array_m[i, 1] = B
        
        i += 1
    
    return A_B_array_m

#### ============================================================================== ####



#### ============================================================================== ####

cpdef double getExpectedWECPower(
    double[:, :] A_B_array_m,
    double power_takeoff_damping_Nsm,
    double fundamental_period_s
):
    """
    Function which takes in A B array [m], the WEC power takeoff damping [N.s/m], and 
    a fundamental (Fourier) period [s], and then computes and returns the expected power 
    output of the WEC [kW]. This is based on eq'n (2.39) of the main report.
    
    Parameters
    ----------
    
    double[:, :] A_B_array_m
        An array of A and B values [m], with the A array being the first column, and the
        B array being the second column.
    
    double power_takeoff_damping_Nsm
        The power takeoff damping [N.s/m].
    
    double fundamental_period_s
        The fundamental (Fourier) period used in modelling the WEC dynamics [s].
    
    Returns
    -------
    
    double
        The expected power output of the WEC [kW].
    """
    
    cdef:
        int i = 0
        int n = 0
        int N = np.array(A_B_array_m).shape[0]
        double A = 0
        double B = 0
        double expected_WEC_power_kW = 0
    
    while i < N:
        n = i + 1
        
        A = A_B_array_m[i, 0]
        B = A_B_array_m[i, 1]
        
        expected_WEC_power_kW += (
            (math.pow(A, 2) + math.pow(B, 2)) *
            math.pow((2 * math.pi * n) / fundamental_period_s, 2)
        )
        
        i += 1
    
    return (power_takeoff_damping_Nsm / 2000) * expected_WEC_power_kW

#### ============================================================================== ####



if __name__ == "__main__":
    pass
