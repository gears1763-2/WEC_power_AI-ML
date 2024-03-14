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


cpdef double getFloatMass(
    double float_inner_diameter_m,
    double float_outer_diameter_m,
    double float_resting_draft_m,
    double fluid_density_kgm3 = 1025
):
    """
    Function which takes in the inner and outer diameters [m] and resting draft [m] of 
    a heave constrained, cylindrical float, and then returns the mass [kg].
    
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


cpdef double getBuoyancyStiffness(
    double float_inner_diameter_m,
    double float_outer_diameter_m,
    double fluid_density_kgm3 = 1025,
    double gravity_ms2 = 9.81
):
    """
    Function which takes in the inner and outer diameters [m] of a heave constrained, 
    cylindrical float, and then returns the buoyancy stiffness [N/m]. This is the 
    k_D term in the main report.
    
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


if __name__ == "__main__":
    pass
