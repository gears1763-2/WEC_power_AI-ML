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
    An example application of the perturbation machine (tensorflow.keras.Sequential)
    to the performance mapping of a heave constrained point absorber.
"""


import math

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as nptype

import sklearn.preprocessing as skl_pp

import tensorflow.keras as tfk

import time

import wave_utils as wu

import wave_energy_converter as wec



#### ============================================================================== ####

@tfk.utils.register_keras_serializable()
def blendedLossFunction(y_true : nptype.ArrayLike, y_pred : nptype.ArrayLike) -> float:
    """
    A blended loss function which returns a weighted sum of the mean squared, mean 
    absolute, and mean squared log errors.
    
    Ref: <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError>
    Ref: <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanAbsoluteError>
    Ref: <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredLogarithmicError>
    
    Parameters
    ----------
    
    y_true : nptype.ArrayLike
        An array like of true target values.
    
    y_pred : nptype.ArrayLike
        An array like of predicted target values.
    
    Returns
    -------
    
    float
        The loss value.
    """
    
    mean_squared_error_weight = 1
    mean_squared_error_term = tfk.losses.MeanSquaredError().call(y_true, y_pred)
    
    mean_absolute_error_weight = 1
    mean_absolute_error_term = tfk.losses.MeanAbsoluteError().call(y_true, y_pred)
    
    mean_squared_log_error_weight = 5
    mean_squared_log_error_term = tfk.losses.MeanSquaredLogarithmicError().call(y_true, y_pred)
    
    blended_loss = (
        (mean_squared_error_weight * mean_squared_error_term) +
        (mean_absolute_error_weight * mean_absolute_error_term) +
        (mean_squared_log_error_weight * mean_squared_log_error_term)
    )
    
    return blended_loss


#### ============================================================================== ####



#### ============================================================================== ####

#   PERTURBATION MACHINE

#   set up normalizer
TRAIN_TEST_ARRAY = np.load("data/perturbation_machine_train_test_split.npz")
INPUT_TRAIN = TRAIN_TEST_ARRAY["input_train"]

STANDARD_SCALER = skl_pp.StandardScaler()
STANDARD_SCALER.fit(INPUT_TRAIN)


#   load perturbation machine
PERTURBATION_MACHINE = tfk.models.load_model("data/perturbation_machine.keras")

#### ============================================================================== ####



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



#### ============================================================================== ####

def getPredictedPower(
    fundamental_period_s : float,
    significant_wave_height_m : float,
    wave_peak_period_s : float,
    sea_depth_m : float,
    float_inner_diameter_m : float,
    float_outer_diameter_m : float,
    float_resting_draft_m : float,
    power_takeoff_stiffness_Nm : float,
    power_takeoff_damping_Nsm: float,
    frequency_array_Hz : nptype.ArrayLike,
    wave_number_array_m : nptype.ArrayLike
) -> float:
    """
    Function which predicts the expected power output of the WEC based on a perturbation 
    of the expected power output under the reduced dynamics.
    
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
    
    frequency_array_Hz : nptype.ArrayLike
        An array of wave component frequencies [Hz].
    
    wave_number_array_m : nptype.ArrayLike
        An array of component wave numbers [1/m].
    
    Returns
    -------
    
    float
        A prediction for expected power output based on the perturbation of the 
        reduced dynamics expected power output.
    """
    
    #   define globals
    global STANDARD_SCALER
    global PERTURBATION_MACHINE
    
    
    #   get reduced dynamics power
    expected_WEC_power_kW = getReducedDynamicsPower(
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
    
    
    #   init extended feature vector
    extended_feature_vector = np.zeros(10)
    
    extended_feature_vector[0] = sea_depth_m
    extended_feature_vector[1] = significant_wave_height_m
    extended_feature_vector[2] = wave_peak_period_s
    extended_feature_vector[3] = float_outer_diameter_m
    extended_feature_vector[4] = power_takeoff_stiffness_Nm
    extended_feature_vector[5] = math.log10(power_takeoff_damping_Nsm)
    
    
    #   compute dimensionless features
    Pi_0 = getPi0(
        frequency_array_Hz,
        wave_number_array_m,
        significant_wave_height_m,
        wave_peak_period_s,
        sea_depth_m,
        float_outer_diameter_m,
        float_inner_diameter_m
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
    
    Pi_1 = getPi1(
        float_mass_kg,
        power_takeoff_stiffness_Nm,
        buoyancy_stiffness_Nm,
        power_takeoff_damping_Nsm
    )
    
    Pi_2 = getPi2(
        Pi_1,
        float_mass_kg,
        power_takeoff_stiffness_Nm,
        buoyancy_stiffness_Nm,
        wave_peak_period_s
    )
    
    Pi_3 = getPi3(
        power_takeoff_damping_Nsm,
        significant_wave_height_m,
        wave_peak_period_s
    )
    
    
    #   add dimensionless features
    extended_feature_vector[6] = math.log10(Pi_0)
    extended_feature_vector[7] = math.log10(Pi_1)
    extended_feature_vector[8] = Pi_2
    extended_feature_vector[9] = math.log10(Pi_3)
    
    
    #   get perturbation value, apply
    perturbation = PERTURBATION_MACHINE.predict(
        STANDARD_SCALER.transform(extended_feature_vector.reshape(1, -1)),
        verbose=0
    )[0][0]
    
    expected_WEC_power_kW *= perturbation
    
    if expected_WEC_power_kW < 0:
        expected_WEC_power_kW = 0
    
    return expected_WEC_power_kW

#### ============================================================================== ####



if __name__ == "__main__":
    start_time = time.time()
    
    #   set fixed values
    sea_depth_m = 30
    
    fundamental_period_s = 900
    
    float_inner_diameter_m = 6
    float_outer_diameter_m = 81.02
    float_resting_draft_m = 2.5
    
    power_takeoff_stiffness_Nm = 0
    power_takeoff_damping_Nsm = 7.229e7
    
    
    
    #   set grid search points
    grid_density = 64
    significant_wave_height_points_m = np.linspace(0, 8, grid_density)
    wave_peak_period_points_s = np.linspace(5, 16, grid_density)
    
    
    
     #   perform grid search
    expected_power_output_grid_kW = np.zeros((grid_density, grid_density))
    
    significant_wave_height_scatter_m = []
    wave_peak_period_scatter_s = []
    expected_power_output_scatter_kW = []
    
    for i in range(0, grid_density):
        wave_peak_period_s = wave_peak_period_points_s[i]
        
        n_components = 4 * math.ceil(fundamental_period_s / wave_peak_period_s)
        
        frequency_array_Hz = np.array(
            [n / fundamental_period_s for n in range(1, n_components + 1)]
        )
        
        wave_number_array_m = wu.getWaveNumberArray(
            frequency_array_Hz,
            sea_depth_m
        )
        
        significant_wave_height_breaking_m = 0.0204 * 9.81 * math.pow(wave_peak_period_s, 2)
        
        for j in range(0, grid_density):
            significant_wave_height_m = significant_wave_height_points_m[j]
            
            significant_wave_height_scatter_m.append(significant_wave_height_m)
            wave_peak_period_scatter_s.append(wave_peak_period_s)
            expected_power_output_scatter_kW.append(0)
            
            if (
                significant_wave_height_m > 0 and 
                significant_wave_height_m <= significant_wave_height_breaking_m
            ):
                expected_power_output_grid_kW[i, j] = getPredictedPower(
                    fundamental_period_s,
                    significant_wave_height_m,
                    wave_peak_period_s,
                    sea_depth_m,
                    float_inner_diameter_m,
                    float_outer_diameter_m,
                    float_resting_draft_m,
                    power_takeoff_stiffness_Nm,
                    power_takeoff_damping_Nsm,
                    frequency_array_Hz,
                    wave_number_array_m
                )
                
                expected_power_output_scatter_kW[-1] = expected_power_output_grid_kW[i, j]
            
            print(
                "performing grid search",
                (i * grid_density) + j + 1,
                "/",
                grid_density * grid_density,
                4 * " ",
                end="\r",
                flush=True
            )
    
    print()
    
    
    
    #   plot grid search results
    wave_peak_period_array_s = np.linspace(
        wave_peak_period_points_s[0],
        wave_peak_period_points_s[-1],
        256
    )
    significant_wave_height_breaking_array_m = 0.0204 * 9.81 * np.power(wave_peak_period_array_s, 2)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(
        wave_peak_period_points_s,
        significant_wave_height_points_m,
        np.transpose(expected_power_output_grid_kW),
        cmap="jet",
        zorder=1
    )
    plt.colorbar(
        label="Expected Power Output [kW]"
    )
    plt.grid(color="C7", alpha=0.5, zorder=2)
    plt.scatter(
        wave_peak_period_scatter_s,
        significant_wave_height_scatter_m,
        s=8,
        c=expected_power_output_scatter_kW,
        cmap="jet",
        alpha=0.5,
        zorder=3,
        label="grid points"
    )
    plt.plot(
        wave_peak_period_array_s,
        significant_wave_height_breaking_array_m,
        color="red",
        linestyle="--",
        zorder=4,
        label="wave breaking front"
    )
    plt.xlim(wave_peak_period_points_s[0], wave_peak_period_points_s[-1])
    plt.xticks(
        [i for i in range(
            round(wave_peak_period_points_s[0]),
            round(wave_peak_period_points_s[-1]) + 1
        )]
    )
    plt.xlabel("Wave Spectral Peak Period [s]")
    plt.ylim(significant_wave_height_points_m[0], significant_wave_height_points_m[-1])
    plt.ylabel("Significant Wave Height [m]")
    plt.legend(
        loc="lower left"
    )
    plt.savefig(
        "../LaTeX/images/example_application_2/grid_search_{}x{}_filled_contours.png".format(
            grid_density, grid_density
        ),
        format="png",
        dpi=128,
        bbox_inches="tight"
    )
    plt.close()
    
    end_time = time.time()
    print("time elapsed =", round(end_time - start_time), "s")
