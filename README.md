# WEC Power AI/ML

    Anthony Truelove MASc, P.Eng.
    email:   gears1763@tutanota.com
    github:  gears1763-2

***See license terms***


This is a project which experiments with applying artificial intelligence and machine 
learning (AI/ML) to the modelling of a point absorber wave energy converter (WEC).

--------


## Contents

In the directory for this project, you should find this README, a LICENSE file, a main 
report `main.pdf`, and the following sub-directories:

    animations/     to hold some WEC animations
    
    Maple/          to hold supporting Maple worksheets
        mw/         to hold the supporting Maple worksheets in native `.mw` format
        pdf/        to hold the supporting Maple worksheets in `.pdf` format
    
    ProteusDS/      to hold an example Proteus DS simulation (used in generating WEC performance data)
    
    Python/         to hold supporting Python scripts and C extensions
        data/       to hold the data table, `numpy` binaries, etc.
        pyx/        to hold supporting Cython files, etc.

--------


## Key Features

  * Reduced dynamics for a heave constrained point absorber are solved exactly (with respect to expected power output).
  * Time domain modelling software is used to generate a data set of more complex dynamics.
  * A perturbation machine (dense neural network) is trained to correct from the reduced to the complex dynamics.
  * Example applications to WEC design optimization and WEC performance mapping are presented.

--------


## Setup

If you run into a `wave_utils` or `wave_energy_converter` module not found error, chances
are you need to build the C extensions for these. See below.

Note that the provided `setup.py` has the following dependencies (listed in `pipreqs.txt`):

    Cython==3.0.4
    setuptools==59.6.0

so you may need to `pip install` these before building the C extensions. Additionally, you
need to make sure that you are set up with a compatible C compiler.

However, if you are all set up, then building the bindings (from `Python/`) is as simple as

    python(3) setup.py build_ext --inplace

depending on your setup (either `python` or `python3`).

### --- Linux (Debian/Ubuntu) Notes ---

You should already be set up with a Python 3 interpreter, so just doing the `pip install`
steps should be sufficient. Additionally, you may also be set up with a compatible
C compiler already. However, if this is not the case, then you can simply

    sudo apt-get install build-essential


### --- Windows Notes ---

On Windows, there are a few things to set up first before building the C extensions;
namely

  * You need to be using Python for Windows, as available from <https://www.python.org/downloads/>. Version 3.10+ should do fine.
  * You need to `pip install` the dependencies listed above (and in `pipreqs.txt`).
  * You need to be using the MSVC compiler (`cl.exe`). This can be obtained by way of installing Visual Studio (as available from <https://visualstudio.microsoft.com/downloads/>; the free community edition is fine) and then selecting the "Desktop development with C++" workload.


### --- Testing Notes ---

To test that the C extensions have been built and are functioning correctly, a testing 
script is provided in `test.py`, so simply

    python(3) test.py

--------
