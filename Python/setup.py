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
    A setup script for building the C extensions (for later import) from the 
    given Cython (.pyx) files. This is essentially a "make all".
"""


from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np
from setuptools import setup, Extension

Options.fast_fail = True

ext_modules = [
    Extension(
        "wave_utils",
        ["pyx/wave_utils.pyx"]
    ),
]

setup(
    ext_modules = cythonize(
        ext_modules,
        nthreads=len(ext_modules),
        compiler_directives = {
            "language_level": "3",
            "profile": False,
            "cdivision": True,
            "boundscheck": False,
            "wraparound": False
        },
        annotate=True,
        force=False
    ),
    zip_safe=False
)
