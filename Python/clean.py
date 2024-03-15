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
    A simple script for cleaning up the project for shipping. This is essentially a
    "make clean".
"""


import os
import shutil


if __name__ == "__main__":
    #   delete build/ (and all contents)
    if os.path.exists("build"):
        shutil.rmtree("build")
        print("removed build" + os.path.sep)
    
    
    #   delete C extensions
    shared_object_list = [
        filename for filename in
        os.listdir(os.getcwd()) if
        ".so" in filename or    # Linux
        ".pyd" in filename      # Windows
    ]
    
    for filename in shared_object_list:
        os.remove(filename)
        print("removed", filename)
    
    
    #   clean out pyx/
    not_pyx_list = [
        filename for filename in
        os.listdir(os.getcwd() + os.path.sep + "pyx") if
        ".pyx" not in filename
    ]
    
    for filename in not_pyx_list:
        os.remove("pyx" + os.path.sep + filename)
        print("removed pyx" + os.path.sep + filename)
    
    
    #   clean up LaTeX/
    LaTeX_clean_list = [
        filename for filename in 
        os.listdir(".." + os.path.sep + "LaTeX") if
        filename != "bib" and
        filename != "images" and
        filename != "tex" and
        ".tex" not in filename and
        ".bib" not in filename and
        ".pdf" not in filename
    ]
    
    for filename in LaTeX_clean_list:
        os.remove(".." + os.path.sep + "LaTeX" + os.path.sep + filename)
        print("removed .." + os.path.sep + "LaTeX" + os.path.sep + filename)
    
    LaTeX_clean_list = [
        filename for filename in 
        os.listdir(".." + os.path.sep + "LaTeX" + os.path.sep + "tex") if
        ".tex" not in filename
    ]
    
    for filename in LaTeX_clean_list:
        os.remove(
            ".." + os.path.sep + "LaTeX" + os.path.sep +
            "tex" + os.path.sep + filename
        )
        print(
            "removed .." + os.path.sep + "LaTeX" + os.path.sep +
            "tex" + os.path.sep + filename
        )
