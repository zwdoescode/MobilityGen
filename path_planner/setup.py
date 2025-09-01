# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension

def get_gpu_info():
    """Detect GPU info using nvidia-smi."""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader,nounits'], 
            universal_newlines=True
        ).strip()
        if output:
            lines = output.split('\n')
            gpus = []
            for line in lines:
                name, cc = [x.strip() for x in line.split(',')]
                major_cc = int(cc.split('.')[0])
                minor_cc = int(cc.split('.')[1]) if '.' in cc else 0
                gpus.append((name, major_cc, minor_cc))
            return gpus
        else:
            return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def find_nvcc():
    """Locate the nvcc binary."""
    cuda_home = os.environ.get('CUDAHOME') or os.environ.get('CUDA_HOME') or '/usr/local/cuda'
    nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
    if os.path.exists(nvcc_path) and os.access(nvcc_path, os.X_OK):
        return nvcc_path
    # Fallback to nvcc in PATH
    try:
        subprocess.check_call(['nvcc', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return 'nvcc'
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

class BuildExtension(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        original_compile = self.compiler._compile

        def custom_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith('.cu'):
                nvcc = find_nvcc()
                if nvcc is None:
                    sys.stderr.write(
                        "Error: nvcc (CUDA compiler) not found! Please install CUDA Toolkit and ensure nvcc is in PATH or set CUDAHOME env.\n"
                    )
                    sys.exit(1)
                filtered_cc_args = [arg for arg in cc_args if arg not in ['-fvisibility=hidden', '-c', '-fPIC']]
                include_args = [arg for arg in filtered_cc_args if arg.startswith('-I')]
                compiler_args = [arg for arg in filtered_cc_args if not arg.startswith('-I')]
                cmd = [
                    nvcc, '-c', src, '-o', obj,
                    '-Xcompiler', ','.join(compiler_args + ['-fPIC']),
                    '-O3', '--use_fast_math'
                ] + include_args
                print("Compiling CUDA:", " ".join(cmd))
                subprocess.check_call(cmd)
            else:
                original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        self.compiler._compile = custom_compile
        build_ext.build_extensions(self)

WITH_GPU = os.environ.get("WITH_GPU") == "1"
if WITH_GPU: 
    gpu_info = get_gpu_info()
    if not gpu_info:
        sys.stderr.write(
            "Error: NVIDIA GPU detected but 'nvidia-smi' failed or not found. "
            "Please ensure NVIDIA drivers are installed properly. Aborting.\n"
        )
        sys.exit(1)
    for name, major_cc, minor_cc in gpu_info:
        print(f"Detected GPU: {name}, Compute Capability: {major_cc}.{minor_cc}")
        if major_cc < 8:
            sys.stderr.write(
                f"Error: Detected GPU '{name}' with compute capability {major_cc}.{minor_cc} is below minimum 8.0 (Ampere). "
                "Please use CPU-only version or upgrade GPU.\n"
            )
            sys.exit(1)
    nvcc_path = find_nvcc()
    if nvcc_path is None:
        sys.stderr.write(
            "Error: nvcc (CUDA compiler) not found! Please install CUDA Toolkit to build GPU version.\n"
        )
        sys.exit(1)
    ext_modules = [
        Pybind11Extension(
            "mobility_gen_path_planner._mobility_gen_path_planner_C",
            ["src/python_bindings.cpp"],
            include_dirs=["src"]
        ),
        Pybind11Extension(
            "path_helper_cuda._path_helper_cuda_op",
            ["src/path_helper_cuda.cu"],
            include_dirs=["src"],
            extra_compile_args=["-O3"],
            extra_link_args=["-L/usr/local/cuda/lib64", "-lcudart"],
        ),
    ]
else:
    ext_modules = [
        Pybind11Extension(
            "mobility_gen_path_planner._mobility_gen_path_planner_C",
            ["src/python_bindings.cpp"],
            include_dirs=["src"]
        ),
    ]

setup(
    name="mobility_gen_op",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

