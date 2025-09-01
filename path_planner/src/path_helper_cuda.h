/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


const double max_dist_threshold = 1e9;

struct MinResult {
    double min_dist = max_dist_threshold;
    double min_dist_along_seg = -1;
    double nearest_x = 0.0;
    double nearest_y = 0.0;
    size_t min_index = -1;
};

class FindNearestCuda {
public:
    FindNearestCuda(py::array_t<double> path_points, py::array_t<double> distances);
    ~FindNearestCuda();

    FindNearestCuda(const FindNearestCuda&) = delete;
    FindNearestCuda& operator=(const FindNearestCuda&) = delete;

    py::tuple find_nearest(py::array_t<double> query_point);
    cudaStream_t getStream() const { return stream_; }

private:
    double* d_path_points_ = nullptr;
    double* d_distances_ = nullptr;
    MinResult* d_reduce_buffer_ = nullptr;

    double* h_out_nearest_point_ = nullptr; // pinned host mem
    double* h_out_dist_along_seg_ = nullptr; // pinned host mem
    double* h_out_dist_to_seg_ = nullptr; // pinned host mem
    size_t* h_out_seg_idx_ = nullptr; // pinned host mem

    int length_ = 0;
    int smCount_ = 0;
    int maxThreadsPerSM_ = 0;
    int maxThreadsPerBlock_ = 0;
    int maxBlocks_ = 0; 
    
    int gridSize_ = 0;   
    int blockSize_ = 0;   
    int device_id_ = 0; //Now we use GPU 0 , and we can accept gpu id in future

    cudaStream_t stream_ = nullptr;

    void allocateAndCopy(py::array_t<double> path_points, py::array_t<double> distances);

   
    void freeDeviceMemory();

    void checkArrays(py::array& path_points, py::array& distances);
};
