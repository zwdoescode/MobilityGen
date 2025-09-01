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


import numpy as np
import numbers
from path_helper_cuda import find_nearest_cuda

def nearest_point_on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    a2b = b - a
    a2c = c - a
    a2b_mag = np.sqrt(np.sum(a2b**2))
    a2b_norm = a2b / (a2b_mag + 1e-6)
    dist = np.dot(a2c, a2b_norm)
    if dist < 0:
        return a, dist
    elif dist > a2b_mag:
        return b, dist
    else:
        return a + a2b_norm * dist, dist

class TestPathHelper:
    def __init__(self, points: np.ndarray):
        self.points = points
        self._init_point_distances()
        self._gpu_op = find_nearest_cuda(self.points,self._point_distances)

    def _init_point_distances(self):
        self._point_distances = np.zeros(len(self.points))
        length = 0.
        for i in range(0, len(self.points) - 1):
            self._point_distances[i] = length
            a = self.points[i]
            b = self.points[i + 1]
            dist = np.sqrt(np.sum((a - b)**2))
            length += dist
        self._point_distances[-1] = length

    def find_nearesat_cpu(self, point):
        min_pt_dist_to_seg = 1e9
        min_pt_seg = None
        min_pt = None
        min_pt_dist_along_path = None

        for a_idx in range(0, len(self.points) - 1):
            b_idx = a_idx + 1
            a = self.points[a_idx]
            b = self.points[b_idx]
            nearest_pt, dist_along_seg = nearest_point_on_segment(a, b, point)
            dist_to_seg = np.sqrt(np.sum((point - nearest_pt)**2))

            if dist_to_seg < min_pt_dist_to_seg:
                min_pt_seg = (a_idx, b_idx)
                min_pt_dist_to_seg = dist_to_seg
                min_pt = nearest_pt
                min_pt_dist_along_path = self._point_distances[a_idx] + dist_along_seg

        
        return min_pt, min_pt_dist_along_path, min_pt_seg, min_pt_dist_to_seg

    def find_nearest_gpu(self, point):
        min_pt, min_pt_dist_along_path, min_pt_dist_to_seg, min_pt_seg = self._gpu_op.find_nearest(point)
        return min_pt, min_pt_dist_along_path, min_pt_seg, min_pt_dist_to_seg

def test_path_helper():
    np.random.seed(42)
    path = np.random.rand(100, 2).astype(np.float64)
    path_helper = TestPathHelper(path)

    for i in range(100):
        test_point = np.random.rand(2).astype(np.float32)
        cpu_result = path_helper.find_nearesat_cpu(test_point)
        gpu_result = path_helper.find_nearest_gpu(test_point)

        equal = all(
            (np.allclose(cpu_result[j], gpu_result[j])
             if isinstance(cpu_result[j], np.ndarray)
             else np.isclose(cpu_result[j], gpu_result[j]) 
             if isinstance(cpu_result[j], numbers.Number)
             else cpu_result[j] == gpu_result[j]
            )
            for j in range(len(cpu_result))
        )
        print(f"Test {i}: Equal = {equal}")
        if not equal:
            print(f"CPU result: {cpu_result}")
            print(f"GPU result: {gpu_result}")
        assert equal, f"Mismatch found in test {i}"

if __name__ == "__main__":
    test_path_helper()
