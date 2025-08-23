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
try:
    from path_helper_cuda import find_nearest_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

def vector_angle(w: np.ndarray, v: np.ndarray):
    delta_angle = np.arctan2(
        w[1] * v[0] - w[0] * v[1], 
        w[0] * v[0] + w[1] * v[1]
    )
    return delta_angle


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
    

class PathHelper:
    def __init__(self, points: np.ndarray):
        self.points = points
        self._init_point_distances()
        self.use_gpu = CUDA_AVAILABLE
        if self.use_gpu:
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

    def point_distances(self):
        return self._point_distances

    def get_path_length(self):
        length = 0.
        for i in range(1, len(self.points)):
            a = self.points[i - 1]
            b = self.points[i]
            dist = np.sqrt(np.sum((a - b)**2))
            length += dist
        return length
    
    def points_x(self):
        return self.points[:, 0]
    
    def points_y(self):
        return self.points[:, 1]
   
    def get_segment_by_distance_and_seg_id(self, distance, seg_id):
        n = len(self.points)
        # Validate seg_id to avoid out-of-range indexing
        if seg_id < 0:
            seg_id = 0
        elif seg_id >= n - 1:
            seg_id = n - 2

        # If distance is less than the distance at seg_id, search backwards
        if distance < self._point_distances[seg_id]:
            # Search backwards to find the first segment where distance fits
            for i in range(seg_id, 0, -1):
                if distance >= self._point_distances[i - 1]:
                    return (i - 1, i)
            # If not found, it means distance is smaller than all points; return first segment
            return (0, 1)
        else:
            # If distance is greater than or equal to the distance at seg_id, search forwards
            for i in range(seg_id, n - 1):
                if distance < self._point_distances[i + 1]:
                    return (i, i + 1)
            # If not found, distance is beyond last point; return last segment
            return (n - 2, n - 1)

    def get_point_by_distance(self, distance, seg_id):
        a_idx, b_idx = self.get_segment_by_distance_and_seg_id(distance, seg_id)
        a, b = self.points[a_idx], self.points[b_idx]
        a_dist, b_dist = self._point_distances[a_idx], self._point_distances[b_idx]
        u = (distance - a_dist) / ((b_dist - a_dist) + 1e-6)
        u = np.clip(u, 0., 1.)
        return a + u * (b - a)
    
    def find_nearest_cpu(self, point):
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

    def find_nearest(self, point):
        if self.use_gpu:
            min_pt, min_pt_dist_along_path, min_pt_dist_to_seg, min_pt_seg = self._gpu_op.find_nearest(point)
            return min_pt, min_pt_dist_along_path, min_pt_seg, min_pt_dist_to_seg
        else:
            return self.find_nearest_cpu(point)
