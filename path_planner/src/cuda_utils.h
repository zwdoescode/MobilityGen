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

#ifndef CUDA_RUNTIME_ERROR_CHECK_H
#define CUDA_RUNTIME_ERROR_CHECK_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                   \
  {                                                                       \
    const cudaError_t error = call;                                       \
    if (error != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__);         \
      fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                            \
    }                                                                     \
  }

#endif // CUDA_RUNTIME_ERROR_CHECK_H
