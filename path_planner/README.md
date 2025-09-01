# MobilityGen - Path Planner

This package contains the C++ implementation and Python bindings for an A* path planner.

## Setup

To install in Python environment

```bash
cd mobility_gen_path_planner
python -m pip install -e .
```

(TODO): Install with OV extension

## Usage
Generate paths methods exhaustively plans all paths from a starting point using Dijkstra's algorithm.

```python
from mobility_gen_path_planner import generate_paths

# Plan all paths connected to start point
start = (0, 0)
freespace = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

output = generate_paths(start, freespace)

# Unroll path to an end point
path = output.unroll_path(end=(2, 2))

# Sample a random path (by selecting random valid endpoint)
end = output.sample_random_end_point()
path = output.unroll_path(end=end)

```

## Path Helper Cuda
The purpose of Path Helper CUDA is to accelerate the performance of the **find nearest** function in the path helper. In the path planning process, this function is often the main bottleneck of Random Path Following when running on the CPU.  
For usage reference, please see the script `test/test_find_nearest_cuda.py`.
