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


import subprocess
import argparse
import glob
import os
import tempfile
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("folder")
parser.add_argument("output_path")
parser.add_argument("--interval", type=int, default=1)

args = parser.parse_args()

images = glob.glob(os.path.join(args.folder, "*.jpg"))

output_dir = tempfile.mkdtemp()

print("Writing selected frames to temp dir")
for path in images:
    frame_index = int(os.path.basename(path).split('.')[0])

    output_path = os.path.join(output_dir, f"{frame_index // args.interval:08d}.jpg")

    shutil.copyfile(path, output_path)

print("Converting to mp4")
subprocess.call([
    "ffmpeg",
    "-framerate",
    "30",
    "-i",
    f"{output_dir}/%08d.jpg",
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    args.output_path
])