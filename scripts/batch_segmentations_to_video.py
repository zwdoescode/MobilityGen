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


import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Optional


def create_fixed_colormap():
    """Create a fixed colormap of 16 distinct pastel colors."""
    colors = [
        [255, 182, 193],  # Light pink
        [176, 224, 230],  # Powder blue
        [255, 218, 185],  # Peach
        [221, 160, 221],  # Plum
        [176, 196, 222],  # Light steel blue
        [152, 251, 152],  # Pale green
        [255, 255, 224],  # Light yellow
        [230, 230, 250],  # Lavender
        [255, 228, 225],  # Misty rose
        [240, 255, 240],  # Honeydew
        [255, 240, 245],  # Lavender blush
        [224, 255, 255],  # Light cyan
        [250, 235, 215],  # Antique white
        [245, 255, 250],  # Mint cream
        [255, 228, 196],  # Bisque
        [240, 248, 255],  # Alice blue
    ]
    return np.array(colors, dtype=np.uint8)


def convert_segmentations_to_video(seg_dir: Path, output_path: Path, fps: int = 30,
                                   normals_dir: Optional[Path] = None, 
                                   depth_dir: Optional[Path] = None) -> bool:
    """
    Convert a directory of segmentation images to a colorized video.
    
    Args:
        seg_dir: Directory containing segmentation PNG files
        output_path: Path where the output video will be saved
        fps: Frames per second for the output video
        normals_dir: Optional directory containing surface normal .npy files
        depth_dir: Optional directory containing 16-bit inverse depth PNG files
    
    Returns:
        True if successful, False otherwise
    """
    # Get sorted list of PNG files
    png_files = sorted(list(seg_dir.glob('*.png')))
    
    if not png_files:
        print(f"  Warning: No PNG files found in {seg_dir}")
        return False

    try:
        # Read first image to get dimensions
        first_img = cv2.imread(str(png_files[0]), cv2.IMREAD_UNCHANGED)
        if first_img is None:
            print(f"  Error: Could not read first image {png_files[0]}")
            return False
            
        height, width = first_img.shape

        # Create fixed colormap
        colormap = create_fixed_colormap()

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"  Error: Could not open video writer for {output_path}")
            return False

        # Process each frame
        for png_file in png_files:
            # Read segmentation image
            seg = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
            if seg is None:
                print(f"  Warning: Could not read {png_file}, skipping")
                continue
            
            # Load corresponding normal map if available
            normal_map = None
            if normals_dir:
                normal_file = normals_dir / (png_file.stem + '.npy')
                if normal_file.exists():
                    normal_map = np.load(str(normal_file))
            
            # Load corresponding depth map if available
            depth_map = None
            if depth_dir:
                depth_file = depth_dir / png_file.name
                if depth_file.exists():
                    depth_map = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
                    if depth_map is not None:
                        depth_map = depth_map / 65535.0

            # Create base colored frame
            colored_frame = colormap[seg % len(colormap)]
            
            # Apply shading based on normal map and depth map
            if normal_map is not None or depth_map is not None:
                # Initialize combined shading
                shading = np.ones((height, width), dtype=float)
                
                # Apply normal-based shading if available
                if normal_map is not None:
                    normal_xyz = 2.0 * normal_map[..., :3] - 1.0
                    light_dir = np.array([-0.5, 0.5, 1])
                    light_dir = light_dir / np.sqrt(np.sum(light_dir**2))
                    normal_shading = np.clip(np.dot(normal_xyz, light_dir), 0, 1)
                    normal_shading /= normal_map[..., 3]
                    normal_shading = 0.5 + 0.5 * normal_shading
                    
                    shading *= normal_shading

                # Apply depth-based shading if available
                if depth_map is not None:
                    # shading *= (depth_map**0.1)
                    pass

                # Apply combined shading to colored frame
                shading = shading.reshape(height, width, 1)
                colored_frame = (colored_frame * shading).astype(np.uint8)

            # Write frame to video
            out.write(colored_frame)

        # Release video writer
        out.release()
        print(f"  ✓ Video saved to {output_path} ({len(png_files)} frames)")
        return True
        
    except Exception as e:
        print(f"  Error: Failed to create video - {str(e)}")
        return False


def process_replay_folder(replay_folder: Path, output_dir: Path, fps: int = 30,
                          use_normals: bool = False, use_depth: bool = False) -> List[Tuple[str, bool]]:
    """
    Process a single replay folder to generate videos for segmentation images.
    
    Args:
        replay_folder: Path to the replay folder
        output_dir: Directory where output videos will be saved
        fps: Frames per second for output videos
        use_normals: Whether to use normal maps for shading
        use_depth: Whether to use depth maps for shading
    
    Returns:
        List of tuples containing (video_name, success_status)
    """
    results = []
    folder_name = replay_folder.name
    
    # Define the segmentation directory to process
    seg_path = 'state/segmentation/robot.front_camera.left.segmentation_image'
    seg_dir = replay_folder / seg_path
    
    if not seg_dir.exists():
        print(f"  Warning: Segmentation directory not found: {seg_dir}")
        output_filename = f'{folder_name}_segmentation.mp4'
        results.append((output_filename, False))
        return results
    
    # Optionally set up normals and depth directories
    normals_dir = None
    if use_normals:
        normals_path = 'state/common'
        normals_dir = replay_folder / normals_path
        if not normals_dir.exists():
            print(f"  Warning: Normals directory not found: {normals_dir}, proceeding without normals")
            normals_dir = None
    
    depth_dir = None
    if use_depth:
        depth_path = 'state/depth/robot.front_camera.left.depth_image'
        depth_dir = replay_folder / depth_path
        if not depth_dir.exists():
            print(f"  Warning: Depth directory not found: {depth_dir}, proceeding without depth")
            depth_dir = None
    
    output_filename = f'{folder_name}_segmentation.mp4'
    output_path = output_dir / output_filename
    
    print(f"  Processing: {seg_path}")
    if normals_dir:
        print(f"    Using normals from: state/normals/robot.front_camera.left.normals_image")
    if depth_dir:
        print(f"    Using depth from: state/depth/robot.front_camera.left.depth_image")
    
    success = convert_segmentations_to_video(seg_dir, output_path, fps, normals_dir, depth_dir)
    results.append((output_filename, success))
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch convert segmentation image sequences to colorized videos from replay folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 batch_segmentations_to_video.py ~/MobilityGenData/replays
  python3 batch_segmentations_to_video.py ~/MobilityGenData/replays --output-dir /path/to/videos
  python3 batch_segmentations_to_video.py ~/MobilityGenData/replays --fps 60
  python3 batch_segmentations_to_video.py ~/MobilityGenData/replays --use-normals --use-depth
        """
    )
    parser.add_argument('replays_dir', type=str, 
                       help='Base directory containing replay folders')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory where output videos will be saved (default: replays_dir/videos)')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Frames per second for output videos (default: 30)')
    parser.add_argument('--use-normals', action='store_true',
                       help='Use surface normal maps for shading (if available)')
    parser.add_argument('--use-depth', action='store_true',
                       help='Use depth maps for shading (if available)')
    args = parser.parse_args()

    replays_dir = Path(args.replays_dir).expanduser()
    
    if not replays_dir.exists():
        raise ValueError(f"Replays directory not found: {replays_dir}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = replays_dir / 'videos'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"FPS: {args.fps}")
    print(f"Use normals: {args.use_normals}, Use depth: {args.use_depth}\n")
    
    # Get all subdirectories in the replays directory
    replay_folders = sorted([d for d in replays_dir.iterdir() if d.is_dir() and d.name != 'videos'])
    
    if not replay_folders:
        print(f"No subdirectories found in {replays_dir}")
        return
    
    print(f"Found {len(replay_folders)} replay folder(s) to process\n")
    
    # Process each replay folder
    all_results = []
    for i, replay_folder in enumerate(replay_folders, 1):
        print(f"[{i}/{len(replay_folders)}] Processing: {replay_folder.name}")
        results = process_replay_folder(replay_folder, output_dir, args.fps, args.use_normals, args.use_depth)
        all_results.extend(results)
        print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    successful = sum(1 for _, success in all_results if success)
    total = len(all_results)
    print(f"Successfully created: {successful}/{total} videos")
    print(f"Output directory: {output_dir}")
    
    if successful < total:
        print("\nFailed conversions:")
        for video_name, success in all_results:
            if not success:
                print(f"  - {video_name}")


if __name__ == "__main__":
    main()

