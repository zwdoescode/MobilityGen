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

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.stage import open_stage
import isaacsim.core.api.objects as objects
from isaacsim.core.utils.stage import add_reference_to_stage


from omni.ext.mobility_gen.occupancy_map import OccupancyMap
from omni.ext.mobility_gen.config import Config, OccupancyMapConfig
from omni.ext.mobility_gen.utils.occupancy_map_utils import occupancy_map_generate_from_prim_async, compute_occupancy_bounds_from_prim_path
from omni.ext.mobility_gen.utils.global_utils import new_stage, new_world, set_viewport_camera
from omni.ext.mobility_gen.scenarios import Scenario, SCENARIOS
from omni.ext.mobility_gen.robots import ROBOTS
from omni.ext.mobility_gen.reader import Reader



def load_scenario(path: str) -> Scenario:
    reader = Reader(path)
    config = reader.read_config()
    robot_type = ROBOTS.get(config.robot_type)
    scenario_type = SCENARIOS.get(config.scenario_type)
    open_stage(os.path.join(path, "stage.usd"))
    prim_utils.delete_prim("/World/robot")
    new_world(physics_dt=robot_type.physics_dt)
    occupancy_map = reader.read_occupancy_map()
    robot = robot_type.build("/World/robot")
    chase_camera_path = robot.build_chase_camera()
    set_viewport_camera(chase_camera_path)
    robot_type = ROBOTS.get(config.robot_type)
    occupancy_map = OccupancyMap.from_ros_yaml(
        ros_yaml_path=os.path.join(path, "occupancy_map", "map.yaml")
    )
    scenario = scenario_type.from_robot_occupancy_map(robot, occupancy_map)
    return scenario


async def build_scenario_from_config(config: Config):
    robot_type = ROBOTS.get(config.robot_type)
    scenario_type = SCENARIOS.get(config.scenario_type)
    new_stage()
    world = new_world(physics_dt=robot_type.physics_dt)
    await world.initialize_simulation_context_async()
    add_reference_to_stage(config.scene_usd,"/World/scene")
    objects.GroundPlane("/World/ground_plane", visible=False)
    robot = robot_type.build("/World/robot")


    if config.occupancy_map_config is None:
        config.occupancy_map_config = OccupancyMapConfig()

    if config.occupancy_map_config.z_min is None:
        config.occupancy_map_config.z_min = robot.occupancy_map_z_min
    
    if config.occupancy_map_config.z_max is None:
        config.occupancy_map_config.z_max = robot.occupancy_map_z_max

    if config.occupancy_map_config.cell_size is None:
        config.occupancy_map_config.cell_size = robot.occupancy_map_cell_size

    if config.occupancy_map_config.origin is None \
            or config.occupancy_map_config.lower_bound is None \
            or config.occupancy_map_config.upper_bound is None:

        origin, lower_bound, upper_bound = compute_occupancy_bounds_from_prim_path(
            prim_path="/World/scene",
            z_min=config.occupancy_map_config.z_min,
            z_max=config.occupancy_map_config.z_max,
            cell_size=config.occupancy_map_config.cell_size
        )
        config.occupancy_map_config.origin = origin
        config.occupancy_map_config.lower_bound = lower_bound
        config.occupancy_map_config.upper_bound = upper_bound


    occupancy_map = await occupancy_map_generate_from_prim_async(
        origin=config.occupancy_map_config.origin,
        lower_bound=config.occupancy_map_config.lower_bound,
        upper_bound=config.occupancy_map_config.upper_bound,
        cell_size=config.occupancy_map_config.cell_size
    )

    chase_camera_path = robot.build_chase_camera()
    set_viewport_camera(chase_camera_path)
    scenario = scenario_type.from_robot_occupancy_map(robot, occupancy_map)
    return scenario