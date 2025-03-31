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


import asyncio
import numpy as np
import os
import datetime
import tempfile
import glob

import omni.ext
import omni.ui as ui

from isaacsim.gui.components.ui_utils import (
    btn_builder,
    cb_builder,
    color_picker_builder,
    dropdown_builder,
    float_builder,
    multi_btn_builder,
    xyz_builder,
)

from omni.ext.mobility_gen.utils.global_utils import save_stage
from omni.ext.mobility_gen.writer import Writer
from omni.ext.mobility_gen.inputs import GamepadDriver, KeyboardDriver
from omni.ext.mobility_gen.scenarios import SCENARIOS, Scenario
from omni.ext.mobility_gen.utils.global_utils import get_world
from omni.ext.mobility_gen.robots import ROBOTS
from omni.ext.mobility_gen.config import Config, OccupancyMapConfig
from omni.ext.mobility_gen.build import build_scenario_from_config


if "MOBILITY_GEN_DATA" in os.environ:
    DATA_DIR = os.environ['MOBILITY_GEN_DATA']
else:
    DATA_DIR = os.path.expanduser("~/MobilityGenData")

RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
SCENARIOS_DIR = os.path.join(DATA_DIR, "scenarios")


class MobilityGenExtension(omni.ext.IExt):

    def on_startup(self, ext_id):

        self.keyboard = KeyboardDriver.connect()
        self.gamepad = GamepadDriver.connect()
        self.scenario: Scenario = None
        self.config: Config = None

        self.count = 0

        self.scenario_path: str | None = None
        self.cached_stage_path: str | None = None

        self.writer: Writer | None = None
        self.step: int = 0
        self.is_recording: bool = False
        self.recording_enabled: bool = False
        self.recording_time: float = 0.

        self._occupancy_map_image_provider = omni.ui.ByteImageProvider()

        self._visualize_window = omni.ui.Window("MobilityGen - Occupancy Map", width=300, height=300)
        with self._visualize_window.frame:
            self._occ_map_frame = ui.Frame()
            self._occ_map_frame.set_build_fn(self.build_occ_map_frame)
            

        self._teleop_window = omni.ui.Window("MobilityGen", width=300, height=300)

        with self._teleop_window.frame:
            with ui.VStack():
                with ui.VStack():
                    with ui.HStack():
                        ui.Label("USD Path / URL")
                        self.scene_usd_field_string_model = ui.SimpleStringModel()
                        self.scene_usd_field = ui.StringField(model=self.scene_usd_field_string_model, height=25)

                    with ui.HStack():
                        ui.Label("Scenario Type")
                        self.scenario_combo_box = ui.ComboBox(0, *SCENARIOS.names())

                    with ui.HStack():
                        ui.Label("Robot Type")
                        self.robot_combo_box = ui.ComboBox(0, *ROBOTS.names())

                    with ui.VStack():
                        self._omap_override = cb_builder(
                            "Manual occupancy bounds",
                            tooltip="If true, use manually specified occupancy map bounds",
                            default_val=False
                        )
                        self._omap_origin = xyz_builder(label="Origin")
                        self._omap_lower_bound = xyz_builder(label="Lower Bound")
                        self._omap_upper_bound = xyz_builder(label="Upper Bound")
                        
                    ui.Button("Build", clicked_fn=self.build_scenario)

                with ui.VStack():
                    self.recording_count_label = ui.Label("")
                    self.recording_dir_label = ui.Label(f"Output directory: {RECORDINGS_DIR}")
                    self.recording_name_label = ui.Label("")
                    self.recording_step_label = ui.Label("")

                    ui.Button("Reset", clicked_fn=self.reset)
                    with ui.HStack():
                        ui.Button("Start Recording", clicked_fn=self.enable_recording)
                        ui.Button("Stop Recording", clicked_fn=self.disable_recording)

        self.update_recording_count()
        self.clear_recording()

    def build_occ_map_frame(self):
        if self.scenario is not None:
            with ui.VStack():
                image_widget = ui.ImageWithProvider(
                    self._occupancy_map_image_provider
                )

    def draw_occ_map(self):
        if self.scenario is not None:
            image = self.scenario.occupancy_map.ros_image().copy().convert("RGBA")
            data = list(image.tobytes())
            self._occupancy_map_image_provider.set_bytes_data(data, [image.width, image.height])
            self._occ_map_frame.rebuild()


    def update_recording_count(self):
        num_recordings = len(glob.glob(os.path.join(RECORDINGS_DIR, "*")))
        self.recording_count_label.text = f"Number of recordings: {num_recordings}"

    def create_config(self):
        config = Config(
            scenario_type=list(SCENARIOS.names())[self.scenario_combo_box.model.get_item_value_model().get_value_as_int()],
            robot_type=list(ROBOTS.names())[self.robot_combo_box.model.get_item_value_model().get_value_as_int()],
            scene_usd=self.scene_usd_field_string_model.as_string
        )
        return config
    
    def scenario_type(self):
        index = self.scenario_combo_box.model.get_item_value_model().get_value_as_int()
        return SCENARIOS.get_index(index)
    
    def on_shutdown(self):
        self.keyboard.disconnect()
        self.gamepad.disconnect()
        world = get_world()
        world.remove_physics_callback("scenario_physics", self.on_physics)

    def start_new_recording(self):
        recording_name = datetime.datetime.now().isoformat()
        recording_path = os.path.join(RECORDINGS_DIR, recording_name)
        writer = Writer(recording_path)
        writer.write_config(self.config)
        writer.write_occupancy_map(self.scenario.occupancy_map)
        writer.copy_stage(self.cached_stage_path)
        self.step = 0
        self.recording_time = 0.
        self.recording_name_label.text = f"Current recording name: {recording_name}"
        self.recording_step_label.text = f"Current recording duration: {self.recording_time:.2f}s"
        self.writer = writer
        self.update_recording_count()
    
    def clear_recording(self):
        self.writer = None
        self.recording_name_label.text = "Current recording name: "
        self.recording_step_label.text = "Current recording duration: "

    def clear_scenario(self):
        self.scenario = None
        self.cached_stage_path = None

    def enable_recording(self):
        if not self.recording_enabled:
            if self.scenario is not None:
                self.start_new_recording()
            self.recording_enabled = True

    def disable_recording(self):
        self.recording_enabled = False
        self.clear_recording()

    def reset(self):
        self.writer = None
        self.scenario.reset()
        if self.recording_enabled:
            self.start_new_recording()

    def on_physics(self, step_size: int):

        if self.scenario is not None:

            is_alive = self.scenario.step(step_size)

            if not is_alive:
                self.reset()
            
            if self.writer is not None:
                state_dict = self.scenario.state_dict_common()
                self.writer.write_state_dict_common(state_dict, step=self.step)
                self.step += 1
                self.recording_time += step_size
                if self.step % 15 == 0:
                    self.recording_step_label.text = f"Current recording duration: {self.recording_time:.2f}s"

    def get_omap_config(self, robot_type):
        z_min = robot_type.occupancy_map_z_min
        z_max = robot_type.occupancy_map_z_max
        cell_size = robot_type.occupancy_map_cell_size
        omap_config = OccupancyMapConfig(
            origin=(
                self._omap_origin[0].get_value_as_float(),
                self._omap_origin[1].get_value_as_float(),
                0.
            ),
            lower_bound=(
                self._omap_lower_bound[0].get_value_as_float(),
                self._omap_lower_bound[1].get_value_as_float(),
                z_min
            ),
            upper_bound=(
                self._omap_upper_bound[0].get_value_as_float(),
                self._omap_upper_bound[1].get_value_as_float(),
                z_max
            ),
            cell_size=cell_size
        )

        if not self._omap_override.get_value_as_bool():
            omap_config.prim_path = "/World/scene"

        return omap_config
    
    def update_ui_from_config(self):
        omap_cfg: OccupancyMapConfig = self.config.occupancy_map_config
        for i in range(3):
            self._omap_origin[i].set_value(omap_cfg.origin[i])
        for i in range(3):
            self._omap_lower_bound[i].set_value(omap_cfg.lower_bound[i])
        for i in range(3):
            self._omap_upper_bound[i].set_value(omap_cfg.upper_bound[i])

    def build_scenario(self):

        async def _build_scenario_async():
            
            self.clear_recording()
            self.clear_scenario()

            config = self.create_config()
            robot_type = ROBOTS.get(config.robot_type)
            config.occupancy_map_config = self.get_omap_config(robot_type)

            self.scenario, self.config = await build_scenario_from_config(config)

            self.update_ui_from_config()

            self.draw_occ_map()
            
            world = get_world()
            await world.reset_async()

            self.scenario.reset()

            world.add_physics_callback("scenario_physics", self.on_physics)

            # cache stage
            self.cached_stage_path = os.path.join(tempfile.mkdtemp(), "stage.usd")
            save_stage(self.cached_stage_path)

            if self.recording_enabled:
                self.start_new_recording()

            # self.scenario.save(path)

        asyncio.ensure_future(_build_scenario_async())