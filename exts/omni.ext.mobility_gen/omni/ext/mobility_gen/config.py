import json
from typing import Literal, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class OccupancyMapConfig:
    prim_path: Optional[str] = None
    cell_size: Optional[float] = None
    origin: Optional[Tuple[float, float, float]] = None
    lower_bound: Optional[Tuple[float, float, float]] = None
    upper_bound: Optional[Tuple[float, float, float]] = None


@dataclass
class Config:
    scenario_type: str
    robot_type: str
    scene_usd: str
    occupancy_map_config: Optional[OccupancyMapConfig] = None

    def to_json(self):
        return json.dumps(asdict(self), indent=2)
    
    @staticmethod
    def from_json(data: str):
        data = json.loads(data)
        data['occupancy_map_config'] = OccupancyMapConfig(**data['occupancy_map_config'])
        return Config(**data)