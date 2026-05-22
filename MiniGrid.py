import random
import time
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Key, Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace

class MiniGrid(MiniGridEnv):
    def __init__(self, size=8, max_steps=300, **kwargs):
        instructions = MissionSpace(
            mission_func=lambda: "You need to find the key before getting to the goal square."
        )

        super().__init__(
            mission_space=instructions,
            width=size,
            height=size,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=5,
            **kwargs
        )
        