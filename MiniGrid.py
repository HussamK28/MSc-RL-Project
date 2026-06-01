import random
import time
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Key, Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace

class MiniGrid(MiniGridEnv):
    def __init__(self, size=12, max_steps=500, **kwargs):
        instructions = MissionSpace(
            mission_func=lambda: "You need to find the key before getting to the goal square."
        )

        super().__init__(
            mission_space=instructions,
            width=size,
            height=size,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=3,
            **kwargs
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        colours = ["red", "blue", "green", "yellow"]
        key_colour = random.choice(colours)
        self.mission = f"To reach the goal, you must have the {key_colour} key!"

        key_cell = self.randomCell()
        self.grid.set(*key_cell, Key(key_colour))

        door = Door(key_colour, is_open=False, is_locked=True)
        door_cell = self.randomCell() 
        self.grid.set(*door_cell, door)

        goal = Goal()
        goal_state = self.randomCell()
        self.grid.set(*goal_state, goal)
        self.place_agent()
    
    def randomCell(self):
        while True:
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            if self.grid.get(x, y) is None:
                return (x, y)


env = MiniGrid(render_mode="human")
num_tests = 10
for episode in range(num_tests):
    print(f"\n=== Episode {episode+1} ===")
    obs, info = env.reset()

    obj_positions = {}
    for x in range(env.width):
        for y in range(env.height):
            obj = env.grid.get(x, y)
            if obj is not None:
                obj_positions[(x, y)] = obj

    env.render()
    time.sleep(2)

print("\n✅ Testing completed. Check positions and randomization visually.")
env.close()