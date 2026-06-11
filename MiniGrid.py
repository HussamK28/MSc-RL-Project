import random
import time
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Key, Goal, Wall, Ball
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace

class MiniGrid(MiniGridEnv):
    def __init__(self, size=24, max_steps=800, **kwargs):
        instructions = MissionSpace(
            mission_func=lambda: "You need to find the key before getting to the goal square."
        )

        super().__init__(
            mission_space=instructions,
            width=size,
            height=size,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=7,
            **kwargs
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        wall1 = width // 3
        wall2 = (2 * width) // 3
        for y_pos in range(1, height-1):
            self.grid.set(wall1, y_pos, Wall())
            self.grid.set(wall2, y_pos, Wall())

        colours = ["red", "blue", "green", "yellow"]
        key1_colour = random.choice(colours)
        key2_colour = random.choice([c for c in colours if c != key1_colour])

        self.mission =(f"To reach the goal, you must first have the {key1_colour} key to unlock door 1!"
        f"You then need to get the {key2_colour} key to unlock door 2 and reach the end goal!")

        door1_pos = random.randint(2, height-3)
        door1 = Door(key1_colour, is_open=False, is_locked=True)
        self.grid.set(wall1, door1_pos, door1)

        door2_pos = random.randint(2, height-3)
        door2 = Door(key2_colour, is_open=False, is_locked=True)
        self.grid.set(wall2, door2_pos, door2)

        key1_x = random.randint(1, wall1-1)
        key1_y = random.randint(1, height-2)
        self.grid.set(key1_x, key1_y, Key(key1_colour))

        key2_x = random.randint(wall1+1, wall2-1)
        key2_y = random.randint(1, height-2)
        self.grid.set(key2_x, key2_y, Key(key2_colour))

        goal_x = random.randint(wall2 + 1, width - 2)
        goal_y = random.randint(1, height - 2)
        self.grid.set(goal_x, goal_y, Goal())

        self.wall1 = wall1
        self.wall2 = wall2

        self.door1_pos = door1_pos
        self.door2_pos = door2_pos

        self.key1_colour = key1_colour
        self.key2_colour = key2_colour

        self.place_agent(top=(1,1), size=(wall1-1, height-2))

        self.npc_pos = self.random_npc_pos()


    def random_npc_pos(self):
        while True:
            npc_x = random.randint(1, self.width - 2)
            npc_y = random.randint(1, self.height - 2)

            if self.grid.get(npc_x, npc_y) is None and (npc_x, npc_y) != tuple(self.agent_pos):
                self.grid.set(npc_x, npc_y, Ball("purple"))
                return (npc_x, npc_y)

        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        old_npc_x, old_npc_y = self.npc_pos
        self.grid.set(old_npc_x, old_npc_y, None)
        while True:
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)

            if self.grid.get(x, y) is None and (x, y) != tuple(self.agent_pos):
                self.grid.set(x, y, Ball("purple"))
                self.npc_pos = (x, y)
                break

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = MiniGrid(render_mode="human")
    num_tests = 10

    for episode in range(num_tests):
        print(f"\n=== Episode {episode+1} ===")
        obs, info = env.reset()
        env.render()
        time.sleep(2)

    print("\n✅ Testing completed. Check positions and randomization visually.")
    env.close()