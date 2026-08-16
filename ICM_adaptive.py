# Import the necessary libraries for my experiments
import os
import pickle
from datetime import datetime

# Imports the PPO algorithm and a callback mechanism for the metrics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Imports the gym and MiniGrid environments
from MiniGrid import MiniGrid
from gymnasium.wrappers import FilterObservation, FlattenObservation
from collections import deque

# Annealing reduces influence of intrinsic motivation as training progress
class IntrinsicAnnealingCallback(BaseCallback):
    def __init__(self, total_timesteps, start=1.0, end=0.15):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.start = start
        self.end = end
    
    def _on_step(self):
        # Calculate fraction of timesteps completed
        fraction = min(self.num_timesteps / self.total_timesteps, 1.0)
        # Scale the intrinsic motivation according to the fraction
        scale = (self.start + fraction * (self.end - self.start))
        self.training_env.envs[0].intrinsic_reward_scale = scale
        return True

# This class provides a callback mechanism so that the metrics described below can be completed at the end of each episode
class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        # This history dictionary stores every metric to be recoreded by the program
        self.history = {"return": [],
                        "success": [],
                        "intrinsic_reward": [],
                        "state_coverage": [], 
                        "extrinsic_return": [], 
                        "key1": [], 
                        "door1": [], 
                        "key2": [], 
                        "door2": [], 
                        "door1_with_key": [],
                        "mean_prediction_error": [],
                        "mean_learning_progress":[],
                        "mean_fast_pred_error":[],
                        "mean_slow_pred_error":[],
                        "positive_lp_fraction": [],
                        "door1_faced_with_key": [],
                        "door2_with_key":[],
                        "door2_faced_with_key": [],
                        "final_room_entered": [],
                        "goal_reached": [],
                        }
    # At the end of every episode, we pop each sub-array from completed_episodes and append the history dictionary with the metrics
    def _on_step(self):
        env = self.training_env.envs[0]

        while len(env.completed_episodes) > 0:
            ep = env.completed_episodes.pop(0)
            for k in self.history:
                self.history[k].append(ep[k])

        return True

# Defines the ICM module
class ICM(nn.Module):
    def __init__(self, observation_dim, action_dim, feature_dim=128):
        super().__init__()
        self.action_dim = action_dim
        # The encoder used to turn the agent’s observation into something to be 
        # learned from such as the action and what should happen in the next state
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.Tanh()
        )
        # The inverse model tries to figure out what action 
        # led to this new state. 
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim*2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        # The forward model works out what action to do next. 
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, observation, next_observation, action):
        # Put the current and next observation through encoder
        phi = self.encoder(observation)
        next_phi = self.encoder(next_observation)
        # Concatenated these observations and send it through inverse model
        inverse_input = torch.cat([phi, next_phi], dim=1)
        predicted_action = self.inverse_model(inverse_input)

        # Encode each action through one hot encoding
        # Concatenate that with current observation
        # Predict the next observation through forward model
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        forward_input = torch.cat([phi, action_onehot], dim=1)
        predicted_next_phi = self.forward_model(forward_input)
        
        # Calculate predicted error and inverse loss and mean forward loss
        forward_error = F.mse_loss(predicted_next_phi, next_phi.detach(), reduction="none").mean(dim=1)
        inverse_loss = F.cross_entropy(predicted_action, action)
        forward_loss = forward_error.mean()

        # ICM loss has 80% weighting
        beta = 0.2
        icm_loss = ((1.0 - beta) * inverse_loss + beta * forward_loss)

        return (forward_error.detach(), icm_loss, forward_loss.detach(), inverse_loss.detach())

# This MetricsWrapper class stores information about each metric that will be recorded at the end of each episode
class MetricsWrapper(gym.Wrapper):
    def __init__(self, env, icm, icm_optimiser, device):
        super().__init__(env)
    # Define ICM, its optimiser, device, previous observations, intrinsic rewards
        self.icm = icm
        self.icm_optimiser = icm_optimiser
        self.device = device
        self.previous_observations = None
        self.intrinsic_reward_scale = 1.0
    # Define short term and long term learning progress and error averages
        self.learning_progress_fast = 0.01
        self.learning_progress_slow = 0.001
        self.fast_pred_error = None
        self.slow_pred_error = None
        self.learning_progress_clip = 0.1
        # The per episode trajectory and all trajectories are stored in separate arrays.
        # Also a heatmap for each seed is also created from a zero-matrix
        self.episode_trajectory = []
        self.all_trajectories = []
        self.visit_heatmap = np.zeros((env.unwrapped.height, env.unwrapped.width))

        # These variables will store data such as return, 
        # intrinsic/extrinsic reward, success from each episode
        self.episode_return = 0
        self.episode_intrinsic_reward = 0
        self.episode_success = 0
        self.episode_extrinsic_return = 0
        self.episode_states = set()
        # Completed_episodes will store all the metrics data collected per episode before sending them to the history dictionary
        self.completed_episodes = []

        # These variables below will store the number of times a particular sub-goal has been achieved
        self.key1_reached = 0
        self.door1_opened = 0
        self.key2_reached = 0
        self.door2_opened = 0

        self.door1_reached_with_key = 0
        self.door1_faced_with_key = 0
        self.door2_reached_with_key = 0
        self.door2_faced_with_key = 0
        self.final_room_entered = 0
        self.goal_reached = 0
        # Discount factor rate
        self.door_shaping_gamma = 0.995

        # These are base scaling rewards for each stage
        # Including a completion bonus upon the end of each stage
        self.base_door1_reward_scale = 0.0025
        self.door1_completion_bonus = 0.05
        self.base_door2_reward_scale = 0.0025
        self.door2_completion_bonus = 0.05
        self.base_entry_reward_scale = 0.0015
        self.base_goal_reward_scale = 0.0015
        self.final_room_entry_bonus = 0.02

        # Base reward scales that increases during bottleneck
        self.door1_reward_scale = self.base_door1_reward_scale
        self.door2_reward_scale = self.base_door2_reward_scale
        self.entry_reward_scale = self.base_entry_reward_scale
        self.goal_reward_scale = self.base_goal_reward_scale

        # Defines the number of recent episodes and scaling for bottleneck stages
        self.bottleneck_window = 100
        self.bottleneck_scale = 2.0
        # Creates a queue of the last 100 episodes for each major stage
        # where bottleneck is present
        self.bottleneck_history = {
            "door1": deque(maxlen=self.bottleneck_window),
            "door2": deque(maxlen=self.bottleneck_window),
            "final_room": deque(maxlen=self.bottleneck_window),
            "goal": deque(maxlen=self.bottleneck_window),
        }
        self.current_bottleneck = None

        self.reset_episode_metrics()

    # This function calculates the learning progress
    def calculate_learning_progress(self, prediction_error):
        # Converts prediction error into a float
        prediction_error = float(prediction_error)
        # Defines both long/short term prediction errors
        if self.fast_pred_error is None:
            self.fast_pred_error = prediction_error
            self.slow_pred_error = prediction_error
            return 0.0
        # Calculates prediction errors by multiplying learning progress by prediction error 
        # and the difficulty rating times prediction error
        self.fast_pred_error = (self.learning_progress_fast * prediction_error + (1.0 - self.learning_progress_fast) * self.fast_pred_error)
        self.slow_pred_error = (self.learning_progress_slow * prediction_error + (1.0 - self.learning_progress_slow) * self.slow_pred_error)

        # Finds difference between prediction errors and returns the value 
        # unless if its negative and then it would return 0
        learning_progress = self.slow_pred_error - self.fast_pred_error
        learning_progress = max(learning_progress, 0.0)

        # Clips LP between 0-0.1
        return float(np.clip(learning_progress,0.0, self.learning_progress_clip))

    # In this function, all the variables defined in the initialisation function are reset
    def reset_episode_metrics(self):
        self.episode_return = 0
        self.episode_intrinsic_reward = 0
        self.episode_success = 0
        self.episode_extrinsic_return = 0
        self.episode_steps = 0
        self.episode_states = set()
        # Set these boolean flags to False
        self.ep_key1 = False
        self.ep_door1 = False
        self.ep_key2 = False
        self.ep_door2 = False

        self.ep_door1_reached_with_key = False
        self.previous_door1_distance = None
        self.ep_door1_faced_with_key = False

        self.ep_door2_reached_with_key = False
        self.previous_door2_distance = None
        self.ep_door2_faced_with_key = False

        self.ep_final_room_entered = False
        self.ep_goal_reached = False
        self.previous_entry_distance = None
        self.previous_goal_distance = None

        # Define arrays
        self.episode_prediction_errors = []
        self.episode_learning_progress = []
        self.episode_fast_errors = []
        self.episode_slow_errors = []
        self.episode_intrinsic_scales = []


    def reset(self, **kwargs):
        # Reset environment
        obs, info = self.env.reset(**kwargs)
        # Normalise previous observations
        self.previous_observations = self.normalise_observations(obs)
        self.reset_episode_metrics()
        # This extracts the agent's starting position and adds it to heatmap
        x, y = self.unwrapped.agent_pos
        self.episode_trajectory = [(x, y)]
        self.visit_heatmap[y, x] += 1

        self.episode_states.add(self.state_key())
        return obs, info

    def state_key(self):
    # Retrieves the agent's current position and whether it is carrying an object
        x, y = self.unwrapped.agent_pos
        carrying = self.unwrapped.carrying
        carried_object = None
        # Retrieves the object type and colour
        if carrying is not None:
            carried_object = (carrying.type, carrying.color)
        # Returns the positioning, direction, object and the status of each door to the state coverage heatmap 
        return int(x), int(y), int(self.unwrapped.agent_dir), carried_object, bool(self.ep_door1), bool(self.ep_door2)
    # Checks if carrying key whose colour matches key1
    def check_if_carrying_key1(self):
        carrying = self.unwrapped.carrying
        return (carrying is not None and carrying.type == "key" and carrying.color == self.unwrapped.key1_colour)
    # calculates distance between agent and door 1
    def distance_to_door1(self):
        agent_x, agent_y = self.unwrapped.agent_pos
        door_x = self.unwrapped.wall1
        door_y = self.unwrapped.door1_pos
        
        return (abs(int(agent_x) - int(door_x)) + abs(int(agent_y) - int(door_y)))

    # Checks if carrying key whose colour matches key2
    def check_if_carrying_key2(self):
        carrying = self.unwrapped.carrying
        return (carrying is not None and carrying.type == "key" and carrying.color == self.unwrapped.key2_colour)
    # calculates distance between agent and door 2
    def distance_to_door2(self):
        agent_x, agent_y = self.unwrapped.agent_pos
        door_x = self.unwrapped.wall2
        door_y = self.unwrapped.door2_pos
        return (abs(int(agent_x) - int(door_x)) + abs(int(agent_y) - int(door_y)))
    # calculates distance between agent and goal
    def distance_to_goal(self):
        agent_x, agent_y = self.unwrapped.agent_pos
        goal_x, goal_y = self.unwrapped.goal_pos
        return (abs(int(agent_x) - int(goal_x)) + abs(int(agent_y) - int(goal_y)))
    # calculates distance between agent and final room
    def distance_to_final_room(self):
        agent_x, agent_y = self.unwrapped.agent_pos
        entry_x = int(self.unwrapped.wall2) + 1
        entry_y = int(self.unwrapped.door2_pos)
        return (abs(int(agent_x) - entry_x) + abs(int(agent_y) - entry_y))

        # This function updates each boolean flag if a sub-goal is completed
    def update_subgoal_metrics(self):
        # Extract the grid and whether agent is carrying something
        grid = self.unwrapped.grid
        carrying = self.unwrapped.carrying
        # If agent is carrying a key that matches the colour of key 1, update the key 1 flag
        if (carrying is not None and carrying.type == "key" and carrying.color == self.unwrapped.key1_colour):
            self.ep_key1 = True
        # If door 1 is open, we update that boolean flag
        door1 = grid.get(self.unwrapped.wall1, self.unwrapped.door1_pos)
        if door1 is not None and door1.is_open:
            self.ep_door1 = True
        # If agent is carrying a key that matches the colour of key 2, update the key 2 flag
        if (carrying is not None and carrying.type == "key" and carrying.color == self.unwrapped.key2_colour):
            self.ep_key2 = True
        # If door 1 is open, we update that boolean flag
        door2 = grid.get(self.unwrapped.wall2, self.unwrapped.door2_pos)
        if door2 is not None and door2.is_open:
            self.ep_door2 = True
                # Gets the agent's current position and uses the Manhattan distance to calculate distance between agent and door1
        agent_x, agent_y = self.unwrapped.agent_pos

        door1_distance = (abs(int(agent_x) - int(self.unwrapped.wall1)) + abs(int(agent_y) - int(self.unwrapped.door1_pos)))
        # Checks to see if agent reaches door1 with key1
        carrying_key1 = (carrying is not None and carrying.type == "key" and carrying.color == self.unwrapped.key1_colour)

        if carrying_key1 and door1_distance == 1:
            self.ep_door1_reached_with_key = True

        front_x, front_y = self.unwrapped.front_pos

        if (carrying_key1 and int(front_x) == int(self.unwrapped.wall1) and int(front_y) == int(self.unwrapped.door1_pos)):
            self.ep_door1_faced_with_key = True
        
        door2_distance = (abs(int(agent_x) - int(self.unwrapped.wall2)) + abs(int(agent_y) - int(self.unwrapped.door2_pos)))
        # Checks to see if agent reaches door1 with key1
        carrying_key2 = (carrying is not None and carrying.type == "key" and carrying.color == self.unwrapped.key2_colour)

        if carrying_key2 and door2_distance == 1:
            self.ep_door2_reached_with_key = True

        
        if (carrying_key2 and int(front_x) == int(self.unwrapped.wall2) and int(front_y) == int(self.unwrapped.door2_pos)):
            self.ep_door2_faced_with_key = True
       # If door 2 is opened and agent as passed it, it has entered final room 
        if (self.ep_door2 and int(agent_x) > int(self.unwrapped.wall2)):
            self.ep_final_room_entered = True
    # Check to see the current status of doors
    def step(self, action):
        door1_was_open = self.ep_door1
        door2_was_open = self.ep_door2
        final_room_was_entered = self.ep_final_room_entered

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.update_subgoal_metrics()

        # Check whether the doors were just opened on this timestep
        # Award completion bonuses if true
        door1_just_opened = (not door1_was_open and self.ep_door1)
        door1_completion_bonus = (self.door1_completion_bonus if door1_just_opened else 0.0)
        door2_just_opened = (not door2_was_open and self.ep_door2)
        door2_completion_bonus = (self.door2_completion_bonus if door2_just_opened else 0.0)
        final_room_just_entered = (not final_room_was_entered and self.ep_final_room_entered)
        final_room_entry_bonus = (self.final_room_entry_bonus if final_room_just_entered else 0.0)


        # Add states to trajectory and heatmap
        self.episode_trajectory.append((int(x), int(y)))
        self.visit_heatmap[y, x] += 1
        # Normalise next observations
        normalised_next_obs = self.normalise_observations(obs)
        # Put current and next observations and actions into tensor
        observation_tensor = torch.tensor(self.previous_observations, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_observation_tensor = torch.tensor(normalised_next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        # Pass variables through ICM NN
        (prediction_error_tensor, icm_loss, forward_loss, inverse_loss) = self.icm(
            observation_tensor,
            next_observation_tensor,
            action_tensor
        )


        # Clear gradients from previous updates
        self.icm_optimiser.zero_grad()
        # Backpropagate forward and ICM loss
        icm_loss.backward()
        # Clip gradients to reduce larger updates
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), max_norm=0.5)
        # Update ICM NN
        self.icm_optimiser.step()
        # Calculate Prediction error & Learning progress
        prediction_error = float(prediction_error_tensor.item())
        
        learning_progress = self.calculate_learning_progress(prediction_error)
        scaled_prediction_error = float(np.clip(prediction_error, 0.0, 0.1))

        # Reward shaping to help motivate the agent
        door1_progress_reward = 0.0
        # If carrying key 1 we check the distance 
        # the closer the agent is to the door the higher the bonus is
        if self.check_if_carrying_key1():
            current_distance = self.distance_to_door1()
            if self.previous_door1_distance is not None:
                previous_potential = -float(self.previous_door1_distance)
                current_potential = -float(current_distance)
                door1_progress_reward = (self.door_shaping_gamma * current_potential - previous_potential)
            self.previous_door1_distance = current_distance
        else:
            self.previous_door1_distance = None
        
        door2_progress_reward = 0.0
        if self.check_if_carrying_key2():
            current_distance = self.distance_to_door2()
            if self.previous_door2_distance is not None:
                previous_potential = -float(self.previous_door2_distance)
                current_potential = -float(current_distance)
                door2_progress_reward = (self.door_shaping_gamma * current_potential - previous_potential)
            self.previous_door2_distance = current_distance
        else:
            self.previous_door2_distance = None
        
        entry_progress_reward = 0.0
        goal_progress_reward = 0.0

        if self.ep_door2 and not self.ep_final_room_entered:
            current_entry_distance = (self.distance_to_final_room())

            if self.previous_entry_distance is not None:
                previous_potential = -float(self.previous_entry_distance)

                current_potential = -float(current_entry_distance)
                entry_progress_reward = (self.door_shaping_gamma * current_potential - previous_potential)
            self.previous_entry_distance = (current_entry_distance)

            self.previous_goal_distance = None

        elif self.ep_final_room_entered:
            current_goal_distance = (self.distance_to_goal())

            if self.previous_goal_distance is not None:
                previous_potential = -float(self.previous_goal_distance)
                current_potential = -float(current_goal_distance)

                goal_progress_reward = (self.door_shaping_gamma * current_potential - previous_potential)
            self.previous_goal_distance = (current_goal_distance)

            self.previous_entry_distance = None

        else:
            self.previous_entry_distance = None
            self.previous_goal_distance = None
        # clip the learning progress between 0 and 1
        scaled_learning_progress = float(np.clip(learning_progress, 0.0, 0.1))
        # Give 80% weighting to LP in comparison to error weight and use that to calculate hybrid reward
        prediction_error_weight = 0.2
        learning_progress_weight = 0.8
        
        hybrid_reward = (prediction_error_weight * scaled_prediction_error + learning_progress_weight * scaled_learning_progress)
        # Using the annealing formula from earlier multiply it by hybrid reward
        intrinsic_reward = hybrid_reward * self.intrinsic_reward_scale
        # Scale each stage reward by the adaptive reward bonus scheme
        scaled_door1_reward = (self.door1_reward_scale * door1_progress_reward)
        scaled_door2_reward = (self.door2_reward_scale * door2_progress_reward)
        scaled_entry_reward = (self.entry_reward_scale * entry_progress_reward)
        scaled_goal_reward = (self.goal_reward_scale * goal_progress_reward)
        curiousity_reward = float(np.clip(intrinsic_reward, 0.0, 0.10))
        # Calculate total shaping reward and add it to curiosity reward
        shaping_reward = (scaled_door1_reward + door1_completion_bonus + scaled_door2_reward + door2_completion_bonus + scaled_entry_reward + scaled_goal_reward + final_room_entry_bonus)
        total_intrinsic_reward = float(np.clip(curiousity_reward + shaping_reward, -0.03, 0.15))
        extrinsic_reward = float(reward)
        total_reward = total_intrinsic_reward + extrinsic_reward

        info["intrinsic_reward"] = total_intrinsic_reward
        info["hybrid_reward"] = hybrid_reward

        # Increment per episode variables 
        self.episode_return += total_reward
        self.episode_intrinsic_reward += total_intrinsic_reward
        self.episode_extrinsic_return += extrinsic_reward
        self.episode_steps += 1

        self.episode_states.add(self.state_key())

        # AAdd per episode PE and LP to arrays
        self.episode_prediction_errors.append(prediction_error)
        self.episode_learning_progress.append(learning_progress)
        self.episode_fast_errors.append(float(self.fast_pred_error))
        self.episode_slow_errors.append(float(self.slow_pred_error))
        self.episode_intrinsic_scales.append(self.intrinsic_reward_scale)

        if reward > 0:
            self.episode_success = 1
            self.ep_goal_reached = True


        done = terminated or truncated
        # If done add to completed episodes array
        if done:
            self.all_trajectories.append(self.episode_trajectory.copy())
            self.completed_episodes.append({
                "return": self.episode_return,
                "success": self.episode_success,
                "intrinsic_reward": self.episode_intrinsic_reward,
                "state_coverage": len(self.episode_states),
                "extrinsic_return": self.episode_extrinsic_return,
                "key1": int(self.ep_key1),
                "door1": int(self.ep_door1),
                "key2": int(self.ep_key2),
                "door2": int(self.ep_door2),
                "door1_with_key": int(self.ep_door1_reached_with_key),
                "mean_prediction_error": (float(np.mean(self.episode_prediction_errors)) if self.episode_prediction_errors else 0.0),
                "mean_learning_progress": (float(np.mean(self.episode_learning_progress)) if self.episode_learning_progress else 0.0),
                "mean_fast_pred_error": (float(np.mean(self.episode_fast_errors)) if self.episode_fast_errors else 0.0),
                "mean_slow_pred_error": (float(np.mean(self.episode_slow_errors)) if self.episode_slow_errors else 0.0),
                "door1_faced_with_key": int(self.ep_door1_faced_with_key),
                "door2_with_key": int(self.ep_door2_reached_with_key),
                "door2_faced_with_key": int(self.ep_door2_faced_with_key),
                "final_room_entered": int(self.ep_final_room_entered),
                "goal_reached": int(self.ep_goal_reached),
    })


            self.key1_reached += int(self.ep_key1)
            self.door1_opened += int(self.ep_door1)
            self.key2_reached += int(self.ep_key2)
            self.door2_opened += int(self.ep_door2)
            self.door1_reached_with_key += int(self.ep_door1_reached_with_key)
            self.door1_faced_with_key += int(self.ep_door1_faced_with_key)
            self.door2_reached_with_key += int(self.ep_door2_reached_with_key)
            self.door2_faced_with_key += int(self.ep_door2_faced_with_key)
            self.final_room_entered += int(self.ep_final_room_entered)
            self.goal_reached += int(self.ep_goal_reached)

            if self.ep_key1:
                self.bottleneck_history["door1"].append(int(self.ep_door1))


            if self.ep_key2:
                self.bottleneck_history["door2"].append(int(self.ep_door2))
            
            if self.ep_door2:
                self.bottleneck_history["final_room"].append(int(self.ep_final_room_entered))
            
            if self.ep_final_room_entered:
                self.bottleneck_history["goal"].append(int(self.ep_goal_reached))

            self.update_bottleneck_rewards()

        self.previous_observations = self.normalise_observations(obs)
        

        return obs, total_reward, terminated, truncated, info

    def normalise_observations(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return obs / 10.0

    
    def adaptive_bottleneck_scale(self, base_scale, success_rate, bottleneck_scale=2.0):
        difficulty = 1.0 - success_rate
        return base_scale * (1.0 + bottleneck_scale * difficulty)
    
    def update_bottleneck_rewards(self):
        success_rates = {}
        for stage, history in self.bottleneck_history.items():
            if len(history) >= 20:
                success_rates[stage] = float(np.mean(history))
        
        if not success_rates:
            self.current_bottleneck = None
            return
        if all(rate>=0.95 for rate in success_rates.values()):
            self.current_bottleneck = None
        else:
            self.current_bottleneck = min(success_rates, key=success_rates.get)
        self.door1_reward_scale = self.base_door1_reward_scale
        self.door2_reward_scale = self.base_door2_reward_scale
        self.entry_reward_scale = self.base_entry_reward_scale
        self.goal_reward_scale = self.base_goal_reward_scale

        if "door1" in success_rates:
            self.door1_reward_scale = self.adaptive_bottleneck_scale(self.base_door1_reward_scale, success_rates["door1"], self.bottleneck_scale)
        if "door2" in success_rates:
            self.door2_reward_scale = self.adaptive_bottleneck_scale(self.base_door2_reward_scale, success_rates["door2"], self.bottleneck_scale)
        if "final_room" in success_rates:
            self.entry_reward_scale = self.adaptive_bottleneck_scale(self.base_entry_reward_scale, success_rates["final_room"], self.bottleneck_scale)
        if "goal" in success_rates:
            self.goal_reward_scale = self.adaptive_bottleneck_scale(self.base_goal_reward_scale, success_rates["goal"], self.bottleneck_scale)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

temp_env = MiniGrid(size=12, max_steps=400, noisy_tv=False, render_mode=None)
temp_env = FilterObservation(temp_env, ["image", "direction"])
temp_env = FlattenObservation(temp_env)

obs_dim = temp_env.observation_space.shape[0]
action_dim = temp_env.action_space.n

icm = ICM(obs_dim, action_dim).to(device)
icm_optimiser = torch.optim.Adam(icm.parameters(), lr=1e-4)

def make_env(icm, icm_optimiser):
    def _init():
        env = MiniGrid(
            size=12,
            max_steps=400,
            noisy_tv=False,
            render_mode=None
        )

        env = FilterObservation(env,["image", "direction"])

        env = FlattenObservation(env)

        env = MetricsWrapper(
            env,
            icm=icm,
            icm_optimiser=icm_optimiser,
            device=device
        )

        return env

    return _init
    

def mean_last(values, window=100):
    if len(values) == 0:
        return 0.0

    window = min(window, len(values))
    return float(np.mean(values[-window:]))

def conditional_last(numerator_values,denominator_values,window=100):
    window = min(window,len(numerator_values),len(denominator_values))

    if window == 0:
        return 0.0

    top = sum(numerator_values[-window:])

    bottom = sum(denominator_values[-window:])

    if bottom == 0:
        return 0.0

    return float(top / bottom)

seeds = [42, 123, 456]

all_seed_results = []

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

for seed in seeds:
    print(f"\nRunning seed: {seed}")

    set_all_seeds(seed)

    icm = ICM(
        obs_dim,
        action_dim
    ).to(device)

    icm_optimiser = torch.optim.Adam(
        icm.parameters(),
        lr=1e-4
    )
    vec_env = DummyVecEnv([make_env(icm,icm_optimiser)])
    vec_env.seed(seed)
    vec_env.reset()
    

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=2.5e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": {
                "pi": [256, 256],
                "vf": [256, 256]
            }
        },
        verbose=1,
        seed=seed
    )

    total_timesteps = 500_000
    callback = MetricsCallback()
    annealing_callback = IntrinsicAnnealingCallback(total_timesteps=total_timesteps, start=1.0, end=0.15)


    model.learn(total_timesteps=total_timesteps,callback=CallbackList([annealing_callback, callback]))

    env = vec_env.envs[0]
    episodes = len(callback.history["success"])

    seed_result = {
        "seed": seed,
        "episodes": episodes,
        "success_rate": (
            np.mean(callback.history["success"])
            if episodes > 0
            else 0.0
        ),
        "coverage": (
            np.mean(callback.history["state_coverage"])
            if episodes > 0
            else 0.0
        ),
        "last_100_coverage": mean_last(
            callback.history["state_coverage"],
            100
        ),
        "key1_rate": (
            env.key1_reached / episodes
            if episodes > 0
            else 0.0
        ),
        "door1_rate": (
            env.door1_opened / episodes
            if episodes > 0
            else 0.0
        ),
        "door1_with_key_rate": (
            env.door1_reached_with_key / episodes
            if episodes > 0
            else 0.0
        ),
        "p_reach_door1_given_key1": (
            env.door1_reached_with_key
            / env.key1_reached
            if env.key1_reached > 0
            else 0.0
        ),
        "p_open_door1_given_reached": (
            env.door1_opened
            / env.door1_reached_with_key
            if env.door1_reached_with_key > 0
            else 0.0
        ),
        "door1_faced_with_key_rate": (
            env.door1_faced_with_key / episodes
            if episodes > 0
            else 0.0
        ),

        "p_face_door1_given_reached": (
            env.door1_faced_with_key
            / env.door1_reached_with_key
            if env.door1_reached_with_key > 0
            else 0.0
        ),

        "p_open_door1_given_faced": (
            env.door1_opened
            / env.door1_faced_with_key
            if env.door1_faced_with_key > 0
            else 0.0
        ),
        "key1_count": env.key1_reached,
        "door1_reached_count": env.door1_reached_with_key,
        "door1_faced_count": env.door1_faced_with_key,
        "door1_opened_count": env.door1_opened,
        "key2_rate": (
            env.key2_reached / episodes
            if episodes > 0
            else 0.0
        ),
        "door2_rate": (
            env.door2_opened / episodes
            if episodes > 0
            else 0.0
        ),
        "door2_with_key_rate": (
            env.door2_reached_with_key / episodes
            if episodes > 0
            else 0.0
        ),
        "p_reach_door2_given_key2": (
            env.door2_reached_with_key
            / env.key2_reached
            if env.key2_reached > 0
            else 0.0
        ),
        "p_open_door2_given_reached": (
            env.door2_opened
            / env.door2_reached_with_key
            if env.door2_reached_with_key > 0
            else 0.0
        ),
        "door2_faced_with_key_rate": (
            env.door2_faced_with_key / episodes
            if episodes > 0
            else 0.0
        ),

        "p_face_door2_given_reached": (
            env.door2_faced_with_key
            / env.door2_reached_with_key
            if env.door2_reached_with_key > 0
            else 0.0
        ),

        "p_open_door2_given_faced": (
            env.door2_opened
            / env.door2_faced_with_key
            if env.door2_faced_with_key > 0
            else 0.0
        ),
        "key2_count": env.key2_reached,
        "door2_reached_count": env.door2_reached_with_key,
        "door2_faced_count": env.door2_faced_with_key,
        "door2_opened_count": env.door2_opened,
        "final_room_rate": (
            env.final_room_entered / episodes
            if episodes > 0
            else 0.0
        ),

        "goal_rate": (
            env.goal_reached / episodes
            if episodes > 0
            else 0.0
        ),

        "p_enter_final_room_given_door2": (
            env.final_room_entered
            / env.door2_opened
            if env.door2_opened > 0
            else 0.0
        ),

        "p_goal_given_final_room": (
            env.goal_reached
            / env.final_room_entered
            if env.final_room_entered > 0
            else 0.0
        ),

        "p_goal_given_door2": (
            env.goal_reached
            / env.door2_opened
            if env.door2_opened > 0
            else 0.0
        ),

        "final_room_count": env.final_room_entered,
        "goal_count": env.goal_reached,
        "time_to_first_success": (
            callback.history["success"].index(1) + 1
            if 1 in callback.history["success"]
            else np.nan
        ),
    }

    all_seed_results.append(seed_result)
    print("\n--------------------------------")
    print(f"RESULTS FOR SEED {seed}")
    print("--------------------------------")

    file_name = datetime.now().strftime(f"goal_seed_{seed}_run_%Y%m%d_%H%M%S")

    save_dir = os.path.join("results", file_name)
    os.makedirs(save_dir, exist_ok=True)

    returns = callback.history["return"]
    successes = callback.history["success"]
    intrinsic_rewards = callback.history["intrinsic_reward"]
    coverages = callback.history["state_coverage"]

    print("Episodes logged:", len(successes))
    print("Average return:", np.mean(returns))
    print("Success rate:", np.mean(successes) * 100, "%")
    print("Average intrinsic reward:", np.mean(intrinsic_rewards))
    print("Average state coverage:", np.mean(coverages))
    print("Average extrinsic return:", np.mean(callback.history["extrinsic_return"]))
    print("Action dim:", action_dim)

    print("*******************")
    print("Success rate over last 100 episodes: ", mean_last(successes, 100) * 100, "%")
    print("Average coverage over last 100 episodes: ", mean_last(coverages, 100))
    print("Average extrinsic return over last 100 episodes: ", mean_last(callback.history["extrinsic_return"], 100))
    print("*******************")

    if 1 in successes:
        print("Time to first success:", successes.index(1) + 1, "episodes")
    else:
        print("Time to first success: not achieved")

    def convert_to_percentage(top, bottom):
        if bottom == 0:
            return 0.0
        else:
            return 100 * top / bottom

    env = vec_env.envs[0]
    episodes = len(successes)

    if episodes > 0:
        print("Picked up key1:", 100 * env.key1_reached / episodes, "%")
        print("Opened door1:", 100 * env.door1_opened / episodes, "%")
        print("Picked up key2:", 100 * env.key2_reached / episodes, "%")
        print("Opened door2:", 100 * env.door2_opened / episodes, "%")
        print("Reached door1 with key1: ", 100 * env.door1_reached_with_key / episodes, "%")
        print("Faced door1 with key1:",100 * env.door1_faced_with_key / episodes,"%")
        print("P(door1 | key1): ", convert_to_percentage(env.door1_opened, env.key1_reached), "%")
        print("P(key2 | door1): ", convert_to_percentage(env.key2_reached, env.door1_opened), "%")
        print("P(door2 | key2): ", convert_to_percentage(env.door2_opened, env.key2_reached), "%")
        print("P(reach door1 with key1 | key1): ", convert_to_percentage(env.door1_reached_with_key, env.key1_reached), "%")
        print("P(open door1 | reached door1 with key1): ", convert_to_percentage(env.door1_opened, env.door1_reached_with_key), "%")
        print("P(face door1 | reached door1):",convert_to_percentage(env.door1_faced_with_key,env.door1_reached_with_key),"%")
        print("P(open door1 | faced door1):",convert_to_percentage(env.door1_opened,env.door1_faced_with_key),"%")
        print("Average positive LP fraction:",np.mean(callback.history["positive_lp_fraction"]))
        print("Mean prediction error:",np.mean(callback.history["mean_prediction_error"]))
        print("Maximum episode mean prediction error:",np.max(callback.history["mean_prediction_error"]))
        print("Mean learning progress:",np.mean(callback.history["mean_learning_progress"]))
        print("Maximum episode mean learning progress:",np.max(callback.history["mean_learning_progress"]))
        print("Key1 over last 100 episodes:",mean_last(callback.history["key1"], 100) * 100,"%")
        print("Door1 over last 100 episodes:", mean_last(callback.history["door1"], 100) * 100,"%")
        print("Key2 over last 100 episodes:",mean_last(callback.history["key2"], 100) * 100,"%")
        print("Door2 over last 100 episodes:",mean_last(callback.history["door2"], 100) * 100,"%")
        print("Reached door2 with key2: ", 100 * env.door2_reached_with_key / episodes, "%")
        print("Faced door2 with key2:",100 * env.door2_faced_with_key / episodes,"%")
        print("P(reach door2 with key2 | key2): ", convert_to_percentage(env.door2_reached_with_key, env.key2_reached), "%")
        print("P(face door2 | reached door2):",convert_to_percentage(env.door2_faced_with_key,env.door2_reached_with_key),"%")
        print("P(open door2 | faced door2):",convert_to_percentage(env.door2_opened,env.door2_faced_with_key),"%")
        print("Entered final room:", convert_to_percentage(env.final_room_entered, episodes), "%")
        print("Reached Goal:", convert_to_percentage(env.goal_reached, episodes),"%")
        print("P(enter final room | Door2 opened):", convert_to_percentage(env.final_room_entered, env.door2_opened),"%")
        print("P(goal | Final room reached):", convert_to_percentage(env.goal_reached, env.final_room_entered),"%")
        print("P(goal | Door2 opened):", convert_to_percentage(env.goal_reached, env.door2_opened),"%")
        print("Final room over last 100 episodes:", mean_last(callback.history['final_room_entered'], 100) * 100,"%")
        print("Goal over last 100 episodes:",mean_last(callback.history["goal_reached"],100) * 100, "%")
        print("P(enter final room | Door2 opened) ""over last 100:",conditional_last(callback.history["final_room_entered"],callback.history["door2"],100) * 100,"%")
        print("P(goal | final room entered) ""over last 100:",conditional_last(callback.history["goal_reached"],callback.history["final_room_entered"],100) * 100,"%")
    


    trajectory = env.all_trajectories[-1]

    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker="o")
    plt.gca().invert_yaxis()
    plt.title("Agent Trajectory")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.grid(True)
    plt.savefig(
        os.path.join(save_dir, "trajectory_graph.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(env.visit_heatmap)
    plt.colorbar(label="Visit count")
    plt.title("Visited State Heatmap")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.savefig(
        os.path.join(save_dir, "heatmap_chart.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    prediction_errors = callback.history["mean_prediction_error"]
    learning_progress_values = callback.history["mean_learning_progress"]

    plt.figure(figsize=(6,6))
    plt.plot(
        prediction_errors,
        label="Mean prediction error"
    )
    plt.plot(
        learning_progress_values,
        label="Mean learning progress"
    )
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Prediction Error vs Learning Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(
            save_dir,
            "prediction_error_vs_learning_progress.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    plt.figure(figsize=(6,6))
    plt.plot(
        callback.history["mean_fast_pred_error"],
        label="Fast error EMA"
    )
    plt.plot(
        callback.history["mean_slow_pred_error"],
        label="Slow error EMA"
    )
    plt.xlabel("Episode")
    plt.ylabel("Prediction error")
    plt.title("Fast and Slow Prediction-Error Averages")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(
            save_dir,
            "fast_vs_slow_error.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    with open(os.path.join(save_dir, "run_trajectories.pkl"), "wb") as f:
        pickle.dump(env.all_trajectories, f)
    np.save(os.path.join(save_dir, "visit_heatmap.npy"),env.visit_heatmap)
    metrics = {
        "seed": seed,
        "episodes_logged": len(successes),
        "success_rate": np.mean(successes),
        "avg_return": np.mean(returns),
        "avg_intrinsic_reward": np.mean(intrinsic_rewards),
        "avg_state_coverage": np.mean(coverages),
        "avg_extrinsic_return": np.mean(callback.history["extrinsic_return"]),
        "time_to_first_success":
            successes.index(1) + 1 if 1 in successes else None,
        
    }

    with open(os.path.join(save_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    vec_env.close()


print("\n==============================")
print("MULTI-SEED RESULTS")
print("==============================")

metrics_to_summarise = [
    "success_rate",
    "coverage",
    "last_100_coverage",
    "key1_rate",
    "door1_rate",
    "door1_with_key_rate",
    "p_reach_door1_given_key1",
    "p_open_door1_given_reached",
    "door1_faced_with_key_rate",
    "p_face_door1_given_reached",
    "p_open_door1_given_faced",
    "time_to_first_success",
    "key2_rate",
    "door2_rate",
    "door2_with_key_rate",
    "door2_faced_with_key_rate",
    "p_reach_door2_given_key2",
    "p_face_door2_given_reached",
    "p_open_door2_given_faced",
    "p_open_door2_given_reached",
    "final_room_rate",
    "goal_rate",
    "p_enter_final_room_given_door2",
    "p_goal_given_final_room",
    "p_goal_given_door2",
]

for metric in metrics_to_summarise:
    values = np.asarray(
        [result[metric] for result in all_seed_results],
        dtype=np.float32
    )

    if metric == "time_to_first_success":
        mean = np.nanmean(values)
        std = np.nanstd(values)
    else:
        mean = np.mean(values)
        std = np.std(values)

    print(
        f"{metric}: "
        f"{mean:.4f} "
        f"± {std:.4f}"
    )

total_episodes = sum(
    result["episodes"]
    for result in all_seed_results
)

total_key1 = sum(
    result["key1_count"]
    for result in all_seed_results
)

total_door1_reached = sum(
    result["door1_reached_count"]
    for result in all_seed_results
)

total_door1_faced = sum(
    result["door1_faced_count"]
    for result in all_seed_results
)

total_door1_opened = sum(
    result["door1_opened_count"]
    for result in all_seed_results
)

print("\n==============================")
print("POOLED EVENT COUNTS")
print("==============================")

print("Total episodes:", total_episodes)
print("Key1 collected:", total_key1)
print("Door1 reached with Key1:", total_door1_reached)
print("Door1 faced with Key1:", total_door1_faced)
print("Door1 opened:", total_door1_opened)

print(
    "P(reach Door1 | Key1):",
    convert_to_percentage(
        total_door1_reached,
        total_key1
    ),
    "%"
)

print(
    "P(face Door1 | reached):",
    convert_to_percentage(
        total_door1_faced,
        total_door1_reached
    ),
    "%"
)

print(
    "P(open Door1 | faced):",
    convert_to_percentage(
        total_door1_opened,
        total_door1_faced
    ),
    "%"
)

total_key2 = sum(
    result["key2_count"]
    for result in all_seed_results
)

total_door2_reached = sum(
    result["door2_reached_count"]
    for result in all_seed_results
)

total_door2_faced = sum(
    result["door2_faced_count"]
    for result in all_seed_results
)

total_door2_opened = sum(
    result["door2_opened_count"]
    for result in all_seed_results
)

total_final_room = sum(
    result["final_room_count"]
    for result in all_seed_results
)

total_goal = sum(
    result["goal_count"]
    for result in all_seed_results
)

print("Key2 collected:", total_key2)

print(
    "Door2 reached with Key2:",
    total_door2_reached
)

print(
    "Door2 faced with Key2:",
    total_door2_faced
)

print("Door2 opened:", total_door2_opened)

print(
    "P(reach Door2 | Key2):",
    convert_to_percentage(
        total_door2_reached,
        total_key2
    ),
    "%"
)

print(
    "P(face Door2 | reached):",
    convert_to_percentage(
        total_door2_faced,
        total_door2_reached
    ),
    "%"
)

print(
    "P(open Door2 | faced):",
    convert_to_percentage(
        total_door2_opened,
        total_door2_faced
    ),
    "%"
)


print(
    "Final room entered:",
    total_final_room
)

print(
    "Goal reached:",
    total_goal
)

print(
    "P(enter final room | Door2 opened):",
    convert_to_percentage(
        total_final_room,
        total_door2_opened
    ),
    "%"
)

print(
    "P(goal | final room entered):",
    convert_to_percentage(
        total_goal,
        total_final_room
    ),
    "%"
)

print(
    "P(goal | Door2 opened):",
    convert_to_percentage(
        total_goal,
        total_door2_opened
    ),
    "%"
)