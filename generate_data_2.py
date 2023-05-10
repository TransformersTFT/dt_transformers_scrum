import torch
from torch.utils.data import Dataset
import gym
import numpy as np
import pandas as pd
import pickle

from prueba_env import env_va
import operative_functions as asf



num_trajectories=10
max_trajectory_length=100
obs_dim=10
action_dim=3
traj_len=1000


def generate_dataset():
    # Initialize empty lists for trajectories
    trajectories = []
    env=env_va()

    for i in range(num_trajectories):
        # Initialize empty lists for current trajectory
        obs_list, action_list, reward_list, info_list = [], [], [], []
        done=False
        while not done:
            state=env.reset()
            obs_list.append(state)
        # Generate a random length for the current trajectory
        

            for t in range(traj_len):
                # Generate random observations, actions, and rewards for each timestep
                action_t = env.action_space.sample()
                print("la accion es................................................................",action_t)
                obs_t, reward_t, done, info_t = env.step(action_t)
                #obs_t = np.random.randn(obs_dim)
                #reward_t = np.random.randn()
                #done_t = np.random.randn()

                # Append the generated values to the lists for the current trajectory
                obs_list.append(obs_t)
                action_list.append(action_t)
                reward_list.append(reward_t)
                info_list.append(info_t)
                if done: 
                    break

            # Append the current trajectory to the list of trajectories
            trajectories.append({'observations': obs_list[:-1], 'actions': action_list, 'rewards': reward_list, "done": done,"info": info_list})

    # Save the trajectories as a pickle file
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(trajectories, f)

    return trajectories


if __name__ == '__main__':
    generate_dataset()