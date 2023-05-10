import torch
from torch.utils.data import Dataset
import gym
import numpy as np
import pandas as pd
import pickle

from prueba_env import env_va
import operative_functions as asf


#in the the D4RL datasets the dictionary contains: obss, actions, returns, done_idxs, rtg, timesteps

def generate_data():
    env  = env_va()
    obs = env.reset()

    obs_list = []
    action_list = []
    reward_list = []
    done_list = []
    info_list = []
    jobs_list = []
    current_time_step_list = []
    costcoils_list = []
    plants_df_list = []
    orders_df_list = []

    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        obs_list.append(obs)
        action_list.append(action)
        reward_list.append(reward)
        done_list.append(done)
        info_list.append(info)
        jobs_list.append(env.jobs)
        current_time_step_list.append(env.current_time_step)
        costcoils_list.append(env.costcoils)
        plants_df_list.append(env.plants_df)
        orders_df_list.append(env.orders_df)

        if done:
            obs = env.reset()

    data = {"obs": obs_list,
            "action": action_list,
            "reward": reward_list,
            "done": done_list,
            "info": info_list,
            "jobs": jobs_list,
            "current_time_step": current_time_step_list,
            "costcoils": costcoils_list,
            "plants_df": plants_df_list,
            "orders_df": orders_df_list}

    with open('va_data.pkl', 'wb') as f:
        pickle.dump(data, f)