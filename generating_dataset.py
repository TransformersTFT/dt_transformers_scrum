import torch
from torch.utils.data import Dataset
import gym
import numpy as np
import pandas as pd
import pickle

from prueba_env import env_va
import operative_functions as asf






def generate_data():




    env = env_va()
 

   
    # Set the number of episodes and steps
    n_episodes = 100
    ep_len=200
    state_dim = env.jobs
    action_dim = env.jobs
        # Initialize the dataset
    dataset = {'observations': np.zeros((n_episodes, ep_len, state_dim)),
           'actions': np.zeros((n_episodes, ep_len, action_dim)),
           'rewards': np.zeros((n_episodes, ep_len)),
           'terminals': np.zeros((n_episodes, ep_len)),
           'next_observations': np.zeros((n_episodes, ep_len, state_dim))}

    dataset = {}
    dataset['obs_list'] = []
    dataset['action_list'] = []
    dataset['reward_list'] = []
    dataset['done_list'] = []
    dataset['info_list'] = []
    dataset['jobs_list'] = []
    dataset['current_time_step_list'] = []
    dataset['costcoils_list'] = []
    dataset['plants_df_list'] = []
    dataset['orders_df_list'] = []

    
    # Fill dataset with random data
    for ep in range(n_episodes):
        for t in range(ep_len):
            action= env.action_space.sample()
            obs, reward, done, info = env.step(action)
            dataset['obs_list']=obs
            dataset['reward_list'] = reward
            dataset['done_list'] = done
            #dataset['next_observations'][ep, t, :] = np.random.rand(state_dim)
            dataset['info_list']= info
            dataset['jobs_list'] = 
            dataset['current_time_step_list'] = []
            dataset['costcoils_list'] = []
            dataset['plants_df_list'] = []
            dataset['orders_df_list'] = []
            
            #if env.done:
                #obs = env.reset()

    # Save dataset as pickle file
    with open('reacher_2d.pkl', 'wb') as f:
        pickle.dump(dataset, f)
