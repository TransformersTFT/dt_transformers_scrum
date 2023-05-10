import numpy as np
import pickle
import prueba_nueva_dataset as pnv
dataset_path='env_va-dataset-v3.pkl'

with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
print(trajectories)
        #pickle deserializes the dataset as json does
        # save all path information into separate lists  
states, traj_lens, returns = [], [], []
    #for path in trajectories:
n_episodes=pnv.n_episodes

    # Convert dictionary keys to integers
for key in trajectories.keys():
    trajectories[key] = {i: trajectories[key][i] for i in range(n_episodes)}
observation = trajectories['observations'][0][1]
reward = trajectories['rewards'][0][2]
print("ehhhhhhhhhhhhhhhhhhhhhhhhhhh")
print(observation)
print(reward)

'''
for path in trajectories:
  print(path)
  print("imprimiendo---------------------------------------------")
  keys=trajectories.keys()
        #print(keys)
        
  print(path['reward'][-1])
  
'''